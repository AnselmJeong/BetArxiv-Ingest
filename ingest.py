#!/usr/bin/env python3
"""
Combined PDF ingestion and processing script.
This script handles batch processing of PDFs from a directory.
"""

import warnings
import os
import asyncio
from pathlib import Path
import argparse
from tqdm import tqdm
import dotenv
import logging
from typing import Optional, List
import re
import json
from pydantic import ValidationError

from docling.document_converter import DocumentConverter
from google import genai
from google.genai import types
from pypdf import PdfReader

from utils.models import DocumentCreate
from utils.db import Database
from utils.api_clients import IdentifierExtractor
from pydantic import BaseModel

dotenv.load_dotenv()

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DSN = os.getenv("DATABASE_URL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_EMBED_MODEL = "gemini-embedding-exp-03-07"
GEMINI_MODEL = "gemini-2.5-flash-preview-05-20"


class PaperMetadata(BaseModel):
    title: str
    authors: List[str]
    journal_name: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    publication_year: Optional[int] = None
    abstract: Optional[str] = None
    keywords: Optional[List[str]] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None


async def pdf_to_markdown(pdf_path: str) -> str:
    """Use docling to convert PDF to Markdown"""
    try:
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        markdown = result.document.export_to_markdown()
        return markdown
    except Exception as e:
        logger.error(f"Docling failed for {pdf_path}: {e}")
        raise


async def extract_pdf_first_page_text(pdf_path: str) -> Optional[str]:
    """Extract text from the first page of a PDF using pypdf."""
    try:
        reader = PdfReader(pdf_path)
        if len(reader.pages) == 0:
            logger.warning(f"PDF has no pages: {pdf_path}")
            return None

        # Extract text from first page
        first_page = reader.pages[0]
        text = first_page.extract_text()

        if not text or len(text.strip()) < 50:  # Ensure we got meaningful text
            logger.warning(f"First page text too short or empty: {pdf_path}")
            return None

        logger.info(
            f"📄 Extracted {len(text)} characters from first page of {pdf_path}"
        )

        # Save debug text file
        # await save_debug_text(pdf_path, text)

        return text

    except Exception as e:
        logger.error(f"Failed to extract first page text from {pdf_path}: {e}")
        return None


async def save_debug_text(pdf_path: str, text: str) -> None:
    """Save extracted PDF text to debug file."""
    try:
        # Create debug directory if it doesn't exist
        debug_dir = Path("debug_pdf_texts")
        debug_dir.mkdir(exist_ok=True)

        # Create filename based on PDF path
        pdf_file = Path(pdf_path)
        debug_filename = f"{pdf_file.stem}_first_page.txt"
        debug_path = debug_dir / debug_filename

        # Write text to debug file
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(f"=== DEBUG: First page text from {pdf_path} ===\n\n")
            f.write(text)
            f.write(f"\n\n=== End of text ({len(text)} characters) ===\n")

        logger.info(f"💾 Saved debug text to: {debug_path}")

    except Exception as e:
        logger.warning(f"Failed to save debug text for {pdf_path}: {e}")
        # Don't raise exception as this is just debug functionality


async def extract_metadata_from_first_page(
    pdf_path: str, identifier_extractor: IdentifierExtractor
) -> Optional[dict]:
    """Extract metadata by looking for identifiers in PDF first page."""
    try:
        # Extract text from first page
        first_page_text = await extract_pdf_first_page_text(pdf_path)
        if not first_page_text:
            return None

        logger.info("🔍 Searching for identifiers in PDF first page...")

        # Try to find identifiers and fetch metadata
        metadata = await identifier_extractor.fetch_metadata_by_identifier(
            first_page_text
        )
        if metadata:
            logger.info(
                "✅ Successfully extracted metadata from first page identifiers"
            )
            return metadata

        # If no identifiers found, try title-based search with first page
        logger.info("🔍 No identifiers in first page, trying title extraction...")
        title = extract_title_from_text(first_page_text)

        if title:
            logger.info(f"📖 Extracted title from first page: {title}")
            metadata = await identifier_extractor.fetch_metadata_by_title(title)
            if metadata:
                logger.info(
                    "✅ Successfully found metadata via first page title search"
                )
                return metadata

        logger.info("❌ First page metadata extraction failed")
        return None

    except Exception as e:
        logger.warning(f"⚠️ Failed to extract metadata from first page: {e}")
        return None


def extract_title_from_text(text: str) -> Optional[str]:
    """Extract title from raw text using enhanced heuristics."""
    lines = text.split("\n")

    # Look for title patterns in first few lines
    for i, line in enumerate(lines[:15]):  # Check first 15 lines
        line = line.strip()

        # Skip empty lines or very short lines
        if not line or len(line) < 10:
            continue

        # Skip lines that look like metadata or headers
        skip_patterns = [
            "abstract",
            "author",
            "doi:",
            "arxiv:",
            "journal",
            "volume",
            "issue",
            "received",
            "accepted",
            "published",
            "copyright",
            "©",
            "university",
            "department",
            "email",
            "@",
            "http",
            "www",
            "proceedings",
            "conference",
        ]

        if any(pattern in line.lower() for pattern in skip_patterns):
            continue

        # Skip lines that are all caps (likely section headers)
        if line.isupper():
            continue

        # Skip lines with too many numbers (likely dates/references)
        if sum(c.isdigit() for c in line) > len(line) * 0.3:
            continue

        # Look for reasonable title length
        if 15 <= len(line) <= 200:
            # Additional checks for title-like characteristics
            # Titles usually have proper capitalization
            words = line.split()
            if len(words) >= 3:  # At least 3 words
                # Check if it looks like a title (proper case, not all lowercase)
                if not line.islower() and any(
                    word[0].isupper() for word in words if word
                ):
                    return line

    return None


async def extract_metadata_from_identifiers_and_title(
    markdown: str, identifier_extractor: IdentifierExtractor
) -> Optional[dict]:
    """Try to extract metadata using identifiers and title-based search."""
    try:
        # First, try to extract using identifiers from the full markdown
        metadata = await identifier_extractor.fetch_metadata_by_identifier(markdown)
        if metadata:
            logger.info("✅ Successfully extracted metadata from identifiers")
            return metadata

        # If no identifiers found, try to extract title first and search by title
        logger.info(
            "🔍 No identifiers found, attempting title extraction for arXiv search"
        )

        # Use a simple regex to extract potential title from markdown
        title = extract_title_from_text(markdown)

        if title:
            logger.info(f"📖 Extracted title for search: {title}")
            metadata = await identifier_extractor.fetch_metadata_by_title(title)
            if metadata:
                logger.info(
                    "✅ Successfully found metadata via title-based arXiv search"
                )
                return metadata

        logger.info("❌ Both identifier and title-based searches failed")
        return None

    except Exception as e:
        logger.warning(f"⚠️ Failed to extract metadata from identifiers/title: {e}")
        return None


async def extract_metadata_llm_fallback(
    markdown: str, genai_client: genai.Client
) -> dict:
    """Fallback LLM-based metadata extraction when identifier-based fails."""
    prompt = f"""
Given the following research paper in Markdown format, extract the following fields as JSON:
- title
- authors (as a list)
- journal_name
- volume
- issue
- publication_year
- abstract
- keywords (as a list)
- doi (if mentioned, extract the DOI identifier)
- arxiv_id (if mentioned, extract the arXiv identifier like "2502.04780v1")

Look carefully for DOI and arXiv identifiers in the text. DOI usually appears as "doi:10.xxxx" or "https://doi.org/10.xxxx". 
arXiv ID usually appears as "arXiv:2024.xxxxx" or similar patterns.

Return only valid JSON matching this schema. Do not include any explanation or extra text.

Markdown:
{markdown}
"""

    # Default values in case of extraction failure
    default_metadata = {
        "title": "Unknown Title",
        "authors": ["Unknown Author"],
        "journal_name": None,
        "volume": None,
        "issue": None,
        "publication_year": None,
        "abstract": None,
        "keywords": [],
        "doi": None,
        "arxiv_id": None,
    }

    try:
        response = await genai_client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PaperMetadata,
            ),
        )
        raw_content = response.text
        logger.debug(f"LLM raw response: {raw_content}")

        # Try to validate with Pydantic
        try:
            data = PaperMetadata.model_validate_json(raw_content).model_dump()
            logger.info("✅ LLM metadata extraction successful")
            return data
        except (ValidationError, json.JSONDecodeError) as validation_error:
            logger.warning(
                f"⚠️ JSON validation failed, trying to parse manually: {validation_error}"
            )

            # Try to parse JSON manually and extract what we can
            try:
                raw_data = json.loads(raw_content)
                extracted_data = {}
                for key in default_metadata.keys():
                    extracted_data[key] = raw_data.get(key, default_metadata[key])

                # Ensure authors is a list
                if not isinstance(extracted_data["authors"], list):
                    if isinstance(extracted_data["authors"], str):
                        extracted_data["authors"] = [extracted_data["authors"]]
                    else:
                        extracted_data["authors"] = default_metadata["authors"]

                # Ensure keywords is a list
                if extracted_data["keywords"] and not isinstance(
                    extracted_data["keywords"], list
                ):
                    if isinstance(extracted_data["keywords"], str):
                        extracted_data["keywords"] = [extracted_data["keywords"]]
                    else:
                        extracted_data["keywords"] = []

                logger.info("✅ Partial LLM metadata extraction successful")
                return extracted_data

            except json.JSONDecodeError:
                logger.warning("⚠️ Could not parse JSON at all, using defaults")
                return default_metadata

    except Exception as e:
        logger.error(f"❌ LLM metadata extraction failed completely: {e}")
        return default_metadata


async def extract_metadata(
    markdown: str,
    genai_client: genai.Client,
    identifier_extractor: IdentifierExtractor,
    pdf_path: str = None,
) -> dict:
    """Main metadata extraction function that tries PDF first page, then identifiers, title search, then LLM fallback."""

    # Strategy 1: Try to extract using PDF first page (if pdf_path provided)
    if pdf_path:
        logger.info("📄 Trying PDF first page metadata extraction...")
        metadata = await extract_metadata_from_first_page(
            pdf_path, identifier_extractor
        )
        if metadata:
            logger.info("📡 Using metadata from PDF first page")

            # Check if we're missing important fields and supplement with LLM if needed
            missing_fields = []
            if not metadata.get("abstract"):
                missing_fields.append("abstract")
            if not metadata.get("keywords"):
                missing_fields.append("keywords")

            if missing_fields:
                logger.info(
                    f"🔍 Supplementing missing fields with LLM: {missing_fields}"
                )
                llm_metadata = await extract_metadata_llm_fallback(
                    markdown, genai_client
                )

                # Fill in missing fields from LLM
                for field in missing_fields:
                    if llm_metadata.get(field):
                        metadata[field] = llm_metadata[field]

            return metadata

    # Strategy 2: Try to extract using identifiers and title-based search from markdown
    metadata = await extract_metadata_from_identifiers_and_title(
        markdown, identifier_extractor
    )

    if metadata:
        # If we got metadata from API, we still might want to enhance it with LLM for missing fields
        logger.info("📡 Using metadata from markdown identifiers")

        # Check if we're missing important fields and supplement with LLM if needed
        missing_fields = []
        if not metadata.get("abstract"):
            missing_fields.append("abstract")
        if not metadata.get("keywords"):
            missing_fields.append("keywords")

        if missing_fields:
            logger.info(f"🔍 Supplementing missing fields with LLM: {missing_fields}")
            llm_metadata = await extract_metadata_llm_fallback(markdown, genai_client)

            # Fill in missing fields from LLM
            for field in missing_fields:
                if llm_metadata.get(field):
                    metadata[field] = llm_metadata[field]

        return metadata

    # Strategy 3: Fallback to LLM-based extraction
    logger.info("🤖 Falling back to LLM-based metadata extraction")
    return await extract_metadata_llm_fallback(markdown, genai_client)


async def get_embedding(text: str, genai_client: genai.Client) -> List[float]:
    """Use Google GenAI to get embedding"""
    # Handle empty or very short text
    if not text or len(text.strip()) < 3:
        logger.warning("Text too short for embedding, returning zero vector")
        return [0.0] * 768

    try:
        response = await genai_client.aio.models.embed_content(
            model=GEMINI_EMBED_MODEL,
            contents=text.strip(),
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=768,
            ),
        )
        embeddings = response.embeddings[0].values
        logger.debug(f"[DEBUG] Embedding response: {embeddings[:5]}...")
        return embeddings
    except Exception as e:
        logger.error(f"Embedding generation failed for text (len={len(text)}): {e}")
        # Fallback: return a zero vector of the expected dimension
        return [0.0] * 768


def ensure_vector(vec, dim=768):
    """Ensure vector is of the correct dimension"""
    if not isinstance(vec, list) or not all(isinstance(x, (float, int)) for x in vec):
        return [0.0] * dim
    if len(vec) != dim:
        return [0.0] * dim
    return vec


async def process_pdf(
    pdf_path: str,
    genai_client: genai.Client,
    base_dir: str,
    identifier_extractor: IdentifierExtractor,
) -> DocumentCreate:
    """Process a PDF file and extract metadata, content, and generate embeddings"""
    pdf_file = Path(pdf_path)
    base_path = Path(base_dir)

    # Extract folder name and file path relative to base_dir
    try:
        # Calculate relative paths
        relative_pdf_path = str(pdf_file.relative_to(base_path))
        folder_name = str(pdf_file.parent.relative_to(base_path))
        if folder_name == ".":
            folder_name = None
    except ValueError:
        # PDF is outside the base directory - use absolute paths as fallback
        relative_pdf_path = str(pdf_file)
        folder_name = str(pdf_file.parent)

    markdown = await pdf_to_markdown(pdf_path)
    # Remove references section and everything after (robust heading/line match)
    pattern = re.compile(
        r"^(#{1,6}\s*)?(references|reference|bibliography|참고문헌)\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    match = pattern.search(markdown)
    if match:
        markdown = markdown[: match.start()].rstrip()

    meta = await extract_metadata(
        markdown, genai_client, identifier_extractor, pdf_path
    )
    title_emb = await get_embedding(meta.get("title", ""), genai_client)
    abstract_emb = await get_embedding(meta.get("abstract", ""), genai_client)

    # Ensure vectors are of the correct dimension
    title_emb = ensure_vector(title_emb, 768)
    abstract_emb = ensure_vector(abstract_emb, 768)

    return DocumentCreate(
        title=meta.get("title", "Unknown"),
        authors=meta.get("authors", []),
        journal_name=meta.get("journal_name"),
        volume=meta.get("volume"),
        issue=meta.get("issue"),
        publication_year=meta.get("publication_year"),
        abstract=meta.get("abstract"),
        keywords=meta.get("keywords", []),
        doi=meta.get("doi"),
        arxiv_id=meta.get("arxiv_id"),
        markdown=markdown,
        summary=None,  # Summary will be generated separately via API
        previous_work=None,
        hypothesis=None,
        distinction=None,
        methodology=None,
        results=None,
        limitations=None,
        implications=None,
        title_embedding=title_emb,
        abstract_embedding=abstract_emb,
        status="processed",
        folder_name=folder_name,
        url=relative_pdf_path,
        rating=None,  # Rating will be set by users via inspect page
    )


def get_all_pdf_paths(base_dir):
    """Get all PDF file paths in a directory recursively"""
    return [str(p) for p in Path(base_dir).rglob("*.pdf")]


async def get_already_ingested_paths(db):
    """Query all URLs (file paths) from the DB"""
    query = "SELECT url FROM documents WHERE url IS NOT NULL"
    async with db.pool.cursor() as cur:
        await cur.execute(query)
        rows = await cur.fetchall()
        return set(row["url"] for row in rows if row["url"])


def convert_to_relative_paths(absolute_paths, base_dir):
    """Convert absolute paths to relative paths for comparison"""
    base_path = Path(base_dir)
    relative_paths = set()

    for abs_path in absolute_paths:
        try:
            # Try to convert to relative path
            rel_path = str(Path(abs_path).relative_to(base_path))
            relative_paths.add(rel_path)
        except ValueError:
            # If path is outside base_dir, keep as absolute
            relative_paths.add(abs_path)

    return relative_paths


async def main(data_root):
    """Main ingestion function"""
    print(f"[DEBUG] Using DATABASE_URL: {DSN}")

    # Check for Google API key
    if not GOOGLE_API_KEY:
        logger.error("❌ GOOGLE_API_KEY environment variable is not set")
        return

    db = Database(DSN)
    await db.connect()

    # Initialize Google GenAI client with API key
    genai_client = genai.Client(api_key=GOOGLE_API_KEY)

    # Initialize identifier extractor for arXiv/DOI lookups
    identifier_extractor = IdentifierExtractor()

    try:
        # Get all PDF absolute paths
        all_pdf_absolute_paths = set(get_all_pdf_paths(data_root))
        print(
            f"[DEBUG] Found {len(all_pdf_absolute_paths)} PDFs in '{data_root}' (including subdirectories)"
        )

        # Convert to relative paths for comparison with DB
        all_pdf_relative_paths = convert_to_relative_paths(
            all_pdf_absolute_paths, data_root
        )

        # Get already ingested relative paths from DB
        already_ingested_relative = set(await get_already_ingested_paths(db))
        print(
            f"[DEBUG] Found {len(already_ingested_relative)} already-ingested PDFs in DB"
        )

        # Find new relative paths to process
        new_relative_paths = all_pdf_relative_paths - already_ingested_relative

        if not new_relative_paths:
            print("No new PDFs to ingest.")
            return

        print(f"Found {len(new_relative_paths)} new PDFs to ingest in '{data_root}'.")

        # Convert back to absolute paths for processing
        base_path = Path(data_root)
        new_absolute_paths = []
        for rel_path in new_relative_paths:
            if Path(rel_path).is_absolute():
                # Already absolute path (outside base_dir)
                new_absolute_paths.append(rel_path)
            else:
                # Convert relative to absolute
                abs_path = str(base_path / rel_path)
                new_absolute_paths.append(abs_path)

        # Keep track of processing statistics
        processed_count = 0
        failed_count = 0

        for pdf_path in tqdm(sorted(new_absolute_paths), desc="Ingesting PDFs"):
            try:
                logger.info(f"🔄 Processing: {pdf_path}")
                document_data = await process_pdf(
                    pdf_path, genai_client, data_root, identifier_extractor
                )
                document_id = await db.insert_document(document_data)
                await db.update_paper_status(document_id, "processed")
                processed_count += 1
                logger.info(f"✅ Successfully ingested: {pdf_path} (ID: {document_id})")

            except Exception as e:
                failed_count += 1
                error_type = type(e).__name__

                # Categorize errors for better debugging
                if "docling" in str(e).lower() or "convert" in str(e).lower():
                    logger.error(f"📄 PDF conversion failed for {pdf_path}: {e}")
                elif "genai" in str(e).lower() or "connection" in str(e).lower():
                    logger.error(f"🤖 LLM service error for {pdf_path}: {e}")
                elif "database" in str(e).lower() or "postgres" in str(e).lower():
                    logger.error(f"💾 Database error for {pdf_path}: {e}")
                else:
                    logger.error(f"❌ Unknown error for {pdf_path} ({error_type}): {e}")

                # Print traceback only for debugging (comment out in production)
                logger.debug(f"Full traceback for {pdf_path}:", exc_info=True)

                # Continue to next file
                continue

        # Print final statistics
        total_attempted = processed_count + failed_count
        print("\n📊 Processing Summary:")
        print(f"   Total attempted: {total_attempted}")
        print(f"   ✅ Successfully processed: {processed_count}")
        print(f"   ❌ Failed: {failed_count}")

        if failed_count > 0:
            success_rate = (processed_count / total_attempted) * 100
            print(f"   📈 Success rate: {success_rate:.1f}%")

    finally:
        # Clean up resources
        await identifier_extractor.close()
        await db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest new PDFs from a data root directory."
    )
    parser.add_argument(
        "--directory",
        "-d",
        type=str,
        default="docs",
        help="Root directory to scan for PDFs (default: 'docs')",
    )
    args = parser.parse_args()
    asyncio.run(main(args.directory))
