"""
API clients for fetching bibliographic metadata from external sources.
"""

import re
import logging
import httpx
import xml.etree.ElementTree as ET
import arxiv
from typing import Dict, Optional, List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class ArxivAPIClient:
    """Client for fetching metadata from arXiv API."""

    BASE_URL = "http://export.arxiv.org/api/query"

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    def extract_arxiv_id(self, text: str) -> Optional[str]:
        """Extract arXiv ID from text using regex patterns."""
        patterns = [
            r"arXiv:\s*(\d{4}\.\d{4,5}(?:v\d+)?)",  # arXiv: 2502.04780v1 (with space)
            r"arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)",  # arXiv:2502.04780v1 or arXiv:2502.04780
            r"arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)",  # arxiv.org/abs/2502.04780v1
            r"arxiv\.org/pdf/(\d{4}\.\d{4,5}(?:v\d+)?)",  # arxiv.org/pdf/2502.04780v1
            # More flexible patterns to catch arXiv IDs in various contexts
            r"(?:^|\s)(\d{4}\.\d{4,5}(?:v\d+)?)(?:\s|$)",  # Standalone ID with word boundaries
            r"(?:paper|preprint|submission)[\s:]*(\d{4}\.\d{4,5}(?:v\d+)?)",  # After keywords
            r"(?:arxiv|arXiv|ARXIV)[\s:]*(\d{4}\.\d{4,5}(?:v\d+)?)",  # Flexible arXiv prefix
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                arxiv_id = match.group(1)
                # Validate arXiv ID format (YYYY.NNNNN with optional version)
                if re.match(r"^\d{4}\.\d{4,5}(?:v\d+)?$", arxiv_id):
                    # Remove version number for API query if present
                    base_id = re.sub(r"v\d+$", "", arxiv_id)
                    # logger.info(f"Found arXiv ID: {arxiv_id} (base: {base_id})")
                    return base_id

        return None

    async def fetch_metadata(self, arxiv_id: str) -> Optional[Dict]:
        """Fetch metadata for an arXiv paper."""
        try:
            url = f"{self.BASE_URL}?id_list={arxiv_id}"
            response = await self.client.get(url)
            response.raise_for_status()

            return self._parse_arxiv_response(response.text)

        except Exception as e:
            # logger.error(f"Error fetching arXiv metadata for {arxiv_id}: {e}")
            return None

    def _parse_arxiv_response(self, xml_content: str) -> Optional[Dict]:
        """Parse arXiv API XML response."""
        try:
            root = ET.fromstring(xml_content)

            # Namespace for arXiv API
            ns = {
                "atom": "http://www.w3.org/2005/Atom",
                "arxiv": "http://arxiv.org/schemas/atom",
            }

            entry = root.find("atom:entry", ns)
            if entry is None:
                # logger.warning("No entry found in arXiv response")
                return None

            # Extract metadata
            title = entry.find("atom:title", ns)
            title_text = title.text.strip() if title is not None else "Unknown Title"

            # Authors
            authors = []
            for author in entry.findall("atom:author", ns):
                name = author.find("atom:name", ns)
                if name is not None:
                    authors.append(name.text.strip())

            # Abstract
            summary = entry.find("atom:summary", ns)
            abstract_text = summary.text.strip() if summary is not None else None

            # Publication date
            published = entry.find("atom:published", ns)
            publication_year = None
            if published is not None:
                try:
                    date_obj = datetime.fromisoformat(
                        published.text.replace("Z", "+00:00")
                    )
                    publication_year = date_obj.year
                except:
                    pass

            # Categories (for keywords)
            categories = []
            for category in entry.findall("atom:category", ns):
                term = category.get("term")
                if term:
                    categories.append(term)

            # arXiv ID from the ID field
            arxiv_url = entry.find("atom:id", ns)
            arxiv_id = None
            if arxiv_url is not None:
                arxiv_id = arxiv_url.text.split("/")[-1]

            return {
                "title": title_text,
                "authors": authors,
                "abstract": abstract_text,
                "publication_year": publication_year,
                "journal_name": "arXiv preprint",  # arXiv papers are preprints
                "keywords": categories,
                "arxiv_id": arxiv_id,
                "doi": None,  # arXiv papers typically don't have DOIs
                "volume": None,
                "issue": None,
            }

        except Exception as e:
            # logger.error(f"Error parsing arXiv response: {e}")
            return None


class ArxivSearchClient:
    """Client for searching arXiv papers by title using the arxiv package."""

    def __init__(self):
        pass

    async def close(self):
        """No resources to close for arxiv package."""
        pass

    def search_by_title(self, title: str, max_results: int = 5) -> Optional[Dict]:
        """Search arXiv by title and return the best match."""
        try:
            # Clean up title for better search
            clean_title = self._clean_title_for_search(title)

            # Try exact title search first
            results = self._search_arxiv_with_query(f'ti:"{clean_title}"', max_results)

            # If no results, try a more flexible search
            if not results:
                # logger.info("🔍 Exact title search failed, trying flexible search...")
                # Try title search without quotes (allows partial matching)
                results = self._search_arxiv_with_query(
                    f"ti:{clean_title}", max_results
                )

            # If still no results, try all fields search
            if not results:
                # logger.info("🔍 Title field search failed, trying all fields...")
                # Search in all fields
                results = self._search_arxiv_with_query(clean_title, max_results)

            if not results:
                # logger.warning(f"No arXiv papers found for title: {title}")
                return None

            # Sort by similarity and return the best match
            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            best_match = results[0]

            # Only return if similarity is above threshold
            if (
                best_match["similarity_score"] > 0.4
            ):  # Lower threshold for more flexibility
                # logger.info(
                #     f"Found arXiv paper by title search: {best_match['title']} (similarity: {best_match['similarity_score']:.2f})"
                # )
                return best_match
            else:
                # logger.warning(
                #     f"Best match similarity too low: {best_match['similarity_score']:.2f}"
                # )
                return None

        except Exception as e:
            # logger.error(f"Failed to search arXiv by title '{title}': {str(e)}")
            return None

    def _search_arxiv_with_query(self, query: str, max_results: int) -> List[Dict]:
        """Perform arXiv search with a specific query and return results."""
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )

            results = []
            for paper in search.results():
                # Calculate similarity score (simple word matching)
                similarity = self._calculate_title_similarity(
                    query.replace("ti:", "").replace('"', ""), paper.title
                )

                # 서지 정보 구성
                bib_data = {
                    "title": paper.title,
                    "authors": [author.name for author in paper.authors],
                    "arxiv_id": paper.get_short_id(),
                    "published": paper.published.strftime("%Y-%m-%d")
                    if paper.published
                    else None,
                    "updated": paper.updated.strftime("%Y-%m-%d")
                    if paper.updated
                    else None,
                    "journal_name": paper.journal_ref
                    if paper.journal_ref
                    else "arXiv preprint",
                    "doi": paper.doi if paper.doi else None,
                    "abstract": paper.summary,
                    "keywords": paper.categories,
                    "primary_category": paper.primary_category,
                    "similarity_score": similarity,
                    "publication_year": paper.published.year
                    if paper.published
                    else None,
                    "volume": None,
                    "issue": None,
                }
                results.append(bib_data)

            return results
        except Exception as e:
            # logger.warning(f"arXiv search failed for query '{query}': {str(e)}")
            return []

    def _clean_title_for_search(self, title: str) -> str:
        """Clean title for better search results."""
        # Remove common formatting and special characters
        clean = re.sub(r"[^\w\s-]", " ", title)
        clean = re.sub(r"\s+", " ", clean)
        return clean.strip()

    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """Calculate simple word-based similarity between two titles."""
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0


class CrossRefAPIClient:
    """Client for fetching metadata from CrossRef API."""

    BASE_URL = "https://api.crossref.org/works"

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    def extract_doi(self, text: str) -> Optional[str]:
        """Extract DOI from text using regex patterns."""
        patterns = [
            r"doi\.org/(10\.[^\s]+)",  # https://doi.org/10.1080/10509585.2015.1092083
            r"doi:\s*(10\.[^\s]+)",  # doi: 10.1080/10509585.2015.1092083 (with space)
            r"DOI:\s*(10\.[^\s]+)",  # DOI: 10.1080/10509585.2015.1092083
            r"(10\.\d{4,}/[^\s]+)",  # Just the DOI itself
            # New patterns to handle whitespace and line breaks within DOIs
            r"doi\.org/(10\.\d{4,}/\s*[^\s]+)",  # Handle space after publisher prefix
            r"doi:\s*(10\.\d{4,}/\s*[^\s]+)",  # doi: 10.3390/ systems...
            r"DOI:\s*(10\.\d{4,}/\s*[^\s]+)",  # DOI: 10.3390/ systems...
            r"(10\.\d{4,}/\s*[a-zA-Z0-9\-\.]+)",  # Handle space in middle of DOI
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                doi = match.group(1)
                # Clean up DOI - remove whitespace and trailing punctuation
                doi = re.sub(r"\s+", "", doi)  # Remove all whitespace
                doi = doi.rstrip(".,;)")

                # Validate DOI format (10.xxxx/something)
                if re.match(r"^10\.\d{4,}/.+", doi):
                    return doi

        return None

    async def fetch_metadata(self, doi: str) -> Optional[Dict]:
        """Fetch metadata for a DOI."""
        try:
            url = f"{self.BASE_URL}/{doi}"
            headers = {
                "Accept": "application/json",
                "User-Agent": "BetArxiv/1.0 (mailto:your-email@example.com)",
            }

            response = await self.client.get(url, headers=headers)
            response.raise_for_status()

            data = response.json()
            return self._parse_crossref_response(data)

        except Exception as e:
            # logger.error(f"Error fetching CrossRef metadata for {doi}: {e}")
            return None

    def _parse_crossref_response(self, data: Dict) -> Optional[Dict]:
        """Parse CrossRef API JSON response."""
        try:
            work = data.get("message", {})

            # Title
            title_list = work.get("title", [])
            title = title_list[0] if title_list else "Unknown Title"

            # Authors
            authors = []
            for author in work.get("author", []):
                given = author.get("given", "")
                family = author.get("family", "")
                if given and family:
                    authors.append(f"{given} {family}")
                elif family:
                    authors.append(family)

            # Abstract
            abstract_list = work.get("abstract")
            abstract = None
            if abstract_list:
                # Remove HTML tags from abstract
                abstract = re.sub(r"<[^>]+>", "", abstract_list)

            # Journal name
            journal_names = work.get("container-title", [])
            journal_name = journal_names[0] if journal_names else None

            # Publication year
            publication_year = None
            date_parts = work.get("published-print", {}).get("date-parts") or work.get(
                "published-online", {}
            ).get("date-parts")
            if date_parts and len(date_parts[0]) > 0:
                publication_year = date_parts[0][0]

            # Volume and issue
            volume = work.get("volume")
            issue = work.get("issue")

            # DOI
            doi = work.get("DOI")

            # Keywords/subjects
            keywords = work.get("subject", [])

            return {
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "publication_year": publication_year,
                "journal_name": journal_name,
                "volume": volume,
                "issue": issue,
                "keywords": keywords,
                "doi": doi,
                "arxiv_id": None,  # CrossRef papers typically don't have arXiv IDs
            }

        except Exception as e:
            # logger.error(f"Error parsing CrossRef response: {e}")
            return None


class IdentifierExtractor:
    """Main class for extracting and fetching metadata using arXiv or DOI."""

    def __init__(self):
        self.arxiv_client = ArxivAPIClient()
        self.arxiv_search_client = ArxivSearchClient()
        self.crossref_client = CrossRefAPIClient()

    async def close(self):
        """Close all API clients."""
        await self.arxiv_client.close()
        await self.arxiv_search_client.close()
        await self.crossref_client.close()

    def extract_identifiers(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract both arXiv ID and DOI from text."""
        arxiv_id = self.arxiv_client.extract_arxiv_id(text)
        doi = self.crossref_client.extract_doi(text)
        return arxiv_id, doi

    async def fetch_metadata_by_identifier(self, text: str) -> Optional[Dict]:
        """
        Extract identifiers from text and fetch metadata.
        Tries arXiv first, then DOI.
        """
        arxiv_id, doi = self.extract_identifiers(text)

        # Try arXiv first (typically more reliable for preprints)
        if arxiv_id:
            # logger.info(f"Found arXiv ID: {arxiv_id}")
            metadata = await self.arxiv_client.fetch_metadata(arxiv_id)
            if metadata:
                return metadata

        # Try DOI if arXiv didn't work
        if doi:
            # logger.info(f"Found DOI: {doi}")
            metadata = await self.crossref_client.fetch_metadata(doi)
            if metadata:
                return metadata

        # logger.warning("No valid identifiers found or metadata retrieval failed")
        return None

    async def fetch_metadata_by_title(self, title: str) -> Optional[Dict]:
        """
        Search for metadata using title-based arXiv search.
        This is useful when no identifiers are found in the text.
        """
        if not title or len(title.strip()) < 5:
            # logger.warning("Title too short or empty for search")
            return None

        # logger.info(f"Searching arXiv by title: {title}")
        # ArxivSearchClient.search_by_title is synchronous, no await needed
        metadata = self.arxiv_search_client.search_by_title(title)
        return metadata

    async def fetch_metadata_comprehensive(
        self, text: str, title: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Comprehensive metadata extraction strategy:
        1. Try to find identifiers in text and fetch metadata
        2. If no identifiers found, try title-based arXiv search
        3. Return None if both fail (caller can fallback to LLM)
        """
        # Strategy 1: Try identifier-based extraction
        metadata = await self.fetch_metadata_by_identifier(text)
        if metadata:
            return metadata

        # Strategy 2: Try title-based arXiv search
        # If title is not provided, try to extract it from text
        search_title = title
        if not search_title:
            search_title = self._extract_title_from_text(text)

        if search_title:
            # logger.info(f"🔍 Attempting title-based search with: {search_title}")
            metadata = await self.fetch_metadata_by_title(search_title)
            if metadata:
                return metadata

        # logger.warning(
        #     "Both identifier-based and title-based metadata extraction failed"
        # )
        return None

    def _extract_title_from_text(self, text: str) -> Optional[str]:
        """Extract title from text using simple heuristics."""
        lines = text.split("\n")

        # Look for title patterns
        for i, line in enumerate(lines):
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Look for markdown headers
            if line.startswith("#"):
                title = line.lstrip("#").strip()
                if len(title) > 10 and len(title) < 200:  # Reasonable title length
                    return title

            # Look for lines that might be titles (early in document, reasonable length)
            if i < 20 and len(line) > 10 and len(line) < 200:
                # Skip lines that look like metadata or citations
                if not any(
                    pattern in line.lower()
                    for pattern in [
                        "abstract",
                        "author",
                        "doi:",
                        "arxiv:",
                        "journal",
                        "volume",
                    ]
                ):
                    # Check if it's not all caps (which might be section headers)
                    if not line.isupper():
                        return line

        return None
