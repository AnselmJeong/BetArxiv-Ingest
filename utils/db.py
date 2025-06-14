import logging
from typing import List, Optional, Any, Dict
from uuid import UUID

import os
import json

import psycopg
from psycopg.rows import dict_row

from .models import (
    DocumentCreate,
    Document,
    DocumentListItem,
    DocumentMetadata,
    DocumentSummary,
    DocumentEmbedding,
    FolderInfo,
    UpdateSummaryRequest,
    UpdateMetadataRequest,
    DocumentListResponse,
    UpdateRatingRequest,
)

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.pool: Optional[psycopg.AsyncConnection] = None

    async def connect(self):
        self.pool = await psycopg.AsyncConnection.connect(
            self.dsn, autocommit=True, row_factory=dict_row
        )
        # logger.info("Connected to PostgreSQL.")

    async def close(self):
        if self.pool:
            await self.pool.close()
            # logger.info("Closed PostgreSQL connection.")

    # Document operations
    async def insert_document(self, document: DocumentCreate) -> UUID:
        """Insert a new document into the database."""
        query = """
            INSERT INTO documents (
                title, authors, journal_name, publication_year, abstract,
                keywords, volume, issue, url, doi, arxiv_id, markdown, summary,
                previous_work, hypothesis, distinction, methodology, results, limitations, implications,
                title_embedding, abstract_embedding, status, folder_name
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            ) RETURNING id
        """
        async with self.pool.cursor() as cur:
            await cur.execute(
                query,
                (
                    document.title,
                    document.authors,
                    document.journal_name,
                    document.publication_year,
                    document.abstract,
                    document.keywords,
                    document.volume,
                    document.issue,
                    document.url,
                    document.doi,
                    document.arxiv_id,
                    document.markdown,
                    document.summary,
                    document.previous_work,
                    document.hypothesis,
                    document.distinction,
                    document.methodology,
                    document.results,
                    document.limitations,
                    document.implications,
                    document.title_embedding,
                    document.abstract_embedding,
                    document.status,
                    document.folder_name,
                ),
            )
            row = await cur.fetchone()
            return row["id"]

    async def get_document(self, document_id: UUID) -> Optional[Document]:
        """Get a document by ID."""
        query = """
            SELECT id, title, authors, journal_name, publication_year,
                   abstract, keywords, volume, issue, url, doi, arxiv_id, markdown,
                   summary, previous_work, hypothesis, distinction, methodology, results, limitations, implications,
                   title_embedding, abstract_embedding, status, folder_name
            FROM documents
            WHERE id = %s
        """
        async with self.pool.cursor() as cur:
            await cur.execute(query, (str(document_id),))
            row = await cur.fetchone()
            if row:
                row_dict = dict(row)
                # Parse embeddings from JSON strings if they exist
                if row_dict.get("title_embedding"):
                    row_dict["title_embedding"] = (
                        json.loads(row_dict["title_embedding"])
                        if isinstance(row_dict["title_embedding"], str)
                        else row_dict["title_embedding"]
                    )
                if row_dict.get("abstract_embedding"):
                    row_dict["abstract_embedding"] = (
                        json.loads(row_dict["abstract_embedding"])
                        if isinstance(row_dict["abstract_embedding"], str)
                        else row_dict["abstract_embedding"]
                    )
                return Document(**row_dict)
            return None

    async def get_document_metadata(
        self, document_id: UUID
    ) -> Optional[DocumentMetadata]:
        """Get document metadata by ID."""
        query = """
            SELECT title, authors, journal_name, publication_year,
                   abstract, keywords, volume, issue, url, doi, arxiv_id, markdown, rating
            FROM documents
            WHERE id = %s
        """
        async with self.pool.cursor() as cur:
            await cur.execute(query, (str(document_id),))
            row = await cur.fetchone()
            if row:
                return DocumentMetadata(**dict(row))
            return None

    async def get_document_summary(
        self, document_id: UUID
    ) -> Optional[DocumentSummary]:
        query = """
        SELECT summary, previous_work, hypothesis, distinction, methodology, results, limitations, implications 
        FROM documents WHERE id=%s
        """
        async with self.pool.cursor() as cur:
            await cur.execute(query, (str(document_id),))
            row = await cur.fetchone()
            if row:
                return DocumentSummary(**row)
            return None

    async def get_document_embedding(
        self, document_id: UUID
    ) -> Optional[DocumentEmbedding]:
        query = "SELECT title_embedding, abstract_embedding FROM documents WHERE id=%s"
        async with self.pool.cursor() as cur:
            await cur.execute(query, (str(document_id),))
            row = await cur.fetchone()
            if row:
                row_dict = dict(row)
                # Parse embeddings from JSON strings if they exist
                if row_dict.get("title_embedding"):
                    row_dict["title_embedding"] = (
                        json.loads(row_dict["title_embedding"])
                        if isinstance(row_dict["title_embedding"], str)
                        else row_dict["title_embedding"]
                    )
                if row_dict.get("abstract_embedding"):
                    row_dict["abstract_embedding"] = (
                        json.loads(row_dict["abstract_embedding"])
                        if isinstance(row_dict["abstract_embedding"], str)
                        else row_dict["abstract_embedding"]
                    )
                return DocumentEmbedding(**row_dict)
            return None

    async def update_document_summary(
        self, document_id: UUID, summary_data: UpdateSummaryRequest
    ) -> bool:
        fields = []
        values = []
        for field, value in summary_data.model_dump(exclude_unset=True).items():
            if value is not None:
                fields.append(f"{field}=%s")
                values.append(value)

        if not fields:
            return False

        query = (
            f"UPDATE documents SET {', '.join(fields)}, updated_at=NOW() WHERE id=%s"
        )
        values.append(str(document_id))

        async with self.pool.cursor() as cur:
            await cur.execute(query, values)
            return cur.rowcount > 0

    async def update_document_metadata(
        self, document_id: UUID, metadata_data: UpdateMetadataRequest
    ) -> bool:
        data = metadata_data.model_dump(exclude_unset=True)
        fields = []
        values = []

        for field, value in data.items():
            if value is not None:
                fields.append(f"{field}=%s")
                values.append(value)

        if not fields:
            return False

        query = (
            f"UPDATE documents SET {', '.join(fields)}, updated_at=NOW() WHERE id=%s"
        )
        values.append(str(document_id))

        async with self.pool.cursor() as cur:
            await cur.execute(query, values)
            return cur.rowcount > 0

    async def update_document_rating(
        self, document_id: UUID, rating_data: UpdateRatingRequest
    ) -> bool:
        """Update the rating for a specific document."""
        query = "UPDATE documents SET rating=%s, updated_at=NOW() WHERE id=%s"
        values = [rating_data.rating, str(document_id)]

        async with self.pool.cursor() as cur:
            await cur.execute(query, values)
            return cur.rowcount > 0

    async def list_documents(
        self,
        skip: int = 0,
        limit: int = 50,
        folder_name: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> DocumentListResponse:
        """List documents with optional filtering."""
        where_conditions = []
        params = []

        if folder_name:
            where_conditions.append("folder_name = %s")
            params.append(folder_name)

        if filters:
            for key, value in filters.items():
                if key in [
                    "title",
                    "authors",
                    "journal_name",
                    "publication_year",
                    "keywords",
                ]:
                    where_conditions.append(f"{key} = %s")
                    params.append(value)

        where_clause = (
            f"WHERE {' AND '.join(where_conditions)}" if where_conditions else ""
        )

        count_query = f"SELECT COUNT(*) FROM documents {where_clause}"
        query = f"""
            SELECT id, title, authors, journal_name, publication_year,
                   volume, issue, url, abstract, keywords, folder_name, doi, arxiv_id, rating
            FROM documents
            {where_clause}
            ORDER BY created_at DESC, title
            LIMIT %s OFFSET %s
        """

        async with self.pool.cursor() as cur:
            await cur.execute(count_query, params)
            total_row = await cur.fetchone()
            total = total_row["count"]

            await cur.execute(query, params + [limit, skip])
            rows = await cur.fetchall()
            documents = [DocumentListItem(**dict(row)) for row in rows]

            return DocumentListResponse(
                documents=documents, total=total, skip=skip, limit=limit
            )

    async def search_documents(
        self,
        query: str,
        folder_name: Optional[str] = None,
        k: int = 4,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search documents using semantic vector similarity."""
        from .utils import get_genai_client, get_embedding

        try:
            # Generate embedding for the search query
            genai_client = get_genai_client()
            query_embedding = await get_embedding(query, genai_client)

            where_conditions = [
                "title_embedding IS NOT NULL AND abstract_embedding IS NOT NULL"
            ]
            where_params = []

            if folder_name:
                where_conditions.append("folder_name = %s")
                where_params.append(folder_name)

            if filters:
                for key, value in filters.items():
                    if key in [
                        "title",
                        "authors",
                        "journal_name",
                        "publication_year",
                        "keywords",
                    ]:
                        where_conditions.append(f"{key} = %s")
                        where_params.append(value)

            where_clause = f"WHERE {' AND '.join(where_conditions)}"

            # Use vector similarity search with both title and abstract embeddings
            search_query = f"""
                WITH similarity_scores AS (
                    SELECT 
                        id,
                        title,
                        abstract,
                        authors,
                        journal_name,
                        publication_year,
                        folder_name,
                        keywords,
                        url,
                        (
                            0.7 * (1 - (title_embedding <=> %s::vector)) +
                            0.3 * (1 - (abstract_embedding <=> %s::vector))
                        ) as similarity_score
                    FROM documents
                    {where_clause}
                )
                SELECT *
                FROM similarity_scores
                WHERE similarity_score >= 0.3  -- Minimum similarity threshold
                ORDER BY similarity_score DESC
                LIMIT %s
            """

            # Parameters: query_embedding (twice for title and abstract), where_params, limit
            search_params = [query_embedding, query_embedding] + where_params + [k]

            async with self.pool.cursor() as cur:
                await cur.execute(search_query, search_params)
                rows = await cur.fetchall()

                results = []
                for row in rows:
                    # Generate snippet from abstract
                    snippet = None
                    if row.get("abstract"):
                        snippet = row["abstract"]
                        # if len(abstract) > 200:
                        #     snippet = abstract[:200] + "..."
                        # else:
                        #     snippet = abstract

                    results.append(
                        {
                            "id": row["id"],
                            "title": row["title"],
                            "authors": row["authors"],
                            "journal_name": row.get("journal_name"),
                            "publication_year": row.get("publication_year"),
                            "folder_name": row.get("folder_name"),
                            "keywords": row.get("keywords"),
                            "similarity_score": round(row["similarity_score"], 3),
                            "snippet": snippet,
                            "url": row.get("url"),
                        }
                    )

                return results

        except Exception as e:
            # logger.error(f"Semantic search failed: {e}")
            # Fallback to simple text search
            return await self._fallback_text_search(query, folder_name, k, filters)

    async def _fallback_text_search(
        self,
        query: str,
        folder_name: Optional[str] = None,
        k: int = 4,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Fallback text search when vector search fails."""
        where_conditions = []
        params = []

        if folder_name:
            where_conditions.append("folder_name = %s")
            params.append(folder_name)

        if filters:
            for key, value in filters.items():
                if key in [
                    "title",
                    "authors",
                    "journal_name",
                    "publication_year",
                    "keywords",
                ]:
                    where_conditions.append(f"{key} = %s")
                    params.append(value)

        where_clause = (
            f"AND {' AND '.join(where_conditions)}" if where_conditions else ""
        )

        search_query = f"""
            SELECT id, title, authors, abstract, journal_name, publication_year, folder_name, keywords, url,
                   (
                       CASE WHEN title ILIKE %s THEN 0.8 ELSE 0 END +
                       CASE WHEN abstract ILIKE %s THEN 0.5 ELSE 0 END +
                       CASE WHEN EXISTS (SELECT 1 FROM unnest(authors) a WHERE a ILIKE %s) THEN 0.3 ELSE 0 END
                   ) as similarity_score
            FROM documents
            WHERE (title ILIKE %s OR abstract ILIKE %s OR EXISTS (SELECT 1 FROM unnest(authors) a WHERE a ILIKE %s))
            {where_clause}
            ORDER BY similarity_score DESC
            LIMIT %s
        """

        search_term = f"%{query}%"
        search_params = (
            [
                search_term,
                search_term,
                search_term,
                search_term,
                search_term,
                search_term,
            ]
            + params
            + [k]
        )

        async with self.pool.cursor() as cur:
            await cur.execute(search_query, search_params)
            rows = await cur.fetchall()

            results = []
            for row in rows:
                # Generate snippet from abstract
                snippet = None
                if row.get("abstract"):
                    abstract = row["abstract"]
                    if len(abstract) > 200:
                        snippet = abstract[:200] + "..."
                    else:
                        snippet = abstract

                results.append(
                    {
                        "id": row["id"],
                        "title": row["title"],
                        "authors": row["authors"],
                        "journal_name": row.get("journal_name"),
                        "publication_year": row.get("publication_year"),
                        "folder_name": row.get("folder_name"),
                        "keywords": row.get("keywords"),
                        "similarity_score": row["similarity_score"],
                        "snippet": snippet,
                        "url": row.get("url"),
                    }
                )

            return results

    async def get_folders(self, base_path: Optional[str] = None) -> List[FolderInfo]:
        query = """
            SELECT folder_name, COUNT(*) as document_count 
            FROM documents 
            WHERE folder_name IS NOT NULL 
            GROUP BY folder_name
            ORDER BY folder_name
        """
        async with self.pool.cursor() as cur:
            await cur.execute(query)
            rows = await cur.fetchall()
            folders = []
            for row in rows:
                folder_name = row["folder_name"]
                document_count = row["document_count"]
                if base_path:
                    folder_path = os.path.join(base_path, folder_name)
                else:
                    folder_path = folder_name
                folders.append(
                    FolderInfo(
                        name=folder_name,
                        path=folder_path,
                        document_count=document_count,
                    )
                )
            return folders

    async def get_status(self, document_id: Optional[UUID] = None):
        if document_id:
            query = "SELECT status FROM documents WHERE id=%s"
            async with self.pool.cursor() as cur:
                await cur.execute(query, (str(document_id),))
                row = await cur.fetchone()
                return row["status"] if row else None
        else:
            query = "SELECT status, COUNT(*) as count FROM documents GROUP BY status"
            async with self.pool.cursor() as cur:
                await cur.execute(query)
                return await cur.fetchall()

    async def find_similar_documents(
        self,
        document_id: UUID,
        limit: int = 10,
        threshold: float = 0.7,
        title_weight: float = 0.75,
        abstract_weight: float = 0.25,
        include_snippet: bool = True,
        folder_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        # Get the reference document's embeddings
        ref_doc = await self.get_document_embedding(document_id)
        if not ref_doc or not ref_doc.title_embedding or not ref_doc.abstract_embedding:
            return []

        return await self.find_similar_documents_by_embeddings(
            ref_doc.title_embedding,
            ref_doc.abstract_embedding,
            limit=limit,
            threshold=threshold,
            title_weight=title_weight,
            abstract_weight=abstract_weight,
            include_snippet=include_snippet,
            folder_name=folder_name,
            exclude_document_id=document_id,
        )

    async def find_similar_documents_by_embeddings(
        self,
        title_embedding: List[float],
        abstract_embedding: List[float],
        limit: int = 10,
        threshold: float = 0.7,
        title_weight: float = 0.75,
        abstract_weight: float = 0.25,
        include_snippet: bool = True,
        folder_name: Optional[str] = None,
        exclude_document_id: Optional[UUID] = None,
    ) -> List[Dict[str, Any]]:
        where_conditions = [
            "title_embedding IS NOT NULL AND abstract_embedding IS NOT NULL"
        ]
        where_params = []

        if folder_name:
            where_conditions.append("folder_name = %s")
            where_params.append(folder_name)

        if exclude_document_id:
            where_conditions.append("id != %s")
            where_params.append(str(exclude_document_id))

        where_clause = f"WHERE {' AND '.join(where_conditions)}"

        query = f"""
            WITH similarity_scores AS (
                SELECT 
                    id,
                    title,
                    abstract,
                    authors,
                    journal_name,
                    publication_year,
                    folder_name,
                    (
                        {title_weight} * (1 - (title_embedding <=> %s::vector)) +
                        {abstract_weight} * (1 - (abstract_embedding <=> %s::vector))
                    ) as similarity
                FROM documents
                {where_clause}
            )
            SELECT *
            FROM similarity_scores
            WHERE similarity >= %s
            ORDER BY similarity DESC
            LIMIT %s
        """

        # Proper parameter order: embeddings first, then threshold, limit, then where clause params
        params = (
            [title_embedding, abstract_embedding] + where_params + [threshold, limit]
        )

        async with self.pool.cursor() as cur:
            await cur.execute(query, params)
            rows = await cur.fetchall()
            return [dict(row) for row in rows]

    async def search_by_keywords(
        self,
        keywords: List[str],
        search_mode: str = "any",  # "any" (OR) or "all" (AND)
        exact_match: bool = False,
        case_sensitive: bool = False,
        folder_name: Optional[str] = None,
        limit: int = 50,
        include_snippet: bool = True,
    ) -> List[Dict[str, Any]]:
        where_conditions = []
        params = []

        # Build keyword search conditions with relevance score calculation
        relevance_parts = []
        keyword_conditions = []

        for i, keyword in enumerate(keywords):
            # Keyword matching in keywords field (weight: 5)
            if exact_match:
                if case_sensitive:
                    keyword_match = (
                        "(SELECT COUNT(*) FROM unnest(keywords) k WHERE k = %s)"
                    )
                    title_match = "(CASE WHEN title = %s THEN 1 ELSE 0 END)"
                    abstract_match = "(CASE WHEN abstract = %s THEN 1 ELSE 0 END)"
                else:
                    keyword_match = "(SELECT COUNT(*) FROM unnest(keywords) k WHERE LOWER(k) = LOWER(%s))"
                    title_match = (
                        "(CASE WHEN LOWER(title) = LOWER(%s) THEN 1 ELSE 0 END)"
                    )
                    abstract_match = (
                        "(CASE WHEN LOWER(abstract) = LOWER(%s) THEN 1 ELSE 0 END)"
                    )
            else:
                if case_sensitive:
                    keyword_match = (
                        "(SELECT COUNT(*) FROM unnest(keywords) k WHERE k LIKE %s)"
                    )
                    title_match = "(CASE WHEN title LIKE %s THEN 1 ELSE 0 END)"
                    abstract_match = "(CASE WHEN abstract LIKE %s THEN 1 ELSE 0 END)"
                else:
                    keyword_match = "(SELECT COUNT(*) FROM unnest(keywords) k WHERE LOWER(k) LIKE LOWER(%s))"
                    title_match = (
                        "(CASE WHEN LOWER(title) LIKE LOWER(%s) THEN 1 ELSE 0 END)"
                    )
                    abstract_match = (
                        "(CASE WHEN LOWER(abstract) LIKE LOWER(%s) THEN 1 ELSE 0 END)"
                    )

            # Build relevance score calculation (uses 3 parameters)
            relevance_parts.append(
                f"(5 * {keyword_match} + 3 * {title_match} + 1 * {abstract_match})"
            )

            # Build filtering conditions (uses 3 more parameters - same values)
            filter_condition = (
                f"({keyword_match} > 0 OR {title_match} > 0 OR {abstract_match} > 0)"
            )
            keyword_conditions.append(filter_condition)

            # Add parameters: 3 for relevance calculation + 3 for filtering = 6 per keyword
            search_term = f"%{keyword}%" if not exact_match else keyword
            params.extend(
                [search_term, search_term, search_term]
            )  # For relevance calculation
            params.extend(
                [search_term, search_term, search_term]
            )  # For filtering condition

        # Calculate total relevance score
        relevance_score = " + ".join(relevance_parts)

        # Build where conditions
        if search_mode == "all":
            where_conditions.append(f"({' AND '.join(keyword_conditions)})")
        else:  # "any"
            where_conditions.append(f"({' OR '.join(keyword_conditions)})")

        if folder_name:
            where_conditions.append("folder_name = %s")
            params.append(folder_name)

        where_clause = (
            f"WHERE {' AND '.join(where_conditions)}" if where_conditions else ""
        )

        query = f"""
            SELECT 
                id, title, authors, journal_name, publication_year,
                abstract, keywords, folder_name, url,
                ({relevance_score}) as relevance_score
            FROM documents
            {where_clause}
            ORDER BY relevance_score DESC, created_at DESC
            LIMIT %s
        """
        params.append(limit)

        async with self.pool.cursor() as cur:
            await cur.execute(query, params)
            rows = await cur.fetchall()

            results = []
            for row in rows:
                result = dict(row)

                # Calculate matched keywords for response
                matched_keywords = []
                for keyword in keywords:
                    search_term = keyword.lower() if not case_sensitive else keyword

                    # Check if keyword matches in any field
                    doc_keywords = [
                        k.lower() if not case_sensitive else k
                        for k in (result.get("keywords") or [])
                    ]
                    doc_title = (
                        result.get("title", "").lower()
                        if not case_sensitive
                        else result.get("title", "")
                    )
                    doc_abstract = (
                        result.get("abstract", "").lower()
                        if not case_sensitive
                        else result.get("abstract", "")
                    )

                    if exact_match:
                        if (
                            search_term in doc_keywords
                            or search_term == doc_title
                            or search_term == doc_abstract
                        ):
                            matched_keywords.append(keyword)
                    else:
                        if (
                            any(search_term in k for k in doc_keywords)
                            or search_term in doc_title
                            or search_term in doc_abstract
                        ):
                            matched_keywords.append(keyword)

                result["matched_keywords"] = matched_keywords
                result["match_score"] = (
                    len(matched_keywords) / len(keywords) * 100
                )  # Percentage

                # Add snippet if requested
                if include_snippet and result.get("abstract"):
                    result["snippet"] = (
                        result["abstract"][:200] + "..."
                        if len(result["abstract"]) > 200
                        else result["abstract"]
                    )

                results.append(result)

            return results

    async def get_all_keywords(
        self, folder_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        where_clause = "WHERE folder_name = %s" if folder_name else ""
        query = f"""
            SELECT DISTINCT unnest(keywords) as keyword, COUNT(*) as count
            FROM documents
            {where_clause}
            GROUP BY keyword
            ORDER BY count DESC
        """
        params = [folder_name] if folder_name else []

        async with self.pool.cursor() as cur:
            await cur.execute(query, params)
            rows = await cur.fetchall()
            return [dict(row) for row in rows]

    async def update_paper_status(self, document_id: UUID, status: str) -> bool:
        """Update the status of a document/paper"""
        query = "UPDATE documents SET status = %s, updated_at = NOW() WHERE id = %s"
        async with self.pool.cursor() as cur:
            await cur.execute(query, (status, str(document_id)))
            return cur.rowcount > 0
