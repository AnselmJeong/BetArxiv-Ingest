-- Database schema for BetArxiv Document Management System
-- Updated for new document-based API structure

CREATE EXTENSION IF NOT EXISTS vector;

-- Main documents table
CREATE TABLE IF NOT EXISTS documents (
    -- Primary identifier
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Core document metadata
    title TEXT NOT NULL,
    authors TEXT[] NOT NULL,
    journal_name TEXT,
    volume TEXT,
    issue TEXT, 
    publication_year INTEGER,
    abstract TEXT,
    keywords TEXT[],
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),

    
    -- Document identifiers
    doi TEXT,  -- DOI identifier for published papers
    arxiv_id TEXT,  -- arXiv identifier for preprints
    
    -- Document content and processing
    markdown TEXT,
    url TEXT,  -- File path to original document
    folder_name TEXT,  -- Relative folder path from base directory
    
    -- AI-generated content sections
    summary TEXT,
    previous_work TEXT,
    hypothesis TEXT,
    distinction TEXT,
    methodology TEXT,
    results TEXT,
    limitations TEXT,
    implications TEXT,
    
    -- Vector embeddings for semantic search
    title_embedding vector(768),
    abstract_embedding vector(768),
    
    -- Full-text search vector (populated by trigger)
    search_vector tsvector,
    
    -- Processing status and metadata
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Function to update search vector
CREATE OR REPLACE FUNCTION update_search_vector() RETURNS trigger AS $$
BEGIN
    NEW.search_vector = 
        setweight(to_tsvector('english', coalesce(NEW.title, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(NEW.abstract, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(NEW.markdown, '')), 'C') ||
        setweight(to_tsvector('english', array_to_string(coalesce(NEW.authors, ARRAY[]::text[]), ' ')), 'B') ||
        setweight(to_tsvector('english', array_to_string(coalesce(NEW.keywords, ARRAY[]::text[]), ' ')), 'B');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to automatically update search_vector
DROP TRIGGER IF EXISTS trigger_update_search_vector ON documents;
CREATE TRIGGER trigger_update_search_vector
    BEFORE INSERT OR UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_search_vector();

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_documents_title ON documents (title);
CREATE INDEX IF NOT EXISTS idx_documents_authors ON documents USING GIN (authors);
CREATE INDEX IF NOT EXISTS idx_documents_keywords ON documents USING GIN (keywords);
CREATE INDEX IF NOT EXISTS idx_documents_folder_name ON documents (folder_name);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents (status);
CREATE INDEX IF NOT EXISTS idx_documents_publication_year ON documents (publication_year);
CREATE INDEX IF NOT EXISTS idx_documents_doi ON documents (doi);
CREATE INDEX IF NOT EXISTS idx_documents_arxiv_id ON documents (arxiv_id);

-- Vector similarity search indexes (requires pgvector extension)
CREATE INDEX IF NOT EXISTS idx_documents_title_embedding ON documents USING hnsw (title_embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_documents_abstract_embedding ON documents USING hnsw (abstract_embedding vector_cosine_ops);

-- Full-text search index (tsvector)
CREATE INDEX IF NOT EXISTS idx_documents_search_vector ON documents USING GIN (search_vector);

-- Add comments for documentation
COMMENT ON TABLE documents IS 'Main table storing research documents and their processed content';
COMMENT ON COLUMN documents.folder_name IS 'Folder path relative to base directory where the document is stored';
COMMENT ON COLUMN documents.url IS 'Full file path to the original document';
COMMENT ON COLUMN documents.doi IS 'DOI identifier for published papers (e.g., 10.1080/10509585.2015.1092083)';
COMMENT ON COLUMN documents.arxiv_id IS 'arXiv identifier for preprints (e.g., 2502.04780v1)';
COMMENT ON COLUMN documents.title_embedding IS 'Vector embedding of document title for semantic search';
COMMENT ON COLUMN documents.abstract_embedding IS 'Vector embedding of document abstract for semantic search';
COMMENT ON COLUMN documents.search_vector IS 'Full-text search vector combining title, abstract, markdown, authors, and keywords with different weights (A=title, B=abstract/authors/keywords, C=markdown) using English text configuration. Automatically updated by trigger.'; 