-- Migration script to add full-text search capabilities to existing documents table
-- Run this script to add tsvector column and trigger to existing data

-- Step 1: Add the search_vector column (not generated, will be populated by trigger)
ALTER TABLE documents 
ADD COLUMN IF NOT EXISTS search_vector tsvector;

-- Step 2: Create function to update search vector
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

-- Step 3: Create trigger to automatically update search_vector on INSERT and UPDATE
DROP TRIGGER IF EXISTS trigger_update_search_vector ON documents;
CREATE TRIGGER trigger_update_search_vector
    BEFORE INSERT OR UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_search_vector();

-- Step 4: Create GIN index for fast full-text search
CREATE INDEX IF NOT EXISTS idx_documents_search_vector 
ON documents USING GIN (search_vector);

-- Step 5: Update existing rows to populate search_vector
UPDATE documents SET search_vector = 
    setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
    setweight(to_tsvector('english', coalesce(abstract, '')), 'B') ||
    setweight(to_tsvector('english', coalesce(markdown, '')), 'C') ||
    setweight(to_tsvector('english', array_to_string(coalesce(authors, ARRAY[]::text[]), ' ')), 'B') ||
    setweight(to_tsvector('english', array_to_string(coalesce(keywords, ARRAY[]::text[]), ' ')), 'B')
WHERE search_vector IS NULL;

-- Step 6: Add comment for documentation
COMMENT ON COLUMN documents.search_vector IS 'Full-text search vector combining title, abstract, markdown, authors, and keywords with different weights (A=title, B=abstract/authors/keywords, C=markdown) using English text configuration. Automatically updated by trigger.';

-- Example usage queries (for reference):
-- 
-- 1. Simple text search:
-- SELECT title, ts_rank(search_vector, plainto_tsquery('english', 'machine learning')) as rank
-- FROM documents 
-- WHERE search_vector @@ plainto_tsquery('english', 'machine learning')
-- ORDER BY rank DESC;
--
-- 2. Advanced search with multiple terms:
-- SELECT title, ts_rank(search_vector, to_tsquery('english', 'neural & network')) as rank
-- FROM documents 
-- WHERE search_vector @@ to_tsquery('english', 'neural & network')
-- ORDER BY rank DESC;
--
-- 3. Search with phrase:
-- SELECT title, ts_rank(search_vector, phraseto_tsquery('english', 'deep learning')) as rank
-- FROM documents 
-- WHERE search_vector @@ phraseto_tsquery('english', 'deep learning')
-- ORDER BY rank DESC;
--
-- 4. Boolean search (AND, OR, NOT):
-- SELECT title, ts_rank(search_vector, to_tsquery('english', 'machine & learning & !supervised')) as rank
-- FROM documents 
-- WHERE search_vector @@ to_tsquery('english', 'machine & learning & !supervised')
-- ORDER BY rank DESC; 