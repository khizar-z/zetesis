CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id uuid PRIMARY KEY,
    entry_slug text NOT NULL,
    entry_title text NOT NULL,
    section_number text NOT NULL,
    section_title text NOT NULL,
    url text NOT NULL,
    chunk_text text NOT NULL,
    chunk_index int NOT NULL,
    embedding vector(768) NOT NULL
);

CREATE TABLE IF NOT EXISTS entries (
    id uuid PRIMARY KEY,
    entry_slug text NOT NULL UNIQUE,
    entry_title text NOT NULL,
    url text NOT NULL,
    section_count int NOT NULL,
    section_outline text NOT NULL,
    intro_text text NOT NULL,
    search_text text NOT NULL,
    embedding vector(768) NOT NULL
);

ALTER TABLE entries
ADD COLUMN IF NOT EXISTS subdiscipline text;

ALTER TABLE entries
ADD COLUMN IF NOT EXISTS graph_embedding vector(768);

CREATE TABLE IF NOT EXISTS explicit_edges (
    source_slug text NOT NULL REFERENCES entries(entry_slug) ON DELETE CASCADE,
    target_slug text NOT NULL REFERENCES entries(entry_slug) ON DELETE CASCADE,
    weight int NOT NULL CHECK (weight > 0),
    PRIMARY KEY (source_slug, target_slug)
);

CREATE INDEX IF NOT EXISTS explicit_edges_source_idx
ON explicit_edges (source_slug);

CREATE INDEX IF NOT EXISTS explicit_edges_target_idx
ON explicit_edges (target_slug);

CREATE TABLE IF NOT EXISTS semantic_edges (
    source_slug text NOT NULL REFERENCES entries(entry_slug) ON DELETE CASCADE,
    target_slug text NOT NULL REFERENCES entries(entry_slug) ON DELETE CASCADE,
    similarity double precision NOT NULL CHECK (similarity >= -1.0 AND similarity <= 1.0),
    PRIMARY KEY (source_slug, target_slug)
);

CREATE INDEX IF NOT EXISTS semantic_edges_source_idx
ON semantic_edges (source_slug);

CREATE INDEX IF NOT EXISTS semantic_edges_target_idx
ON semantic_edges (target_slug);

ALTER TABLE semantic_edges
DROP CONSTRAINT IF EXISTS semantic_edges_check;
