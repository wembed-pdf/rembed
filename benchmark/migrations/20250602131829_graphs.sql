-- Migration: Create graphs table
-- Up migration

CREATE TABLE graphs (
    graph_id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Original generation parameters
    n INTEGER NOT NULL CHECK (n > 0),
    deg INTEGER NOT NULL CHECK (deg > 0),
    wseed INTEGER NOT NULL,
    pseed INTEGER NOT NULL,
    sseed INTEGER NOT NULL,
    
    -- Processed graph metrics (after largest component extraction)
    processed_n INTEGER NOT NULL CHECK (processed_n > 0),
    processed_avg_degree double precision NOT NULL CHECK (processed_avg_degree > 0),
    
    -- File storage
    file_path TEXT NOT NULL,
    checksum CHAR(64) NOT NULL, -- SHA-256 hex string is always 64 characters
    
    -- Ensure file paths are unique
    CONSTRAINT unique_file_path UNIQUE (file_path),
    
    -- Ensure parameter combinations are unique (prevent duplicate graphs)
    CONSTRAINT unique_graph_params UNIQUE (n, deg, wseed, pseed, sseed)
);

-- Create indexes for common queries
CREATE INDEX idx_graphs_created_at ON graphs (created_at);
CREATE INDEX idx_graphs_n_deg ON graphs (n, deg);
CREATE INDEX idx_graphs_processed_metrics ON graphs (processed_n, processed_avg_degree);
CREATE INDEX idx_graphs_seeds ON graphs (wseed, pseed, sseed);

-- Down migration (uncomment when needed)
-- DROP TABLE IF EXISTS graphs;
