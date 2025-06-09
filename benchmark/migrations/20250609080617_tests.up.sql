-- Migration: Create tests table for correctness testing
-- Up migration

CREATE TABLE tests (
    result_id BIGINT NOT NULL REFERENCES position_results(result_id) ON DELETE CASCADE PRIMARY KEY,
    file_path TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Ensure each result_id has only one test file
    CONSTRAINT unique_test_result UNIQUE (result_id)
);

-- Create indexes for common queries
CREATE INDEX idx_tests_result_id ON tests (result_id);
CREATE INDEX idx_tests_created_at ON tests (created_at);
