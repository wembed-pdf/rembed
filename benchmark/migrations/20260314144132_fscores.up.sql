-- Migration: create fscores table
-- Up migration

CREATE TABLE fscores (
    result_id BIGINT NOT NULL REFERENCES position_results(result_id) ON DELETE CASCADE,
    iteration_number INTEGER NOT NULL,
    recall DOUBLE PRECISION NOT NULL,
    prec DOUBLE PRECISION NOT NULL,
    fscore DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (result_id, iteration_number)
);

-- Create index for common queries
CREATE INDEX idx_fscore_fscore ON fscores (fscore);
CREATE INDEX idx_fscore_result_id ON fscores (result_id);