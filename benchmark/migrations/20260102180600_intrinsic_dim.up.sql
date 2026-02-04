-- Migration: create intrinsic_dim table
-- Up migration

CREATE TABLE intrinsic_dim (
    result_id BIGINT NOT NULL REFERENCES position_results(result_id) ON DELETE CASCADE,
    iteration_number INTEGER NOT NULL,
    intrinsic_dimension DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (result_id, iteration_number)
);

-- Create index for common queries
CREATE INDEX idx_intrinsic_dim_dimension ON intrinsic_dim (intrinsic_dimension);
CREATE INDEX idx_intrinsic_dim_result_id ON intrinsic_dim (result_id);