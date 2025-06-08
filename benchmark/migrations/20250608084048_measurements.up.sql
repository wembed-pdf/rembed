-- Migration: Create measurements table
-- Up migration

CREATE TABLE measurements (
    measurement_id BIGSERIAL PRIMARY KEY,
    code_state_id BIGINT NOT NULL REFERENCES code_states(code_state_id) ON DELETE CASCADE,
    result_id BIGINT NOT NULL REFERENCES position_results(result_id) ON DELETE CASCADE,
    iteration_number INTEGER NOT NULL,
    sample_count INTEGER NOT NULL,
    hostname TEXT NOT NULL,
    architecture TEXT NOT NULL,
    benchmark_type TEXT NOT NULL CHECK (benchmark_type IN ('construction', 'sparse_query', 'light_nodes', 'heavy_nodes')),
    
    -- Performance measurements (mean values)
    wall_time_mean BIGINT NOT NULL, -- Nanoseconds
    wall_time_stddev BIGINT NOT NULL, -- Nanoseconds
    instruction_count_mean DOUBLE PRECISION NOT NULL,
    instruction_count_stddev DOUBLE PRECISION NOT NULL,
    cycles_mean DOUBLE PRECISION NOT NULL,
    cycles_stddev DOUBLE PRECISION NOT NULL,
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Ensure we don't duplicate measurements for the same configuration
    CONSTRAINT unique_measurement UNIQUE (code_state_id, result_id, iteration_number, benchmark_type)
);

-- Create indexes for common queries
CREATE INDEX idx_measurements_code_state_id ON measurements (code_state_id);
CREATE INDEX idx_measurements_result_id ON measurements (result_id);
CREATE INDEX idx_measurements_iteration_number ON measurements (iteration_number);
CREATE INDEX idx_measurements_benchmark_type ON measurements (benchmark_type);
CREATE INDEX idx_measurements_hostname ON measurements (hostname);
CREATE INDEX idx_measurements_architecture ON measurements (architecture);
CREATE INDEX idx_measurements_created_at ON measurements (created_at);

-- Indexes for performance analysis
CREATE INDEX idx_measurements_wall_time ON measurements (wall_time_mean);
CREATE INDEX idx_measurements_instructions ON measurements (instruction_count_mean);
CREATE INDEX idx_measurements_cycles ON measurements (cycles_mean);

-- Composite index for filtering by common criteria
CREATE INDEX idx_measurements_filter ON measurements (benchmark_type, architecture, hostname, created_at);

