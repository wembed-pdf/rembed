-- Add specialized indexes for pie chart queries on underlying tables
-- These indexes target the specific WHERE clause patterns used in pie chart generation

-- Index on measurements table for hostname and benchmark_type filtering
CREATE INDEX IF NOT EXISTS idx_measurements_pie_filter ON measurements 
(hostname, benchmark_type);

-- Index on graphs table for the graph parameters used in pie charts
CREATE INDEX IF NOT EXISTS idx_graphs_pie_params ON graphs 
(deg, dim, ple, alpha);

-- Index on position_results for embedding dimension filtering
CREATE INDEX IF NOT EXISTS idx_position_results_embedding_dim ON position_results 
(embedding_dim);

-- Composite index on measurements for iteration filtering
CREATE INDEX IF NOT EXISTS idx_measurements_iteration_filter ON measurements 
(code_state_id, result_id, benchmark_type, hostname, iteration_number);
