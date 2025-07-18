CREATE OR REPLACE VIEW measurement_results_view AS
WITH ranked_code_states AS (
    SELECT 
        code_state_id,
        data_structure_name,
        ROW_NUMBER() OVER (
            PARTITION BY data_structure_name 
            ORDER BY created_at DESC
        ) as code_state_rank
    FROM code_states
),
ranked_iterations AS (
    SELECT 
        measurement_id,
        ROW_NUMBER() OVER (
            PARTITION BY code_state_id, result_id, benchmark_type, hostname 
            ORDER BY iteration_number DESC
        ) as iteration_rank
    FROM measurements
)
SELECT 
    -- Measurement data
    m.measurement_id,
    m.iteration_number,
    m.sample_count,
    m.hostname,
    m.architecture,
    m.benchmark_type,
    m.wall_time_mean,
    m.wall_time_stddev,
    m.instruction_count_mean,
    m.instruction_count_stddev,
    m.cycles_mean,
    m.cycles_stddev,
    m.created_at as measurement_created_at,
    
    -- Code state information
    m.code_state_id,
    cs.checksum as code_checksum,
    cs.data_structure_name,
    cs.created_at as code_state_created_at,
    
    -- Repository information
    rs.repo_state_id,
    rs.commit_hash,
    rs.commit_message,
    rs.timestamp as commit_timestamp,
    
    -- Position result information
    m.result_id,
    pr.embedding_dim,
    pr.dim_hint,
    pr.max_iterations,
    pr.actual_iterations,
    pr.seed as embedding_seed,
    pr.file_path as result_file_path,
    pr.checksum as result_checksum,
    
    -- Graph information and generation parameters
    g.graph_id,
    g.n,
    g.deg,
    g.ple,
    g.dim,
    g.alpha,
    g.wseed,
    g.pseed,
    g.sseed,
    g.processed_n,
    g.processed_avg_degree,
    g.file_path as graph_file_path,
    
    -- Computed flags
    (rcs.code_state_rank = 1) as is_newest_code_state,
    (ri.iteration_rank = 1) as is_last_iteration

FROM measurements m
    JOIN code_states cs ON m.code_state_id = cs.code_state_id
    JOIN ranked_code_states rcs ON cs.code_state_id = rcs.code_state_id
    JOIN ranked_iterations ri ON m.measurement_id = ri.measurement_id
    JOIN repository_states rs ON cs.repo_state_id = rs.repo_state_id
    JOIN position_results pr ON m.result_id = pr.result_id
    JOIN graphs g ON pr.graph_id = g.graph_id;

