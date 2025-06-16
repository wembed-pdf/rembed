CREATE OR REPLACE FUNCTION sync_measurements_diffie_hellman()
RETURNS TABLE(diffie_to_hellman INTEGER, hellman_to_diffie INTEGER) AS $$
DECLARE
    count_d_to_h INTEGER;
    count_h_to_d INTEGER;
BEGIN
    -- Copy from diffie to hellman
    INSERT INTO measurements (
        code_state_id,
        result_id,
        iteration_number,
        sample_count,
        hostname,
        architecture,
        benchmark_type,
        wall_time_mean,
        wall_time_stddev,
        instruction_count_mean,
        instruction_count_stddev,
        cycles_mean,
        cycles_stddev
    )
    SELECT 
        m.code_state_id,
        m.result_id,
        m.iteration_number,
        m.sample_count,
        'hellman',
        m.architecture,
        m.benchmark_type,
        m.wall_time_mean,
        m.wall_time_stddev,
        m.instruction_count_mean,
        m.instruction_count_stddev,
        m.cycles_mean,
        m.cycles_stddev
    FROM measurements m
    WHERE m.hostname = 'diffie'
    ON CONFLICT (code_state_id, result_id, iteration_number, benchmark_type, hostname) DO NOTHING;
    
    GET DIAGNOSTICS count_d_to_h = ROW_COUNT;
    
    -- Copy from hellman to diffie
    INSERT INTO measurements (
        code_state_id,
        result_id,
        iteration_number,
        sample_count,
        hostname,
        architecture,
        benchmark_type,
        wall_time_mean,
        wall_time_stddev,
        instruction_count_mean,
        instruction_count_stddev,
        cycles_mean,
        cycles_stddev
    )
    SELECT 
        m.code_state_id,
        m.result_id,
        m.iteration_number,
        m.sample_count,
        'diffie',
        m.architecture,
        m.benchmark_type,
        m.wall_time_mean,
        m.wall_time_stddev,
        m.instruction_count_mean,
        m.instruction_count_stddev,
        m.cycles_mean,
        m.cycles_stddev
    FROM measurements m
    WHERE m.hostname = 'hellman'
    ON CONFLICT (code_state_id, result_id, iteration_number, benchmark_type, hostname) DO NOTHING;
    
    GET DIAGNOSTICS count_h_to_d = ROW_COUNT;
    
    RETURN QUERY SELECT count_d_to_h, count_h_to_d;
END;
$$ LANGUAGE plpgsql;
