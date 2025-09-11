-- PostgreSQL initialization script for Smart Traffic Management System
-- This script sets up the database with proper configuration

-- Create database if it doesn't exist (handled by POSTGRES_DB environment variable)
-- CREATE DATABASE traffic_db;

-- Connect to the traffic database
\c traffic_db;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Set timezone
SET timezone = 'UTC';

-- Create custom types
DO $$ BEGIN
    CREATE TYPE traffic_light_state AS ENUM ('red', 'yellow', 'green', 'flashing_red', 'flashing_yellow', 'off');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE system_status AS ENUM ('operational', 'degraded', 'maintenance', 'offline', 'error');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE optimization_status AS ENUM ('pending', 'in_progress', 'completed', 'failed', 'cancelled');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE event_level AS ENUM ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Create indexes for better performance
-- These will be created by SQLAlchemy, but we can add additional ones here

-- Create a function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create a function to clean up old data
CREATE OR REPLACE FUNCTION cleanup_old_data()
RETURNS void AS $$
BEGIN
    -- Delete traffic data older than 30 days
    DELETE FROM traffic_data 
    WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '30 days';
    
    -- Delete system events older than 90 days
    DELETE FROM system_events 
    WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '90 days';
    
    -- Delete API logs older than 30 days
    DELETE FROM api_logs 
    WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '30 days';
    
    -- Delete health checks older than 7 days
    DELETE FROM health_checks 
    WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '7 days';
END;
$$ language 'plpgsql';

-- Create a function to get traffic statistics
CREATE OR REPLACE FUNCTION get_traffic_stats(intersection_id_param TEXT, hours_param INTEGER DEFAULT 24)
RETURNS TABLE (
    total_vehicles BIGINT,
    avg_speed NUMERIC,
    peak_hour INTEGER,
    peak_vehicles BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COALESCE(SUM((lane_counts->>'north_lane')::INTEGER + 
                     (lane_counts->>'south_lane')::INTEGER + 
                     (lane_counts->>'east_lane')::INTEGER + 
                     (lane_counts->>'west_lane')::INTEGER), 0) as total_vehicles,
        COALESCE(AVG(avg_speed), 0) as avg_speed,
        EXTRACT(HOUR FROM timestamp)::INTEGER as peak_hour,
        COALESCE(SUM((lane_counts->>'north_lane')::INTEGER + 
                     (lane_counts->>'south_lane')::INTEGER + 
                     (lane_counts->>'east_lane')::INTEGER + 
                     (lane_counts->>'west_lane')::INTEGER), 0) as peak_vehicles
    FROM traffic_data 
    WHERE intersection_id = intersection_id_param 
      AND created_at >= CURRENT_TIMESTAMP - INTERVAL '1 hour' * hours_param
    GROUP BY EXTRACT(HOUR FROM timestamp)
    ORDER BY peak_vehicles DESC
    LIMIT 1;
END;
$$ language 'plpgsql';

-- Create a function to get intersection performance metrics
CREATE OR REPLACE FUNCTION get_intersection_performance(intersection_id_param TEXT, days_param INTEGER DEFAULT 7)
RETURNS TABLE (
    total_optimizations BIGINT,
    successful_optimizations BIGINT,
    avg_confidence NUMERIC,
    avg_improvement NUMERIC,
    success_rate NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*) as total_optimizations,
        COUNT(*) FILTER (WHERE status = 'completed') as successful_optimizations,
        COALESCE(AVG(confidence_score), 0) as avg_confidence,
        COALESCE(AVG(expected_improvement), 0) as avg_improvement,
        COALESCE(
            (COUNT(*) FILTER (WHERE status = 'completed')::NUMERIC / COUNT(*)) * 100, 
            0
        ) as success_rate
    FROM optimization_results 
    WHERE intersection_id = intersection_id_param 
      AND created_at >= CURRENT_TIMESTAMP - INTERVAL '1 day' * days_param;
END;
$$ language 'plpgsql';

-- Create a view for real-time traffic status
CREATE OR REPLACE VIEW real_time_traffic AS
SELECT 
    i.id as intersection_id,
    i.name as intersection_name,
    i.status as intersection_status,
    td.timestamp,
    td.lane_counts,
    td.avg_speed,
    td.weather_condition,
    td.confidence_score,
    td.created_at as data_created_at
FROM intersections i
LEFT JOIN LATERAL (
    SELECT *
    FROM traffic_data 
    WHERE traffic_data.intersection_id = i.id
    ORDER BY traffic_data.created_at DESC
    LIMIT 1
) td ON true;

-- Create a view for signal optimization history
CREATE OR REPLACE VIEW signal_optimization_history AS
SELECT 
    or.id as optimization_id,
    or.intersection_id,
    i.name as intersection_name,
    or.algorithm_used,
    or.confidence_score,
    or.expected_improvement,
    or.optimization_time,
    or.status,
    or.created_at,
    or.applied_at
FROM optimization_results or
JOIN intersections i ON i.id = or.intersection_id
ORDER BY or.created_at DESC;

-- Create a view for system health summary
CREATE OR REPLACE VIEW system_health_summary AS
SELECT 
    component,
    status,
    COUNT(*) as check_count,
    MAX(created_at) as last_check,
    AVG(response_time) as avg_response_time
FROM health_checks 
WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '1 hour'
GROUP BY component, status
ORDER BY component, last_check DESC;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE traffic_db TO traffic_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO traffic_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO traffic_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO traffic_user;

-- Create a scheduled job to clean up old data (requires pg_cron extension)
-- SELECT cron.schedule('cleanup-old-data', '0 2 * * *', 'SELECT cleanup_old_data();');

-- Insert some sample data for testing
INSERT INTO intersections (id, name, location_lat, location_lng, lanes, status) VALUES
('junction-1', 'Main Street & First Avenue', 20.2961, 85.8245, '["north_lane", "south_lane", "east_lane", "west_lane"]', 'operational'),
('junction-2', 'Second Street & Oak Avenue', 20.2971, 85.8255, '["north_lane", "south_lane", "east_lane", "west_lane"]', 'operational'),
('junction-3', 'Third Street & Pine Avenue', 20.2981, 85.8265, '["north_lane", "south_lane", "east_lane", "west_lane"]', 'operational')
ON CONFLICT (id) DO NOTHING;

-- Create initial health check record
INSERT INTO health_checks (component, status, message, response_time, details) VALUES
('database', 'healthy', 'Database connection successful', 0.001, '{"version": "15.0", "connections": 5}'),
('redis', 'healthy', 'Redis connection successful', 0.002, '{"version": "7.0", "memory": "1.2M"}'),
('system', 'healthy', 'System resources normal', 0.001, '{"cpu": 15.5, "memory": 45.2}')
ON CONFLICT DO NOTHING;

-- Create initial configuration
INSERT INTO configurations (key, value, value_type, description, is_sensitive) VALUES
('system.version', '1.0.0', 'string', 'System version', false),
('system.maintenance_mode', 'false', 'bool', 'Maintenance mode flag', false),
('traffic.default_signal_timing', '30', 'int', 'Default signal timing in seconds', false),
('traffic.max_signal_timing', '300', 'int', 'Maximum signal timing in seconds', false),
('api.rate_limit_requests', '100', 'int', 'API rate limit requests per minute', false),
('api.rate_limit_window', '60', 'int', 'API rate limit window in seconds', false)
ON CONFLICT (key) DO NOTHING;

-- Create indexes for better performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_traffic_data_intersection_created 
ON traffic_data (intersection_id, created_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_traffic_data_timestamp 
ON traffic_data (timestamp DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_optimization_results_intersection_status 
ON optimization_results (intersection_id, status, created_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_system_events_type_created 
ON system_events (event_type, created_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_api_logs_status_created 
ON api_logs (status_code, created_at DESC);

-- Analyze tables for better query planning
ANALYZE;
