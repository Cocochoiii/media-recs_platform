-- Media Recommender Database Schema
-- PostgreSQL initialization script

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    external_id VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Demographics
    age_group VARCHAR(20),
    gender VARCHAR(20),
    location VARCHAR(255),
    device_type VARCHAR(50),
    subscription_tier VARCHAR(20) DEFAULT 'free',
    
    -- Preferences (JSON)
    preferences JSONB DEFAULT '{}',
    
    -- Flags
    is_active BOOLEAN DEFAULT true,
    is_cold_start BOOLEAN DEFAULT true
);

CREATE INDEX idx_users_external_id ON users(external_id);
CREATE INDEX idx_users_created_at ON users(created_at);

-- Items table
CREATE TABLE IF NOT EXISTS items (
    id SERIAL PRIMARY KEY,
    external_id VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Content
    title VARCHAR(500) NOT NULL,
    description TEXT,
    
    -- Metadata
    genre VARCHAR(100)[],
    tags VARCHAR(100)[],
    duration_seconds INTEGER,
    release_date DATE,
    
    -- Popularity
    popularity_score FLOAT DEFAULT 0.0,
    view_count INTEGER DEFAULT 0,
    rating_avg FLOAT DEFAULT 0.0,
    rating_count INTEGER DEFAULT 0,
    
    -- Content embeddings (stored as binary)
    embedding BYTEA,
    
    -- Flags
    is_active BOOLEAN DEFAULT true
);

CREATE INDEX idx_items_external_id ON items(external_id);
CREATE INDEX idx_items_genre ON items USING GIN(genre);
CREATE INDEX idx_items_tags ON items USING GIN(tags);
CREATE INDEX idx_items_popularity ON items(popularity_score DESC);
CREATE INDEX idx_items_title_trgm ON items USING GIN(title gin_trgm_ops);

-- Interactions table (partitioned by month)
CREATE TABLE IF NOT EXISTS interactions (
    id BIGSERIAL,
    user_id INTEGER NOT NULL REFERENCES users(id),
    item_id INTEGER NOT NULL REFERENCES items(id),
    
    interaction_type VARCHAR(50) NOT NULL,  -- view, click, purchase, rating
    rating FLOAT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Context
    device_type VARCHAR(50),
    session_id VARCHAR(255),
    
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- Create partitions for last 12 months
DO $$
DECLARE
    start_date DATE := DATE_TRUNC('month', CURRENT_DATE - INTERVAL '11 months');
    end_date DATE;
    partition_name TEXT;
BEGIN
    FOR i IN 0..12 LOOP
        end_date := start_date + INTERVAL '1 month';
        partition_name := 'interactions_' || TO_CHAR(start_date, 'YYYY_MM');
        
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS %I PARTITION OF interactions
             FOR VALUES FROM (%L) TO (%L)',
            partition_name,
            start_date,
            end_date
        );
        
        start_date := end_date;
    END LOOP;
END $$;

CREATE INDEX idx_interactions_user_id ON interactions(user_id);
CREATE INDEX idx_interactions_item_id ON interactions(item_id);
CREATE INDEX idx_interactions_user_item ON interactions(user_id, item_id);
CREATE INDEX idx_interactions_type ON interactions(interaction_type);

-- Recommendations log table
CREATE TABLE IF NOT EXISTS recommendation_logs (
    id BIGSERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    
    -- Request info
    request_id UUID DEFAULT uuid_generate_v4(),
    model_version VARCHAR(50),
    num_recommendations INTEGER,
    
    -- Response
    recommended_items INTEGER[],
    scores FLOAT[],
    
    -- Metrics
    latency_ms FLOAT,
    cache_hit BOOLEAN DEFAULT false,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_rec_logs_user_id ON recommendation_logs(user_id);
CREATE INDEX idx_rec_logs_created_at ON recommendation_logs(created_at);

-- Model metadata table
CREATE TABLE IF NOT EXISTS model_versions (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    version VARCHAR(50) NOT NULL,
    
    -- Metrics
    ndcg_10 FLOAT,
    hr_10 FLOAT,
    coverage FLOAT,
    diversity FLOAT,
    
    -- Training info
    training_samples INTEGER,
    training_duration_seconds INTEGER,
    
    -- Storage
    model_path VARCHAR(500),
    
    -- Status
    is_active BOOLEAN DEFAULT false,
    deployed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(model_name, version)
);

-- A/B test experiments table
CREATE TABLE IF NOT EXISTS experiments (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    description TEXT,
    
    -- Variants
    control_model VARCHAR(100),
    treatment_model VARCHAR(100),
    traffic_split FLOAT DEFAULT 0.5,  -- Fraction going to treatment
    
    -- Status
    status VARCHAR(20) DEFAULT 'draft',  -- draft, running, completed, cancelled
    started_at TIMESTAMP WITH TIME ZONE,
    ended_at TIMESTAMP WITH TIME ZONE,
    
    -- Results
    results JSONB,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- User experiment assignments
CREATE TABLE IF NOT EXISTS experiment_assignments (
    id SERIAL PRIMARY KEY,
    experiment_id INTEGER REFERENCES experiments(id),
    user_id INTEGER REFERENCES users(id),
    variant VARCHAR(20) NOT NULL,  -- control, treatment
    assigned_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(experiment_id, user_id)
);

-- Functions

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER items_updated_at
    BEFORE UPDATE ON items
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- Update item popularity after interaction
CREATE OR REPLACE FUNCTION update_item_popularity()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE items
    SET 
        view_count = view_count + 1,
        rating_avg = CASE 
            WHEN NEW.rating IS NOT NULL THEN
                (rating_avg * rating_count + NEW.rating) / (rating_count + 1)
            ELSE rating_avg
        END,
        rating_count = CASE 
            WHEN NEW.rating IS NOT NULL THEN rating_count + 1
            ELSE rating_count
        END,
        popularity_score = (
            SELECT LOG(COUNT(*) + 1) + COALESCE(AVG(rating), 3) / 5
            FROM interactions
            WHERE item_id = NEW.item_id
            AND created_at > CURRENT_TIMESTAMP - INTERVAL '30 days'
        )
    WHERE id = NEW.item_id;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_popularity_after_interaction
    AFTER INSERT ON interactions
    FOR EACH ROW
    EXECUTE FUNCTION update_item_popularity();

-- Mark user as not cold start after sufficient interactions
CREATE OR REPLACE FUNCTION update_cold_start_status()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE users
    SET is_cold_start = false
    WHERE id = NEW.user_id
    AND is_cold_start = true
    AND (
        SELECT COUNT(*) FROM interactions 
        WHERE user_id = NEW.user_id
    ) >= 5;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER check_cold_start_after_interaction
    AFTER INSERT ON interactions
    FOR EACH ROW
    EXECUTE FUNCTION update_cold_start_status();

-- Views

-- Active items with popularity
CREATE OR REPLACE VIEW active_items_ranked AS
SELECT 
    id,
    external_id,
    title,
    genre,
    tags,
    popularity_score,
    view_count,
    rating_avg,
    rating_count,
    RANK() OVER (ORDER BY popularity_score DESC) as popularity_rank
FROM items
WHERE is_active = true;

-- User interaction summary
CREATE OR REPLACE VIEW user_interaction_summary AS
SELECT 
    user_id,
    COUNT(*) as total_interactions,
    COUNT(DISTINCT item_id) as unique_items,
    AVG(rating) as avg_rating,
    MAX(created_at) as last_interaction,
    ARRAY_AGG(DISTINCT interaction_type) as interaction_types
FROM interactions
GROUP BY user_id;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO recommender;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO recommender;
