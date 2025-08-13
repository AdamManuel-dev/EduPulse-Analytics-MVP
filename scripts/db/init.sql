-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create schema for the application
CREATE SCHEMA IF NOT EXISTS edupulse;

-- Set search path
SET search_path TO edupulse, public;

-- Create students table
CREATE TABLE IF NOT EXISTS students (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    district_id VARCHAR(50) UNIQUE NOT NULL,
    grade_level INTEGER CHECK (grade_level >= 0 AND grade_level <= 12),
    enrollment_date DATE NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create student_features time-series table
CREATE TABLE IF NOT EXISTS student_features (
    student_id UUID REFERENCES students(id) ON DELETE CASCADE,
    feature_date DATE NOT NULL,
    attendance_rate FLOAT CHECK (attendance_rate >= 0 AND attendance_rate <= 1),
    gpa_current FLOAT CHECK (gpa_current >= 0 AND gpa_current <= 5),
    discipline_incidents INTEGER DEFAULT 0 CHECK (discipline_incidents >= 0),
    feature_vector FLOAT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (student_id, feature_date)
);

-- Convert student_features to hypertable
SELECT create_hypertable('student_features', 'feature_date', 
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE);

-- Create predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    student_id UUID REFERENCES students(id) ON DELETE CASCADE,
    prediction_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    risk_score FLOAT CHECK (risk_score >= 0 AND risk_score <= 1),
    risk_category VARCHAR(20) CHECK (risk_category IN ('low', 'medium', 'high', 'critical')),
    confidence FLOAT CHECK (confidence >= 0 AND confidence <= 1),
    risk_factors JSONB DEFAULT '[]',
    model_version VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create training_feedback table
CREATE TABLE IF NOT EXISTS training_feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    prediction_id UUID REFERENCES predictions(id) ON DELETE CASCADE,
    outcome_date DATE,
    outcome_type VARCHAR(50),
    feedback_type VARCHAR(50) CHECK (feedback_type IN ('true_positive', 'false_positive', 'true_negative', 'false_negative')),
    educator_notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create attendance_records table
CREATE TABLE IF NOT EXISTS attendance_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    student_id UUID REFERENCES students(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    status VARCHAR(20) CHECK (status IN ('present', 'absent', 'tardy', 'excused')),
    period INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(student_id, date, period)
);

-- Create grades table
CREATE TABLE IF NOT EXISTS grades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    student_id UUID REFERENCES students(id) ON DELETE CASCADE,
    course_id VARCHAR(50) NOT NULL,
    course_name VARCHAR(200),
    grade_value FLOAT CHECK (grade_value >= 0 AND grade_value <= 100),
    grade_letter VARCHAR(2),
    submission_date DATE NOT NULL,
    assignment_type VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create discipline_incidents table
CREATE TABLE IF NOT EXISTS discipline_incidents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    student_id UUID REFERENCES students(id) ON DELETE CASCADE,
    incident_date DATE NOT NULL,
    severity_level INTEGER CHECK (severity_level >= 1 AND severity_level <= 5),
    incident_type VARCHAR(100),
    description TEXT,
    resolution VARCHAR(200),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create model_metadata table
CREATE TABLE IF NOT EXISTS model_metadata (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_version VARCHAR(50) UNIQUE NOT NULL,
    model_type VARCHAR(50),
    training_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    performance_metrics JSONB DEFAULT '{}',
    hyperparameters JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better query performance
CREATE INDEX idx_students_district_id ON students(district_id);
CREATE INDEX idx_students_grade_level ON students(grade_level);
CREATE INDEX idx_student_features_student_date ON student_features(student_id, feature_date DESC);
CREATE INDEX idx_predictions_student_id ON predictions(student_id);
CREATE INDEX idx_predictions_date ON predictions(prediction_date DESC);
CREATE INDEX idx_predictions_risk_score ON predictions(risk_score DESC);
CREATE INDEX idx_attendance_student_date ON attendance_records(student_id, date DESC);
CREATE INDEX idx_grades_student_date ON grades(student_id, submission_date DESC);
CREATE INDEX idx_discipline_student_date ON discipline_incidents(student_id, incident_date DESC);

-- Create materialized view for student risk summary
CREATE MATERIALIZED VIEW IF NOT EXISTS student_risk_summary AS
SELECT 
    s.id as student_id,
    s.district_id,
    s.grade_level,
    p.risk_score as latest_risk_score,
    p.risk_category as latest_risk_category,
    p.prediction_date as last_assessment_date,
    COUNT(DISTINCT a.date) FILTER (WHERE a.status = 'absent' AND a.date >= CURRENT_DATE - INTERVAL '30 days') as absences_30d,
    AVG(g.grade_value) FILTER (WHERE g.submission_date >= CURRENT_DATE - INTERVAL '90 days') as avg_grade_90d,
    COUNT(DISTINCT d.id) FILTER (WHERE d.incident_date >= CURRENT_DATE - INTERVAL '30 days') as incidents_30d
FROM students s
LEFT JOIN LATERAL (
    SELECT risk_score, risk_category, prediction_date 
    FROM predictions 
    WHERE student_id = s.id 
    ORDER BY prediction_date DESC 
    LIMIT 1
) p ON true
LEFT JOIN attendance_records a ON s.id = a.student_id
LEFT JOIN grades g ON s.id = g.student_id
LEFT JOIN discipline_incidents d ON s.id = d.student_id
GROUP BY s.id, s.district_id, s.grade_level, p.risk_score, p.risk_category, p.prediction_date;

-- Create index on materialized view
CREATE INDEX idx_risk_summary_risk_score ON student_risk_summary(latest_risk_score DESC);

-- Function to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_students_updated_at BEFORE UPDATE ON students
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions (adjust as needed for production)
GRANT ALL PRIVILEGES ON SCHEMA edupulse TO edupulse_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA edupulse TO edupulse_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA edupulse TO edupulse_user;