# Feature Extraction Module

## Overview

The feature extraction module is responsible for transforming raw student data into numerical features suitable for machine learning. It implements a modular, extensible architecture with separate extractors for different data domains.

## Architecture

```
FeaturePipeline (Orchestrator)
       │
       ├── AttendanceFeatureExtractor
       ├── GradesFeatureExtractor
       └── DisciplineFeatureExtractor
```

## Components

### FeaturePipeline

**Location**: `src/features/pipeline.py`

The main orchestrator that coordinates all feature extractors.

**Key Methods:**

- `extract_features(student_id, start_date, end_date)` - Extract all features for a student
- `batch_extract(student_ids, start_date, end_date)` - Batch feature extraction
- `get_cached_features(student_id, date)` - Retrieve cached features

**Features:**
- Parallel extraction from multiple data sources
- Redis caching with TTL
- Database persistence
- Batch processing optimization

### BaseFeatureExtractor

**Location**: `src/features/base.py`

Abstract base class providing common functionality for all extractors.

**Key Methods:**
- `extract()` - Abstract method for feature extraction
- `validate_data()` - Data validation
- `handle_missing_values()` - Missing value imputation
- `normalize_features()` - Feature normalization

### AttendanceFeatureExtractor

**Location**: `src/features/attendance.py`

Extracts 14 attendance-related features.

**Features Generated:**

1. **Basic Metrics** (6 features):
   - `attendance_rate`: Overall attendance rate (0-1)
   - `absence_count`: Total absences
   - `tardy_count`: Total tardies
   - `excused_absence_rate`: Proportion of excused absences
   - `consecutive_absences`: Max consecutive absence days
   - `monday_absence_rate`: Monday-specific absence pattern

2. **Trend Analysis** (4 features):
   - `attendance_trend`: Linear trend coefficient
   - `attendance_volatility`: Standard deviation of weekly rates
   - `improving_attendance`: Boolean indicator of improvement
   - `attendance_streak`: Current attendance streak days

3. **Pattern Detection** (4 features):
   - `chronic_absenteeism`: Risk indicator (>10% absence)
   - `friday_pattern`: End-of-week absence pattern
   - `post_break_pattern`: Post-vacation absence spike
   - `period_specific_absence`: Class-specific patterns

### GradesFeatureExtractor

**Location**: `src/features/grades.py`

Extracts 15 academic performance features.

**Features Generated:**

1. **Current Performance** (5 features):
   - `current_gpa`: Current GPA (0-4.0)
   - `core_subject_avg`: Math, English, Science average
   - `failing_courses`: Number of failing grades
   - `grade_volatility`: Consistency across subjects
   - `assignment_completion_rate`: Homework submission rate

2. **Trends** (5 features):
   - `gpa_trend`: GPA change over time
   - `improvement_rate`: Grade improvement velocity
   - `subject_drop_indicator`: Sudden performance drops
   - `assessment_performance`: Test vs homework delta
   - `participation_score`: Class participation metrics

3. **Risk Indicators** (5 features):
   - `at_risk_courses`: Courses with D or F
   - `credit_deficiency`: Behind on credits
   - `math_struggle`: Math-specific difficulty
   - `reading_level_gap`: Below grade reading level
   - `standardized_test_gap`: State test performance

### DisciplineFeatureExtractor

**Location**: `src/features/discipline.py`

Extracts 13 behavioral and discipline features.

**Features Generated:**

1. **Incident Metrics** (5 features):
   - `incident_count`: Total discipline incidents
   - `severity_score`: Weighted by incident severity
   - `detention_count`: Number of detentions
   - `suspension_days`: Total suspension days
   - `behavioral_referrals`: Teacher referral count

2. **Pattern Analysis** (4 features):
   - `incident_trend`: Increasing/decreasing pattern
   - `incident_frequency`: Incidents per month
   - `escalation_pattern`: Severity escalation
   - `time_of_day_pattern`: When incidents occur

3. **Behavioral Indicators** (4 features):
   - `positive_behavior_ratio`: Positive vs negative
   - `peer_conflict_indicator`: Social issues
   - `authority_conflict`: Adult relationship issues
   - `intervention_response`: Response to interventions

## Feature Vector Structure

The complete feature vector contains 42 dimensions:

```python
feature_vector = {
    # Attendance (0-13)
    'attendance_rate': 0.92,
    'absence_count': 5,
    'tardy_count': 3,
    # ... 11 more
    
    # Grades (14-28)
    'current_gpa': 3.2,
    'core_subject_avg': 85.5,
    'failing_courses': 0,
    # ... 12 more
    
    # Discipline (29-41)
    'incident_count': 2,
    'severity_score': 0.3,
    'detention_count': 1,
    # ... 10 more
}
```

## Data Processing Pipeline

### 1. Data Collection
```python
# Gather raw data from multiple sources
attendance_data = get_attendance_records(student_id)
grade_data = get_grade_records(student_id)
discipline_data = get_discipline_records(student_id)
```

### 2. Temporal Windowing
```python
# Apply sliding window for time-series analysis
window_size = 30  # days
windows = create_sliding_windows(data, window_size)
```

### 3. Feature Extraction
```python
# Extract features for each window
features = []
for window in windows:
    window_features = extractor.extract(window)
    features.append(window_features)
```

### 4. Normalization
```python
# Normalize features to [0, 1] range
normalized = StandardScaler().fit_transform(features)
```

### 5. Validation
```python
# Validate feature quality
assert all(0 <= f <= 1 for f in normalized)
assert len(normalized) == 42
```

## Caching Strategy

### Redis Cache Structure
```
Key: features:{student_id}:{date}
Value: JSON-encoded feature vector
TTL: 3600 seconds (1 hour)
```

### Cache Invalidation
- On new data ingestion
- On manual refresh request
- After TTL expiration

## Missing Data Handling

### Imputation Strategies

1. **Attendance**: Forward-fill with decay
2. **Grades**: Mean imputation by subject
3. **Discipline**: Zero-fill (absence of incidents)

### Minimum Data Requirements
- At least 30 days of attendance data
- At least 1 grading period of grades
- No minimum for discipline (can be empty)

## Performance Optimization

### Batch Processing
```python
# Process multiple students in parallel
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(extract, sid) for sid in student_ids]
    results = [f.result() for f in futures]
```

### Database Query Optimization
- Use indexed columns for filtering
- Batch queries with IN clauses
- Leverage TimescaleDB time-series functions

### Memory Management
- Stream large datasets instead of loading into memory
- Use generators for batch processing
- Clear cache for processed batches

## Usage Examples

### Single Student Feature Extraction
```python
from src.features.pipeline import FeaturePipeline

pipeline = FeaturePipeline()
features = pipeline.extract_features(
    student_id="STU123456",
    start_date=datetime(2024, 9, 1),
    end_date=datetime(2024, 12, 31)
)
print(f"Extracted {len(features)} features")
```

### Batch Extraction with Caching
```python
student_ids = ["STU001", "STU002", "STU003"]
features = pipeline.batch_extract(
    student_ids=student_ids,
    start_date=datetime(2024, 9, 1),
    end_date=datetime(2024, 12, 31),
    use_cache=True
)
```

### Custom Feature Extractor
```python
from src.features.base import BaseFeatureExtractor

class CustomExtractor(BaseFeatureExtractor):
    def extract(self, data):
        # Custom extraction logic
        return features
```

## Testing

### Unit Tests
```bash
pytest tests/unit/test_feature_extractors.py
```

### Integration Tests
```bash
pytest tests/integration/test_feature_pipeline.py
```

### Performance Tests
```bash
pytest tests/performance/test_feature_extraction.py -v
```

## Monitoring

### Key Metrics
- Feature extraction latency
- Cache hit rate
- Missing data percentage
- Feature distribution statistics

### Alerts
- Extraction time > 5 seconds
- Cache hit rate < 80%
- Missing data > 20%

## Future Enhancements

1. **Additional Features**:
   - Social media sentiment
   - Extracurricular participation
   - Parent engagement metrics

2. **Advanced Processing**:
   - Feature selection algorithms
   - Automated feature engineering
   - Deep learning embeddings

3. **Performance**:
   - GPU acceleration for batch processing
   - Distributed extraction with Spark
   - Real-time streaming features

---

*For implementation details, see the [source code](../../src/features/).*