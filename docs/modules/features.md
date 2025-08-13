# Feature Engineering Module Documentation

## Overview

The Feature Engineering module extracts and transforms raw student data into 42 meaningful features used by the machine learning model for dropout risk prediction. These features capture attendance patterns, academic performance, behavioral indicators, and demographic factors.

## Feature Categories

### 1. Attendance Features (12 features)

These features capture student attendance patterns and trends.

| Feature Name | Description | Range | Calculation |
|-------------|------------|-------|-------------|
| `attendance_rate_30d` | Attendance rate over last 30 days | [0, 1] | (Days Present) / (Total School Days) |
| `attendance_rate_90d` | Attendance rate over last 90 days | [0, 1] | (Days Present) / (Total School Days) |
| `absence_rate_30d` | Absence rate over last 30 days | [0, 1] | (Days Absent) / (Total School Days) |
| `absence_rate_90d` | Absence rate over last 90 days | [0, 1] | (Days Absent) / (Total School Days) |
| `tardy_rate_30d` | Tardiness rate over last 30 days | [0, 1] | (Days Tardy) / (Total School Days) |
| `tardy_rate_90d` | Tardiness rate over last 90 days | [0, 1] | (Days Tardy) / (Total School Days) |
| `consecutive_absences` | Maximum consecutive days absent | [0, ∞) | Count of sequential absences |
| `attendance_trend` | Trend in attendance (improving/declining) | [-1, 1] | Linear regression slope |
| `excused_absence_ratio` | Ratio of excused to total absences | [0, 1] | (Excused Absences) / (Total Absences) |
| `monday_absence_rate` | Absence rate specifically on Mondays | [0, 1] | (Monday Absences) / (Total Mondays) |
| `friday_absence_rate` | Absence rate specifically on Fridays | [0, 1] | (Friday Absences) / (Total Fridays) |
| `attendance_volatility` | Consistency of attendance patterns | [0, ∞) | Standard deviation of weekly attendance |

### 2. Academic Performance Features (15 features)

Features related to grades, academic progress, and performance trends.

| Feature Name | Description | Range | Calculation |
|-------------|------------|-------|-------------|
| `current_gpa` | Current grade point average | [0, 4] | Weighted average of grades |
| `gpa_trend` | Change in GPA over time | [-4, 4] | Current GPA - Previous GPA |
| `math_grade_avg` | Average grade in math courses | [0, 100] | Mean of all math grades |
| `english_grade_avg` | Average grade in English courses | [0, 100] | Mean of all English grades |
| `science_grade_avg` | Average grade in science courses | [0, 100] | Mean of all science grades |
| `homework_completion_rate` | Rate of homework submissions | [0, 1] | (Submitted) / (Assigned) |
| `test_score_avg` | Average test scores | [0, 100] | Mean of all test scores |
| `quiz_score_avg` | Average quiz scores | [0, 100] | Mean of all quiz scores |
| `assignment_late_rate` | Rate of late submissions | [0, 1] | (Late Submissions) / (Total Submissions) |
| `failed_courses_count` | Number of failed courses | [0, ∞) | Count of courses with F grade |
| `grade_volatility` | Consistency of grades | [0, ∞) | Standard deviation of grades |
| `improvement_rate` | Rate of grade improvement | [-1, 1] | (Improved Grades) / (Total Grades) |
| `course_difficulty_avg` | Average difficulty of enrolled courses | [1, 5] | Mean difficulty rating |
| `credit_completion_rate` | Rate of credit completion | [0, 1] | (Earned Credits) / (Attempted Credits) |
| `semester_grade_trend` | Trend across semesters | [-1, 1] | Linear regression slope |

### 3. Behavioral Features (7 features)

Features capturing disciplinary incidents and behavioral patterns.

| Feature Name | Description | Range | Calculation |
|-------------|------------|-------|-------------|
| `discipline_incidents_count` | Total disciplinary incidents | [0, ∞) | Count of all incidents |
| `discipline_severity_avg` | Average severity of incidents | [0, 5] | Mean severity score |
| `discipline_trend` | Trend in disciplinary issues | [-1, 1] | Change rate over time |
| `suspension_count` | Number of suspensions | [0, ∞) | Count of suspension events |
| `detention_count` | Number of detentions | [0, ∞) | Count of detention events |
| `positive_behavior_count` | Positive behavior recognitions | [0, ∞) | Count of positive reports |
| `behavior_improvement_rate` | Rate of behavioral improvement | [-1, 1] | (Current - Previous) / Previous |

### 4. Demographic Features (8 features)

Student demographic and socioeconomic indicators.

| Feature Name | Description | Range | Calculation |
|-------------|------------|-------|-------------|
| `age` | Current age in years | [5, 20] | Current Date - Date of Birth |
| `grade_level` | Current grade level | [1, 12] | Enrolled grade |
| `years_enrolled` | Years at current school | [0, 12] | Current Date - Enrollment Date |
| `gender_encoded` | Gender (encoded) | {0, 1} | Binary encoding |
| `ethnicity_risk_factor` | Ethnicity-based risk factor | [0, 1] | Statistical risk mapping |
| `socioeconomic_score` | Socioeconomic status indicator | [0, 1] | Normalized SES score |
| `special_ed_flag` | Special education status | {0, 1} | Binary flag |
| `english_learner_flag` | English learner status | {0, 1} | Binary flag |

## Feature Extraction Pipeline

### Architecture

```python
class FeaturePipeline:
    """Main feature extraction pipeline"""
    
    def __init__(self, db_session):
        self.db = db_session
        self.extractors = [
            AttendanceFeatureExtractor(),
            GradesFeatureExtractor(),
            DisciplineFeatureExtractor(),
            DemographicFeatureExtractor()
        ]
    
    def extract_features(self, student_id: str, reference_date: date) -> np.ndarray:
        """Extract all features for a student"""
        features = []
        for extractor in self.extractors:
            features.extend(extractor.extract(student_id, reference_date, self.db))
        return np.array(features)
```

### Feature Extractors

#### Attendance Feature Extractor

```python
class AttendanceFeatureExtractor:
    """Extract attendance-related features"""
    
    def extract(self, student_id: str, reference_date: date, db: Session):
        # Get attendance records
        records = db.query(AttendanceRecord).filter(
            AttendanceRecord.student_id == student_id,
            AttendanceRecord.date <= reference_date
        ).all()
        
        # Calculate features
        features = {
            'attendance_rate_30d': self._calculate_rate(records, 30, 'present'),
            'absence_rate_30d': self._calculate_rate(records, 30, 'absent'),
            'tardy_rate_30d': self._calculate_rate(records, 30, 'tardy'),
            'consecutive_absences': self._max_consecutive(records, 'absent'),
            'attendance_trend': self._calculate_trend(records),
            # ... more features
        }
        
        return list(features.values())
```

#### Academic Feature Extractor

```python
class GradesFeatureExtractor:
    """Extract academic performance features"""
    
    def extract(self, student_id: str, reference_date: date, db: Session):
        # Get grade records
        grades = db.query(Grade).filter(
            Grade.student_id == student_id,
            Grade.submission_date <= reference_date
        ).all()
        
        # Calculate GPA
        current_gpa = self._calculate_gpa(grades)
        
        # Subject-specific averages
        math_avg = self._subject_average(grades, 'MATH')
        english_avg = self._subject_average(grades, 'ENG')
        science_avg = self._subject_average(grades, 'SCI')
        
        # Performance metrics
        homework_rate = self._completion_rate(grades, 'homework')
        test_avg = self._assignment_average(grades, 'test')
        
        return [current_gpa, math_avg, english_avg, science_avg, homework_rate, test_avg, ...]
```

## Feature Normalization

All features are normalized to ensure consistent scaling:

```python
class FeatureNormalizer:
    """Normalize features to [0, 1] or standard scale"""
    
    def __init__(self):
        self.scalers = {
            'minmax': MinMaxScaler(),
            'standard': StandardScaler(),
            'robust': RobustScaler()
        }
    
    def normalize(self, features: np.ndarray, method='minmax'):
        """Normalize feature values"""
        scaler = self.scalers[method]
        return scaler.fit_transform(features.reshape(-1, 1)).flatten()
```

### Normalization Rules

| Feature Type | Normalization Method | Reason |
|-------------|---------------------|---------|
| Rates/Percentages | None (already [0,1]) | Natural bounds |
| Counts | Min-Max | Bound to [0,1] |
| Grades | Division by 100 | Convert to [0,1] |
| Trends | Tanh | Bound to [-1,1] |
| Demographics | One-hot or ordinal | Categorical handling |

## Feature Importance

Based on model analysis, the most important features for dropout prediction:

### Top 10 Most Important Features

1. **attendance_rate_30d** (0.152) - Recent attendance is strongest predictor
2. **current_gpa** (0.098) - Academic performance indicator
3. **consecutive_absences** (0.087) - Pattern of continuous absence
4. **discipline_incidents_count** (0.074) - Behavioral issues
5. **homework_completion_rate** (0.068) - Engagement indicator
6. **gpa_trend** (0.061) - Direction of academic performance
7. **socioeconomic_score** (0.055) - Economic factors
8. **attendance_trend** (0.048) - Attendance trajectory
9. **failed_courses_count** (0.043) - Academic struggles
10. **grade_volatility** (0.039) - Consistency of performance

## Feature Engineering Best Practices

### 1. Handling Missing Data

```python
def handle_missing(self, value, feature_type):
    """Handle missing values appropriately"""
    
    strategies = {
        'attendance': 0,  # Assume absent if no record
        'grade': np.nan,  # Will be imputed with mean
        'demographic': 'unknown',  # Categorical unknown
        'behavioral': 0  # Assume no incidents
    }
    
    if value is None:
        return strategies.get(feature_type, 0)
    return value
```

### 2. Temporal Features

```python
def create_temporal_features(self, records, window_sizes=[7, 30, 90]):
    """Create features for multiple time windows"""
    
    features = {}
    for window in window_sizes:
        window_data = self._filter_by_window(records, window)
        features[f'mean_{window}d'] = np.mean(window_data)
        features[f'std_{window}d'] = np.std(window_data)
        features[f'trend_{window}d'] = self._calculate_trend(window_data)
    
    return features
```

### 3. Feature Interactions

```python
def create_interaction_features(self, base_features):
    """Create interaction features"""
    
    interactions = []
    
    # Attendance × GPA interaction
    interactions.append(
        base_features['attendance_rate'] * base_features['current_gpa']
    )
    
    # Discipline × Socioeconomic interaction
    interactions.append(
        base_features['discipline_count'] * base_features['socioeconomic_score']
    )
    
    return interactions
```

## Feature Validation

### Data Quality Checks

```python
def validate_features(self, features: np.ndarray):
    """Validate extracted features"""
    
    checks = {
        'no_nan': not np.isnan(features).any(),
        'no_inf': not np.isinf(features).any(),
        'correct_length': len(features) == 42,
        'valid_ranges': self._check_ranges(features)
    }
    
    if not all(checks.values()):
        raise ValueError(f"Feature validation failed: {checks}")
    
    return True
```

### Range Validation

```python
FEATURE_RANGES = {
    'attendance_rate_30d': (0, 1),
    'current_gpa': (0, 4),
    'discipline_incidents_count': (0, float('inf')),
    'age': (5, 20),
    # ... all 42 features
}

def _check_ranges(self, features):
    """Check if features are within expected ranges"""
    for i, (name, (min_val, max_val)) in enumerate(FEATURE_RANGES.items()):
        if not min_val <= features[i] <= max_val:
            logger.warning(f"Feature {name} out of range: {features[i]}")
            return False
    return True
```

## Performance Optimization

### Caching Strategy

```python
@lru_cache(maxsize=1000)
def get_cached_features(student_id: str, date_str: str):
    """Cache extracted features"""
    return extract_features(student_id, datetime.fromisoformat(date_str))
```

### Batch Processing

```python
def extract_features_batch(self, student_ids: List[str], reference_date: date):
    """Extract features for multiple students efficiently"""
    
    # Bulk load data
    all_attendance = db.query(AttendanceRecord).filter(
        AttendanceRecord.student_id.in_(student_ids)
    ).all()
    
    # Group by student
    grouped = defaultdict(list)
    for record in all_attendance:
        grouped[record.student_id].append(record)
    
    # Extract features per student
    features = {}
    for student_id in student_ids:
        features[student_id] = self.extract_single(grouped[student_id])
    
    return features
```

## Feature Updates

### Adding New Features

1. Define feature in appropriate extractor class
2. Add to feature name list
3. Update normalization rules
4. Retrain model with new feature set
5. Update documentation

### Feature Deprecation

1. Mark feature as deprecated in code
2. Maintain backward compatibility
3. Log warnings when deprecated features are used
4. Remove after grace period

## Monitoring & Debugging

### Feature Distribution Monitoring

```python
def monitor_feature_distributions(self, features_batch):
    """Monitor feature distributions for drift"""
    
    for i, feature_name in enumerate(FEATURE_NAMES):
        values = features_batch[:, i]
        
        stats = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'nulls': np.isnan(values).sum()
        }
        
        # Check for anomalies
        if stats['std'] == 0:
            logger.warning(f"Feature {feature_name} has zero variance")
        if stats['nulls'] > len(values) * 0.1:
            logger.warning(f"Feature {feature_name} has >10% missing values")
```

## Related Documentation

- [Feature Extraction Module](./features.md) - Implementation details
- [ML Pipeline](../ML_PIPELINE.md) - How features are used in training
- [API Reference](../API_REFERENCE.md) - Feature-related endpoints