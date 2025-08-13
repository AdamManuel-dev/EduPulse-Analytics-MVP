# Python Client Examples

This guide provides comprehensive examples of interacting with the EduPulse Analytics API using Python.

## Installation

First, install the required packages:

```bash
pip install requests pandas numpy matplotlib seaborn
```

## Basic Setup

### Client Configuration

```python
import requests
import json
import pandas as pd
from datetime import datetime, date
from typing import Dict, List, Optional

class EduPulseClient:
    """Python client for EduPulse Analytics API"""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
        # Set up authentication if provided
        if api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            })
    
    def _request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make HTTP request to API"""
        url = f"{self.base_url}{endpoint}"
        response = self.session.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> Dict:
        """Check API health status"""
        return self._request('GET', '/health')
```

## Authentication

### Login and Token Management

```python
class AuthenticatedClient(EduPulseClient):
    """Client with authentication support"""
    
    def login(self, username: str, password: str) -> str:
        """Authenticate and get access token"""
        response = self._request(
            'POST',
            '/auth/login',
            json={'username': username, 'password': password}
        )
        
        # Store token
        self.token = response['access_token']
        self.session.headers.update({
            'Authorization': f'Bearer {self.token}'
        })
        
        return self.token
    
    def refresh_token(self, refresh_token: str) -> str:
        """Refresh access token"""
        response = self._request(
            'POST',
            '/auth/refresh',
            json={'refresh_token': refresh_token}
        )
        
        self.token = response['access_token']
        self.session.headers.update({
            'Authorization': f'Bearer {self.token}'
        })
        
        return self.token
```

## Prediction Examples

### Single Student Risk Prediction

```python
def predict_student_risk(client: EduPulseClient, student_id: str) -> Dict:
    """
    Get dropout risk prediction for a single student
    
    Args:
        client: API client instance
        student_id: Unique student identifier
    
    Returns:
        Prediction results with risk score and factors
    """
    response = client._request(
        'POST',
        '/api/v1/predictions/predict',
        json={
            'student_id': student_id,
            'include_factors': True
        }
    )
    
    # Parse response
    result = {
        'student_id': response['student_id'],
        'risk_score': response['risk_score'],
        'risk_category': response['risk_category'],
        'confidence': response['confidence'],
        'factors': response.get('contributing_factors', [])
    }
    
    return result

# Example usage
client = EduPulseClient('http://localhost:8000')
prediction = predict_student_risk(client, 'STU001')

print(f"Student: {prediction['student_id']}")
print(f"Risk Score: {prediction['risk_score']:.2%}")
print(f"Category: {prediction['risk_category']}")
print("\nTop Risk Factors:")
for factor in prediction['factors'][:3]:
    print(f"  - {factor['factor']}: {factor['details']}")
```

### Batch Predictions

```python
def predict_batch(client: EduPulseClient, student_ids: List[str]) -> pd.DataFrame:
    """
    Get predictions for multiple students
    
    Args:
        client: API client instance
        student_ids: List of student IDs
    
    Returns:
        DataFrame with prediction results
    """
    response = client._request(
        'POST',
        '/api/v1/predictions/predict-batch',
        json={
            'student_ids': student_ids,
            'top_k': len(student_ids)
        }
    )
    
    # Convert to DataFrame
    df = pd.DataFrame(response['predictions'])
    df['risk_score'] = df['risk_score'].round(3)
    
    return df

# Example usage
student_ids = ['STU001', 'STU002', 'STU003', 'STU004', 'STU005']
predictions_df = predict_batch(client, student_ids)

# Display results
print(predictions_df.sort_values('risk_score', ascending=False))

# Visualize risk distribution
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
predictions_df['risk_category'].value_counts().plot(kind='bar')
plt.title('Risk Category Distribution')
plt.xlabel('Risk Category')
plt.ylabel('Number of Students')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### Historical Predictions

```python
def get_prediction_history(
    client: EduPulseClient,
    student_id: str,
    days: int = 30
) -> pd.DataFrame:
    """
    Get historical predictions for a student
    
    Args:
        client: API client instance
        student_id: Student identifier
        days: Number of days of history
    
    Returns:
        DataFrame with historical predictions
    """
    response = client._request(
        'GET',
        f'/api/v1/predictions/history/{student_id}',
        params={'days': days}
    )
    
    # Convert to DataFrame with datetime index
    df = pd.DataFrame(response['predictions'])
    df['prediction_date'] = pd.to_datetime(df['prediction_date'])
    df.set_index('prediction_date', inplace=True)
    
    return df

# Example: Track risk trend
history_df = get_prediction_history(client, 'STU001', days=90)

# Plot risk trend
plt.figure(figsize=(12, 6))
plt.plot(history_df.index, history_df['risk_score'], marker='o')
plt.axhline(y=0.7, color='r', linestyle='--', label='High Risk Threshold')
plt.axhline(y=0.3, color='g', linestyle='--', label='Low Risk Threshold')
plt.title(f'Risk Score Trend for Student STU001')
plt.xlabel('Date')
plt.ylabel('Risk Score')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## Student Management

### List Students with Filtering

```python
def list_students(
    client: EduPulseClient,
    grade_level: Optional[int] = None,
    risk_category: Optional[str] = None,
    limit: int = 100
) -> pd.DataFrame:
    """
    List students with optional filtering
    
    Args:
        client: API client instance
        grade_level: Filter by grade (1-12)
        risk_category: Filter by risk level (low, medium, high, critical)
        limit: Maximum number of results
    
    Returns:
        DataFrame with student information
    """
    params = {'limit': limit}
    if grade_level:
        params['grade_level'] = grade_level
    if risk_category:
        params['risk_category'] = risk_category
    
    response = client._request(
        'GET',
        '/api/v1/students',
        params=params
    )
    
    return pd.DataFrame(response['students'])

# Example: Get all high-risk 10th graders
high_risk_sophomores = list_students(
    client,
    grade_level=10,
    risk_category='high'
)

print(f"Found {len(high_risk_sophomores)} high-risk 10th graders")
print(high_risk_sophomores[['student_id', 'first_name', 'last_name', 'risk_score']])
```

### Create New Student

```python
def create_student(client: EduPulseClient, student_data: Dict) -> Dict:
    """
    Create a new student record
    
    Args:
        client: API client instance
        student_data: Student information
    
    Returns:
        Created student record
    """
    response = client._request(
        'POST',
        '/api/v1/students',
        json=student_data
    )
    
    return response

# Example
new_student = {
    'student_id': 'STU999',
    'first_name': 'John',
    'last_name': 'Doe',
    'grade_level': 9,
    'date_of_birth': '2008-05-15',
    'enrollment_date': '2023-08-20',
    'gender': 'M',
    'ethnicity': 'Hispanic',
    'socioeconomic_status': 'middle'
}

created = create_student(client, new_student)
print(f"Created student: {created['student_id']}")
```

## Analytics and Reporting

### Risk Distribution Analysis

```python
def analyze_risk_distribution(client: EduPulseClient) -> Dict:
    """
    Analyze risk distribution across all students
    """
    # Get all students
    all_students = list_students(client, limit=1000)
    
    # Calculate statistics
    stats = {
        'total_students': len(all_students),
        'risk_categories': all_students['risk_category'].value_counts().to_dict(),
        'avg_risk_score': all_students['risk_score'].mean(),
        'std_risk_score': all_students['risk_score'].std(),
        'high_risk_count': len(all_students[all_students['risk_category'] == 'high']),
        'critical_risk_count': len(all_students[all_students['risk_category'] == 'critical'])
    }
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Risk category distribution
    categories = list(stats['risk_categories'].keys())
    counts = list(stats['risk_categories'].values())
    axes[0].bar(categories, counts, color=['green', 'yellow', 'orange', 'red'])
    axes[0].set_title('Risk Category Distribution')
    axes[0].set_xlabel('Risk Category')
    axes[0].set_ylabel('Number of Students')
    
    # Risk score histogram
    axes[1].hist(all_students['risk_score'], bins=20, edgecolor='black')
    axes[1].axvline(stats['avg_risk_score'], color='red', linestyle='--', 
                    label=f"Mean: {stats['avg_risk_score']:.2f}")
    axes[1].set_title('Risk Score Distribution')
    axes[1].set_xlabel('Risk Score')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    return stats
```

### Intervention Priority List

```python
def get_intervention_priorities(
    client: EduPulseClient,
    min_risk_score: float = 0.7,
    limit: int = 20
) -> pd.DataFrame:
    """
    Get prioritized list of students needing intervention
    """
    # Get all students
    students = list_students(client, limit=500)
    
    # Filter high-risk students
    high_risk = students[students['risk_score'] >= min_risk_score].copy()
    
    # Get additional details for each student
    for idx, student in high_risk.iterrows():
        prediction = predict_student_risk(client, student['student_id'])
        high_risk.at[idx, 'top_factor'] = prediction['factors'][0]['factor'] if prediction['factors'] else 'Unknown'
        high_risk.at[idx, 'confidence'] = prediction['confidence']
    
    # Sort by risk score and confidence
    high_risk['priority_score'] = high_risk['risk_score'] * high_risk['confidence']
    high_risk = high_risk.sort_values('priority_score', ascending=False)
    
    # Select top students
    priority_list = high_risk.head(limit)[
        ['student_id', 'first_name', 'last_name', 'grade_level', 
         'risk_score', 'risk_category', 'top_factor', 'priority_score']
    ]
    
    return priority_list

# Example usage
priorities = get_intervention_priorities(client, min_risk_score=0.75)
print("Students Requiring Immediate Intervention:")
print(priorities.to_string(index=False))
```

## Model Training

### Trigger Model Retraining

```python
def train_model(
    client: EduPulseClient,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001
) -> Dict:
    """
    Trigger model retraining
    """
    response = client._request(
        'POST',
        '/api/v1/training/train',
        json={
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'early_stopping': True,
            'validation_split': 0.2
        }
    )
    
    job_id = response['job_id']
    print(f"Training job started: {job_id}")
    
    # Monitor training progress
    import time
    while True:
        status = client._request('GET', f'/api/v1/training/status/{job_id}')
        
        print(f"Status: {status['status']}")
        print(f"Progress: {status['progress']}%")
        print(f"Current Epoch: {status.get('current_epoch', 'N/A')}")
        print(f"Training Loss: {status.get('training_loss', 'N/A')}")
        print("-" * 40)
        
        if status['status'] in ['completed', 'failed']:
            break
        
        time.sleep(10)  # Check every 10 seconds
    
    return status
```

## Error Handling

### Robust API Client

```python
class RobustEduPulseClient(EduPulseClient):
    """Client with retry logic and error handling"""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None, max_retries: int = 3):
        super().__init__(base_url, api_key)
        self.max_retries = max_retries
    
    def _request_with_retry(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make request with automatic retry on failure"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return self._request(method, endpoint, **kwargs)
            
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Rate limited
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                elif e.response.status_code >= 500:  # Server error
                    print(f"Server error. Retrying... (Attempt {attempt + 1})")
                    time.sleep(1)
                else:
                    raise  # Don't retry client errors
                
                last_error = e
            
            except requests.exceptions.ConnectionError as e:
                print(f"Connection error. Retrying... (Attempt {attempt + 1})")
                time.sleep(2)
                last_error = e
        
        raise last_error or Exception("Max retries exceeded")
```

## Data Export

### Export Predictions to CSV

```python
def export_predictions_csv(
    client: EduPulseClient,
    output_file: str,
    grade_levels: Optional[List[int]] = None
):
    """
    Export all predictions to CSV file
    """
    # Get all students
    students = list_students(client, limit=10000)
    
    # Filter by grade if specified
    if grade_levels:
        students = students[students['grade_level'].isin(grade_levels)]
    
    # Get predictions for all students
    all_predictions = []
    
    for student_id in students['student_id']:
        try:
            prediction = predict_student_risk(client, student_id)
            all_predictions.append(prediction)
        except Exception as e:
            print(f"Error predicting for {student_id}: {e}")
    
    # Create DataFrame
    predictions_df = pd.DataFrame(all_predictions)
    
    # Merge with student info
    result = pd.merge(
        students,
        predictions_df,
        on='student_id',
        how='left'
    )
    
    # Export to CSV
    result.to_csv(output_file, index=False)
    print(f"Exported {len(result)} predictions to {output_file}")

# Example usage
export_predictions_csv(
    client,
    'predictions_export.csv',
    grade_levels=[9, 10, 11, 12]
)
```

## Webhook Integration

### Set Up Webhook for High-Risk Alerts

```python
def setup_risk_webhook(client: EduPulseClient, webhook_url: str, threshold: float = 0.8):
    """
    Configure webhook for high-risk student alerts
    """
    response = client._request(
        'POST',
        '/api/v1/webhooks',
        json={
            'url': webhook_url,
            'event_type': 'high_risk_detected',
            'config': {
                'risk_threshold': threshold,
                'include_factors': True,
                'batch_notifications': False
            }
        }
    )
    
    print(f"Webhook configured: {response['webhook_id']}")
    return response['webhook_id']

# Example webhook handler (Flask)
from flask import Flask, request

app = Flask(__name__)

@app.route('/webhook/high-risk', methods=['POST'])
def handle_high_risk_alert():
    data = request.json
    
    student_id = data['student_id']
    risk_score = data['risk_score']
    factors = data['contributing_factors']
    
    # Send email alert
    send_alert_email(
        to='counselor@school.edu',
        subject=f'High Risk Alert: Student {student_id}',
        body=f'Student {student_id} has risk score {risk_score:.2%}\n'
             f'Top factors: {factors[:3]}'
    )
    
    return {'status': 'received'}, 200
```

## Complete Example Script

```python
#!/usr/bin/env python3
"""
Complete example script for EduPulse Analytics API
"""

import argparse
import json
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='EduPulse Analytics Client')
    parser.add_argument('--api-url', default='http://localhost:8000',
                       help='API base URL')
    parser.add_argument('--api-key', help='API key for authentication')
    parser.add_argument('--action', required=True,
                       choices=['predict', 'batch', 'analyze', 'export'],
                       help='Action to perform')
    parser.add_argument('--student-id', help='Student ID for prediction')
    parser.add_argument('--output', help='Output file for exports')
    
    args = parser.parse_args()
    
    # Initialize client
    client = RobustEduPulseClient(args.api_url, args.api_key)
    
    # Perform action
    if args.action == 'predict':
        if not args.student_id:
            print("Error: --student-id required for prediction")
            return
        
        result = predict_student_risk(client, args.student_id)
        print(json.dumps(result, indent=2))
    
    elif args.action == 'batch':
        students_df = list_students(client, limit=100)
        predictions = predict_batch(client, students_df['student_id'].tolist())
        print(predictions)
    
    elif args.action == 'analyze':
        stats = analyze_risk_distribution(client)
        print(json.dumps(stats, indent=2))
    
    elif args.action == 'export':
        if not args.output:
            args.output = f"export_{datetime.now().strftime('%Y%m%d')}.csv"
        
        export_predictions_csv(client, args.output)

if __name__ == '__main__':
    main()
```

## Next Steps

- Review the [JavaScript Client Examples](./javascript_client.md) for frontend integration
- See [Data Import Examples](./data_import.md) for bulk data loading
- Check the [API Reference](../API_REFERENCE.md) for complete endpoint documentation