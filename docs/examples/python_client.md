# Python Client Examples

This guide provides comprehensive examples of interacting with the EduPulse Analytics API using Python.

## Installation

Before you begin, you'll need to install the required Python packages that enable HTTP communication, data processing, and visualization:

```bash
# Install core dependencies for API communication and data analysis
pip install requests pandas numpy matplotlib seaborn

# Alternative: Install with specific versions for reproducibility
pip install requests==2.31.0 pandas==2.1.4 numpy==1.26.4 matplotlib==3.8.2 seaborn==0.13.0
```

**What each package does:**
- `requests`: HTTP library for making API calls to EduPulse
- `pandas`: Data manipulation and analysis (handling student data tables)
- `numpy`: Numerical computing (statistical calculations on risk scores)
- `matplotlib`: Creating charts and graphs for data visualization
- `seaborn`: Statistical data visualization (enhanced plots and styling)

**System requirements:**
- Python 3.8 or higher
- Internet connection for API access
- Minimum 256MB RAM for data processing (1GB+ recommended for large datasets)

## Basic Setup

### Client Configuration

This section shows how to create a reusable client class for interacting with the EduPulse API. The client handles authentication, request formatting, and error handling automatically.

```python
import requests
import json
import pandas as pd
from datetime import datetime, date
from typing import Dict, List, Optional

class EduPulseClient:
    """
    Python client for EduPulse Analytics API
    
    This class provides a high-level interface to the EduPulse API, handling:
    - HTTP session management for connection reuse (faster repeated requests)
    - Automatic authentication header setup
    - URL construction and request formatting
    - Basic error handling with meaningful exceptions
    
    Usage:
        # For development/testing (no authentication)
        client = EduPulseClient('http://localhost:8000')
        
        # For production (with API key authentication)
        client = EduPulseClient('https://api.edupulse.com', api_key='your-key-here')
    """
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """
        Initialize the API client
        
        Args:
            base_url: EduPulse API server URL (e.g., 'http://localhost:8000')
            api_key: Optional API key for authentication (required for production)
        
        The client creates a persistent HTTP session which:
        - Reuses TCP connections (significant performance improvement for multiple requests)
        - Automatically handles cookies and connection pooling
        - Applies headers to all requests from this client instance
        """
        self.base_url = base_url.rstrip('/')  # Remove trailing slash to avoid double slashes in URLs
        self.session = requests.Session()     # Create persistent session for efficiency
        
        # Configure authentication if API key is provided
        # JWT Bearer tokens are the standard for stateless API authentication
        if api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {api_key}',           # Standard Bearer token format
                'Content-Type': 'application/json',             # Tell server we're sending JSON
                'Accept': 'application/json',                   # Tell server we expect JSON responses
                'User-Agent': 'EduPulse-Python-Client/1.0'     # Identify our client for server logs
            })
    
    def _request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """
        Make HTTP request to API with automatic error handling
        
        Args:
            method: HTTP method ('GET', 'POST', 'PUT', 'DELETE')
            endpoint: API endpoint path (e.g., '/api/v1/predict')
            **kwargs: Additional arguments passed to requests (json, params, etc.)
        
        Returns:
            Dict: Parsed JSON response from the API
            
        Raises:
            HTTPError: For 4xx/5xx HTTP status codes
            ConnectionError: For network connectivity issues
            TimeoutError: For requests that take too long
            JSONDecodeError: For invalid JSON responses
        
        This method centralizes error handling so individual API methods don't need
        to implement retry logic or parse error responses.
        """
        url = f"{self.base_url}{endpoint}"
        
        # Set reasonable timeout if not specified (prevents hanging requests)
        if 'timeout' not in kwargs:
            kwargs['timeout'] = 30  # 30 seconds is reasonable for ML inference
        
        try:
            response = self.session.request(method, url, **kwargs)
            
            # Raise exception for HTTP error status codes (4xx, 5xx)
            # This makes error handling consistent across all API methods
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            # Add context to HTTP errors for easier debugging
            error_detail = "Unknown error"
            try:
                error_response = e.response.json()
                error_detail = error_response.get('message', error_response.get('detail', str(e)))
            except:
                pass  # If error response isn't JSON, use default message
            
            raise requests.exceptions.HTTPError(
                f"API request failed ({e.response.status_code}): {error_detail}",
                response=e.response
            ) from e
    
    def health_check(self) -> Dict:
        """
        Check API health status - useful for monitoring and debugging
        
        Returns:
            Dict with keys:
            - 'status': 'healthy' or 'unhealthy'
            - 'version': API version string
            - 'timestamp': ISO timestamp when check was performed
            - 'uptime': Server uptime in seconds (if available)
        
        This is typically the first method you should call to verify:
        1. The API server is running and reachable
        2. Your client is configured with the correct URL
        3. Basic connectivity is working before attempting complex operations
        
        Example response:
            {
                "status": "healthy",
                "version": "1.0.0", 
                "timestamp": "2025-08-13T10:30:00Z",
                "uptime": 86400
            }
        """
        return self._request('GET', '/health')
```

## Authentication

### Login and Token Management

For production use, EduPulse uses JWT (JSON Web Token) authentication to secure API access. This section demonstrates how to handle user login, token storage, and automatic token refresh.

```python
import time
from datetime import datetime, timedelta

class AuthenticatedClient(EduPulseClient):
    """
    Extended client with full authentication support including automatic token refresh
    
    JWT Authentication Flow:
    1. Login with username/password to get access token + refresh token
    2. Use access token for API requests (valid for 1 hour)
    3. When access token expires, use refresh token to get new access token
    4. Refresh tokens are valid for 30 days
    
    This class handles the complexity of token lifecycle management automatically.
    """
    
    def __init__(self, base_url: str):
        """Initialize without API key - we'll get tokens via login"""
        super().__init__(base_url)
        self.access_token = None
        self.refresh_token = None
        self.token_expires_at = None
    
    def login(self, username: str, password: str) -> str:
        """
        Authenticate with username/password and obtain JWT tokens
        
        Args:
            username: User's email or username (usually counselor@school.edu format)
            password: User's password
        
        Returns:
            str: Access token for immediate use
            
        Raises:
            HTTPError: If credentials are invalid (401) or user account is disabled
            
        Security Note:
            - Passwords are sent over HTTPS only (never use HTTP in production)
            - Tokens are stored in memory only (not persisted to disk)
            - Failed login attempts are rate-limited by the server
        
        Example:
            client = AuthenticatedClient('https://api.edupulse.com')
            try:
                token = client.login('counselor@school.edu', 'secure_password')
                print(f"Login successful. Token expires in 1 hour.")
            except requests.HTTPError as e:
                if e.response.status_code == 401:
                    print("Invalid credentials")
                else:
                    print(f"Login failed: {e}")
        """
        response = self._request(
            'POST',
            '/auth/login',
            json={
                'username': username, 
                'password': password,
                'remember_me': True  # Request longer-lived refresh token
            }
        )
        
        # Extract tokens from response
        self.access_token = response['access_token']
        self.refresh_token = response['refresh_token']
        
        # Calculate when token expires (server returns expires_in seconds)
        expires_in_seconds = response.get('expires_in', 3600)  # Default 1 hour
        self.token_expires_at = datetime.now() + timedelta(seconds=expires_in_seconds)
        
        # Update session headers for future requests
        self.session.headers.update({
            'Authorization': f'Bearer {self.access_token}'
        })
        
        print(f"✓ Authenticated successfully. Token expires at {self.token_expires_at.strftime('%Y-%m-%d %H:%M:%S')}")
        return self.access_token
    
    def refresh_token_if_needed(self) -> bool:
        """
        Automatically refresh access token if it's expired or about to expire
        
        Returns:
            bool: True if token was refreshed, False if refresh wasn't needed
            
        This method is called automatically before each API request, so you
        typically don't need to call it manually. However, it's useful for
        long-running scripts that might span multiple hours.
        
        Buffer time: Tokens are refreshed 5 minutes before actual expiration
        to prevent race conditions where token expires between check and use.
        """
        if not self.refresh_token:
            return False
            
        # Check if token expires within 5 minutes (300 seconds buffer)
        if self.token_expires_at and datetime.now() + timedelta(seconds=300) > self.token_expires_at:
            try:
                new_token = self.refresh_token_manual(self.refresh_token)
                print(f"✓ Token automatically refreshed. New expiration: {self.token_expires_at.strftime('%Y-%m-%d %H:%M:%S')}")
                return True
            except Exception as e:
                print(f"⚠ Token refresh failed: {e}. You may need to login again.")
                return False
        
        return False
    
    def refresh_token_manual(self, refresh_token: str) -> str:
        """
        Manually refresh access token using refresh token
        
        Args:
            refresh_token: Valid refresh token from previous login
            
        Returns:
            str: New access token
            
        Raises:
            HTTPError: If refresh token is invalid or expired
            
        Use this method when:
        - Implementing your own token refresh logic
        - Refresh token is stored separately (e.g., database, config file)
        - You want explicit control over when tokens are refreshed
        """
        response = self._request(
            'POST',
            '/auth/refresh',
            json={'refresh_token': refresh_token}
        )
        
        # Update stored tokens
        self.access_token = response['access_token']
        if 'refresh_token' in response:
            self.refresh_token = response['refresh_token']  # Server may issue new refresh token
        
        # Update expiration time
        expires_in_seconds = response.get('expires_in', 3600)
        self.token_expires_at = datetime.now() + timedelta(seconds=expires_in_seconds)
        
        # Update session headers
        self.session.headers.update({
            'Authorization': f'Bearer {self.access_token}'
        })
        
        return self.access_token
    
    def _request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """
        Override parent method to include automatic token refresh
        
        This ensures that every API request automatically refreshes the token
        if needed, making the client "self-healing" for long-running applications.
        """
        # Try to refresh token before making request
        self.refresh_token_if_needed()
        
        # Make the actual request using parent implementation
        return super()._request(method, endpoint, **kwargs)
    
    def logout(self) -> None:
        """
        Invalidate current tokens and clear authentication
        
        This is important for security - it tells the server to blacklist
        the current tokens so they can't be used if compromised.
        """
        try:
            if self.refresh_token:
                self._request(
                    'POST',
                    '/auth/logout',
                    json={'refresh_token': self.refresh_token}
                )
        except Exception as e:
            print(f"Logout API call failed: {e}")
        
        # Clear stored tokens regardless of API call result
        self.access_token = None
        self.refresh_token = None
        self.token_expires_at = None
        
        # Remove authorization header
        if 'Authorization' in self.session.headers:
            del self.session.headers['Authorization']
        
        print("✓ Logged out successfully")

# Example usage showing complete authentication lifecycle
def demo_authentication():
    """Demonstrate complete authentication workflow"""
    
    # Initialize client
    client = AuthenticatedClient('http://localhost:8000')
    
    try:
        # Step 1: Login
        print("🔐 Logging in...")
        client.login('demo@school.edu', 'demo_password')
        
        # Step 2: Use API (token automatically managed)
        print("📊 Making API calls...")
        health = client.health_check()
        print(f"API Status: {health['status']}")
        
        # Step 3: Simulate long-running process
        print("⏱ Simulating long-running process...")
        time.sleep(2)  # In real app, this might be hours
        
        # Token refresh happens automatically on next request
        health2 = client.health_check()
        print(f"API still accessible: {health2['status']}")
        
        # Step 4: Clean logout
        print("🚪 Logging out...")
        client.logout()
        
    except Exception as e:
        print(f"❌ Authentication demo failed: {e}")

# Example for storing tokens securely (for production apps)
def secure_token_storage_example():
    """
    Example of secure token storage for production applications
    
    In production, you should:
    1. Store tokens in encrypted form
    2. Use secure storage (keyring, environment variables, encrypted files)
    3. Never log tokens or include them in error messages
    4. Implement proper token rotation policies
    """
    import keyring  # pip install keyring
    import os
    
    client = AuthenticatedClient('https://api.edupulse.com')
    
    # Try to load existing token from secure storage
    stored_refresh_token = keyring.get_password("edupulse", "refresh_token")
    
    if stored_refresh_token:
        try:
            # Attempt to use stored refresh token
            client.refresh_token_manual(stored_refresh_token)
            print("✓ Resumed session with stored token")
        except:
            print("⚠ Stored token invalid, need to login again")
            stored_refresh_token = None
    
    if not stored_refresh_token:
        # Fresh login required
        username = os.getenv('EDUPULSE_USERNAME')  # From environment variable
        password = keyring.get_password("edupulse", username)  # From secure keyring
        
        client.login(username, password)
        
        # Store refresh token securely for next time
        keyring.set_password("edupulse", "refresh_token", client.refresh_token)
        print("✓ Tokens stored securely for future use")
```

## Prediction Examples

### Single Student Risk Prediction

This is the core functionality of EduPulse - generating dropout risk predictions for individual students. The prediction combines 42 different behavioral features into a single risk score that helps counselors prioritize their intervention efforts.

```python
def predict_student_risk(client: EduPulseClient, student_id: str, include_explanations: bool = True) -> Dict:
    """
    Get dropout risk prediction for a single student with comprehensive analysis
    
    Args:
        client: API client instance (must be authenticated for production)
        student_id: Unique student identifier (format: 'STU' + digits, e.g., 'STU001234')
        include_explanations: Whether to include detailed factor explanations (default: True)
    
    Returns:
        Dict containing:
        - student_id: Echoed student identifier
        - risk_score: Probability of dropout (0.0 = no risk, 1.0 = certain dropout)
        - risk_category: Human-readable category ('low', 'medium', 'high', 'critical')
        - confidence: Model's confidence in prediction (0.0-1.0, higher = more reliable)
        - factors: List of contributing factors with weights and explanations
        - recommendations: Suggested interventions based on risk level
    
    Raises:
        HTTPError: 
            - 404: Student not found in system
            - 400: Invalid student ID format or insufficient historical data
            - 403: User lacks permission to access this student's data
        
    Performance Note:
        - Typical response time: 50-150ms (depends on data volume and server load)
        - Predictions are cached for 1 hour to improve performance
        - Model inference uses GPU acceleration when available
    
    Business Context:
        This prediction represents the likelihood that a student will drop out within
        the next academic year based on their behavioral patterns over the past 20 weeks.
        The model considers attendance rates, academic performance trends, disciplinary
        incidents, and social engagement factors.
    """
    try:
        response = client._request(
            'POST',
            '/api/v1/predictions/predict',
            json={
                'student_id': student_id,
                'include_factors': include_explanations,      # Get detailed factor breakdown
                'include_confidence': True,                   # Include model confidence score
                'include_historical_context': True           # Include comparison to past predictions
            }
        )
        
        # Parse and enrich the response with business logic
        result = {
            'student_id': response['student_id'],
            'risk_score': round(response['risk_score'], 3),  # Round to 3 decimal places for readability
            'risk_category': response['risk_category'],
            'confidence': round(response['confidence'], 3),
            'prediction_date': response.get('timestamp', datetime.now().isoformat()),
            'factors': response.get('contributing_factors', [])
        }
        
        # Add business-friendly interpretations
        result['risk_percentage'] = f"{result['risk_score']:.1%}"  # Format as percentage
        result['urgency_level'] = _interpret_urgency(result['risk_score'])
        result['recommended_actions'] = _get_intervention_recommendations(result['risk_category'])
        result['next_review_date'] = _calculate_next_review_date(result['risk_score'])
        
        # Add historical context if available
        if 'historical_trend' in response:
            result['trend'] = response['historical_trend']  # 'improving', 'stable', 'declining'
        
        return result
        
    except requests.exceptions.HTTPError as e:
        # Provide helpful error messages for common scenarios
        if e.response.status_code == 404:
            raise StudentNotFoundError(
                f"Student {student_id} not found. Possible causes:\n"
                f"- Student ID format is incorrect (should be 'STU' + digits)\n"
                f"- Student is not enrolled in the current academic year\n"
                f"- Student data hasn't been imported into EduPulse yet"
            )
        elif e.response.status_code == 400:
            error_detail = e.response.json().get('detail', 'Unknown validation error')
            raise InsufficientDataError(
                f"Cannot predict risk for {student_id}: {error_detail}\n"
                f"Common causes:\n"
                f"- Student enrolled less than 4 weeks ago (insufficient data)\n"
                f"- Missing critical data (attendance, grades, or enrollment info)\n"
                f"- Student transferred from another district recently"
            )
        else:
            raise PredictionError(f"Prediction failed for {student_id}: {e}")

def _interpret_urgency(risk_score: float) -> str:
    """Convert numeric risk score to actionable urgency level"""
    if risk_score < 0.25:
        return "Low Priority - Continue regular monitoring"
    elif risk_score < 0.50:
        return "Medium Priority - Schedule check-in within 2 weeks"
    elif risk_score < 0.75:
        return "High Priority - Intervention needed within 1 week"
    else:
        return "Critical Priority - Immediate intervention required"

def _get_intervention_recommendations(risk_category: str) -> List[str]:
    """Get specific, actionable intervention recommendations"""
    recommendations = {
        'low': [
            "Continue regular academic progress monitoring",
            "Maintain positive reinforcement for good attendance",
            "Monthly check-in with student to maintain engagement"
        ],
        'medium': [
            "Schedule student meeting within 2 weeks to assess needs",
            "Contact parents/guardians to discuss any concerns",
            "Review academic progress and identify areas for additional support",
            "Consider peer mentoring or study group participation"
        ],
        'high': [
            "PRIORITY: Schedule immediate meeting with student and parents",
            "Develop formal intervention plan with specific, measurable goals",
            "Daily check-ins with counselor or designated staff member",
            "Assess need for academic accommodations or schedule modifications",
            "Connect family with social services if barriers to attendance exist"
        ],
        'critical': [
            "URGENT: Emergency meeting with student, parents, and intervention team",
            "Assess for immediate safety concerns or crisis situations",
            "Daily monitoring with mandatory check-ins",
            "Consider intensive support services (mental health, case management)",
            "Develop emergency academic plan to prevent immediate failure",
            "Coordinate with district social worker and external support agencies"
        ]
    }
    return recommendations.get(risk_category, ["Contact system administrator - unknown risk category"])

def _calculate_next_review_date(risk_score: float) -> str:
    """Calculate when this student should be reviewed again"""
    if risk_score < 0.25:
        days = 30  # Monthly review for low risk
    elif risk_score < 0.50:
        days = 14  # Bi-weekly review for medium risk
    elif risk_score < 0.75:
        days = 7   # Weekly review for high risk
    else:
        days = 3   # Every 3 days for critical risk
    
    next_date = datetime.now() + timedelta(days=days)
    return next_date.strftime('%Y-%m-%d')

# Comprehensive usage example with error handling and business logic
def demo_student_prediction():
    """
    Demonstrate complete student risk prediction workflow
    
    This example shows how a school counselor might use the prediction API
    in their daily workflow, including error handling and action planning.
    """
    
    # Initialize client (in production, use authentication)
    client = EduPulseClient('http://localhost:8000')
    
    # List of students to check (typically from your student information system)
    students_to_check = [
        'STU001234',  # Known student
        'STU999999',  # Non-existent student (will demonstrate error handling)
        'STU005678'   # Another known student
    ]
    
    print("🎯 EduPulse Student Risk Assessment Report")
    print("=" * 60)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    successful_predictions = []
    failed_predictions = []
    
    for student_id in students_to_check:
        print(f"📊 Analyzing student {student_id}...")
        
        try:
            # Get prediction with full context
            prediction = predict_student_risk(client, student_id, include_explanations=True)
            successful_predictions.append(prediction)
            
            # Display results in counselor-friendly format
            print(f"✅ {prediction['student_id']}")
            print(f"   Risk Score: {prediction['risk_percentage']} ({prediction['risk_category']} risk)")
            print(f"   Confidence: {prediction['confidence']:.1%}")
            print(f"   Urgency: {prediction['urgency_level']}")
            print(f"   Next Review: {prediction['next_review_date']}")
            
            # Show top 3 risk factors
            if prediction['factors']:
                print(f"   Top Risk Factors:")
                for i, factor in enumerate(prediction['factors'][:3], 1):
                    print(f"     {i}. {factor['factor']}: {factor['details']} (weight: {factor['weight']:.2f})")
            
            # Show recommended actions for high-risk students
            if prediction['risk_score'] > 0.5:  # Medium risk or higher
                print(f"   🚨 RECOMMENDED ACTIONS:")
                for action in prediction['recommended_actions'][:3]:  # Show top 3 actions
                    print(f"     • {action}")
            
            print()
            
        except (StudentNotFoundError, InsufficientDataError, PredictionError) as e:
            failed_predictions.append({'student_id': student_id, 'error': str(e)})
            print(f"❌ {student_id}: {e}")
            print()
        
        except Exception as e:
            failed_predictions.append({'student_id': student_id, 'error': f"Unexpected error: {e}"})
            print(f"⚠️ {student_id}: Unexpected error - {e}")
            print()
    
    # Summary report
    print("📋 SUMMARY REPORT")
    print("-" * 30)
    print(f"✅ Successful predictions: {len(successful_predictions)}")
    print(f"❌ Failed predictions: {len(failed_predictions)}")
    
    if successful_predictions:
        # Calculate summary statistics
        risk_scores = [p['risk_score'] for p in successful_predictions]
        avg_risk = sum(risk_scores) / len(risk_scores)
        high_risk_count = sum(1 for score in risk_scores if score > 0.7)
        
        print(f"📈 Average risk score: {avg_risk:.1%}")
        print(f"🚨 Students requiring immediate attention: {high_risk_count}")
        
        # Sort by risk score for priority list
        priority_students = sorted(successful_predictions, key=lambda x: x['risk_score'], reverse=True)
        
        if priority_students[0]['risk_score'] > 0.5:
            print(f"\n🎯 HIGHEST PRIORITY STUDENT:")
            top_student = priority_students[0]
            print(f"   {top_student['student_id']} - {top_student['risk_percentage']} risk")
            print(f"   Immediate action required: {top_student['recommended_actions'][0]}")
    
    return successful_predictions, failed_predictions

# Custom exception classes for better error handling
class StudentNotFoundError(Exception):
    """Raised when student ID doesn't exist in the system"""
    pass

class InsufficientDataError(Exception):
    """Raised when student lacks sufficient data for prediction"""
    pass

class PredictionError(Exception):
    """Raised for general prediction failures"""
    pass

# Example: Simple prediction for immediate use
if __name__ == "__main__":
    # Quick example - predict single student
    try:
        client = EduPulseClient('http://localhost:8000')
        result = predict_student_risk(client, 'STU001')
        
        print(f"🎯 Risk Assessment for {result['student_id']}")
        print(f"Risk Level: {result['risk_percentage']} ({result['risk_category']})")
        print(f"Urgency: {result['urgency_level']}")
        print(f"Next Review: {result['next_review_date']}")
        
        if result['factors']:
            print(f"\nTop Risk Factor: {result['factors'][0]['factor']}")
            print(f"Details: {result['factors'][0]['details']}")
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
```

### Batch Predictions

Batch predictions are essential for processing large groups of students efficiently. This is typically used for daily risk assessments, monthly reports, or analyzing entire grade levels. The batch API is optimized for high throughput and can process hundreds of students in a single request.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Optional

def predict_batch(
    client: EduPulseClient, 
    student_ids: List[str], 
    include_factors: bool = False,
    async_processing: bool = False,
    chunk_size: int = 50
) -> pd.DataFrame:
    """
    Get predictions for multiple students with intelligent batching and error handling
    
    Args:
        client: API client instance (must be authenticated)
        student_ids: List of student IDs to process (max 1000 per request)
        include_factors: Whether to include detailed risk factors (increases response time)
        async_processing: Whether to use async processing for very large batches
        chunk_size: Number of students to process per API call (optimize for your network)
    
    Returns:
        DataFrame with columns:
        - student_id: Student identifier
        - risk_score: Dropout probability (0.0-1.0)
        - risk_category: Human-readable risk level
        - confidence: Model confidence in prediction
        - prediction_date: When prediction was generated
        - factors: Risk factors (if include_factors=True)
    
    Performance Notes:
        - Batch processing is 5-10x faster than individual predictions
        - Typical throughput: 200-500 students/minute (depends on server load)
        - Large batches (>100 students) automatically use async processing
        - Results are cached for 1 hour to avoid duplicate processing
    
    Business Use Cases:
        - Daily risk assessment for entire school
        - Monthly counselor reports
        - Grade-level intervention planning
        - District-wide risk monitoring
    """
    
    if not student_ids:
        return pd.DataFrame()
    
    # Validate input size
    if len(student_ids) > 1000:
        raise ValueError(
            f"Too many students ({len(student_ids)}). Maximum is 1000 per batch. "
            f"Consider processing in chunks or using async processing."
        )
    
    print(f"🔄 Processing {len(student_ids)} student predictions...")
    
    # For very large batches, process in chunks to avoid timeouts
    if len(student_ids) > chunk_size:
        return _process_batch_in_chunks(
            client, student_ids, include_factors, chunk_size
        )
    
    try:
        # Single batch request for smaller groups
        response = client._request(
            'POST',
            '/api/v1/predictions/predict-batch',
            json={
                'student_ids': student_ids,
                'include_factors': include_factors,
                'include_confidence': True,
                'async': async_processing,
                'priority': 'normal'  # Options: 'low', 'normal', 'high', 'urgent'
            }
        )
        
        # Handle async processing
        if async_processing and 'job_id' in response:
            return _wait_for_async_batch(client, response['job_id'])
        
        # Process synchronous results
        predictions = response.get('predictions', [])
        failed = response.get('failed', [])
        
        if failed:
            print(f"⚠️ {len(failed)} predictions failed:")
            for failure in failed[:5]:  # Show first 5 failures
                print(f"   - {failure['student_id']}: {failure['error']}")
            if len(failed) > 5:
                print(f"   ... and {len(failed) - 5} more failures")
        
        # Convert to DataFrame with enhanced information
        if not predictions:
            print("❌ No successful predictions generated")
            return pd.DataFrame()
        
        df = pd.DataFrame(predictions)
        
        # Enhance DataFrame with calculated fields
        df = _enrich_predictions_dataframe(df)
        
        print(f"✅ Successfully processed {len(df)} predictions")
        print(f"📊 Risk distribution: {_format_risk_summary(df)}")
        
        return df
        
    except Exception as e:
        print(f"❌ Batch prediction failed: {e}")
        raise

def _process_batch_in_chunks(
    client: EduPulseClient, 
    student_ids: List[str], 
    include_factors: bool,
    chunk_size: int
) -> pd.DataFrame:
    """
    Process large batches in smaller chunks to avoid timeouts and memory issues
    
    This approach is more reliable for large datasets and provides progress feedback.
    """
    all_results = []
    chunks = [student_ids[i:i + chunk_size] for i in range(0, len(student_ids), chunk_size)]
    
    print(f"📦 Processing in {len(chunks)} chunks of {chunk_size} students each")
    
    for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
        try:
            chunk_df = predict_batch(client, chunk, include_factors, async_processing=False, chunk_size=chunk_size)
            if not chunk_df.empty:
                all_results.append(chunk_df)
        except Exception as e:
            print(f"⚠️ Chunk {i+1} failed: {e}")
            continue
    
    if not all_results:
        return pd.DataFrame()
    
    # Combine all chunks
    combined_df = pd.concat(all_results, ignore_index=True)
    print(f"✅ Combined {len(combined_df)} predictions from {len(all_results)} successful chunks")
    
    return combined_df

def _wait_for_async_batch(client: EduPulseClient, job_id: str, timeout_minutes: int = 10) -> pd.DataFrame:
    """
    Wait for async batch job to complete and return results
    
    Async processing is used automatically for large batches (>100 students)
    or when the server is under high load.
    """
    import time
    
    timeout_seconds = timeout_minutes * 60
    start_time = time.time()
    check_interval = 5  # Check every 5 seconds
    
    print(f"⏳ Waiting for async batch job {job_id} to complete...")
    
    while time.time() - start_time < timeout_seconds:
        try:
            status_response = client._request('GET', f'/api/v1/predictions/batch-status/{job_id}')
            
            status = status_response['status']
            progress = status_response.get('progress', 0)
            
            if status == 'completed':
                results = status_response['results']
                df = pd.DataFrame(results['predictions'])
                df = _enrich_predictions_dataframe(df)
                
                print(f"✅ Async batch completed: {len(df)} predictions")
                return df
                
            elif status == 'failed':
                error_msg = status_response.get('error', 'Unknown error')
                raise Exception(f"Batch job failed: {error_msg}")
                
            elif status in ['running', 'pending']:
                print(f"🔄 Job {status}: {progress}% complete")
                time.sleep(check_interval)
                
            else:
                raise Exception(f"Unknown job status: {status}")
                
        except KeyboardInterrupt:
            print(f"\n⚠️ Interrupted. Job {job_id} may still be running on server.")
            raise
        except Exception as e:
            print(f"❌ Error checking job status: {e}")
            raise
    
    raise TimeoutError(f"Batch job {job_id} timed out after {timeout_minutes} minutes")

def _enrich_predictions_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calculated fields and business-friendly columns to predictions DataFrame
    """
    if df.empty:
        return df
    
    # Ensure numeric columns are properly typed
    df['risk_score'] = pd.to_numeric(df['risk_score'], errors='coerce').round(3)
    df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce').round(3)
    
    # Add business-friendly columns
    df['risk_percentage'] = df['risk_score'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
    df['urgency_level'] = df['risk_score'].apply(_interpret_urgency)
    df['next_review_days'] = df['risk_score'].apply(lambda x: 30 if x < 0.25 else 14 if x < 0.5 else 7 if x < 0.75 else 3)
    
    # Add priority ranking (1 = highest priority)
    df['priority_rank'] = df['risk_score'].rank(method='dense', ascending=False)
    
    # Ensure prediction_date is datetime
    if 'prediction_date' in df.columns:
        df['prediction_date'] = pd.to_datetime(df['prediction_date'], errors='coerce')
    else:
        df['prediction_date'] = pd.Timestamp.now()
    
    # Sort by risk score (highest first)
    df = df.sort_values('risk_score', ascending=False).reset_index(drop=True)
    
    return df

def _format_risk_summary(df: pd.DataFrame) -> str:
    """Format risk distribution summary for display"""
    if df.empty:
        return "No data"
    
    risk_counts = df['risk_category'].value_counts()
    total = len(df)
    
    summary_parts = []
    for category in ['critical', 'high', 'medium', 'low']:
        if category in risk_counts:
            count = risk_counts[category]
            pct = count / total * 100
            summary_parts.append(f"{category}: {count} ({pct:.1f}%)")
    
    return ", ".join(summary_parts)

# Comprehensive example showing real-world batch processing workflow
def demo_batch_predictions():
    """
    Demonstrate complete batch prediction workflow for a school counselor
    
    This example shows how to:
    1. Process predictions for multiple grade levels
    2. Generate priority intervention lists
    3. Create visual reports for administrators
    4. Export results for staff meetings
    """
    
    client = EduPulseClient('http://localhost:8000')
    
    print("🏫 Daily Risk Assessment Report Generation")
    print("=" * 60)
    
    # Simulate getting student lists from student information system
    grade_9_students = [f"STU{900 + i:03d}" for i in range(1, 26)]   # 25 students
    grade_10_students = [f"STU{1000 + i:03d}" for i in range(1, 31)] # 30 students
    grade_11_students = [f"STU{1100 + i:03d}" for i in range(1, 28)] # 27 students
    grade_12_students = [f"STU{1200 + i:03d}" for i in range(1, 33)] # 32 students
    
    all_results = {}
    
    # Process each grade level separately for better organization
    grade_levels = {
        "Grade 9": grade_9_students,
        "Grade 10": grade_10_students,
        "Grade 11": grade_11_students,
        "Grade 12": grade_12_students
    }
    
    for grade_name, students in grade_levels.items():
        print(f"\n📊 Processing {grade_name} ({len(students)} students)...")
        
        try:
            # Process batch with progress tracking
            predictions_df = predict_batch(
                client, 
                students, 
                include_factors=True,  # Get detailed factors for intervention planning
                chunk_size=20          # Process in smaller chunks for reliability
            )
            
            if not predictions_df.empty:
                all_results[grade_name] = predictions_df
                
                # Show immediate summary for this grade
                high_risk_count = len(predictions_df[predictions_df['risk_score'] > 0.7])
                avg_risk = predictions_df['risk_score'].mean()
                
                print(f"   ✅ {len(predictions_df)} predictions completed")
                print(f"   🚨 {high_risk_count} students need immediate attention")
                print(f"   📈 Average risk score: {avg_risk:.1%}")
                
            else:
                print(f"   ❌ No predictions generated for {grade_name}")
                
        except Exception as e:
            print(f"   ❌ Failed to process {grade_name}: {e}")
            continue
    
    if not all_results:
        print("\n❌ No predictions were successfully generated")
        return
    
    # Combine all grades for school-wide analysis
    all_predictions = pd.concat(all_results.values(), ignore_index=True)
    
    # Generate comprehensive analysis
    print(f"\n📋 SCHOOL-WIDE RISK ASSESSMENT SUMMARY")
    print("=" * 50)
    print(f"Total Students Analyzed: {len(all_predictions)}")
    print(f"Average Risk Score: {all_predictions['risk_score'].mean():.1%}")
    print(f"Risk Distribution: {_format_risk_summary(all_predictions)}")
    
    # Identify top priority students across all grades
    critical_students = all_predictions[all_predictions['risk_score'] > 0.8]
    high_risk_students = all_predictions[
        (all_predictions['risk_score'] > 0.6) & (all_predictions['risk_score'] <= 0.8)
    ]
    
    print(f"\n🚨 IMMEDIATE ACTION REQUIRED:")
    print(f"   Critical Risk (>80%): {len(critical_students)} students")
    print(f"   High Risk (60-80%): {len(high_risk_students)} students")
    
    if len(critical_students) > 0:
        print(f"\n🎯 TOP 5 CRITICAL RISK STUDENTS:")
        top_critical = critical_students.head(5)
        for _, student in top_critical.iterrows():
            print(f"   • {student['student_id']}: {student['risk_percentage']} risk")
            if 'factors' in student and student['factors']:
                top_factor = student['factors'][0] if isinstance(student['factors'], list) else "Multiple factors"
                print(f"     Primary concern: {top_factor}")
    
    # Create visualizations for administrative report
    _create_batch_visualizations(all_results, all_predictions)
    
    # Generate exportable report
    report_file = f"risk_assessment_report_{pd.Timestamp.now().strftime('%Y%m%d')}.csv"
    all_predictions.to_csv(report_file, index=False)
    print(f"\n📄 Detailed report exported to: {report_file}")
    
    return all_results, all_predictions

def _create_batch_visualizations(grade_results: Dict[str, pd.DataFrame], all_predictions: pd.DataFrame):
    """
    Create comprehensive visualizations for batch prediction results
    
    Generates charts that administrators and counselors can use in meetings
    and reports to stakeholders.
    """
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a comprehensive dashboard
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('EduPulse Risk Assessment Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Risk Distribution by Grade Level
    grade_risk_data = []
    for grade, df in grade_results.items():
        for category in ['low', 'medium', 'high', 'critical']:
            count = len(df[df['risk_category'] == category])
            grade_risk_data.append({'Grade': grade, 'Risk Category': category, 'Count': count})
    
    grade_risk_df = pd.DataFrame(grade_risk_data)
    grade_risk_pivot = grade_risk_df.pivot(index='Grade', columns='Risk Category', values='Count').fillna(0)
    
    grade_risk_pivot.plot(kind='bar', stacked=True, ax=axes[0, 0], 
                         color=['green', 'yellow', 'orange', 'red'])
    axes[0, 0].set_title('Risk Distribution by Grade Level')
    axes[0, 0].set_xlabel('Grade Level')
    axes[0, 0].set_ylabel('Number of Students')
    axes[0, 0].legend(title='Risk Category', bbox_to_anchor=(1.05, 1))
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Risk Score Distribution (Histogram)
    axes[0, 1].hist(all_predictions['risk_score'], bins=20, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(all_predictions['risk_score'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {all_predictions["risk_score"].mean():.2f}')
    axes[0, 1].axvline(0.7, color='orange', linestyle='--', label='High Risk Threshold')
    axes[0, 1].set_title('Risk Score Distribution')
    axes[0, 1].set_xlabel('Risk Score')
    axes[0, 1].set_ylabel('Number of Students')
    axes[0, 1].legend()
    
    # 3. Confidence vs Risk Score Scatter
    scatter = axes[0, 2].scatter(all_predictions['confidence'], all_predictions['risk_score'], 
                               alpha=0.6, c=all_predictions['risk_score'], cmap='Reds')
    axes[0, 2].set_title('Prediction Confidence vs Risk Score')
    axes[0, 2].set_xlabel('Model Confidence')
    axes[0, 2].set_ylabel('Risk Score')
    plt.colorbar(scatter, ax=axes[0, 2], label='Risk Score')
    
    # 4. Grade-wise Average Risk Scores
    grade_avg_risk = {grade: df['risk_score'].mean() for grade, df in grade_results.items()}
    grades = list(grade_avg_risk.keys())
    avg_risks = list(grade_avg_risk.values())
    
    bars = axes[1, 0].bar(grades, avg_risks, color=['lightblue', 'lightgreen', 'lightyellow', 'lightcoral'])
    axes[1, 0].set_title('Average Risk Score by Grade')
    axes[1, 0].set_xlabel('Grade Level')
    axes[1, 0].set_ylabel('Average Risk Score')
    axes[1, 0].set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, avg_risks):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{value:.2f}', ha='center', va='bottom')
    
    # 5. Priority Distribution (Next Review Timeline)
    review_counts = all_predictions['next_review_days'].value_counts().sort_index()
    review_labels = {3: 'Immediate (3 days)', 7: 'Weekly (7 days)', 
                    14: 'Bi-weekly (14 days)', 30: 'Monthly (30 days)'}
    
    pie_labels = [review_labels.get(days, f'{days} days') for days in review_counts.index]
    axes[1, 1].pie(review_counts.values, labels=pie_labels, autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title('Review Schedule Distribution')
    
    # 6. Top Risk Factors (if available)
    if 'factors' in all_predictions.columns:
        # This would need to be implemented based on actual factor data structure
        axes[1, 2].text(0.5, 0.5, 'Risk Factor Analysis\n(Implementation depends on\nfactor data structure)', 
                       ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Top Risk Factors')
    else:
        axes[1, 2].text(0.5, 0.5, 'Risk Factor Analysis\nNot Available', 
                       ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Top Risk Factors')
    
    plt.tight_layout()
    
    # Save the dashboard
    dashboard_file = f"risk_dashboard_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.png"
    plt.savefig(dashboard_file, dpi=300, bbox_inches='tight')
    print(f"📊 Dashboard saved as: {dashboard_file}")
    
    # Display the dashboard
    plt.show()

# Quick batch example for immediate use
def quick_batch_example():
    """Simple example for testing batch predictions quickly"""
    
    client = EduPulseClient('http://localhost:8000')
    
    # Small test batch
    test_students = ['STU001', 'STU002', 'STU003', 'STU004', 'STU005']
    
    try:
        print("🧪 Running quick batch test...")
        results_df = predict_batch(client, test_students, include_factors=False)
        
        if not results_df.empty:
            print(f"\n📊 Results Preview:")
            print(results_df[['student_id', 'risk_percentage', 'risk_category', 'urgency_level']].head())
            
            # Quick statistics
            high_risk = len(results_df[results_df['risk_score'] > 0.7])
            print(f"\n📈 Quick Stats:")
            print(f"   Total students: {len(results_df)}")
            print(f"   High risk: {high_risk}")
            print(f"   Average risk: {results_df['risk_score'].mean():.1%}")
        else:
            print("❌ No results generated")
            
    except Exception as e:
        print(f"❌ Quick test failed: {e}")

if __name__ == "__main__":
    # Run quick test
    quick_batch_example()
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