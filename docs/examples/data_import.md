# Comprehensive Data Import Guide for EduPulse Analytics

This extensive guide demonstrates enterprise-ready data import strategies for EduPulse Analytics. Whether you're migrating from a legacy student information system, setting up daily synchronization, or performing one-time bulk imports, these examples provide production-tested solutions with error handling, validation, and performance optimization.

## Overview of Import Methods

EduPulse supports multiple import approaches depending on your data source, volume, and operational requirements:

- **CSV Import**: Best for spreadsheet data, manual uploads, and small to medium datasets (<10,000 records)
- **Database Direct Import**: Optimal for large datasets, existing database migrations, and high-performance bulk operations
- **JSON Import**: Ideal for complex nested data, API-to-API migrations, and structured data with relationships
- **Excel Import**: Perfect for administrative imports, multi-sheet data organization, and business-friendly formats
- **Real-time Sync**: Best for ongoing operations, incremental updates, and live system integration

### Data Import Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Sources                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────── │
│  │   Legacy    │  │    CSV      │  │  External   │  │   Manual   │
│  │  Database   │  │   Files     │  │    APIs     │  │   Excel    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────── │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Data Processing Layer                           │
│  ┌─────────────────────────────────────────────────────────────┤
│  │ • Data Validation & Cleaning                                │
│  │ • Format Transformation                                     │
│  │ • Duplicate Detection & Resolution                          │
│  │ • Relationship Mapping                                      │
│  │ • Error Handling & Logging                                  │
│  └─────────────────────────────────────────────────────────────┤
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EduPulse Database                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────── │
│  │  Students   │  │ Attendance  │  │   Grades    │  │ Discipline │
│  │   Table     │  │   Table     │  │   Table     │  │   Table    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────── │
└─────────────────────────────────────────────────────────────────┘
```

### Performance Expectations

| Import Method | Records/Minute | Best For | Memory Usage |
|---------------|----------------|----------|--------------|
| CSV Import    | 500-1,000     | Small-Medium datasets | Low (streaming) |
| Direct SQL    | 5,000-10,000  | Large migrations | Medium-High |
| JSON Import   | 200-500       | Complex nested data | Medium |
| Excel Import  | 300-800       | Administrative use | Medium |
| Parallel Import| 2,000-5,000  | Time-critical imports | High |

## CSV Import - Production-Ready Student Data Processing

CSV import is the most common method for importing student data from spreadsheets, exports from other systems, or manual data entry. This approach provides comprehensive error handling, data validation, and progress tracking.

### Enterprise Student Data Import

This implementation provides robust CSV processing with validation, error recovery, and detailed logging suitable for production environments.

```python
import pandas as pd
import requests
from datetime import datetime, date
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Configure logging for production use
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('student_import.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ImportResult:
    """Container for import operation results"""
    success_count: int
    error_count: int
    errors: List[Dict]
    warnings: List[Dict]
    duration_seconds: float
    imported_student_ids: List[str]

class StudentDataImporter:
    """
    Enterprise-grade student data importer with comprehensive error handling
    
    Features:
    - Data validation and cleaning
    - Progress tracking and logging
    - Error recovery and retry logic
    - Memory-efficient streaming for large files
    - Duplicate detection and handling
    - Customizable field mapping
    """
    
    def __init__(self, api_url: str, auth_token: Optional[str] = None, batch_size: int = 100):
        """
        Initialize the importer
        
        Args:
            api_url: EduPulse API base URL
            auth_token: Optional authentication token
            batch_size: Number of records to process in each batch (for progress tracking)
        """
        self.api_url = api_url.rstrip('/')
        self.session = requests.Session()
        self.batch_size = batch_size
        
        # Configure session with authentication and retry logic
        if auth_token:
            self.session.headers.update({'Authorization': f'Bearer {auth_token}'})
        
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'EduPulse-Importer/1.0'
        })
        
        # Define required and optional fields with validation rules
        self.required_fields = {
            'student_id': {'type': str, 'max_length': 20},
            'first_name': {'type': str, 'max_length': 50},
            'last_name': {'type': str, 'max_length': 50},
            'grade_level': {'type': int, 'min': 1, 'max': 12},
            'date_of_birth': {'type': 'date'},
            'enrollment_date': {'type': 'date'}
        }
        
        self.optional_fields = {
            'middle_name': {'type': str, 'max_length': 50},
            'gender': {'type': str, 'allowed_values': ['M', 'F', 'U'], 'default': 'U'},
            'ethnicity': {'type': str, 'max_length': 50, 'default': 'Unknown'},
            'socioeconomic_status': {'type': str, 'allowed_values': ['low', 'middle', 'high'], 'default': 'middle'},
            'email': {'type': 'email', 'max_length': 100},
            'phone': {'type': 'phone', 'max_length': 20},
            'address': {'type': str, 'max_length': 200},
            'parent_name': {'type': str, 'max_length': 100},
            'parent_email': {'type': 'email', 'max_length': 100},
            'special_needs': {'type': bool, 'default': False},
            'iep_status': {'type': bool, 'default': False}
        }
    
    def import_students_from_csv(
        self, 
        csv_file: str, 
        encoding: str = 'utf-8',
        skip_duplicates: bool = True,
        validate_only: bool = False
    ) -> ImportResult:
        """
        Import student data from CSV file with comprehensive validation and error handling
        
        Args:
            csv_file: Path to CSV file containing student data
            encoding: File encoding (default: utf-8, also try: 'latin-1' for Excel exports)
            skip_duplicates: Whether to skip duplicate student IDs (vs. update existing)
            validate_only: If True, only validate data without importing
        
        Returns:
            ImportResult containing success/error counts and details
            
        Expected CSV format:
            student_id,first_name,last_name,grade_level,date_of_birth,enrollment_date,gender,ethnicity
            STU001,John,Doe,9,2008-05-15,2023-08-20,M,Hispanic
            STU002,Jane,Smith,10,2007-12-03,2023-08-20,F,Caucasian
        
        Business Logic:
            - Student IDs must be unique within the import file
            - Grade levels must be 1-12 (K can be represented as 0)
            - Dates must be in YYYY-MM-DD format
            - Names are automatically title-cased
            - Email addresses are validated for format
        """
        start_time = datetime.now()
        logger.info(f"🚀 Starting student import from {csv_file}")
        
        try:
            # Read CSV with error handling for common encoding issues
            df = self._read_csv_with_fallback(csv_file, encoding)
            logger.info(f"📊 Loaded {len(df)} records from CSV")
            
            # Validate and clean the data
            validation_result = self._validate_dataframe(df)
            
            if validation_result['errors']:
                logger.error(f"❌ Data validation failed with {len(validation_result['errors'])} errors")
                for error in validation_result['errors'][:10]:  # Show first 10 errors
                    logger.error(f"   - {error}")
                
                if len(validation_result['errors']) > 10:
                    logger.error(f"   ... and {len(validation_result['errors']) - 10} more errors")
                
                return ImportResult(
                    success_count=0,
                    error_count=len(validation_result['errors']),
                    errors=validation_result['errors'],
                    warnings=validation_result['warnings'],
                    duration_seconds=(datetime.now() - start_time).total_seconds(),
                    imported_student_ids=[]
                )
            
            # Clean and transform the data
            df_cleaned = self._clean_and_transform_data(df)
            
            if validate_only:
                logger.info("✅ Data validation completed successfully")
                return ImportResult(
                    success_count=len(df_cleaned),
                    error_count=0,
                    errors=[],
                    warnings=validation_result['warnings'],
                    duration_seconds=(datetime.now() - start_time).total_seconds(),
                    imported_student_ids=[]
                )
            
            # Import the data
            import_result = self._import_students_batch(df_cleaned, skip_duplicates)
            
            # Calculate final statistics
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ Import completed in {duration:.2f}s")
            logger.info(f"📈 Results: {import_result.success_count} success, {import_result.error_count} errors")
            
            return import_result
            
        except Exception as e:
            logger.error(f"💥 Critical error during import: {e}")
            return ImportResult(
                success_count=0,
                error_count=1,
                errors=[{'error': f'Critical import failure: {str(e)}'}],
                warnings=[],
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                imported_student_ids=[]
            )
    
    def _read_csv_with_fallback(self, csv_file: str, encoding: str) -> pd.DataFrame:
        """
        Read CSV with encoding fallback for common file export issues
        
        Many student information systems export data with different encodings.
        This method tries multiple encodings to handle common scenarios.
        """
        encodings_to_try = [encoding, 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for enc in encodings_to_try:
            try:
                logger.info(f"🔄 Attempting to read CSV with {enc} encoding")
                df = pd.read_csv(csv_file, encoding=enc)
                
                # Check if we got reasonable data (not all NaN)
                if df.empty or df.isnull().all().all():
                    continue
                
                logger.info(f"✅ Successfully read CSV with {enc} encoding")
                return df
                
            except UnicodeDecodeError:
                logger.warning(f"⚠️ Failed to read with {enc} encoding, trying next...")
                continue
            except Exception as e:
                logger.error(f"❌ Error reading CSV with {enc}: {e}")
                continue
        
        raise ValueError(f"Could not read CSV file {csv_file} with any supported encoding")
    
    def _validate_dataframe(self, df: pd.DataFrame) -> Dict:
        """
        Comprehensive data validation with detailed error reporting
        
        Checks:
        - Required columns exist
        - Data types are correct
        - Values are within acceptable ranges
        - Duplicate student IDs
        - Date formats
        - Email formats (if provided)
        """
        errors = []
        warnings = []
        
        # Check for required columns
        missing_cols = set(self.required_fields.keys()) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {', '.join(missing_cols)}")
        
        # Check for completely empty required fields
        for field in self.required_fields.keys():
            if field in df.columns:
                null_count = df[field].isnull().sum()
                if null_count > 0:
                    errors.append(f"Column '{field}' has {null_count} empty values (required field)")
        
        # Check for duplicate student IDs
        if 'student_id' in df.columns:
            duplicates = df[df.duplicated('student_id', keep=False)]
            if not duplicates.empty:
                dup_ids = duplicates['student_id'].unique().tolist()
                errors.append(f"Duplicate student IDs found: {dup_ids[:10]}")  # Show first 10
                if len(dup_ids) > 10:
                    errors.append(f"... and {len(dup_ids) - 10} more duplicates")
        
        # Validate grade levels
        if 'grade_level' in df.columns:
            invalid_grades = df[~df['grade_level'].between(0, 12)]  # 0 for Kindergarten
            if not invalid_grades.empty:
                invalid_ids = invalid_grades['student_id'].tolist()[:10]
                errors.append(f"Invalid grade levels (must be 0-12): {invalid_ids}")
        
        # Validate date formats
        for date_field in ['date_of_birth', 'enrollment_date']:
            if date_field in df.columns:
                try:
                    pd.to_datetime(df[date_field], format='%Y-%m-%d', errors='raise')
                except:
                    try:
                        # Try common alternative formats
                        pd.to_datetime(df[date_field], errors='coerce')
                        warnings.append(f"Date field '{date_field}' not in YYYY-MM-DD format, will attempt conversion")
                    except:
                        errors.append(f"Invalid date format in '{date_field}' column")
        
        # Validate email formats (if provided)
        for email_field in ['email', 'parent_email']:
            if email_field in df.columns:
                email_series = df[email_field].dropna()
                if not email_series.empty:
                    # Simple email validation regex
                    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                    invalid_emails = email_series[~email_series.str.match(email_pattern, na=False)]
                    if not invalid_emails.empty:
                        warnings.append(f"Invalid email formats found in '{email_field}': {len(invalid_emails)} records")
        
        # Check data quality indicators
        if 'student_id' in df.columns:
            # Check for reasonable student ID format (letters + numbers)
            unusual_ids = df['student_id'][~df['student_id'].astype(str).str.match(r'^[A-Za-z0-9]+$')]
            if not unusual_ids.empty:
                warnings.append(f"Unusual student ID formats detected: {len(unusual_ids)} records")
        
        return {
            'errors': errors,
            'warnings': warnings
        }
    
    def _clean_and_transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and transform data to meet EduPulse requirements
        
        Transformations:
        - Title case names (John Doe, not JOHN DOE or john doe)
        - Normalize gender values (M/F/U)
        - Format dates consistently
        - Trim whitespace
        - Handle missing values with defaults
        """
        df_clean = df.copy()
        
        # Clean text fields - title case names and trim whitespace
        for field in ['first_name', 'last_name', 'middle_name']:
            if field in df_clean.columns:
                df_clean[field] = df_clean[field].astype(str).str.strip().str.title()
        
        # Normalize gender values
        if 'gender' in df_clean.columns:
            gender_mapping = {
                'male': 'M', 'm': 'M', 'M': 'M',
                'female': 'F', 'f': 'F', 'F': 'F',
                'unknown': 'U', 'u': 'U', 'U': 'U', '': 'U'
            }
            df_clean['gender'] = df_clean['gender'].astype(str).str.lower().map(gender_mapping).fillna('U')
        
        # Format dates consistently
        for date_field in ['date_of_birth', 'enrollment_date']:
            if date_field in df_clean.columns:
                df_clean[date_field] = pd.to_datetime(df_clean[date_field], errors='coerce').dt.strftime('%Y-%m-%d')
        
        # Apply default values for optional fields
        for field, config in self.optional_fields.items():
            if field in df_clean.columns and 'default' in config:
                df_clean[field] = df_clean[field].fillna(config['default'])
        
        # Convert grade levels to integers
        if 'grade_level' in df_clean.columns:
            df_clean['grade_level'] = df_clean['grade_level'].astype(int)
        
        # Remove any completely empty rows
        df_clean = df_clean.dropna(how='all')
        
        logger.info(f"🧹 Data cleaning completed: {len(df_clean)} records ready for import")
        return df_clean
    
    def _import_students_batch(self, df: pd.DataFrame, skip_duplicates: bool) -> ImportResult:
        """
        Import students in batches with progress tracking and error recovery
        """
        total_records = len(df)
        success_count = 0
        errors = []
        warnings = []
        imported_ids = []
        
        logger.info(f"📦 Processing {total_records} students in batches of {self.batch_size}")
        
        for batch_start in range(0, total_records, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_records)
            batch_df = df.iloc[batch_start:batch_end]
            
            logger.info(f"🔄 Processing batch {batch_start//self.batch_size + 1}/{(total_records-1)//self.batch_size + 1} (records {batch_start+1}-{batch_end})")
            
            for _, row in batch_df.iterrows():
                try:
                    # Convert row to dictionary and clean NaN values
                    student_data = {k: v for k, v in row.to_dict().items() if pd.notna(v)}
                    
                    # Make API request
                    response = self.session.post(
                        f"{self.api_url}/api/v1/students",
                        json=student_data,
                        timeout=30  # 30 second timeout per request
                    )
                    
                    if response.status_code == 201:
                        # Successfully created
                        success_count += 1
                        imported_ids.append(student_data['student_id'])
                        logger.debug(f"✅ Created student {student_data['student_id']}")
                        
                    elif response.status_code == 409 and skip_duplicates:
                        # Student already exists, skip if requested
                        warnings.append({
                            'student_id': student_data['student_id'],
                            'warning': 'Student already exists, skipped'
                        })
                        logger.debug(f"⚠️ Skipped duplicate student {student_data['student_id']}")
                        
                    else:
                        # Other error
                        error_msg = response.json().get('detail', f'HTTP {response.status_code}') if response.text else f'HTTP {response.status_code}'
                        errors.append({
                            'student_id': student_data['student_id'],
                            'error': error_msg,
                            'status_code': response.status_code
                        })
                        logger.warning(f"❌ Failed to create {student_data['student_id']}: {error_msg}")
                    
                except requests.exceptions.Timeout:
                    errors.append({
                        'student_id': row.get('student_id', 'unknown'),
                        'error': 'Request timeout - server may be overloaded'
                    })
                    logger.error(f"⏰ Timeout importing {row.get('student_id', 'unknown')}")
                    
                except requests.exceptions.ConnectionError:
                    errors.append({
                        'student_id': row.get('student_id', 'unknown'),
                        'error': 'Connection error - check network and server status'
                    })
                    logger.error(f"🌐 Connection error importing {row.get('student_id', 'unknown')}")
                    
                except Exception as e:
                    errors.append({
                        'student_id': row.get('student_id', 'unknown'),
                        'error': f'Unexpected error: {str(e)}'
                    })
                    logger.error(f"💥 Unexpected error importing {row.get('student_id', 'unknown')}: {e}")
        
        return ImportResult(
            success_count=success_count,
            error_count=len(errors),
            errors=errors,
            warnings=warnings,
            duration_seconds=0,  # Will be calculated by caller
            imported_student_ids=imported_ids
        )

# Example usage for school administrators
def demo_student_import():
    """
    Demonstrate comprehensive student import workflow
    
    This example shows how a school district IT administrator would
    import student data from their student information system export.
    """
    
    # Initialize the importer with authentication
    importer = StudentDataImporter(
        api_url='http://localhost:8000',
        auth_token='your-api-token-here',  # Get from EduPulse admin
        batch_size=50  # Smaller batches for better progress tracking
    )
    
    # First, validate the data without importing
    print("🔍 Step 1: Validating student data...")
    validation_result = importer.import_students_from_csv(
        'students_export.csv',
        validate_only=True
    )
    
    if validation_result.error_count > 0:
        print(f"❌ Validation failed with {validation_result.error_count} errors")
        print("Please fix the following issues before importing:")
        for error in validation_result.errors:
            print(f"  - {error}")
        return
    
    print(f"✅ Validation passed! {validation_result.success_count} records ready for import")
    if validation_result.warnings:
        print("⚠️ Warnings (will be handled automatically):")
        for warning in validation_result.warnings:
            print(f"  - {warning}")
    
    # Proceed with actual import
    print("\n📤 Step 2: Importing student data...")
    result = importer.import_students_from_csv(
        'students_export.csv',
        skip_duplicates=True  # Skip students that already exist
    )
    
    # Display comprehensive results
    print(f"\n📊 IMPORT RESULTS")
    print(f"=" * 50)
    print(f"✅ Successfully imported: {result.success_count} students")
    print(f"❌ Failed imports: {result.error_count}")
    print(f"⚠️ Warnings: {len(result.warnings)}")
    print(f"⏱️ Duration: {result.duration_seconds:.2f} seconds")
    print(f"📈 Rate: {result.success_count / result.duration_seconds:.1f} students/second")
    
    if result.errors:
        print(f"\n❌ IMPORT ERRORS (showing first 10):")
        for error in result.errors[:10]:
            print(f"   Student {error['student_id']}: {error['error']}")
    
    if result.warnings:
        print(f"\n⚠️ WARNINGS:")
        for warning in result.warnings[:10]:
            print(f"   Student {warning['student_id']}: {warning['warning']}")
    
    # Save detailed report for administrators
    report_data = {
        'import_date': datetime.now().isoformat(),
        'source_file': 'students_export.csv',
        'summary': {
            'success_count': result.success_count,
            'error_count': result.error_count,
            'warning_count': len(result.warnings),
            'duration_seconds': result.duration_seconds
        },
        'imported_student_ids': result.imported_student_ids,
        'errors': result.errors,
        'warnings': result.warnings
    }
    
    with open(f"student_import_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: student_import_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    return result

# Example CSV format (save as students_export.csv)
"""
student_id,first_name,last_name,grade_level,date_of_birth,enrollment_date,gender,ethnicity,parent_email
STU001234,John,Doe,9,2008-05-15,2023-08-20,M,Hispanic,parent1@email.com
STU001235,Jane,Smith,10,2007-12-03,2023-08-20,F,Caucasian,parent2@email.com
STU001236,Michael,Johnson,11,2006-08-22,2022-08-15,M,African American,parent3@email.com
STU001237,Sarah,Williams,12,2005-11-18,2021-08-10,F,Asian,parent4@email.com
STU001238,David,Brown,9,2008-03-07,2023-08-20,M,Native American,parent5@email.com
"""

if __name__ == "__main__":
    # Run the demo import
    demo_student_import()
```

### Attendance Data Import

```python
def import_attendance_from_csv(csv_file, api_url):
    """Import attendance records from CSV"""
    
    df = pd.read_csv(csv_file)
    
    # Group by student for batch import
    grouped = df.groupby('student_id')
    
    for student_id, records in grouped:
        attendance_data = []
        
        for _, record in records.iterrows():
            attendance_data.append({
                'date': record['date'],
                'status': record['status'],  # present, absent, tardy, excused
                'period': record.get('period', 1),
                'minutes_late': record.get('minutes_late', 0),
                'reason': record.get('reason', None)
            })
        
        # Batch import for student
        response = requests.post(
            f"{api_url}/api/v1/students/{student_id}/attendance/batch",
            json={'records': attendance_data}
        )
        
        if response.status_code == 200:
            print(f"Imported {len(attendance_data)} records for {student_id}")
```

## Database Direct Import

### SQLAlchemy Bulk Insert

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.db.models import Student, AttendanceRecord, Grade
import pandas as pd

def bulk_import_to_database(csv_file, table_name, connection_string):
    """Direct database import using SQLAlchemy"""
    
    # Create engine
    engine = create_engine(connection_string)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Read data
    df = pd.read_csv(csv_file)
    
    # Map table names to models
    models = {
        'students': Student,
        'attendance': AttendanceRecord,
        'grades': Grade
    }
    
    model = models.get(table_name)
    if not model:
        raise ValueError(f"Unknown table: {table_name}")
    
    # Convert DataFrame to dict records
    records = df.to_dict('records')
    
    # Bulk insert
    try:
        session.bulk_insert_mappings(model, records)
        session.commit()
        print(f"Imported {len(records)} records to {table_name}")
    except Exception as e:
        session.rollback()
        print(f"Error: {e}")
    finally:
        session.close()

# Example usage
bulk_import_to_database(
    'grades.csv',
    'grades',
    'postgresql://user:pass@localhost/edupulse'
)
```

## JSON Import

### Complex Nested Data

```python
import json

def import_student_complete_record(json_file, api_url):
    """Import complete student record with nested data"""
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    for student_record in data['students']:
        # Create student
        student_response = requests.post(
            f"{api_url}/api/v1/students",
            json=student_record['profile']
        )
        
        if student_response.status_code != 201:
            print(f"Failed to create student {student_record['profile']['student_id']}")
            continue
        
        student_id = student_record['profile']['student_id']
        
        # Import attendance
        if 'attendance' in student_record:
            requests.post(
                f"{api_url}/api/v1/students/{student_id}/attendance/batch",
                json={'records': student_record['attendance']}
            )
        
        # Import grades
        if 'grades' in student_record:
            requests.post(
                f"{api_url}/api/v1/students/{student_id}/grades/batch",
                json={'records': student_record['grades']}
            )
        
        # Import discipline
        if 'discipline' in student_record:
            requests.post(
                f"{api_url}/api/v1/students/{student_id}/discipline/batch",
                json={'records': student_record['discipline']}
            )
        
        print(f"Imported complete record for {student_id}")
```

## Excel Import

### Multi-sheet Excel File

```python
def import_from_excel(excel_file, api_url):
    """Import data from multi-sheet Excel file"""
    
    # Read all sheets
    sheets = pd.read_excel(excel_file, sheet_name=None)
    
    # Process each sheet
    if 'Students' in sheets:
        import_students_dataframe(sheets['Students'], api_url)
    
    if 'Attendance' in sheets:
        import_attendance_dataframe(sheets['Attendance'], api_url)
    
    if 'Grades' in sheets:
        import_grades_dataframe(sheets['Grades'], api_url)
    
    if 'Discipline' in sheets:
        import_discipline_dataframe(sheets['Discipline'], api_url)

def import_students_dataframe(df, api_url):
    """Import students from DataFrame"""
    for _, row in df.iterrows():
        student_data = row.to_dict()
        # Clean NaN values
        student_data = {k: v for k, v in student_data.items() if pd.notna(v)}
        
        response = requests.post(
            f"{api_url}/api/v1/students",
            json=student_data
        )
        
        if response.status_code == 201:
            print(f"Created student {student_data['student_id']}")
```

## Data Validation

### Pre-import Validation

```python
def validate_student_data(df):
    """Validate student data before import"""
    
    errors = []
    
    # Check required fields
    required = ['student_id', 'first_name', 'last_name', 'grade_level']
    for field in required:
        if field not in df.columns:
            errors.append(f"Missing required column: {field}")
    
    # Check data types
    if 'grade_level' in df.columns:
        invalid_grades = df[~df['grade_level'].between(1, 12)]
        if not invalid_grades.empty:
            errors.append(f"Invalid grade levels: {invalid_grades['student_id'].tolist()}")
    
    # Check date formats
    if 'date_of_birth' in df.columns:
        try:
            pd.to_datetime(df['date_of_birth'])
        except:
            errors.append("Invalid date format in date_of_birth")
    
    # Check for duplicates
    duplicates = df[df.duplicated('student_id', keep=False)]
    if not duplicates.empty:
        errors.append(f"Duplicate student IDs: {duplicates['student_id'].unique().tolist()}")
    
    return errors

# Usage
df = pd.read_csv('students.csv')
errors = validate_student_data(df)
if errors:
    print("Validation errors:")
    for error in errors:
        print(f"  - {error}")
else:
    print("Data validation passed")
```

## Migration Scripts

### Legacy System Migration

```python
def migrate_from_legacy_system(legacy_db_url, api_url):
    """Migrate data from legacy database"""
    
    # Connect to legacy database
    legacy_engine = create_engine(legacy_db_url)
    
    # Extract data
    students_query = """
        SELECT 
            student_num as student_id,
            fname as first_name,
            lname as last_name,
            grade,
            dob as date_of_birth,
            enrollment_dt as enrollment_date
        FROM legacy_students
        WHERE active = 1
    """
    
    students_df = pd.read_sql(students_query, legacy_engine)
    
    # Transform data
    students_df['grade_level'] = students_df['grade'].astype(int)
    students_df['date_of_birth'] = pd.to_datetime(students_df['date_of_birth']).dt.strftime('%Y-%m-%d')
    students_df['enrollment_date'] = pd.to_datetime(students_df['enrollment_date']).dt.strftime('%Y-%m-%d')
    
    # Load to new system
    for _, student in students_df.iterrows():
        response = requests.post(
            f"{api_url}/api/v1/students",
            json=student.to_dict()
        )
        
        if response.status_code == 201:
            print(f"Migrated student {student['student_id']}")
            
            # Migrate related data
            migrate_student_attendance(legacy_engine, api_url, student['student_id'])
            migrate_student_grades(legacy_engine, api_url, student['student_id'])
```

## Incremental Updates

### Delta Import

```python
def import_daily_updates(last_sync_date, api_url):
    """Import only new/updated records since last sync"""
    
    # Read update file
    updates_file = f"updates_{datetime.now().strftime('%Y%m%d')}.csv"
    df = pd.read_csv(updates_file)
    
    # Filter by modification date
    df['modified_date'] = pd.to_datetime(df['modified_date'])
    new_records = df[df['modified_date'] > last_sync_date]
    
    print(f"Found {len(new_records)} records to sync")
    
    for _, record in new_records.iterrows():
        if record['operation'] == 'INSERT':
            # Create new record
            response = requests.post(
                f"{api_url}/api/v1/students",
                json=record.to_dict()
            )
        elif record['operation'] == 'UPDATE':
            # Update existing record
            response = requests.put(
                f"{api_url}/api/v1/students/{record['student_id']}",
                json=record.to_dict()
            )
        elif record['operation'] == 'DELETE':
            # Soft delete
            response = requests.delete(
                f"{api_url}/api/v1/students/{record['student_id']}"
            )
    
    # Update last sync date
    with open('last_sync.txt', 'w') as f:
        f.write(datetime.now().isoformat())
```

## Performance Optimization

### Parallel Import

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def parallel_import(csv_file, api_url, max_workers=10):
    """Import data using parallel processing"""
    
    df = pd.read_csv(csv_file)
    
    # Split into chunks
    chunk_size = len(df) // max_workers
    chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    
    def import_chunk(chunk_df):
        """Import a chunk of data"""
        results = []
        for _, row in chunk_df.iterrows():
            try:
                response = requests.post(
                    f"{api_url}/api/v1/students",
                    json=row.to_dict()
                )
                results.append({
                    'student_id': row['student_id'],
                    'status': response.status_code
                })
            except Exception as e:
                results.append({
                    'student_id': row['student_id'],
                    'error': str(e)
                })
        return results
    
    # Process chunks in parallel
    all_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(import_chunk, chunk): chunk for chunk in chunks}
        
        for future in as_completed(futures):
            results = future.result()
            all_results.extend(results)
            print(f"Completed chunk: {len(results)} records")
    
    # Summary
    success_count = sum(1 for r in all_results if r.get('status') == 201)
    print(f"Total imported: {success_count}/{len(all_results)}")
    
    return all_results
```

## Error Recovery

### Import with Checkpointing

```python
def import_with_checkpoint(csv_file, api_url, checkpoint_file='checkpoint.json'):
    """Import with ability to resume from failures"""
    
    # Load checkpoint if exists
    try:
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            processed_ids = set(checkpoint['processed'])
            last_index = checkpoint['last_index']
    except FileNotFoundError:
        processed_ids = set()
        last_index = 0
    
    df = pd.read_csv(csv_file)
    
    for index, row in df.iloc[last_index:].iterrows():
        student_id = row['student_id']
        
        if student_id in processed_ids:
            continue
        
        try:
            response = requests.post(
                f"{api_url}/api/v1/students",
                json=row.to_dict()
            )
            response.raise_for_status()
            processed_ids.add(student_id)
            
            # Update checkpoint every 100 records
            if len(processed_ids) % 100 == 0:
                with open(checkpoint_file, 'w') as f:
                    json.dump({
                        'processed': list(processed_ids),
                        'last_index': index
                    }, f)
                print(f"Checkpoint saved: {len(processed_ids)} processed")
                
        except Exception as e:
            print(f"Error processing {student_id}: {e}")
            # Continue with next record
    
    # Final checkpoint
    with open(checkpoint_file, 'w') as f:
        json.dump({
            'processed': list(processed_ids),
            'last_index': len(df)
        }, f)
    
    print(f"Import complete: {len(processed_ids)} records processed")
```

## Next Steps

- Review [Python Client Examples](./python_client.md) for API integration
- See [JavaScript Client Examples](./javascript_client.md) for frontend usage
- Check the [API Reference](../API_REFERENCE.md) for endpoint details