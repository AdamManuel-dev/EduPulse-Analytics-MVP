# Data Import Examples

This guide provides examples for importing student data into the EduPulse Analytics system.

## CSV Import

### Student Data Import

```python
import pandas as pd
import requests
from datetime import datetime

def import_students_from_csv(csv_file, api_url):
    """Import students from CSV file"""
    
    # Read CSV
    df = pd.read_csv(csv_file)
    
    # Required columns
    required_columns = [
        'student_id', 'first_name', 'last_name', 'grade_level',
        'date_of_birth', 'enrollment_date'
    ]
    
    # Validate columns
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    # Import each student
    success_count = 0
    errors = []
    
    for _, row in df.iterrows():
        student_data = {
            'student_id': row['student_id'],
            'first_name': row['first_name'],
            'last_name': row['last_name'],
            'grade_level': int(row['grade_level']),
            'date_of_birth': row['date_of_birth'],
            'enrollment_date': row['enrollment_date'],
            'gender': row.get('gender', 'U'),
            'ethnicity': row.get('ethnicity', 'Unknown'),
            'socioeconomic_status': row.get('socioeconomic_status', 'middle')
        }
        
        try:
            response = requests.post(
                f"{api_url}/api/v1/students",
                json=student_data
            )
            response.raise_for_status()
            success_count += 1
        except Exception as e:
            errors.append({
                'student_id': row['student_id'],
                'error': str(e)
            })
    
    print(f"Imported {success_count} students")
    if errors:
        print(f"Errors: {len(errors)}")
        for error in errors[:5]:
            print(f"  - {error['student_id']}: {error['error']}")
    
    return success_count, errors

# Example usage
success, errors = import_students_from_csv('students.csv', 'http://localhost:8000')
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