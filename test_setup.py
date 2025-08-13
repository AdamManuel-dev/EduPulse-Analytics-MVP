#!/usr/bin/env python
"""
Test script to verify the EduPulse setup.
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from src.db import models
        from src.models import schemas
        from src.config.settings import get_settings
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_database():
    """Test database connection."""
    print("\nTesting database connection...")
    try:
        from src.db.database import engine
        from sqlalchemy import text
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("✓ Database connection successful")
            
            # Check if tables exist
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'edupulse' 
                LIMIT 5
            """))
            tables = [row[0] for row in result]
            print(f"✓ Found {len(tables)} tables: {tables[:5]}")
            return True
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False

def test_redis():
    """Test Redis connection."""
    print("\nTesting Redis connection...")
    try:
        import redis
        from src.config.settings import get_settings
        settings = get_settings()
        
        r = redis.from_url(str(settings.redis_url))
        r.ping()
        print("✓ Redis connection successful")
        return True
    except Exception as e:
        print(f"✗ Redis connection failed: {e}")
        return False

def test_api():
    """Test API endpoints."""
    print("\nTesting API endpoints...")
    try:
        import requests
        
        # Try to connect to the API
        response = requests.get("http://localhost:8001/health", timeout=2)
        if response.status_code == 200:
            print("✓ API health check successful")
            return True
        else:
            print(f"✗ API returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("ℹ API not running (start with: uvicorn src.api.main:app --reload)")
        return False
    except Exception as e:
        print(f"✗ API test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("EduPulse Setup Verification")
    print("=" * 50)
    
    results = {
        "Imports": test_imports(),
        "Database": test_database(),
        "Redis": test_redis(),
        "API": test_api()
    }
    
    print("\n" + "=" * 50)
    print("Summary:")
    print("-" * 50)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:15} {status}")
    
    all_passed = all(results.values())
    print("-" * 50)
    
    if all_passed:
        print("✓ All tests passed! Setup is complete.")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())