import pytest
import pandas as pd
import numpy as np
from core.processing.cleaner import EnterpriseCleaner

@pytest.fixture
def sample_data():
    """Provides a synthetic raw dataset for unit testing."""
    return pd.DataFrame({
        "id": [1, 2, 2, 3],  # Includes duplicate
        "title": ["Data Engineer", "Sr Analyst", "Sr Analyst", "Junior Dev"],
        "company": ["TechCorp", "DataInc", "DataInc", "DevLLC"],
        "category": ["Data", "Analytics", "Analytics", "Dev"],
        "tags": ["Python, SQL", "Excel, Tablue", "Excel, Tablue", "Java"],
        "salary": ["$120,000", "80k", "80k", "$45"]  # Diverse formats
    })

def test_duplicate_removal(sample_data):
    """Ensures EnterpriseCleaner removes duplicates by ID."""
    cleaner = EnterpriseCleaner()
    cleaned = cleaner._vectorized_clean(sample_data)
    assert len(cleaned) == 3

def test_salary_parsing(sample_data):
    """Validates consistent extraction of numeric salaries."""
    cleaner = EnterpriseCleaner()
    cleaned = cleaner._vectorized_clean(sample_data)
    
    # 120k string to numeric
    assert cleaned.loc[cleaned["id"] == 1, "salary_numeric"].iloc[0] == 120000.0
    
    # 80k to numeric
    assert cleaned.loc[cleaned["id"] == 2, "salary_numeric"].iloc[0] == 80000.0
    
    # $45 hourly to annual (~$86k)
    assert cleaned.loc[cleaned["id"] == 3, "salary_numeric"].iloc[0] > 80000

def test_schema_validation():
    """Ensures exceptions are raised for missing columns."""
    from core.quality.profiler import DataProfiler
    df_broken = pd.DataFrame({"oops": [1]})
    
    with pytest.raises(ValueError, match="Missing columns"):
        DataProfiler.validate_schema(df_broken, ["id", "salary"])
