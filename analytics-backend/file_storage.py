import os
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Storage directory for persistent files
STORAGE_DIR = "data_storage"

def ensure_storage_directory():
    """Ensure the storage directory exists."""
    os.makedirs(STORAGE_DIR, exist_ok=True)

def save_merged_csv(merged_df: pd.DataFrame) -> str:
    """
    Save the merged dataframe to persistent storage.
    
    Args:
        merged_df: The merged dataframe to save
        
    Returns:
        Path to the saved file
    """
    ensure_storage_directory()
    file_path = os.path.join(STORAGE_DIR, "merged.csv")
    merged_df.to_csv(file_path, index=False)
    logger.info(f"Saved merged.csv to {file_path}")
    return file_path

def load_merged_csv() -> Optional[pd.DataFrame]:
    """
    Load the merged dataframe from persistent storage.
    
    Returns:
        The merged dataframe if it exists, None otherwise
    """
    file_path = os.path.join(STORAGE_DIR, "merged.csv")
    if os.path.exists(file_path):
        logger.info(f"Loading merged.csv from {file_path}")
        return pd.read_csv(file_path)
    else:
        logger.warning(f"merged.csv not found at {file_path}")
        return None

def load_provider_roster() -> Optional[pd.DataFrame]:
    """
    Load the provider roster with errors CSV file.
    
    Returns:
        The provider roster dataframe if it exists, None otherwise
    """
    file_path = os.path.join(STORAGE_DIR, "provider_roster_with_errors.csv")
    if os.path.exists(file_path):
        logger.info(f"Loading provider_roster_with_errors.csv from {file_path}")
        return pd.read_csv(file_path)
    else:
        logger.warning(f"provider_roster_with_errors.csv not found at {file_path}")
        return None

def save_provider_roster(roster_df: pd.DataFrame) -> str:
    """
    Save the provider roster dataframe to persistent storage.
    
    Args:
        roster_df: The provider roster dataframe to save
        
    Returns:
        Path to the saved file
    """
    ensure_storage_directory()
    file_path = os.path.join(STORAGE_DIR, "provider_roster_with_errors.csv")
    roster_df.to_csv(file_path, index=False)
    logger.info(f"Saved provider_roster_with_errors.csv to {file_path}")
    return file_path

def check_required_files_exist() -> dict:
    """
    Check if required files exist in storage.
    
    Returns:
        Dictionary with file existence status
    """
    merged_path = os.path.join(STORAGE_DIR, "merged.csv")
    roster_path = os.path.join(STORAGE_DIR, "provider_roster_with_errors.csv")
    
    return {
        "merged_csv_exists": os.path.exists(merged_path),
        "provider_roster_exists": os.path.exists(roster_path),
        "merged_csv_path": merged_path,
        "provider_roster_path": roster_path
    }
