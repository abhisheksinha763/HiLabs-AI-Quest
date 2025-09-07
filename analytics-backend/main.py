from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from typing import List
import pandas as pd
import io
import logging
import re
import numpy as np
from fuzzywuzzy import fuzz, process
import os
import shutil
from data_processing import clean_roster, get_state_city_distribution, calculate_compliance_status, data_quality_by_state, generate_expired_license_compliance_report
from file_storage import save_merged_csv, load_merged_csv, load_provider_roster, save_provider_roster, check_required_files_exist
from intelligent_query_service import IntelligentQueryService
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Intelligent Query service
CSV_PATH = "/Users/adityagupta/Code/analytics-backend/data_storage/cleaned_roster.csv"
intelligent_query_service = IntelligentQueryService(csv_path=CSV_PATH)

app = FastAPI(
    title="CSV Analytics Backend",
    description="FastAPI backend for processing CSV files from Next.js frontend",
    version="1.0.0"
)

# Allow your Next.js origin (dev + prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "https://your-site.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_TYPES = {"text/csv", "application/vnd.ms-excel", "application/octet-stream"}  # browsers vary

def merge_datasets(file_paths):
    """
    Merge multiple provider datasets with the mock_npi_registry dataset.
    
    Parameters:
        file_paths: list
            x+1 file paths where:
            - First x are provider license datasets
            - Last one is mock_npi_registry dataset
    
    Returns:
        final_df: pd.DataFrame
            Merged dataframe
    """
    
    if len(file_paths) < 2:
        raise ValueError("Need at least 2 files: provider dataset(s) and the mock_npi_registry file")
    
    # Step 1: Load provider datasets (first x files)
    provider_dfs = [pd.read_csv(path) for path in file_paths[:-1]]
    merged_df = pd.concat(provider_dfs, ignore_index=True)
    
    # Step 2: Load mock_npi_registry (last file)
    df_ref_db_3 = pd.read_csv(file_paths[-1])
    
    # Step 3: Detect license number columns
    lic_num_cols = [col for col in df_ref_db_3.columns if "lic" in col.lower() and "num" in col.lower()]
    
    # Step 4: Create unified license_number column
    df_ref_db_3["license_number"] = df_ref_db_3[lic_num_cols].bfill(axis=1).iloc[:, 0]
    
    # Step 5: Merge
    final_df = pd.merge(
        merged_df,
        df_ref_db_3,
        on="license_number",
        how="left"
    )
    
    return final_df

def find_invalid_phones(df_provider_db):
    """
    This function checks for formatting issues in the 'practice_phone' column.
    Valid formats:
        1. +1 (XXX) XXX-XXXX
        2. (XXX) XXX-XXXX
    Returns:
        - A list of dictionaries with invalid phone numbers along with full_name and license_number
        - Count of invalid phone numbers
    """
    
    # Define regex patterns for valid phone formats
    pattern1 = re.compile(r'^\+1 \(\d{3}\) \d{3}-\d{4}$')
    pattern2 = re.compile(r'^\(\d{3}\) \d{3}-\d{4}$')
    
    invalid_entries = []
    
    for _, row in df_provider_db.iterrows():
        phone = str(row.get('practice_phone', '')).strip()
        
        if not (pattern1.match(phone) or pattern2.match(phone)):
            invalid_entries.append({
                'license_number': row.get('license_number', ''),
                'provider_name': row.get('full_name', ''),
                'phone_number': phone
            })
    
    invalid_count = len(invalid_entries)
    
    return invalid_entries, invalid_count

def gupta_use(df_provider_db, df_merged, weights=(0.25, 0.25, 0.25, 0.25)):
    def format_score(df, threshold=60):
        cols = list(df.columns)
        best_npi = process.extractOne("npi", cols, scorer=fuzz.partial_ratio)
        npi_col = best_npi[0] if best_npi else None
        best_phone = process.extractOne("phone", cols, scorer=fuzz.partial_ratio)
        phone_col = best_phone[0] if best_phone else None
        
        if npi_col is None and phone_col is None:
            return None, []
        
        total_rows = len(df)
        if total_rows == 0:
            return 1.0, []
        
        # Use the new find_invalid_phones function to get bad_phone_list
        bad_phone_list, phone_invalid_count = find_invalid_phones(df)
        
        invalid_count = 0
        for _, row in df.iterrows():
            invalid = False
            if npi_col is not None:
                val = row[npi_col]
                if not pd.notna(val) or not str(val).isnumeric():
                    invalid = True
            if phone_col is not None:
                val = row[phone_col]
                if not pd.notna(val):
                    invalid = True
                else:
                    s = re.sub(r'[\s\-\(\)]', '', str(val))
                    if s.startswith('+'):
                        s_clean = s[1:]
                        if not (s_clean.isdigit() and len(s_clean) == 11 and s_clean.startswith('1')):
                            invalid = True
                    else:
                        if not (s.isdigit() and len(s) == 10):
                            invalid = True
            if invalid:
                invalid_count += 1
        
        score = 1 - (invalid_count / total_rows)
        return score, bad_phone_list

    def completeness_score(df):
        total_cells = df.shape[0] * df.shape[1]
        if total_cells == 0:
            return 1.0
        missing_cells = df.isna().sum().sum() + (df == '').sum().sum()
        score = (missing_cells / total_cells)
        return 1 - score

    def redundancy_score(df, threshold=70):
        cols = list(df.columns)
        res = process.extractOne("license number", cols, scorer=fuzz.partial_ratio)
        if not res:
            return None, 0
        best = res[0]
        score = res[1]
        if score < threshold:
            return None, 0
        lic_col = best
        total_count = len(df[lic_col])
        unique_count = df[lic_col].nunique(dropna=True)
        if total_count == 0:
            return 0.0, 0
        redundancy = (unique_count / total_count)
        num_duplicates = total_count - unique_count
        return redundancy, num_duplicates

    def accuracy(df_provider_db, df_merged):
        def norm(s):
            if pd.isna(s):
                return ""
            s = str(s)
            s = re.sub(r'\s+', ' ', s).strip().lower()
            return s
        def norm_npi(x):
            if pd.isna(x):
                return ""
            if isinstance(x, (np.integer, int)):
                return str(int(x))
            if isinstance(x, (np.floating, float)):
                if float(x).is_integer():
                    return str(int(round(x)))
                s = repr(x)
            else:
                s = str(x)
            s = s.strip().lower()
            s = re.sub(r'[^0-9a-z]+', '', s)
            if s == "":
                return ""
            if s.isdigit():
                s = s.lstrip('0')
                if s == "":
                    s = "0"
            return s
        cols = df_provider_db.columns
        best_npi = process.extractOne("npi", cols, scorer=fuzz.partial_ratio)
        best_lic = process.extractOne("license number", cols, scorer=fuzz.partial_ratio)
        best_name = process.extractOne("full name", cols, scorer=fuzz.partial_ratio)
        if best_npi is None or best_lic is None or best_name is None:
            return None
        npi_col = best_npi[0]
        license_col = best_lic[0]
        full_col = best_name[0]
        if not (npi_col and license_col and full_col):
            return None
        cols_merged = df_merged.columns
        best_npi_m = process.extractOne("npi", cols_merged, scorer=fuzz.partial_ratio)
        best_lic_m = process.extractOne("license number", cols_merged, scorer=fuzz.partial_ratio)
        best_name_m = process.extractOne("provider name", cols_merged, scorer=fuzz.partial_ratio)
        if best_npi_m is None or best_lic_m is None or best_name_m is None:
            return None
        npi_col_m = best_npi_m[0]
        license_col_m = best_lic_m[0]
        full_col_m = best_name_m[0]
        if not (npi_col_m and license_col_m and full_col_m):
            return None
        merged_map = {}
        for _, r in df_merged.iterrows():
            k = (norm_npi(r[npi_col_m]), norm(r[license_col_m]))
            merged_map.setdefault(k, []).append(norm(r[full_col_m]))
        total_rows = len(df_provider_db)
        if total_rows == 0:
            return 1.0
        valid_count = 0
        for _, r in df_provider_db.iterrows():
            key = (norm_npi(r[npi_col]), norm(r[license_col]))
            if key in merged_map:
                prov_name = norm(r[full_col])
                candidates = merged_map[key]
                if not candidates:
                    continue
                best_score = max(fuzz.ratio(prov_name, cand) for cand in candidates)
                if best_score >= 70:
                    valid_count += 1
        return valid_count / total_rows

    # Calculate individual scores
    f, bad_phone_list = format_score(df_provider_db)
    r, num_duplicates = redundancy_score(df_provider_db)
    c = completeness_score(df_provider_db)
    a = accuracy(df_provider_db, df_merged)
    
    # Calculate data quality score
    data_quality = weights[0]*f + weights[1]*r + weights[2]*c + weights[3]*a
    
    # Calculate additional metrics
    total_providers = len(df_provider_db)
    unique_providers = df_provider_db.nunique().sum() if total_providers > 0 else 0
    percentage_duplicate = (num_duplicates / total_providers * 100) if total_providers > 0 else 0
    
    return data_quality, num_duplicates, percentage_duplicate, unique_providers, bad_phone_list

@app.get("/")
async def root():
    return {"message": "CSV Analytics Backend is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "csv-analytics-backend"}

@app.post("/process")
async def process_csv(files: List[UploadFile] = File(...)):
    """
    Process 4 CSV files: provider_roster_with_errors.csv, ny_medical_license_database.csv, 
    ca_medical_license_database.csv, and mock_npi_registry.csv
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    if len(files) != 4:
        raise HTTPException(status_code=400, detail="Exactly 4 files required: provider_roster_with_errors.csv, ny_medical_license_database.csv, ca_medical_license_database.csv, mock_npi_registry.csv")
    
    # Log received files
    logger.info(f"Received {len(files)} file(s) for processing:")
    for i, f in enumerate(files):
        logger.info(f"  File {i+1}: {f.filename} (Content-Type: {f.content_type})")
    
    # Validate file types
    for f in files:
        if f.content_type not in ALLOWED_TYPES:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {f.content_type}. Only CSV files are allowed."
            )

    try:
        # Create temporary directory for processing
        temp_dir = "temp_processing"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save uploaded files temporarily
        file_paths = []
        provider_file_path = None
        
        for f in files:
            # Read file content
            raw = await f.read()
            
            # Save to temporary file
            temp_file_path = os.path.join(temp_dir, f.filename)
            with open(temp_file_path, 'wb') as temp_file:
                temp_file.write(raw)
            
            file_paths.append(temp_file_path)
            
            # Identify provider roster file
            if "provider_roster_with_errors" in f.filename:
                provider_file_path = temp_file_path
        
        # Identify file order: ny_medical, ca_medical, mock_npi_registry
        merge_file_paths = []
        for f in files:
            temp_file_path = os.path.join(temp_dir, f.filename)
            if "ny_medical_license_database" in f.filename:
                merge_file_paths.append(temp_file_path)
            elif "ca_medical_license_database" in f.filename:
                merge_file_paths.append(temp_file_path)
            elif "mock_npi_registry" in f.filename:
                merge_file_paths.append(temp_file_path)
        
        if len(merge_file_paths) != 3:
            raise HTTPException(status_code=400, detail="Missing required files: ny_medical_license_database.csv, ca_medical_license_database.csv, mock_npi_registry.csv")
        
        # Step 1: Merge datasets (File 3, File 4, File 2)
        logger.info("Starting dataset merge process...")
        merged_df = merge_datasets(merge_file_paths)
        
        # Save merged.csv to temporary location and persistent storage
        merged_csv_path = os.path.join(temp_dir, "merged.csv")
        merged_df.to_csv(merged_csv_path, index=False)
        logger.info(f"Saved merged dataset to {merged_csv_path}")
        
        # Step 2: Load provider roster file
        provider_df = pd.read_csv(provider_file_path)
        
        # Save to persistent storage for demographics endpoint
        save_merged_csv(merged_df)
        save_provider_roster(provider_df)
        
        # Save provider file as CSV (already is CSV, but for consistency)
        provider_csv_path = os.path.join(temp_dir, "provider_roster.csv")
        provider_df.to_csv(provider_csv_path, index=False)
        logger.info(f"Saved provider roster to {provider_csv_path}")
        
        # Step 3: Run data quality analysis
        logger.info("Running data quality analysis...")
        data_quality, num_duplicates, percentage_duplicate, unique_providers, bad_phone_list = gupta_use(provider_df, merged_df)
        
        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dir)
        
        # Return results
        return {
            "status": "success",
            "message": "Successfully processed all files and completed data quality analysis",
            "data_quality_score": float(data_quality) if data_quality is not None else None,
            "num_duplicates": int(num_duplicates) if num_duplicates is not None else 0,
            "percentage_duplicate": float(percentage_duplicate),
            "unique_providers": int(unique_providers),
            "bad_phone_list": bad_phone_list,
            "provider_file_info": {
                "filename": os.path.basename(provider_file_path),
                "rows": len(provider_df),
                "columns": len(provider_df.columns)
            },
            "merged_file_info": {
                "rows": len(merged_df),
                "columns": len(merged_df.columns)
            }
        }
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        logger.error(f"Error processing files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

@app.get("/demographics")
async def get_demographics():
    """
    Process demographics data using saved merged.csv and provider_roster_with_errors.csv files.
    Returns state-city distribution and compliance status.
    """
    try:
        # Check if required files exist
        file_status = check_required_files_exist()
        
        if not file_status["merged_csv_exists"]:
            raise HTTPException(
                status_code=404, 
                detail="merged.csv not found. Please run the /process endpoint first to generate the merged dataset."
            )
        
        if not file_status["provider_roster_exists"]:
            raise HTTPException(
                status_code=404, 
                detail="provider_roster_with_errors.csv not found. Please run the /process endpoint first."
            )
        
        # Load required datasets
        logger.info("Loading datasets for demographics analysis...")
        merged_df = load_merged_csv()
        roster_df = load_provider_roster()
        
        if merged_df is None or roster_df is None:
            raise HTTPException(
                status_code=500, 
                detail="Failed to load required datasets from storage."
            )
        
        # Step 1: Clean the roster using merged data
        logger.info("Cleaning roster data...")
        cleaned_df, removed_count = clean_roster(roster_df, merged_df)
        logger.info(f"Cleaned roster: removed {removed_count} rows, {len(cleaned_df)} rows remaining")
        
        # Step 2: Get state-city distribution
        logger.info("Calculating state-city distribution...")
        state_city_distribution = get_state_city_distribution(cleaned_df)
        
        # Step 3: Calculate compliance status
        logger.info("Calculating compliance status...")
        compliance_status = calculate_compliance_status(cleaned_df, merged_df)
        
        # Return combined results
        return {
            "status": "success",
            "message": "Demographics analysis completed successfully",
            "data": {
                "state_city_distribution": state_city_distribution,
                "compliance_status": compliance_status
            },
            "processing_info": {
                "original_roster_count": len(roster_df),
                "cleaned_roster_count": len(cleaned_df),
                "removed_count": removed_count,
                "merged_dataset_count": len(merged_df)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in demographics analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing demographics: {str(e)}")

# -----------------------
# Chat API schemas
# -----------------------
class ChatRequest(BaseModel):
    query: str

# -----------------------
# Chat endpoint (streaming)
# -----------------------
@app.post("/chat")
def chat(req: ChatRequest):
    """
    Chat endpoint that answers questions based on the cleaned roster CSV data.
    Uses intelligent query processing with dynamic pandas code generation.
    """
    try:
        logger.info(f"Processing intelligent query: {req.query[:100]}...")
        
        # Process query using intelligent query service
        response = intelligent_query_service.process_query(req.query)
        
        def stream():
            yield response
        
        return StreamingResponse(stream(), media_type="text/plain")
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        def error_stream():
            yield f"[Error] Failed to process your question: {str(e)}"
        return StreamingResponse(error_stream(), media_type="text/plain")

@app.post("/reindex")
def reindex_csv():
    """
    Reinitialize the intelligent query service with updated CSV data.
    Call this after updating the CSV data.
    """
    try:
        global intelligent_query_service
        intelligent_query_service = IntelligentQueryService(csv_path=CSV_PATH)
        dataset_summary = intelligent_query_service.get_dataset_summary()
        return {"status": "success", "message": f"Reinitialized with {dataset_summary.get('total_rows', 0)} rows"}
    except Exception as e:
        logger.error(f"Error reinitializing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reinitializing CSV: {str(e)}")

@app.get("/geographics")
async def get_geographics():
    """
    Get data quality metrics by state using provider_roster_with_errors.csv and merged.csv.
    Returns state-level data quality scores including phone format, redundancy, completeness, and accuracy.
    """
    try:
        # Check if required files exist
        file_status = check_required_files_exist()
        
        if not file_status["merged_csv_exists"]:
            raise HTTPException(
                status_code=404, 
                detail="merged.csv not found. Please run the /process endpoint first to generate the merged dataset."
            )
        
        if not file_status["provider_roster_exists"]:
            raise HTTPException(
                status_code=404, 
                detail="provider_roster_with_errors.csv not found. Please run the /process endpoint first."
            )
        
        # Load required datasets
        logger.info("Loading datasets for geographic data quality analysis...")
        merged_df = load_merged_csv()
        roster_df = load_provider_roster()
        
        if merged_df is None or roster_df is None:
            raise HTTPException(
                status_code=500, 
                detail="Failed to load required datasets from storage."
            )
        
        # Run data quality analysis by state
        logger.info("Running data quality analysis by state...")
        state_quality_scores = data_quality_by_state(roster_df, merged_df)
        
        # Return results
        return {
            "status": "success",
            "message": "Geographic data quality analysis completed successfully",
            "data": state_quality_scores,
            "processing_info": {
                "total_states_analyzed": len(state_quality_scores),
                "provider_roster_count": len(roster_df),
                "merged_dataset_count": len(merged_df)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in geographic analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing geographic data quality: {str(e)}")

@app.get("/expiry")
async def get_expired_license_compliance():
    """
    Generate comprehensive compliance report for expired licenses.
    Uses merged.csv and cleaned_roster.csv from data_storage directory.
    """
    try:
        # Define file paths
        merged_csv_path = "/Users/adityagupta/Code/analytics-backend/data_storage/merged.csv"
        cleaned_roster_path = "/Users/adityagupta/Code/analytics-backend/data_storage/cleaned_roster.csv"
        
        # Check if required files exist
        if not os.path.exists(merged_csv_path):
            raise HTTPException(
                status_code=404, 
                detail="merged.csv not found in data_storage directory. Please run the /process endpoint first."
            )
        
        if not os.path.exists(cleaned_roster_path):
            raise HTTPException(
                status_code=404, 
                detail="cleaned_roster.csv not found in data_storage directory. Please run the /process endpoint first."
            )
        
        # Load datasets
        logger.info("Loading datasets for expired license compliance analysis...")
        merged_df = pd.read_csv(merged_csv_path)
        cleaned_roster_df = pd.read_csv(cleaned_roster_path)
        
        # Generate expired license compliance report
        logger.info("Generating expired license compliance report...")
        compliance_report = generate_expired_license_compliance_report(merged_df, cleaned_roster_df)
        
        # Return results
        return {
            "status": "success",
            "message": "Expired license compliance report generated successfully",
            "data": compliance_report,
            "processing_info": {
                "merged_dataset_count": len(merged_df),
                "cleaned_roster_count": len(cleaned_roster_df)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in expired license compliance analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing expired license compliance: {str(e)}")

@app.get("/export")
async def export_cleaned_roster():
    """
    Export the cleaned_roster.csv file from data_storage directory.
    Returns the CSV file as a downloadable attachment.
    """
    try:
        # Define file path
        cleaned_roster_path = "/Users/adityagupta/Code/analytics-backend/data_storage/cleaned_roster.csv"
        
        # Check if file exists
        if not os.path.exists(cleaned_roster_path):
            raise HTTPException(
                status_code=404, 
                detail="cleaned_roster.csv not found in data_storage directory. Please run the /process endpoint first to generate the cleaned roster."
            )
        
        logger.info(f"Exporting cleaned_roster.csv from {cleaned_roster_path}")
        
        # Return file as downloadable response
        return FileResponse(
            path=cleaned_roster_path,
            filename="cleaned_roster.csv",
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=cleaned_roster.csv"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting cleaned roster: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error exporting file: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
