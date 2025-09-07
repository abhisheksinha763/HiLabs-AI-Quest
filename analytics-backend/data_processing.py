import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any
import os
import re
import numpy as np
from fuzzywuzzy import fuzz


def standardize_phone_number( phone: str) -> str:
    digits = re.sub(r'\D', '', str(phone))  # Extract only digits
    
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"  # (XXX) XXX-XXXX
    elif len(digits) == 11 and digits[0] == '1':
        return f"({digits[1:4]}) {digits[4:7]}-{digits[7:]}"  # Remove +1 country code
    else:
        return None  # Keep null if invalid length

def update_provider_addresses(df_final, df_merged):

    df_merged_dedup = df_merged.drop_duplicates(
        subset=['license_number', 'specialty'], keep='last')


    merged = df_final.merge(
        df_merged_dedup[[
            'license_number', 'specialty',
            'provider_business_mailing_address_line1',
            'provider_business_mailing_address_line2',
            'provider_business_practice_location_address_line1',
            'provider_business_practice_location_address_line2'
        ]],
        how='left',
        left_on=['license_number', 'primary_specialty'],
        right_on=['license_number', 'specialty'])

    
    merged['mailing_address_line1'] = merged['provider_business_mailing_address_line1'].combine_first(
        merged['mailing_address_line1'])
    merged['mailing_address_line2'] = merged['provider_business_mailing_address_line2'].combine_first(
        merged['mailing_address_line2'])
    merged['practice_address_line1'] = merged['provider_business_practice_location_address_line1'].combine_first(
        merged['practice_address_line1'])
    merged['practice_address_line2'] = merged['provider_business_practice_location_address_line2'].combine_first(
        merged['practice_address_line2'])

    return merged[df_final.columns]

  

def clean_roster(roster_df, merged_df):
    initial_count = len(roster_df)
    roster_df['npi'] = roster_df['npi'].astype('string')
    
    # Deduplicate valid licenses with NPI
    valid_licenses = merged_df[['license_number', 'address_state', 'npi']].drop_duplicates()
    
    # First: keep rows where license_number matches
    temp = roster_df.merge(
        valid_licenses, 
        on="license_number", 
        how="inner",
        suffixes=('', '_merged')
    )
    
    # Second: filter rows with both license_state and NPI validation
    # License state validation: Keep if license_state is NaN OR address_state is NaN OR they match
    license_valid = (
        temp['license_state'].isna() | 
        temp['address_state'].isna() |
        (temp['license_state'] == temp['address_state'])
    )

    # NPI validation: Keep if npi is NaN OR npi_merged is NaN OR they match
    npi_valid = (
        temp['npi'].isna() | 
        temp['npi_merged'].isna() |
        (temp['npi'] == temp['npi_merged'])
    )
    
    # Apply both validations
    final_df = temp[license_valid & npi_valid].drop(columns=['address_state', 'npi_merged'])
    removed_count = initial_count - len(final_df)
    final_df = final_df.merge(merged_df[['license_number', 'status']], on='license_number', how='left')
    final_df = update_provider_addresses(final_df,merged_df)
    final_df['practice_phone'] = final_df['practice_phone'].apply(standardize_phone_number)
    return final_df, removed_count


def get_state_city_distribution(final_df):
    """Return array of dictionaries with state-city provider distribution"""
    
    result = []
    
    # Group by license_state
    for state in final_df['license_state'].dropna().unique():
        state_df = final_df[final_df['license_state'] == state]
        
        # Count unique license numbers in state
        state_total = state_df['license_number'].nunique()
        
        # Count unique license numbers by city within state
        city_counts = state_df.groupby(state_df['practice_city'].str.upper())['license_number'].nunique().to_dict()
        
        result.append({
            state: state_total,
            'cities': city_counts
        })
    
    return result


def calculate_compliance_status(final_df: pd.DataFrame, merged_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate percentage of providers with valid, active licenses.
    
    Args:
        final_df: Cleaned dataframe from clean_roster function
        merged_df: Merged dataframe from progress API
        
    Returns:
        Dictionary with compliance statistics
    """
    # Merge to get status information
    df_with_status = final_df.merge(
        merged_df[['license_number', 'status', 'expiration_date']], 
        on='license_number', 
        how='left'
    )
    
    # Convert expiration to datetime
    df_with_status['license_expiration_merged'] = pd.to_datetime(
        df_with_status['expiration_date'], errors='coerce'
    )
    
    # Count truly active licenses: status='active' AND not expired
    current_date = datetime.now()
    active_condition = (
        (df_with_status['status'] == 'Active') & 
        (df_with_status['license_expiration_merged'] > current_date)
    )
    
    total_providers = len(final_df)
    active_licenses = active_condition.sum()
    inactive_licenses = (df_with_status['status'] == 'Inactive').sum()
    suspended_licenses = (~df_with_status['status'].isin(['Active', 'Expired', 'Inactive'])).sum()
    expired_licenses = total_providers - active_licenses - inactive_licenses - suspended_licenses
    
    active_percentage = ((active_licenses) / (total_providers) * 100).round(2)
    inactive_percentage = ((inactive_licenses) / (total_providers) * 100).round(2)
    expired_percentage = ((expired_licenses) / (total_providers) * 100).round(2)
    suspended_percentage = ((suspended_licenses) / (total_providers) * 100).round(2)
    
    return {
        'total_providers': int(total_providers),
        'active_licenses': int(active_licenses),
        'inactive_licenses': int(inactive_licenses),
        'suspended_licenses': int(suspended_licenses),
        'expired_licenses': int(expired_licenses),
        'active_percentage': float(active_percentage),
        'inactive_percentage': float(inactive_percentage),
        'suspended_percentage': float(suspended_percentage),
        'expired_percentage': float(expired_percentage)
    }


def data_quality_by_state(df_provider_db, df_merged, weights=(0.25, 0.25, 0.25, 0.25)):
    """
    Calculate data quality metrics by state for provider database.
    
    Args:
        df_provider_db: Provider roster dataframe
        df_merged: Merged reference dataframe
        weights: Tuple of weights for (phone_format, redundancy, completeness, accuracy)
    
    Returns:
        Dictionary with state-level data quality scores and metrics
    """

    def phone_format_score(df):
        phone_col = "practice_phone"
        if phone_col not in df.columns:
            return 1.0, []
        total_rows = len(df)
        if total_rows == 0:
            return 1.0, []
        pattern1 = r'^\(\d{3}\) \d{3}-\d{4}$'
        pattern2 = r'^\+1 \(\d{3}\) \d{3}-\d{4}$'
        bad_rows = []
        valid_count = 0
        for _, row in df.iterrows():
            val = row[phone_col]
            if not pd.notna(val):
                bad_rows.append({
                    "license_number": row.get("license_number"),
                    "full_name": row.get("full_name"),
                    "phone": val
                })
                continue
            s = str(val).strip()
            if re.match(pattern1, s) or re.match(pattern2, s):
                valid_count += 1
            else:
                bad_rows.append({
                    "license_number": row.get("license_number"),
                    "full_name": row.get("full_name"),
                    "phone": val
                })
        score = valid_count / total_rows
        return score, bad_rows

    def completeness_score(df):
        total_cells = df.shape[0] * df.shape[1]
        if total_cells == 0:
            return 1.0
        missing_cells = df.isna().sum().sum() + (df == '').sum().sum()
        return 1 - (missing_cells / total_cells)

    def redundancy_metrics(df):
        lic_col = "license_number"
        if lic_col not in df.columns:
            return 0.0, 0
        total_count = len(df[lic_col])
        if total_count == 0:
            return 0.0, 0
        unique_count = df[lic_col].nunique(dropna=True)
        ratio = unique_count / total_count
        duplicate_count = total_count - unique_count
        return ratio, duplicate_count

    def accuracy(df_provider_db_subset, df_merged):
        npi_col = "npi"
        license_col = "license_number"
        full_col = "full_name"
        state_col = "practice_state"
        npi_col_m = "npi"
        license_col_m = "license_number"
        full_col_m = "provider_name"
        state_col_m = "provider_business_practice_location_address_state"

        def norm(s):
            if pd.isna(s):
                return ""
            s = str(s).strip().lower()
            s = re.sub(r'\s+', ' ', s)
            return s

        def norm_npi(x):
            if pd.isna(x):
                return ""
            if isinstance(x, (int, np.integer)):
                return str(int(x))
            if isinstance(x, (float, np.floating)):
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

        merged_map = {}
        for _, r in df_merged.iterrows():
            key = (norm_npi(r.get(npi_col_m)), norm(r.get(license_col_m)), norm(r.get(state_col_m)))
            merged_map.setdefault(key, []).append(norm(r.get(full_col_m)))

        total_rows = len(df_provider_db_subset)
        if total_rows == 0:
            return 1.0

        valid_count = 0
        for _, r in df_provider_db_subset.iterrows():
            key = (norm_npi(r.get(npi_col)), norm(r.get(license_col)), norm(r.get(state_col)))
            if key in merged_map:
                prov_name = norm(r.get(full_col))
                candidates = merged_map[key]
                if not candidates:
                    continue
                best_score = max(fuzz.ratio(prov_name, cand) for cand in candidates)
                if best_score >= 70:
                    valid_count += 1

        return valid_count / total_rows

    state_scores = {}
    states = df_provider_db['license_state'].dropna().unique()

    for state in states:
        df_state = df_provider_db[df_provider_db['license_state'] == state]

        phone_score, bad_phones = phone_format_score(df_state)
        redundancy_ratio, duplicate_license_numbers = redundancy_metrics(df_state)
        completeness = completeness_score(df_state)
        acc = accuracy(df_state, df_merged)
        overall = weights[0]*phone_score + weights[1]*redundancy_ratio + weights[2]*completeness + weights[3]*acc

        state_scores[state] = {
            "overall_score": overall,
            "phone_format_score": phone_score,
            "bad_phone_list": bad_phones,
            "duplicate_license_numbers": int(duplicate_license_numbers),
            "accuracy_score": acc
        }

    return state_scores


def generate_expired_license_compliance_report(merged_df, final_df):
    """Generate comprehensive compliance report for expired licenses"""
    
    # Filter expired licenses from merged database
    expired_licenses = merged_df[merged_df['status'] == 'Expired'].copy()
    expired_licenses['expiration_date'] = pd.to_datetime(expired_licenses['expiration_date'])
    
    # Get current date for analysis
    current_date = datetime.now()
    
    # 1. SUMMARY STATISTICS
    total_licenses = len(merged_df)
    total_expired = len(expired_licenses)
    expired_percentage = round((total_expired / total_licenses * 100), 2)
    
    state_distribution = expired_licenses['address_state'].value_counts()
    
    # 4. SPECIALTY BREAKDOWN
    expired_with_specialty = expired_licenses.merge(
        final_df[['license_number', 'primary_specialty']], 
        on='license_number', 
        how='left'
    )
    
    specialty_distribution = expired_with_specialty['primary_specialty'].value_counts()
    
    # 5. TIMELINE ANALYSIS
    # Calculate days since expiration
    expired_licenses['days_expired'] = (current_date - expired_licenses['expiration_date']).dt.days
    
    # Categorize by expiration timeframe
    recently_expired = int((expired_licenses['days_expired'] <= 30).sum())
    expired_1_6_months = int(((expired_licenses['days_expired'] > 30) & 
                         (expired_licenses['days_expired'] <= 180)).sum())
    expired_6_12_months = int(((expired_licenses['days_expired'] > 180) & 
                          (expired_licenses['days_expired'] <= 365)).sum())
    expired_over_year = int((expired_licenses['days_expired'] > 365).sum())
    
    # 6. DETAILED EXPIRED LICENSE LIST
    expired_details = expired_licenses.merge(
        final_df[['license_number', 'first_name', 'last_name', 'primary_specialty']], 
        on='license_number', 
        how='left'
    ).sort_values('expiration_date')
    
    # Clean expired_details to handle NaN values
    expired_details_clean = expired_details.head(20).fillna('').to_dict('records')
    
    # Return summary data for further analysis
    return {
        'total_licenses': int(total_licenses),
        'total_expired': int(total_expired),
        'expired_percentage': float(expired_percentage) if not pd.isna(expired_percentage) else 0.0,
        'state_distribution': {str(k): int(v) for k, v in state_distribution.to_dict().items() if pd.notna(k)},
        'specialty_distribution': {str(k): int(v) for k, v in specialty_distribution.to_dict().items() if pd.notna(k)},
        'timeline_breakdown': {
            'recently_expired': recently_expired,
            'expired_1_6_months': expired_1_6_months,
            'expired_6_12_months': expired_6_12_months,
            'expired_over_year': expired_over_year
        },
        'expired_details': expired_details_clean
    }
