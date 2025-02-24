import pandas as pd
import re
from pathlib import Path

def clean_date(date_str):
    """Robust date parser with multiple format support"""
    formats = [
        '%d-%m-%Y', '%m/%d/%Y', '%Y.%m.%d',
        '%d-%b-%Y', '%b %d, %Y', '%Y%m%d'
    ]
    
    for fmt in formats:
        try:
            return pd.to_datetime(date_str, format=fmt, exact=True)
        except:
            continue
    return pd.NaT

def preprocess_gold_data(input_path, output_dir='data/processed'):
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load raw data
    raw_df = pd.read_csv(input_path)
    
    # Standardize column names
    raw_df.columns = (
        raw_df.columns
        .str.strip()
        .str.upper()
        .str.replace('[^A-Z0-9]', '_', regex=True)
    )
    
    # Handle actual column names
    column_mapping = {
        'UNITED_STATES_USD_': 'USD_PRICE',
        'UNITED_STATES_USD': 'USD_PRICE',
        'PRICE_USD_': 'USD_PRICE'
    }
    raw_df = raw_df.rename(columns=column_mapping).drop(columns=['SOUTH_AFRICA_ZAR_', 'EU_'], errors='ignore')
    
    if 'USD_PRICE' not in raw_df.columns:
        available = list(raw_df.columns)
        raise KeyError(
            f"USD price column not found. Available columns: {available}\n"
            "Please rename your price column to 'USD_Price' in the raw data file."
        )
    
    # Verify required columns
    required_cols = {'DATE', 'USD_PRICE'}
    missing = required_cols - set(raw_df.columns)
    if missing:
        raise KeyError(f"Missing columns in raw data: {missing}")
    
    # Clean dates
    raw_df['Clean_Date'] = raw_df['DATE'].apply(clean_date)
    invalid_df = raw_df[raw_df['Clean_Date'].isna()]
    valid_df = raw_df.dropna(subset=['Clean_Date'])
    
    # Save invalid entries
    invalid_path = Path(output_dir) / 'invalid_entries.csv'
    invalid_df.to_csv(invalid_path, index=False)
    
    # Process valid data
    valid_df['DATE'] = valid_df['Clean_Date']
    valid_df = valid_df.drop(columns=['Clean_Date'])
    
    # Date validation
    date_validation = (
        valid_df['DATE'].dt.year.between(1979, 2025) &
        valid_df['DATE'].dt.month.between(1, 12) &
        valid_df['DATE'].dt.day.between(1, 31)
    )
    invalid_dates = valid_df[~date_validation]
    valid_df = valid_df[date_validation]
    
    # Save invalid dates
    invalid_dates_path = Path(output_dir) / 'invalid_dates.csv'
    invalid_dates.to_csv(invalid_dates_path, index=False)
    
    # Feature engineering
    processed_df = valid_df.sort_values('DATE').set_index('DATE')
    processed_df['MA_7'] = processed_df['USD_PRICE'].rolling(7).mean()
    processed_df['Volatility_30'] = processed_df['USD_PRICE'].pct_change().rolling(30).std()
    
    # Save processed data
    version = pd.Timestamp.now().strftime('%Y%m%d')
    output_path = Path(output_dir) / f'gold_processed_{version}.csv'
    processed_df.to_csv(output_path)
    
    return {
        'processed_path': str(output_path),
        'invalid_count': len(invalid_df),
        'invalid_dates_count': len(invalid_dates),
        'processed_count': len(processed_df)
    }

if __name__ == '__main__':
    result = preprocess_gold_data('data/1979-2021.csv')
    print(f"Processed {result['processed_count']} valid records")
    print(f"Found {result['invalid_count']} invalid entries (saved to data/processed/invalid_entries.csv)")
    print(f"Found {result['invalid_dates_count']} invalid dates (saved to data/processed/invalid_dates.csv)")
    print(f"Clean data saved to: {result['processed_path']}")
