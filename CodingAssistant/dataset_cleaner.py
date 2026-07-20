import os
import sys
import pandas as pd

def clean_tabular_dataset(input_file: str, output_file: str = None) -> pd.DataFrame:
    """
    Cleans tabular datasets (CSV / Excel / Parquet / JSON).
    - Removes duplicate rows
    - Removes completely empty rows/columns
    - Trims whitespace from text columns
    """
    if not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
        return None

    print(f"📊 Loading dataset: {input_file}")
    
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    elif input_file.endswith('.json') or input_file.endswith('.jsonl'):
        df = pd.read_json(input_file, lines=input_file.endswith('.jsonl'))
    elif input_file.endswith('.xlsx') or input_file.endswith('.xls'):
        df = pd.read_excel(input_file)
    else:
        print("❌ Unsupported format. Supported: CSV, JSON, JSONL, XLSX")
        return None

    initial_rows = len(df)
    print(f"🔹 Initial rows: {initial_rows}")

    # 1. Remove duplicate rows
    df = df.drop_duplicates()
    dedup_rows = len(df)
    print(f"✂️ Removed {initial_rows - dedup_rows} duplicate rows.")

    # 2. Drop completely empty rows and columns
    df = df.dropna(how='all')
    df = df.dropna(how='all', axis=1)

    # 3. Strip leading/trailing whitespaces in string fields
    str_cols = df.select_dtypes(include=['object', 'string']).columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()

    print(f"✅ Cleaned rows remaining: {len(df)}")

    if not output_file:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_cleaned{ext}"

    if output_file.endswith('.csv'):
        df.to_csv(output_file, index=False)
    elif output_file.endswith('.json'):
        df.to_json(output_file, orient='records', indent=2)

    print(f"💾 Cleaned dataset saved to: {output_file}")
    return df

def remove_corrupt_or_empty_files(folder_path: str):
    """
    Scans a folder and deletes empty (0-byte) or corrupted files.
    """
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return

    removed_count = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            # Remove 0-byte empty files
            if os.path.getsize(file_path) == 0:
                os.remove(file_path)
                print(f"🗑️ Removed empty file: {file_path}")
                removed_count += 1

    print(f"🧹 Cleaned up {removed_count} corrupt/empty files in '{folder_path}'.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
        if os.path.isdir(target):
            remove_corrupt_or_empty_files(target)
        else:
            clean_tabular_dataset(target)
    else:
        print("Usage:")
        print("  python dataset_cleaner.py <dataset_file.csv|json>")
        print("  python dataset_cleaner.py <folder_path>")
