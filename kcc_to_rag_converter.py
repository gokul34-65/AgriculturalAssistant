"""
Simple Script to Convert KCC CSV Data to RAG Format
Works with Kisan Call Centre dataset from data.gov.in or KCC-CHAKSHU portal
"""

import pandas as pd
import os
import sys

def convert_kcc_to_rag(csv_path, output_dir="knowledge_base"):
    """
    Convert Kisan Call Centre CSV to RAG-ready text files
    
    Args:
        csv_path: Path to downloaded KCC CSV file
        output_dir: Directory to save converted files
    """
    
    print("="*70)
    print("  Kisan Call Centre to RAG Converter")
    print("="*70)
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"❌ Error: File not found at {csv_path}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read CSV
    print(f"\n📂 Reading CSV file: {csv_path}")
    try:
        df = pd.read_csv(csv_path, encoding='utf-8', low_memory=False)
        print(f"✓ Loaded {len(df)} records")
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        print("💡 Try: df = pd.read_csv('{csv_path}', encoding='latin1')")
        return False
    
    # Display columns found
    print(f"\n📊 Columns found in CSV:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i}. {col}")
    
    # Try to identify column names (they vary across datasets)
    question_col = None
    answer_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'query' in col_lower or 'question' in col_lower:
            question_col = col
        if 'answer' in col_lower or 'kccans' in col_lower or 'response' in col_lower:
            answer_col = col
    
    if not question_col or not answer_col:
        print("\n⚠️  Could not auto-detect question/answer columns")
        print("Available columns:", list(df.columns))
        question_col = input("Enter QUESTION column name: ").strip()
        answer_col = input("Enter ANSWER column name: ").strip()
    
    print(f"\n✓ Using columns:")
    print(f"   Question: {question_col}")
    print(f"   Answer: {answer_col}")
    
    # Filter for Tamil Nadu if state column exists
    if 'StateName' in df.columns or 'State' in df.columns:
        state_col = 'StateName' if 'StateName' in df.columns else 'State'
        tn_filter = df[state_col].str.contains('Tamil Nadu', na=False, case=False)
        df_tn = df[tn_filter]
        
        print(f"\n🔍 Filtering for Tamil Nadu:")
        print(f"   Total records: {len(df)}")
        print(f"   Tamil Nadu records: {len(df_tn)}")
        
        if len(df_tn) > 0:
            use_tn = input("\nUse only Tamil Nadu data? (y/n): ").strip().lower()
            if use_tn == 'y':
                df = df_tn
    
    # Convert to RAG format
    print(f"\n🔄 Converting to RAG format...")
    output_file = os.path.join(output_dir, "kisan_call_centre_qa.txt")
    
    count = 0
    skipped = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, row in df.iterrows():
            try:
                question = str(row.get(question_col, '')).strip()
                answer = str(row.get(answer_col, '')).strip()
                
                # Skip if empty or invalid
                if not question or not answer or question == 'nan' or answer == 'nan':
                    skipped += 1
                    continue
                
                if len(question) < 10 or len(answer) < 10:
                    skipped += 1
                    continue
                
                # Write Q&A pair
                f.write(f"QUESTION:\n{question}\n\n")
                f.write(f"ANSWER:\n{answer}\n\n")
                
                # Add metadata if available
                metadata_written = False
                metadata_fields = {
                    'Crop': ['Crop'],
                    'District': ['DistrictName', 'District'],
                    'State': ['StateName', 'State'],
                    'Query Type': ['QueryType', 'Type'],
                    'Category': ['Category'],
                    'Season': ['Season'],
                    'Sector': ['Sector']
                }
                
                for label, possible_cols in metadata_fields.items():
                    for col in possible_cols:
                        if col in df.columns and pd.notna(row.get(col)):
                            if not metadata_written:
                                f.write("METADATA:\n")
                                metadata_written = True
                            f.write(f"  - {label}: {row.get(col)}\n")
                            break
                
                if metadata_written:
                    f.write("\n")
                
                f.write("="*80 + "\n\n")
                
                count += 1
                
                if count % 100 == 0:
                    print(f"   Processed {count} Q&A pairs...")
                    
            except Exception as e:
                skipped += 1
                continue
    
    print(f"\n✅ Conversion Complete!")
    print(f"   ✓ Successfully converted: {count} Q&A pairs")
    print(f"   ⚠ Skipped (invalid): {skipped} records")
    print(f"   📄 Output file: {output_file}")
    print(f"   💾 File size: {os.path.getsize(output_file) / 1024:.2f} KB")
    
    # Show sample
    print(f"\n📋 Sample from converted file:")
    print("-"*70)
    with open(output_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()[:30]  # First 30 lines
        print(''.join(lines))
    print("-"*70)
    
    return True

def show_usage():
    """Display usage instructions"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║         KCC Data to RAG Converter - Usage Instructions          ║
╚══════════════════════════════════════════════════════════════════╝

STEP 1: Download KCC Dataset
   Visit: https://kcc-chakshu.icar-web.com/6_data_extract.php
   - Select State: Tamil Nadu
   - Select Year: 2023, 2024
   - Click Download CSV
   - Save file (e.g., kcc_tamil_nadu.csv)

STEP 2: Run This Script
   python kcc_to_rag_converter.py <path_to_csv>
   
   Example:
   python kcc_to_rag_converter.py Downloads/kcc_tamil_nadu.csv

STEP 3: Build RAG Index
   python rag_cli_ollama.py --mode build --docs-dir ./knowledge_base

STEP 4: Start Querying
   python rag_cli_ollama.py --mode query --model gemma3:4b

═══════════════════════════════════════════════════════════════════

Alternative Data Sources:
1. data.gov.in: https://www.data.gov.in/dataset-group-name/kisan-call-centre
2. Kaggle: https://www.kaggle.com/datasets/daskoushik/farmers-call-query-data-qa
3. HuggingFace: https://huggingface.co/datasets/KisanVaani/agriculture-qa-english-only

═══════════════════════════════════════════════════════════════════
    """)

def main():
    if len(sys.argv) < 2:
        show_usage()
        
        # Interactive mode
        print("\n🔄 INTERACTIVE MODE")
        csv_path = input("\nEnter path to CSV file: ").strip()
        
        if not csv_path:
            print("❌ No file provided. Exiting.")
            return
    else:
        csv_path = sys.argv[1]
    
    # Convert
    success = convert_kcc_to_rag(csv_path)
    
    if success:
        print("\n" + "="*70)
        print("  🎉 SUCCESS! Your data is ready for RAG!")
        print("="*70)
        print("\nNext Steps:")
        print("1. Check the file: knowledge_base/kisan_call_centre_qa.txt")
        print("2. Build RAG index:")
        print("   python rag_cli_ollama.py --mode build --docs-dir ./knowledge_base")
        print("3. Test your system:")
        print("   python rag_cli_ollama.py --mode query --model gemma3:4b")
        print("="*70)
    else:
        print("\n❌ Conversion failed. Please check errors above.")

if __name__ == "__main__":
    main()
