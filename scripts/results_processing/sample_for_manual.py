
import numpy as np
import pandas as pd 
import os
import regex as re



def sort_and_add_columns(df):
    # Ensure 'model_name' and 'id' columns are first
    sorted_columns = ['model_name', 'id', "ground_truth"]

    # Separate columns into two groups
    columns_with_f1 = [col for col in df.columns if col.endswith('_f1')]
    columns_without_f1 = [col for col in df.columns if col not in ['model_name', 'id', "ground_truth"] and not col.endswith('_f1')]

    # Sort columns alphabetically
    columns_without_f1_sorted = sorted(columns_without_f1)
    
    # Add columns that don't end with _f1 and their corresponding _manual columns
    for col in columns_without_f1_sorted:
        if col != "rephrasedQuestions":
            sorted_columns.append(col)
            
            # Add columns that end with _f1 and have the same prefix
            prefix = '_'.join(col.split('_')[:2])
            print(prefix)
            for f1_col in columns_with_f1  :
                if f1_col.startswith(prefix):
                    print("exsists")
                    sorted_columns.append(f1_col)
            
            if col not in ['model_name', 'id', "ground_truth"]:
                manual_col = f"{prefix}_manual"
                sorted_columns.append(manual_col)
                df[manual_col] = ""  # Add the _manual column to the DataFrame with NaN values

    return df[sorted_columns + [col for col in df.columns if col not in sorted_columns]]

def ensure_all_columns(datasets):
    # Get the union of all columns across all datasets
    all_columns = sorted(set().union(*[ds.columns for ds in datasets]))
    print(f"Number of original datasets: {len(datasets)}")

    # Ensure each dataset has all columns
    new_datasets = []
    for i, ds in enumerate(datasets):
        for col in all_columns:
            if col not in ds.columns:
                ds[col] = "missing_info"
        new_datasets.append(ds)
    print(f"Number of modified datasets: {len(new_datasets)}")
    return new_datasets

# Function to sample and chunk the datasets
def sample_and_chunk(datasets, sample_size=32, chunk_size=8):
    datasets = ensure_all_columns(datasets)
    chunks = []
    # Sample 10 results from each dataset
    print("DS LENNNN"+ str(len(datasets)))
    samples = [df.sample(n=sample_size, random_state=42).reset_index(drop=True) for df in datasets]
    # samples = [32 samples first_df ,32 samples first_df ]
    # Create 4 tables, each containing 24 results (8 from each model)
    tables = []
    for i in range(1,4):
        # Concatenate the sampled data
        table = pd.concat([samples[j][chunk_size *(i-1):chunk_size*i] for j in range(len(samples))], axis=0).reset_index(drop=True)
        tables.append(table)
    
    return tables


def filter_columns(df):
    # Keep only columns that contain "rephrased", "model_name", or "id"
    filtered_columns = [col for col in df.columns if "rephrased" in col or col in ["model_name", "id", "ground_truth"]]
    return df[filtered_columns]

def fill_manual_columns(tables):
    for table in tables:
        for col in table.columns:
            if col.endswith("manual"):
                table[col] = table[col].fillna("")
    return tables

def extract_answer_content(text):
    pattern = r"'answer_content': '([^']*)'"
    match = re.search(pattern, text)
    return match.group(1) if match else None

def fix_specific_datasets_results(table, keyword='Pad_token', model_names = ['llama', 'gemma','mistral']):
    # Create new consolidated columns
    new_columns = [keyword, f'{keyword}_f1', f'{keyword}_manual']
    table = table.assign(**{col: '' for col in new_columns})

    # Process and consolidate relevant columns
    for col in table.columns:
        if keyword in col:
            target_col = next((nc for nc in new_columns if nc in col), keyword)
            table[target_col] += table[col].astype(str).str.replace('missing_info', '')

    # Keep columns that don't contain the keyword, or contain the keyword but not any model name
    cols_to_keep = [col for col in table.columns if 
                    keyword not in col or 
                    (col in new_columns and not any(model in col.lower() for model in model_names))]

    # Reorder columns, putting new columns at the end
    cols_to_keep = [col for col in cols_to_keep if col not in new_columns] + new_columns
    return table[cols_to_keep] 


PATH_F1 = r"/cs/labs/tomhope/nirm/MusiQue/csv_evaluations/combined_scores.csv"
PATH_PREDICTIONS = r"/cs/labs/tomhope/nirm/MusiQue/csv_evaluations/csv_pred/predictions_of_seam.csv"

if __name__ == "__main__":
    datasets = []
    for WANTED_MODEL_OR_DS in ["Gemma", "Llama", "Mistral"]:
        input_path = f"/cs/labs/tomhope/nirm/MusiQue/csv_evaluations/results_by_model_or_ds/{WANTED_MODEL_OR_DS}_summary.csv"
        if not os.path.exists(input_path):
            print(f"failed on {WANTED_MODEL_OR_DS}")

        df_results = pd.read_csv(input_path)
        df_results['model_name'] = WANTED_MODEL_OR_DS
        df_results['ground_truth'] = df_results['ground_truth'].apply(extract_answer_content)

        df_results =filter_columns(df_results)
        df_results = sort_and_add_columns(df_results)
        datasets.append(df_results)
        print(df_results["model_name"].unique)
        
    sampled_tables = sample_and_chunk(datasets)
    sampled_tables = fill_manual_columns(sampled_tables)

    for i, table in enumerate(sampled_tables):
        table.columns = [col.replace("rephrasedQuestions","")  for col in table.columns]
        table = fix_specific_datasets_results(table, keyword='Pad_token')
        table.to_csv(f'/cs/labs/tomhope/lihish400_/SEAM_PROJECT/scripts/tbl_manual/table_{i+1}.csv', index=False)
        print(f"Table {i+1} saved to table_{i+1}.csv")
        # print(table.head())  # Display the first few rows of each table to verify
        print(table["model_name"].unique())  # Display the first few rows of each table to verify

        