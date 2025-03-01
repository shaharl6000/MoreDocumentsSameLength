import json
import os
import csv
import pandas as pd
import numpy as np
company_to_model = {"google": "Gemma",
                    "mistralai": "Mistral" , 
                    "meta-llama": "Llama"}
PATH_TO_PRED = r"/cs/labs/tomhope/nirm/MusiQue/eval/" #r"/cs/labs/tomhope/nirm/MusiQue/output_pred/"
PATH_TO_CSV = r"/cs/labs/tomhope/nirm/MusiQue/csv_evaluations/combined_scores.csv"

def find_json_files(directory):
    json_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json') and not file.startswith("metadata"):
                json_files.append(os.path.join(root, file))
    return json_files

def extract_scores_from_json(file_path):
    scores_dict = {}
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            print(f"Reading file: {file_path}")  # Debug: Print file being read
            if 'models' in data:
                for model_name, model_data in data['models'].items():
                    ids = model_data['ids'][0]
                    if 'scores' in model_data and isinstance(model_data['scores'], list) and 'ids' in model_data:
                        for score_dict in model_data['scores']:
                            if 'all_f1' in score_dict:
                                all_f1_scores = score_dict['all_f1']
                                if len(all_f1_scores) == len(ids):
                                    path_parts = file_path.split(os.sep)
                                    eval_index = path_parts.index("eval") + 1
                                    model_name = company_to_model[path_parts[eval_index]]  # The folder after "eval"
                                    dataset_name = path_parts[-2]
                                    column_name = f"{model_name}_{dataset_name}_f1"
                                    for id, score in zip(ids, all_f1_scores):
                                        if id not in scores_dict:
                                            scores_dict[id] = {}
                                        scores_dict[id][column_name] = np.round(score, 4)
                                else:
                                    print(f"Mismatch between length of ids and all_f1_scores in model: {model_name}")
                    else:
                        print(f"Missing 'scores' list or 'ids' in model: {model_name}")  # Debug: Print missing keys
            else:
                print(f"Missing 'models' key in file: {file_path}")  # Debug: Print missing keys
    except Exception as e:
        print(f"Error reading {file_path}: {e}")  # Debug: Print error

    return scores_dict

def merge_scores(existing_scores, new_scores):
    for id, scores in new_scores.items():
        if id not in existing_scores:
            existing_scores[id] = scores
        else:
            existing_scores[id].update(scores)
    return existing_scores

def save_scores_to_csv(all_scores_dict, output_csv):
    try:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, 'w', newline='') as csvfile:  # Open file in write mode to overwrite
            writer = csv.writer(csvfile)
            # Collect all unique file names (columns)
            all_columns = set()
            for scores in all_scores_dict.values():
                all_columns.update(scores.keys())
            # header = ['id'] + list(all_columns)
            header = ['id' if col == '' else col for col in ['id'] + sorted(all_columns)]  # Replace empty strings with 'id'

            writer.writerow(header)  # Writing header

            # Write data rows
            for id, scores in all_scores_dict.items():
                row = [id] + [scores.get(column, '') for column in header[1:]]
                writer.writerow(row)
        print(f"Scores saved to {output_csv}")
    except Exception as e:
        print(f"Error writing to CSV: {e}")

def print_first_10_rows(output_csv):
    try:
        with open(output_csv, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i < 10:
                    print(row)
                else:
                    break
    except Exception as e:
        print(f"Error reading CSV: {e}")

def get_f1_score(PATH_TO_PRED, PATH_TO_CSV):
    json_files = find_json_files(PATH_TO_PRED)
    print("JSON files found:")
    print(json_files)

    # Initialize a dictionary to hold all scores
    all_scores_dict = {}

    # Process each JSON file and collect scores
    for file_path in json_files:
        scores_dict = extract_scores_from_json(file_path)
        all_scores_dict = merge_scores(all_scores_dict, scores_dict)
    for key in all_scores_dict:
        print(all_scores_dict[key])
    # Save all collected scores to the CSV file
    df = pd.DataFrame(all_scores_dict).T
    df.to_csv(PATH_TO_CSV, index=True)
    print_first_10_rows(PATH_TO_CSV)

if __name__ == "__main__":
    get_f1_score(PATH_TO_PRED, PATH_TO_CSV)
