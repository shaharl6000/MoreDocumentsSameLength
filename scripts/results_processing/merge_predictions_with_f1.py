import os
import csv
import pandas as pd

PATH_F1 = r"/cs/labs/tomhope/nirm/MusiQue/csv_evaluations/combined_scores.csv"
PATH_PRED = r"/cs/labs/tomhope/nirm/MusiQue/csv_evaluations/csv_pred/predictions_of_seam.csv"
OUTPUT_PATH = "/cs/labs/tomhope/nirm/MusiQue/csv_evaluations/results_by_model_or_ds/"
# MODEL_TO_DS = "models_across_datasets"
# DS_TO_MODEL = "dataset_across_models"

WANTED_MODEL_OR_DS = "Gemma"
MODEL_TO_DS = True

if __name__ == "__main__":

    df_f1 = pd.read_csv(PATH_F1)
    df_results = pd.read_csv(PATH_PRED)
    if MODEL_TO_DS:
        selection_idx = 0
    else:
        selection_idx = 1

    print(df_f1.columns)
    print(df_results.columns)
    selected_columns_f1 = ['Unnamed: 0']+[col for col in df_f1.columns if col.split("_")[selection_idx] == WANTED_MODEL_OR_DS]
    selected_columns_results =["id"] + ["ground_truth"]+ [col for col in df_results.columns if col.split("_")[selection_idx] == WANTED_MODEL_OR_DS]

    df_f1= df_f1[selected_columns_f1]
    df_results = df_results[selected_columns_results]
    
    merged_df = pd.merge(df_results, df_f1, left_on='id', right_on='Unnamed: 0', how='left')
    
    not_to_sort_cols = ["id","ground_truth"]
    columns_to_sort = [col for col in merged_df.columns if col not in not_to_sort_cols]
    sorted_columns = sorted(columns_to_sort, reverse=True)
    final_columns = not_to_sort_cols + sorted_columns
    
    output_file = os.path.join(OUTPUT_PATH, f"{WANTED_MODEL_OR_DS}_summary.csv")
    print(f"Merged file saved to {output_file}")
    merged_df.reindex(columns=final_columns)

    merged_df.columns = merged_df.columns.str.replace(f"_{WANTED_MODEL_OR_DS}", f"")
    merged_df.columns = merged_df.columns.str.replace(f"{WANTED_MODEL_OR_DS}_", f"")

    merged_df.to_csv(output_file, index=False)

    summary_file = os.path.join(OUTPUT_PATH, f"{WANTED_MODEL_OR_DS}_summary_sampled.csv")
    merged_df.sample(100, replace=False, random_state=0).to_csv(summary_file, index=False)



# from compare_model_among_datasets import process_model_results
# from extract_f1_from_predictions import get_f1_score
# path_model_to_ds_results = f"/cs/labs/tomhope/nirm/MusiQue/csv_evals/models_2_datasets/"
# path_dataset_results = f"/cs/labs/tomhope/nirm/MusiQue/csv_evals/datasets_2_models/"
# path_models_to_ds_f1 = f"/cs/labs/tomhope/nirm/MusiQue/csv_evals/models_2_datasets_f1/"
# path_to_models_f1 =  f"/cs/labs/tomhope/nirm/MusiQue/csv_evals/datasets_2_models_f1/"
# general_input_path = f"/cs/labs/tomhope/nirm/MusiQue/"
# models = ['model1', 'model2', 'model3']
# datasets = ['dataset1', 'dataset2', 'dataset3']



# def find_csv_files(directory):
#     json_files = []
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             if file.endswith('.csv'):
#                 json_files.append(os.path.join(root, file))
#     return json_files

# def process_model_dataset_combinations(models, datasets , csv_files):
#         f1_model = find_csv_files(path_to_models_f1)
#         results_model = find_csv_files(path_model_results)

# def process_dataset_model_combinations(models, datasets , csv_files):
#         f1_model = find_csv_files(path_to_dataset_f1)
#         results_dataset = find_csv_files(path_model_results)

# def process_full_csv_for_given_folder(model_name , path):
#     process_model_results(input_directory = general_input_path , 
#                             model_name= model_name, 
#                             output_directory = path_model_to_ds_results)

#     get_f1_score(PATH_TO_EVALS =general_input_path ,
#                  PATH_TO_CSV =path_models_to_ds_f1 ,
#                   name_of_csv=f"{model_name}_f1_among_ds")

#     output_path_models =  os.path.join(path_model_to_ds_results, f"{model_name}_results_across_datasets.json")
#     f1_path =  os.path.join(path_models_to_ds_f1, f"{model_name}_f1_among_ds.csv")
#     print("files found")

# if __name__ == "__main__":
#     process_full_csv_for_given_folder("Mistral-7B-Instruct" , general_input_path)
#     # for model in models:
#     #     model_on_all_datasets()
            
#     # for dataset in datasets:
#     #     dataset_on_all_models()
