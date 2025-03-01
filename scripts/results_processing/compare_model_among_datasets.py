import json
import os
import re
from glob import glob
import csv
import numpy as np
import pandas as pd
company_to_model = {"google": "Gemma",
                    "mistralai": "Mistral" , 
                    "meta-llama": "Llama"}
MODELS_NAME_MAPPING = {
    "Meta-Llama-3-8B-Instruct": "Llama3-8B",
    "Meta-Llama-3-70B-Instruct": "Llama3-70B",
    "gemma-1.1-7b-it": "Gemma1.1-7B",
    "gemma-1.1-2b-it": "Gemma1.1-2B",
    "Mistral-7B-Instruct-v0.2": "Mistral-7B",
    "Mixtral-8x7B-Instruct-v0.1": "Mixtral-8x7B",
    "Mixtral-8x22B-Instruct-v0.1": "Mixtral-8x22B",
}

class Eval:
    def __init__(self, id_key, predictions_dir):
        self.id_key = id_key
        self.predictions_dir = predictions_dir

    def _evaluate(self, data, model_name, sample_index):
        raise NotImplementedError("This method needs to be implemented by subclasses")

    def _evaluate_new(self, data, model_name, sample_index):
        raise NotImplementedError("This method needs to be implemented by subclasses")

    def eval_all_in_dir(self):
        sample_lengths = {}
        all_results = {}
        
        for f_name in glob(f'{self.predictions_dir}/**/*.json', recursive=True):
            if "results" in f_name:
                continue
            model_name = os.path.splitext(os.path.basename(f_name))[0]
            source_dataset = os.path.normpath(f_name).split(os.sep)[-4]
            print("Evaluating model:", model_name, "source dataset:", source_dataset)
            
            with open(f_name, 'rt') as f:
                predictions = json.load(f)

            current_ids = []
            for pred in predictions:
                id_sample = str(pred[self.id_key])
                current_ids.append(id_sample)
                if id_sample not in sample_lengths:
                    length = len(pred["final_msgs"][0]['content'])
                    sample_lengths[id_sample] = length

            results = self._evaluate_new(predictions, model_name, source_dataset)
            if model_name not in all_results:
                all_results[model_name] = {}
            for id_sample, answer in results.items():
                if id_sample not in all_results[model_name]:
                    all_results[model_name][id_sample] = {}
                all_results[model_name][id_sample][source_dataset] = answer

        return all_results, sample_lengths

    @staticmethod
    def get_only_response(prediction):
        pred = prediction["prediction"]
        if '[/INST]' in pred:
            pred = pred[pred.find("[/INST]")+len("[/INST]"):].strip()
        elif "So, the answer is:" in pred:
            pred = pred[pred.rfind("So, the answer is:") + len("So, the answer is:"):].strip()
            if pred == "":
                # print(prediction["prediction"])
                pred = "No answer"
        elif "Aspect-based summary:" in pred:
            pred = pred[pred.rfind("Aspect-based summary:") + len("Aspect-based summary:"):].strip()
        else:
            pred = pred[pred.rfind("*Answer*:") + len("*Answer*:"):].strip()
        return pred

class QA(Eval):

    def __init__(self, id_key, predictions_dir):
        self.correlations = []
        super().__init__(id_key, predictions_dir)

    def _evaluate(self, predictions, model_name, sample_index):
        follow_format = []
        for sample in predictions:
            gt = sample["target"]
            pred = self.postprocess(sample)
            follow_format.append(type(pred) is dict)
        self.correlations.append([np.mean(follow_format), np.mean([])])
        metrics = {"all_f1": []}
        return metrics
    
    def _evaluate_new(self, predictions, model_name, source_dataset):
        id_to_answers = {}
        for sample in predictions:
            gt = sample["target"]
            id_sample = sample["id"]
            pred = self.postprocess(sample)
            pred = extract_answer_content(pred)
            
            if id_sample not in id_to_answers:
                id_to_answers[id_sample] = {}
            id_to_answers[id_sample] = pred
        return id_to_answers

    def parse(self, ground_truth_answer, predicted_answer):
        gt = self._extract_answer_from_dict(ground_truth_answer)
        if type(predicted_answer) is dict:
            pred = self._extract_answer_from_dict(predicted_answer)
        else:
            pred = predicted_answer
        return gt, pred
    
 
    @staticmethod
    def _extract_answer_from_dict(answer_dict):
        if answer_dict != "**":
            if "is_answerable" in answer_dict:
                if answer_dict["is_answerable"]:
                    answer = answer_dict["answer_content"]
                else:
                    answer = answer_dict["is_answerable"]
                return answer
        else:
            return "Not answerable"

    def postprocess(self, pred):
        only_response = self.get_only_response(pred)
        answer_dict = re.search(r'\{.*\}', only_response)
        if answer_dict is None:
            return only_response

        str_dict = answer_dict.group(0)
        str_dict = str_dict.replace("'s", "\\'s").replace("'t", "\\'t").replace("s' ", "s\\' ")
        str_dict = str_dict.replace("\\\\_", "_").replace("\\_", "_")
        try:
            answer = eval(str_dict)
        except Exception as e:
            try:
                str_dict = str_dict.replace("}", "'}")
                answer = eval(str_dict)
            except Exception as e:
                answer = only_response
        return answer

def find_json_files(directory , model_name ):
    json_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if "results" not in file and file.endswith('.json') and not file.startswith("metadata"):
                # if model_name in file:
                json_files.append(os.path.join(root, file))
    print("Found JSON files:")
    print(json_files)
    return json_files


def merge_csv_files(file1, file2, model_name, output_dir):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    merged_df = pd.merge(df1, df2, left_on='id', right_on='ID', how='left')
    output_file = os.path.join(output_dir, f"{model_name}_full_results.csv")
    merged_df.to_csv(output_file, index=False)
    print(f"Merged file saved to {output_file}")

def extract_answer_content(result):
    pattern = r"['\"]?answer_content['\"]?\s*:\s*['\"]?([^'\"]*)['\"]?"
    if type(result) is not str:
        result = str(result)

    match = re.search(pattern, result)
    if match:
        answer_content = match.group(1)
        return answer_content
    else:
        return result 

def add_ground_truth(id_to_answers, json_files):
    with open(json_files[0], 'rt') as f:
        predictions = json.load(f)
        for sample in predictions:
            # print(sample)
            gt = sample["target"]
            id_sample = sample["id"]
            if id_sample not in id_to_answers:
                id_to_answers[id_sample] = {}
            id_to_answers[id_sample]["ground_truth"] = gt
            
    return id_to_answers

def fill_in_model_answers(file_path, model_name, source_dataset ,qa_evaluator, id_to_answers):            
    print("Evaluating model:", model_name, "source dataset:", source_dataset)
    with open(file_path, 'rt') as f:
        predictions = json.load(f)

    column_name = f"{model_name}_{source_dataset}"
    model_answers = qa_evaluator._evaluate_new(predictions, model_name, source_dataset)
    for key, value in model_answers.items():
        if key not in id_to_answers:
            id_to_answers[key] = {}
        id_to_answers[key][column_name] = value
    return id_to_answers

def dict_to_csv(data, csv_output_path ,csv_filename):
    # Collect all model names
    models = set()
    for scores in data.values():
        models.update(scores.keys())

    models = sorted(models)
    
    # Ensure the directory exists
    output_csv = os.path.join(csv_output_path, csv_filename)
    os.makedirs(csv_output_path, exist_ok=True)

    # Write the dictionary data to the CSV file
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id'] + models
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for id, scores in data.items():
            row = {'id': id}
            for model in models:
                row[model] = scores.get(model, '')  # Use an empty string if the model score is missing
            writer.writerow(row)

    print(f"Data has been written to {output_csv}")
    data = pd.read_csv(output_csv)
    data = data.astype(str)
    average_lengths = data.applymap(len).mean()
    print("Average row length in each column:")
    print(average_lengths)



def process_model_results(input_directory, model_name, output_directory):
    id_to_answers = {}
    qa_evaluator = QA("id", input_directory)
    json_files = find_json_files(input_directory, model_name)
    id_to_answers = add_ground_truth(id_to_answers, json_files)

    for file_path in json_files:
        path_parts = file_path.split(os.sep)
        dataset_index = path_parts.index("output_pred") + 1
        dataset_name = path_parts[dataset_index]
        model_name = company_to_model[path_parts[-2]]

        id_to_answers = fill_in_model_answers(file_path,
                                               model_name,
                                                 dataset_name ,
                                                 qa_evaluator,
                                                   id_to_answers)
        
                                        
        
        # Print or save the results as needed
        print("\nFinal id_to_answers:")
        dict_to_csv(id_to_answers, output_directory, f"predictions_of_seam.csv")
                                                                           
        # ouput_json_path = f"{model_name}_results_across_datasets.json"
        # output_file = os.path.join(input_directory, ouput_json_path)
        # with open(output_file, 'wt') as f:
        # json.dump(id_to_answers, f)
                                                                       
                                                                          
    # Print or save the results as needed
    print("\nFinal id_to_answers:")
    dict_to_csv(id_to_answers, output_directory, f"all_predictions_of_seam.csv")

    # print(id_to_answers)

    # ouput_json_path = f"{model_name}_results_across_datasets.json"
    # output_file = os.path.join(input_directory, ouput_json_path)
    # with open(output_file, 'wt') as f:
    #     json.dump(id_to_answers, f)


if __name__ == "__main__":
    # input_directory = r"C:\Users\LIHI\Downloads\output_pred-20240723T144539Z-001\output_pred"
    # model_name = "Mistral-7B-Instruct"
   
    input_directory = r"/cs/labs/tomhope/nirm/MusiQue/output_pred/"
    model_name = "not_in_use"
    output_directory = r"/cs/labs/tomhope/nirm/MusiQue/csv_evaluations/csv_pred/"
    process_model_results(input_directory, model_name,output_directory)

    # merge_csv_files("/cs/labs/tomhope/lihish400_/SEAM_PROJECT/scripts/gemma-1.1-2b.csv",
    #                  "/cs/labs/tomhope/lihish400_/SEAM_PROJECT/scripts/gamma_baseline.csv",
    #                    "gamma",
    #                      "/cs/labs/tomhope/lihish400_/SEAM_PROJECT/scripts")