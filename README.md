# More Documents, Same Length: Isolating the Challenge of Multiple Documents in RAG
This repository contains code and datasets for our paper on the effects of document multiplicity while the context size is fixed in Retrieval-Augmented Generation (RAG) systems.
For detailed methodology, experiments, and analysis, please refer to the full paper : Paper.pdf.

## :bulb: High-Level Conclusions
Our results indicate that including more documents in the input complicates the task in retrieval settings, highlighting the need for systems to balance relevance and diversity to minimize conflicts. Future models could benefit from mechanisms that identify and discard conflicting information while leveraging the benefits of document variety.
<div align="center">
  <img src="/Main_Figure.png" alt="Alt text" width="450">
</div>


## :desktop_computer:  Reproduction Instructions:

### Download the different benchmark datasets
Our custom benchmark datasets include a control set, the original dataset, and variants with replaced distractors for varying document multiplicity. 
Download them here [here](https://drive.google.com/file/d/1z6L0Xl0zhRoOOpwD5WuQI9ukSaEgCraM/view?usp=drive_link) or regenerate using ```bash scripts/create_various_sets.py```
 .



### Prepare the environment

To set up the running environment, run the following command:
```bash
git clone git@github.cs.huji.ac.il:lihish400-1/CATBM.git
cd CATBM
export PYTHONPATH=./
python3.11 -m venv <PATH_TO_VENV>
source <PATH_TO_VENV>/bin/activate
pip install -r requirements.txt
```

### Run predictions
For running in inference on the chosen benchmark dataset you need to define for each benchmark data set a config file under configuration folder [`files/configuration/predict.json`](files/configuration/predict.json).

The `predict.json` file contains the path to the generated benchmark from previous step, the batch size, and the decoding temperature for the LLMs.

We supply two option for running the code with small models (the code run locally), with large model (the code run with Together platform)

To run prediction with the small models, run the following command:
```bash
python scripts/run_model_predictions.py --config <PATH_TO_CONFIG> --model_name <MODEL_NAME>
```

For the large model add ['together_api_key.py'](together_api_key.py) under the root path and define: API_KEY = XXXXX

then run the following command:

```bash
python scripts/run_model_predictions.py --config <PATH_TO_CONFIG> --model_name <MODEL_NAME> --run_together
```

### Evaluate the predictions

To evaluate the predictions, you can use [`scripts/evaluate_dataset.py`](scripts/evaluate_dataset.py) by providing 
the path to the predictions from previous step, and output path where all results will be saved.

```bash
python scripts/evaluate_dataset.py --predictions_dir <OUTPUT_PATH_FROM_PREV_STEP> --output_path <RESULT_OUTPUT> --ds_name MusiQue
```

## :newspaper: Citation

If you use this code or the datasets in your research, please cite:

```
@inproceedings{author2024more,
  title={More Documents, Same Length: Isolating the Challenge of Multiple Documents in RAG},
  author={Author, A. and Author, B.},
  booktitle={Proceedings of ACL 2024},
  year={2024}
}
```


