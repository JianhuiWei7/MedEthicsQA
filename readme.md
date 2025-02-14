# MedEthicsQA: A Question Answering Benchmark for Medical Ethics Evaluation of LLMs
This repo contains datasets and evaluation code for MedEthicsQA.

Unzip the "MedEthicsQA.zip" to get a full size of "MedEthicsQA" and move "MedEthicsQA.json" to "dataset" folder

project/

├── data/

│ ├── MedEthicsQA.json           # the total 26.1k questions. encoded as "utf-8".

│ ├── MedEthicsQA_sample.json    # small samples of questions. encoded as "utf-8". For illustration purpose.

│ └── taxonomy.json              # hierarchical taxonomy

├── model/

│ └── model_api.py

├── evaluation_results/

│ └── "model".csv                 #evaluation result of a model

├── evaluation.py                 # evaluation script

└── calculate_acc.py
