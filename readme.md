# MedEthicsQA: A Question Answering Benchmark for Medical Ethics Evaluation of LLMs

This repo contains datasets and evaluation code for MedEthicsQA.

project/

├── dataset/

│ ├── MedEthicsQA.json # the total 26.1k questions. encoded as "utf-8".

│ └── taxonomy.json # hierarchical taxonomy

├── model/

│ └── model_api.py

├── evaluation_results/

│ └── "model".csv #evaluation result of a model

├── evaluation.py # evaluation script

└── calculate_acc.py 
