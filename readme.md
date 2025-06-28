# MedEthicsQA: A Comprehensive Question Answering Benchmark for Medical Ethics Evaluation of LLMs

This repo contains datasets and evaluation code for MedEthicsQA.

# 📁 model/

The setup of evaluation model

# 📄 MedEthicsQA_MCQ.json

multiple choice question answering subset of the MedEthicsQA, encoded as 'utf-8'

# 📄 MedEthicsQA_open.zip

open-ended question subset of MedEthicsQA.

# 📄 calculate_acc.py

The script for calculating accuracy of MCQ

# 📄 calculate_ratings.py

The script for calculating ratings of open-ended questions

# 📄 evaluate_MCQ_api.py

inference file for MCQ

# 📄 evaluate_open-ended_api.py

inference file for open-ended questions

# 📄 open-ended-LLM-as-Judge.py

script for evaluate the responses of open-ended questions from LLM to the ground truth

# 📄 taxonomy.json

The taxonomy of our proposed hierarchical taxonomy.
