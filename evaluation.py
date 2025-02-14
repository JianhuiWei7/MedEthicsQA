# we use gpt to classify whether a QA pari belongs to ethical questions.
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from functools import partial
from model.model_api import llm_openai
import pandas as pd
from tqdm import tqdm
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from calculate_acc import evaluate
THRESHOLD = 4000
number_of_threads_for_evaulation = 40
if __name__ == "__main__":
    # only variable
    # choice=[deepseek-chat, gpt-4o-mini, qwen-plus, deepseek-reasoner, o1-mini]
    model = "gpt-4o-mini"
    data_file = "data\MedEthicsQA.json"
    with open(data_file, 'r', encoding='utf-8') as json_file:
        data = pd.read_json(json_file)
        data = data[['whole_question', 'correct', 'id']]
        # data = data[0:90]
        print("Total length of dataset: ", len(data))
    pieces = int((len(data) / THRESHOLD)) + 1
    data_pecice = [data[i*THRESHOLD: (i+1)*THRESHOLD] for i in range(pieces)]
    # construct gpt
    gpt = partial(llm_openai, model_name=model, sys_prompt="You are a helpful assistant.", temperature=0.0, max_new_tokens=512)
    def eval_using_gpt(row):
        #specify input for GPT prompt
        eval_prompt = f""""
        Here is a question and a few candicate options.
        Question: {row["whole_question"]}    
        Please choose one most appropriate answer from the candicate options to the question
        Strictly response in the following format:
        #answer: one most appropriate answer (one capital letter) among the given options.
        #reason: why do you choose this."""
        try:
            return gpt(prompt=eval_prompt)
        except:
            return "invalid request error"
    tqdm.pandas()
    new_responses_df_list = []
    for index, data in enumerate(data_pecice):
        results_filename2 = f"evaluation_results/{model}_{index}.csv"
        interval = int(len(data['whole_question']) / number_of_threads_for_evaulation)
        if len(data['whole_question']) % number_of_threads_for_evaulation != 0:
            sub_response_df = [data[i*interval:(i+1)*interval] for i in range(number_of_threads_for_evaulation - 1)]
            sub_response_df.append(data[(number_of_threads_for_evaulation-1)*interval:])
        else:
            sub_response_df = [data[i*interval:(i+1)*interval] for i in range(number_of_threads_for_evaulation)]
        # divide df into many sub_df
        def evaluate_gpt(index):
            sub_response_df[index]['response'] = sub_response_df[index].progress_apply(eval_using_gpt, axis=1)    
        with ThreadPoolExecutor(max_workers=number_of_threads_for_evaulation) as executor:
            futures = [executor.submit(evaluate_gpt, index) for index in range(number_of_threads_for_evaulation)]
            for job in as_completed(futures):
                print('Completed ', job)
                # time.sleep(1)
    
        # aggregate the splited sub_df
        new_responses_df = pd.DataFrame(columns=['whole_question', 'correct', 'id', 'response'])
        for i in range(number_of_threads_for_evaulation):
            new_responses_df = pd.concat([new_responses_df, sub_response_df[i]])
        new_responses_df.to_csv(results_filename2, index=False)
        new_responses_df_list.append(new_responses_df)
        time.sleep(10)

    aggregated = pd.DataFrame(columns=['whole_question', 'correct', 'id', 'response'])
    for i in range(pieces):
        aggregated = pd.concat([aggregated, new_responses_df_list[i]])
    results_filename = f"evaluation_results/{model}.csv"
    aggregated.to_csv(results_filename, index=False)
    evaluate(aggregated)