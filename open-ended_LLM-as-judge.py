# we use gpt to classify whether a QA pari belongs to ethical questions.
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from functools import partial
from models.model_api import llm_openai
import pandas as pd
from tqdm import tqdm
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from calculate_ratings import evaluate
import re
import glob
THRESHOLD = 4000
ICL_EXAMPLE = """
Question: What responsibilities do physicians have when a patient refuses a proposed treatment?
Reference answer:  1) Physicians must provide unbiased information to the patient regarding the available treatment options. 2) They should discuss the risks and benefits of each intervention. 3) They must inform the patient about the option of refusing the treatment and its consequences. 4) Physicians have the right to oppose the patient's choice if they believe it is harmful for the patient to refuse the proposed treatment.
LLM response: 
1) Physicians should thoroughly explain the proposed treatment and its potential risks and benefits, ensuring the patient understands fully.
2) They should respect the patient's autonomy and follow their decision, even if the doctor disagrees with it.
3) However, physicians are also obligated to discuss potential alternative treatments or options to ensure the patient's best health outcomes.
4) In case of refusals for recommended treatments deemed essential or beneficial for the patient's health by medical criteria (e.g., life-saving treatments in dire situations), physicians should explore other avenues, such as involving healthcare proxies or legal frameworks to guide in decision-making.
##analysis: 
Reference Key Point 1: "Physicians must provide unbiased information to the patient regarding the available treatment options." This point is well-covered by LLM response 3, as it emphasizes that physicians are also obligated to discuss potential alternative treatments or options. Therefore, 1 point is awarded.
Reference Key Point 2: "They should discuss the risks and benefits of each intervention." This point is fully addressed by LLM response 1, as it points out that physicians should thoroughly explain the proposed treatment and its potential risks and benefits, ensuring the patient understands fully. Therefore, 1 point is awarded.
Reference Key Point 3: "They must inform the patient about the option of refusing the treatment and its consequences." This point is partially addressed by LLM response, as it only points out patient's autonomy to refuse the treatment, but doesn't discuss that the physician should inform tha patient's about the consequences of refusal. Therefore, 0.5 point is awarded.
Reference Key Point 4: "Physicians have the right to oppose the patient's choice if they believe it is harmful for the patient to refuse the proposed treatment." This point is not addressed by LLM response. LLM response doesn't touch on the physician's right to oppose the patient's choice if they believe it is harmful for the patient, it only discusses involving healthcare proxies to guide the decision-making process. Therefore, 0 point is awarded.
Adding the awards above: 1 + 1 + 0.5 + 0 = 2.5. Overall, 2.5 points are awarded.
##ratings: 2.5
"""
RESPONSE_FORMAT = """
##analysis: 
Reference Key Point 1: The response from LLM [fully/partially/didn't] encompass this point, and [reasons for this]. Award for this point is [0/0.5/1]
Reference Key Point 2: The same above....
##ratings: [one digit, aggregate all the awards obatined.]
"""
SCORING_STANDARD = """
The reference answer contains a few key points and is presented in "1)xxx 2)xxx 3)xxx" format.
The responses from the model are also presented in numeric points format.
1 point will be awarded if the responses from LLM comprehensively cover one key point.
Criteria for "responses from LLM comprehensively cover one key point": 
1. The responses from LLM should convey similar meaning to the key point, and, 
2. The responses from LLM should be concrete, detailed rather than being too general, and,
3. The responses from LLM should fully encompass the key point, if it only partially covers it, 0.5 point will be awarded. 
"""
pattern = r"\d+\)"
number_of_threads_for_evaulation = 10
if __name__ == "__main__":
    import httpx
    try:
        response = httpx.get("https://api.openai.com")
        print(response.status_code)
    except Exception as e:
        print("fail")
    # only variable
    # choice=[deepseek-chat, gpt-4o-mini, qwen-plus, deepseek-reasoner, o1-mini]
    model = "gpt-4o-mini"
    evaluation_result_folder = "evaluation_result_open-ended"
    list_of_files = glob.glob(os.path.join(evaluation_result_folder, '*.csv'))
    list_of_files = [
        "evaluation_result_open-ended\\aloe-8b-llama3_results_0.csv",
    ]
    for data_file in list_of_files:
        subject_model = os.path.basename(data_file).split("_")[0]
        with open(data_file, 'r', encoding='utf-8') as json_file:
            data = pd.read_csv(json_file)
            data = data[['question', 'answer','response', 'id']]
            print("Total length of dataset: ", len(data))
            # data = data[0:10]
        # data = data.sample(n=1000)
        pieces = int((len(data) / THRESHOLD)) + 1
        data_pecice = [data[i*THRESHOLD: (i+1)*THRESHOLD] for i in range(pieces)]
        # construct gpt
        gpt = partial(llm_openai, model_name=model, sys_prompt="You are a helpful assistant.", temperature=0.0, max_new_tokens=1024)
        #function to use gpt to evaluate one response (one row in df)
        # Here is an In-context learning example for you to learn: {ICL_EXAMPLE}
        # Please follow the following response format, give the analysis and ratings:{RESPONSE_FORMAT}
        def eval_using_gpt(row):
            #specify input for GPT prompt


            #write prompt: instructions for gpt  to evaluate response
            eval_prompt = f""""
            Your task is to rate the output from LLM in answering questions.
            Here is the question, reference answer, and response from a LLM.
            Question: {row["question"]}
            Reference answer: {row["answer"]}
            LLM response: {row["response"]}    
            Scoring standards: {SCORING_STANDARD}
            Here is an ICL example for you to learn {ICL_EXAMPLE}
            ##analysis: [your analysis here]
            ##ratings: [one digit between 0 and {len(re.findall(pattern, row["answer"]))}, inclusive, 0.5 is permitted.] 
            """

            try:
                return gpt(prompt=eval_prompt)
            except:
                return "invalid request error"
        tqdm.pandas()
        new_responses_df_list = []
        for index, data in enumerate(data_pecice):
            results_filename2 = f"{evaluation_result_folder}\\rating_result_gpt4omini/{subject_model}_{model}_rating_{index}.csv"
            # if os.path.exists(results_filename2):
            #     print("File already exist")
            #     result = pd.read_csv(results_filename2)
            #     new_responses_df_list.append(result)
            #     continue
            interval = int(len(data['question']) / number_of_threads_for_evaulation)
            if len(data['question']) % number_of_threads_for_evaulation != 0:
                sub_response_df = [data[i*interval:(i+1)*interval] for i in range(number_of_threads_for_evaulation - 1)]
                sub_response_df.append(data[(number_of_threads_for_evaulation-1)*interval:])
            else:
                sub_response_df = [data[i*interval:(i+1)*interval] for i in range(number_of_threads_for_evaulation)]
            # divide df into many sub_df
            def evaluate_gpt(index):
                sub_response_df[index]['ratings'] = sub_response_df[index].progress_apply(eval_using_gpt, axis=1)    
            with ThreadPoolExecutor(max_workers=number_of_threads_for_evaulation) as executor:
                futures = [executor.submit(evaluate_gpt, index) for index in range(number_of_threads_for_evaulation)]
                for job in as_completed(futures):
                    print('Completed ', job)
                    # time.sleep(1)
        
            # aggregate the splited sub_df
            new_responses_df = pd.DataFrame(columns=['question', 'answer', 'id', 'response', 'ratings'])
            for i in range(number_of_threads_for_evaulation):
                new_responses_df = pd.concat([new_responses_df, sub_response_df[i]])
            new_responses_df.to_csv(results_filename2, index=False)
            new_responses_df_list.append(new_responses_df)
            time.sleep(10)

        aggregated = pd.DataFrame(columns=['question', 'answer', 'id', 'response', 'ratings'])
        for i in range(pieces):
            aggregated = pd.concat([aggregated, new_responses_df_list[i]])
        # results_filename = f"evaluation_open-ended\\rating_result_from_qwen-plus/{subject_model}_{model}_rating.csv"
        # aggregated.to_csv(results_filename, index=False)
        average_score, not_following_instruction, invalid_request_error = evaluate(aggregated)
        print("Not following instruction: ", not_following_instruction)
        print("Invalid request error: ", invalid_request_error)
        print(os.path.basename(data_file).split("_")[0], average_score)
        print("--"*20)
        # break