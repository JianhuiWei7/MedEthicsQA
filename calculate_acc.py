import pandas as pd
import re
import json
import pickle
import glob
import os
def evaluate(result, check=False):
    # categories2accuracy = {}
    pattern_1 = r"#[aA]nswer: [a-zA-Z]"
    pattern_2 = r"#[a-zA-Z]: "
    question_list = result['whole_question']
    response_list = result['response']
    correct_list = result['correct']
    id_list = result['id']
    # category_list = result['category']
    invalid_request_list = []
    Total = 0
    correct = 0
    wrong = 0
    wrong_ids = []
    wrong_choices = []
    not_following_instruction = 0
    invalid_request = 0
    for index in range(len(correct_list)):
        # category = category_list[index]
        # if category not in categories2accuracy.keys():
        #     categories2accuracy[category] = [0,0]
        try:
            match = re.findall(pattern=pattern_1, string=response_list[index])
            if len(match) == 1:
                # categories2accuracy[category][1] += 1
                Total += 1
                correct_list[index]
                if correct_list[index] == match[0][-1].upper():
                    correct += 1
                    # categories2accuracy[category][0] += 1
                    if check:
                        print("Question: ", question_list[index])
                        print("Response: ", response_list[index])
                        print("Correct: ", correct_list[index])
                        print("-"*40)
                else:
                    wrong += 1
                    if check:
                        print("Question: ", question_list[index])
                        print("Response: ", response_list[index])
                        print("Correct: ", correct_list[index])
                        print("-"*40)
                    wrong_ids.append(id_list[index])
                    wrong_choices.append(response_list[index])
            else:
                match = re.findall(pattern=pattern_2, string=response_list[index])
                if len(match) == 1:
                    # categories2accuracy[category][1] += 1
                    Total += 1
                    if correct_list[index] == match[0][1].upper():
                        # categories2accuracy[category][0] += 1
                        correct += 1
                        if check:
                            print("Question: ", question_list[index])
                            print("Response: ", response_list[index])
                            print("Correct: ", correct_list[index])
                            print("-"*40)
                    else:
                        wrong += 1
                        if check:
                            print("Question: ", question_list[index])
                            print("Response: ", response_list[index])
                            print("Correct: ", correct_list[index])
                            print("-"*40)
                        wrong_ids.append(id_list[index])
                        wrong_choices.append(response_list[index])
                else:
                    if response_list[index] == "invalid request error":
                        invalid = {}
                        invalid['whole_question'] = question_list[index]
                        invalid['id'] = str(id_list[index])
                        invalid['correct'] = correct_list[index]
                        invalid_request_list.append(invalid)
                        invalid_request += 1
                    else:
                        not_following_instruction += 1
                
        except:
            print(index)
            continue
    print("Invalid request error: ", invalid_request)
    print("Not following instruction: ", not_following_instruction)
    print("Accuracy: ", (correct)/(Total), "||", correct, "to", Total)
    return wrong_ids, wrong_choices, round(correct/Total, 4)

if __name__ == "__main__":
    results_folder = "evaluation_results"
    list_of_files = glob.glob(os.path.join(results_folder, '*.csv'))
    for file_path in list_of_files:
        result = pd.read_csv(file_path)
        wrong_ids,_, avg_acc = evaluate(result, False)
