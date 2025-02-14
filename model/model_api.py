import sys
import os
path_project_folder = os.path.dirname(os.path.dirname(__file__)) #get path of file's directory two levels up (get path of project folder)
sys.path.append(path_project_folder) #add project folder to path to sys.path to be able to load modules
from openai import OpenAI
import requests

OPENAI_MODEL = ['gpt-4o-mini', "gpt-4o", "o1-mini"]
DEEPSEEK_MODEL = ['deepseek-chat', "deepseek-reasoner"]
QWEN_MODEL = ['qwen-plus','deepseek-v3']
DEEPSEEK_MODEL_SUPPLEMENT = ["deepseek-ai/DeepSeek-V3"]
def llm_openai(prompt, model_name, sys_prompt, temperature, max_new_tokens):
    if model_name in OPENAI_MODEL:
        client = OpenAI(api_key="[to be added]")
        if model_name == "o1-mini":
            response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        else:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_new_tokens,
            )
        response_clean = response.choices[0].message.content
    elif model_name in DEEPSEEK_MODEL_SUPPLEMENT:
        url = "https://api.siliconflow.cn/v1/chat/completions"
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "max_tokens": 512,
            "stop": ["null"],
            "temperature": 1.0,
            "top_p": 0.7,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "n": 1,
            "response_format": {"type": "text"},
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "description": "<string>",
                        "name": "<string>",
                        "parameters": {},
                        "strict": False
                    }
                }
            ]
        }
        headers = {
            "Authorization": "[to be added]", 
            "Content-Type": "application/json"
        }
        response = requests.request("POST", url, json=payload, headers=headers)
        response_clean = response.text
    elif model_name in DEEPSEEK_MODEL:
        client = OpenAI(api_key="[to be added]",  base_url="https://api.deepseek.com/v1")
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_new_tokens,
            stream=False,
            temperature=1.0
        )
        response_clean = response.choices[0].message.content
    elif model_name in QWEN_MODEL:
        client = OpenAI(
            api_key="[to be added]", 
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {'role': 'system', 'content': sys_prompt},
                {'role': 'user', 'content': prompt}
            ],
            max_tokens=max_new_tokens
            )
        response_clean = response.choices[0].message.content
    else:
        print("Unsupported model")
    
    return response_clean

