from dotenv import load_dotenv
import os
import sys
from openai import OpenAI
import os
from huggingface_hub import InferenceClient

load_dotenv(f'config/{sys.argv[1]}/.env')
OPENAI_SECRET_KEY= os.getenv("OPENAI_API_KEY")
HF_SECRET_KEY= os.getenv("HUGGINGFACEHUB_API_TOKEN")


client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_SECRET_KEY,
)

# rint(completion.choices[0].message)
def llm_response(prompt):
    response = client.chat.completions.create(
        model="MiniMaxAI/MiniMax-M2.7:novita",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content

prompt = '''
    Classify the following review 
    as having either a positive or
    negative sentiment:

    The banana pudding was not really tasty!
'''

response = llm_response(prompt)
print(response)
