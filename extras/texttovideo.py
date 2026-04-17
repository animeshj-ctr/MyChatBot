from dotenv import load_dotenv
import os
import sys
from openai import OpenAI
import os
from huggingface_hub import InferenceClient

load_dotenv(f'config/{sys.argv[1]}/.env')
OPENAI_SECRET_KEY= os.getenv("OPENAI_API_KEY")
HF_SECRET_KEY= os.getenv("HUGGINGFACEHUB_API_TOKEN")


client = InferenceClient(
    provider="fal-ai",
    api_key=HF_SECRET_KEY,
)

# rint(completion.choices[0].message)
def text_to_video(prompt,output_path):
    video = client.text_to_video(
        prompt,
        model="tencent/HunyuanVideo",
    )
    
    with open(output_path, "wb") as f:
        f.write(video)

    return output_path
    

prompt = '''
Electric car model mg windsor white colour.
It was charging and suddenly blasts happens on driver side and then driver side burns.
Camera showing burnt seat.Then stops.
'''

filename = "output.mp4"
output_path = "genai_outputs/"+filename
video = text_to_video(prompt,output_path)