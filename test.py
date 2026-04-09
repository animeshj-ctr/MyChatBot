from dotenv import load_dotenv
import os
import sys

load_dotenv(f'config/{sys.argv[1]}/.env')
OPENAI_SECRET_KEY= os.getenv("OPENAI_API_KEY")
HF_SECRET_KEY= os.getenv("HUGGINGFACEHUB_API_TOKEN")

print(OPENAI_SECRET_KEY)
print(HF_SECRET_KEY)