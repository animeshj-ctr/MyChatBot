from openai import OpenAI
import time

client = OpenAI(
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama",
)

def llm_response(prompt):
    for i in range(3):
        try:
            res = client.chat.completions.create(
                model="mistral",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                # messages = [
                #     {
                #         "role": "system",
                #         "content": "You are a senior software architect."
                #     },
                #     {
                #         "role": "user",
                #         "content": "Explain microservices"
                #     },
                #     {
                #         "role": "assistant",
                #         "content": "Microservices is an architecture style..."
                #     },
                #     {
                #         "role": "user",
                #         "content": "Give an example"
                #     }
                # ],
                temperature=0,
            )
            return (res.choices[0].message.content)
            break
        except Exception as e:
            print("Retrying...", e)
            time.sleep(2)

prompt = '''
    Classify the following review 
    as having either a positive or
    negative sentiment:

    The banana pudding was really tasty!
'''

response = llm_response(prompt)
print(response)