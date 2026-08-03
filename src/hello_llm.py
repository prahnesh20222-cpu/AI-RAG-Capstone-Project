import os, sys
from dotenv import load_dotenv
from openai import OpenAI

# Load OPENAI_API_KEY from the environment — never hard-code keys
load_dotenv() #loads the content of a file named .env and setsup session env variables.

#Construct a new synchronous OpenAI client instance.
'''
This automatically infers the following arguments from their 
corresponding environment variables if they are not provided
'''
client = OpenAI() 

print("OpenAI API Key loaded from environment:", "OPENAI_API_KEY" in os.environ)
#print(f'base url used is {os.environ.get("OPENAI_BASE_URL")}')
def ask(question):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are concise."},
            {"role": "user",   "content": question},
        ],
    )
    #print(resp)
    return resp.choices[0].message.content

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Say hello."
    print(ask(q))