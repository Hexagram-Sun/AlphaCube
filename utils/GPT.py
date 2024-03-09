import openai
import os

# os.environ['http_proxy'] = '127.0.0.1:10809'
# os.environ['https_proxy'] = '127.0.0.1:10809'

assert os.path.exists('utils/api_key.txt'), 'utils/api_key.txt not found'
with open('utils/api_key.txt', 'r') as f:
    openai.api_key = f.read()

def gpt(prompt):
    
    messages = [
        {
            "role": "user",
            "content": prompt,
        },
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        max_tokens=2048,
        temperature=0.5
    )

    return response["choices"][0]["message"]["content"]

if __name__ == "__main__":
    user_input = "hello"
    gpt_response = gpt(user_input)
    print(gpt_response)
