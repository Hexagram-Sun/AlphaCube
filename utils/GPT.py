import openai
import os
import time

if os.path.exists('utils/proxy.txt'):
    with open('utils/proxy.txt', 'r') as f: proxy = f.read().strip()
    os.environ['http_proxy'] = proxy
    os.environ['https_proxy'] = proxy

assert os.path.exists('utils/api_key.txt'), 'utils/api_key.txt not found'
with open('utils/api_key.txt', 'r') as f:
    openai.api_key = f.read()

def gpt(prompt):
    while 1:
        try:
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
            resp = response["choices"][0]["message"]["content"]
            with open('./logs/chat_history.txt', 'a') as f:
                f.write(f'User: {prompt}\n\nGPT: {resp}\n\n\n\n')
            return resp
        except Exception as e:
            try: print('error:', response, 'retrying...')
            except: print(e)
            time.sleep(5)
            

if __name__ == "__main__":
    user_input = "hello"
    gpt_response = gpt(user_input)
    print(gpt_response)
