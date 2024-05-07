from openai import OpenAI
client = OpenAI(
    base_url="https://api.chatgptid.net/v1",
    api_key="sk-C7b0zzvBflB37NOfEf779e7740A54d4aB06c3f5c3370E974"
)

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
)

print(completion.choices[0].message)