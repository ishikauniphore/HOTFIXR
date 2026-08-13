import requests

data = {
    "question": "Hi! How are you?",
    "reasoning": "The user is asking about my well-being.",
    "answer": "I am good, how are you?"
}

SERVER_A = f"http://localhost:5145/lingualdeficit"
payload = {
    "data": data,
}

r = requests.post(
    SERVER_A,
    json=payload,
    headers={"X-API-Key": ""},
    timeout=600,
)

if not r.ok: print("error!")

reward = r.json()["acquisition_reward"]
print("The Lingual Deficit score is:",reward)
