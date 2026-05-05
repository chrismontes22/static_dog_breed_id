import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Read credentials from environment
API_BASE_URL = f"https://api.cloudflare.com/client/v4/accounts/{os.getenv('CF_ACCOUNT_ID')}/ai/run/"
headers = {"Authorization": f"Bearer {os.getenv('CF_API_TOKEN')}"}


def run(model, inputs):
    input_payload = { "messages": inputs }  # renamed to avoid shadowing built-in 'input'
    response = requests.post(f"{API_BASE_URL}{model}", headers=headers, json=input_payload)
    return response.json()


inputs = [
    { "role": "system", "content": "You are a friendly assistant" },
    { "role": "user", "content": "please write me a cooking recipe for pizza" }
]

output = run("@cf/google/gemma-4-26b-a4b-it", inputs)
print(output)