import json
import os
import requests

BASE = os.getenv('SERVER_URL', 'http://127.0.0.1:8080').rstrip('/')
s = requests.Session()

def check(path):
    r = s.get(BASE + path, timeout=10)
    print(path, r.status_code, r.text[:350])
    r.raise_for_status()

check('/health')
check('/slots')
check('/metrics')
models = s.get(BASE + '/v1/models', timeout=10).json()
model = models['data'][0]['id']
print('Model:', model)
payload = {'model': model, 'messages': [{'role':'user','content':'Reply with exactly: ready'}], 'max_tokens': 32, 'temperature':0}
r = s.post(BASE + '/v1/chat/completions', json=payload, timeout=180)
r.raise_for_status()
print('Generation:', json.dumps(r.json(), indent=2)[:1500])
# Tool-call request: this tests compatibility, not guaranteed model correctness.
payload = {
  'model': model,
  'messages': [{'role':'user','content':'What is the weather in San Diego? Call the weather tool.'}],
  'tools': [{'type':'function','function':{'name':'get_weather','description':'Retrieve weather for a city','parameters':{'type':'object','properties':{'city':{'type':'string'}},'required':['city']}}}],
  'tool_choice': 'auto', 'max_tokens': 256, 'temperature': 0
}
r = s.post(BASE + '/v1/chat/completions', json=payload, timeout=180)
r.raise_for_status()
print('Tool test:', json.dumps(r.json()['choices'][0]['message'], indent=2))
