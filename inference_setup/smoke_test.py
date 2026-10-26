"""Basic HTTP/LLM and tool-call smoke checks for a running llama-server."""
import json
import os
import requests

BASE = os.getenv('SERVER_URL', 'http://127.0.0.1:8080').rstrip('/')
s = requests.Session()

def check(path):
    r = s.get(BASE + path, timeout=15)
    print(path, r.status_code, r.text[:350])
    r.raise_for_status()

for path in ('/health', '/slots', '/metrics'):
    check(path)

r = s.get(BASE + '/v1/models', timeout=15)
r.raise_for_status()
model = r.json()['data'][0]['id']
print('Model:', model)

payload = {'model': model, 'messages': [{'role': 'user', 'content': 'Reply with exactly: ready'}], 'max_tokens': 64, 'temperature': 0}
r = s.post(BASE + '/v1/chat/completions', json=payload, timeout=180)
r.raise_for_status()
message = r.json()['choices'][0]['message']
print('Basic generation:', json.dumps(message, indent=2))
assert message.get('content'), 'Basic completion returned no text'

payload = {
    'model': model,
    'messages': [{'role': 'user', 'content': 'Use the add tool to compute 17 + 25.'}],
    'tools': [{'type': 'function', 'function': {
        'name': 'add', 'description': 'Add two integers',
        'parameters': {'type': 'object', 'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}}, 'required': ['a', 'b']}
    }}],
    'tool_choice': 'required', 'max_tokens': 256, 'temperature': 0
}
r = s.post(BASE + '/v1/chat/completions', json=payload, timeout=180)
r.raise_for_status()
message = r.json()['choices'][0]['message']
print('Tool response:', json.dumps(message, indent=2))
calls = message.get('tool_calls') or []
assert calls, 'Tool-call request returned no structured tool_calls'
call = calls[0]
assert call['function']['name'] == 'add', 'Unexpected tool name'
args = json.loads(call['function']['arguments'])
assert args['a'] == 17 and args['b'] == 25, f'Unexpected tool arguments: {args}'
print('PASS: basic completion and structured tool call')
