
# llama.cpp Inference API

This guide shows how to send requests to our Qwen3-1.7B model using `llama.cpp`.

The server provides an OpenAI-compatible API.

## 1. Start the server

From the project root on EC2:

```bash
bash inference_setup/start_server.sh
```

By default, the API is available at:

```
http://127.0.0.1:8080
```

The examples work either on the EC2 instance or on a laptop with an SSH tunnel established to port 8080.

Install the Python dependency:

```bash
python -m pip install -r inference_setup/requirements.txt
```

## 2. Basic Question

Send a normal question to the model.

```python
import requests

url = "http://127.0.0.1:8080/v1/chat/completions"

payload = {
    "messages": [
        {
            "role": "user",
            "content": "What is machine learning?"
        }
    ],
    "max_tokens": 200,
    "temperature": 0
}

response = requests.post(url, json=payload, timeout=120)
response.raise_for_status()

data = response.json()
print(data["choices"][0]["message"]["content"])
```

The model generates a text response.

## 3. Tool Calling

Tool calling lets the model request that an external function be executed.

For example, we can give the model an `add` function.

### Step 1: Define the tool

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"}
                },
                "required": ["a", "b"]
            }
        }
    }
]
```

### Step 2: Request a tool call

```python
import requests
import json

url = "http://127.0.0.1:8080/v1/chat/completions"

messages = [
    {
        "role": "user",
        "content": "Use the add tool to compute 17 + 25."
    }
]

payload = {
    "messages": messages,
    "tools": tools,
    "tool_choice": "required",
    "max_tokens": 200,
    "temperature": 0
}

response = requests.post(url, json=payload, timeout=120)
response.raise_for_status()

message = response.json()["choices"][0]["message"]

print(json.dumps(message.get("tool_calls", []), indent=2))
```

Expected single entry in the returned `tool_calls` list (simplified):

```json
{
    "id": "call_123",
    "type": "function",
    "function": {
        "name": "add",
        "arguments": "{\"a\":17,\"b\":25}"
    }
}
```

The model only generates the tool call. It does not execute the Python function.

### Step 3: Execute the tool and return its result

```python
def add(a, b):
    return a + b

if not message.get("tool_calls"):
    raise RuntimeError("Model did not return a tool call")

tool_call = message["tool_calls"][0]
arguments = json.loads(tool_call["function"]["arguments"])

if tool_call["function"]["name"] != "add":
    raise ValueError("Unexpected tool name")
result = add(**arguments)

messages.append(message)
messages.append({
    "role": "tool",
    "tool_call_id": tool_call["id"],
    "content": str(result)
})

response = requests.post(
    url,
    json={
        "messages": messages,
        "max_tokens": 200,
        "temperature": 0
    },
    timeout=120
)
response.raise_for_status()

print(response.json()["choices"][0]["message"]["content"])
```

The final answer should communicate that the result is 42.

## 4. Important Notes

- **Endpoint:** Both examples use `/v1/chat/completions`.
- **Normal inference:** The model returns text.
- **Tool calling:** The model returns a function name and arguments. The application executes the function.
- **`tool_choice="required"`:** Forces the model to request a tool call.
- **`tool_choice="auto"`:** Allows the model to decide whether to request a tool.
- **`--jinja`:** Keep this enabled in `start_server.sh` for the model's tool-calling chat template.
- **Security:** Only execute explicitly allowed functions. Validate tool names and arguments before calling them.
- **Access:** The server currently binds to `127.0.0.1`, so you must run requests on EC2 or use SSH port forwarding.