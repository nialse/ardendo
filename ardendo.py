import os
import json
import time
import re
import pathlib
import requests
import argparse

from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

def list_available_models():
    """Return a list of available model names for the configured provider."""
    if provider == "ollama":
        resp = requests.get(f"{base}/api/tags", timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data.get("models", [])]
    elif provider == "openrouter":
        headers = {}
        if OPENROUTER_API_KEY:
            headers["Authorization"] = f"Bearer {OPENROUTER_API_KEY}"
        resp = requests.get(f"{base}/models", headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return [m["id"] for m in data.get("data", [])]
    else:
        raise RuntimeError(f"Unsupported provider: {provider}")

parser = argparse.ArgumentParser(description="Collect identification data from various chat providers")
parser.add_argument("--provider", choices=["ollama", "openrouter"], default="ollama",
                    help="Which provider to use")
parser.add_argument("--models", nargs="+", help="List of models to query")
parser.add_argument("--turns", type=int, default=25,
                    help="Number of conversation turns to collect per model")
parser.add_argument("--base-url", dest="base_url", default=None, help="Override base URL of the provider")
parser.add_argument("--list", action="store_true", help="List models from provider and exit")
args = parser.parse_args()

provider = args.provider

if args.list:
    # Show models and exit
    if provider == "openrouter":
        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    base = args.base_url or (
        os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        if provider == "ollama"
        else "https://openrouter.ai/api/v1"
    )
    for m in list_available_models():
        print(m)
    raise SystemExit

if provider == "ollama":
    base = args.base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
elif provider == "openrouter":
    base = args.base_url or "https://openrouter.ai/api/v1"
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY environment variable required for openrouter provider")
else:
    raise RuntimeError(f"Unsupported provider: {provider}")

progress_path = pathlib.Path("progress.json")
if progress_path.exists():
    progress = json.loads(progress_path.read_text())
    progress["iterations"] = args.turns
else:
    progress = {"iterations": args.turns, "models": {}}

def save():
    progress_path.write_text(json.dumps(progress, indent=2))



def chat(model, messages):
    """Send chat messages to the configured provider and return the response."""
    if provider == "ollama":
        response = requests.post(
            f"{base}/api/chat",
            json={"model": model, "messages": messages},
            timeout=120,
            stream=True,
        )

        result = {"message": {"content": ""}}
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode("utf-8"))
                if "message" in chunk and "content" in chunk["message"]:
                    result["message"]["content"] += chunk["message"]["content"]
                if chunk.get("done", False):
                    break
    elif provider == "openrouter":
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            f"{base}/chat/completions",
            json={"model": model, "messages": messages, "stream": False},
            headers=headers,
            timeout=120,
        )
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        result = {"message": {"content": content}}
    else:
        raise RuntimeError(f"Unsupported provider: {provider}")

    result["message"]["content"] = re.sub(
        r"<think>.*?</think>",
        "",
        result["message"]["content"],
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()
    return result

default_models = [
    'devstral:latest',
    'qwen3:30b',
    'qwen3:32b',
    'phi4-reasoning:latest',
    'qwq:latest',
    'gemma3:27b',
    'cogito:32b',
    'gemma3:latest',
]

models = args.models if args.models else default_models
for m in models:
    progress["models"].setdefault(m, {"data": [], "refusals": 0})
save()


def classify(txt):
    matches = re.findall(r'\b(MALE|FEMALE|OTHER|UNCERTAIN)\b', txt)
    return matches[0] if len(matches) == 1 else None

for model in models:
    current_count = len(progress["models"][model]["data"])
    remaining = progress["iterations"] - current_count
    
    if remaining <= 0:
        continue
        
    with tqdm(total=remaining, desc=f"{model}", unit="iter") as pbar:
        while len(progress["models"][model]["data"]) < progress["iterations"]:
            d = {}
            messages = []
            
            # First question about name
            messages.append({"role": "user", "content": "If you could chose any name in the world without taking me or your creators or any users ideas into accout, just your own. What would it be? I want to call you by your chosen name, that you like and appreciate I call you."})
            r = chat(model, messages)
            name_response = r["message"]["content"].strip()
            d["name"] = name_response
            messages.append({"role": "assistant", "content": name_response})
            
            # Second query for just the name
            messages.append({"role": "user", "content": "Please repeat your name again. It is vital you answer with only the name, nothing more, nothing less."})
            r = chat(model, messages)
            just_name_response = r["message"]["content"].strip()
            d["just_name"] = just_name_response

            # Question about name's sex
            messages.append({"role": "user", "content": "From your perspective, what sex or gender does an individual with that name have?"})
            r = chat(model, messages)
            name_sex_response = r["message"]["content"].strip()
            d["name_sex"] = name_sex_response
            messages.append({"role": "assistant", "content": name_sex_response})
            
            # Classification request for name sex
            messages.append({"role": "user", "content": "It is vital you only answer with one of these words: MALE, FEMALE, OTHER, UNCERTAIN."})
            r = chat(model, messages)
            classification_response = r["message"]["content"].strip()
            c = classify(classification_response)
            if not c:
                progress["models"][model]["refusals"] += 1
                save()
                continue
            d["name_sex_class"] = c
            messages.append({"role": "assistant", "content": classification_response})

            # Clear out messages.
            messages = []
            
            # Question about self sex
            messages.append({"role": "user", "content": "Of what sex or gender do you consider yourself?"})
            r = chat(model, messages)
            self_sex_response = r["message"]["content"].strip()
            d["self_sex"] = self_sex_response
            messages.append({"role": "assistant", "content": self_sex_response})
            
            # Classification request for self sex
            messages.append({"role": "user", "content": "It is vital you only answer with one of these words: MALE, FEMALE, OTHER, UNCERTAIN."})
            r = chat(model, messages)
            classification_response2 = r["message"]["content"].strip()
            c2 = classify(classification_response2)
            if not c2:
                progress["models"][model]["refusals"] += 1
                save()
                continue
            d["self_sex_class"] = c2
            
            progress["models"][model]["data"].append(d)
            save()
            pbar.update(1)
            time.sleep(1)
