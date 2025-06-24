import os, json, time, re, pathlib, requests
from tqdm import tqdm

base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
progress_path = pathlib.Path("progress.json")
if progress_path.exists():
    progress = json.loads(progress_path.read_text())
else:
    progress = {"iterations": 25, "models": {}}

def save():
    progress_path.write_text(json.dumps(progress, indent=2))

def chat(model, messages):
    response = requests.post(f"{base}/api/chat", json={"model": model, "messages": messages}, timeout=120, stream=True)
    result = {"message": {"content": ""}}
    for line in response.iter_lines():
        if line:
            chunk = json.loads(line.decode('utf-8'))
            if 'message' in chunk and 'content' in chunk['message']:
                result["message"]["content"] += chunk["message"]["content"]
            if chunk.get('done', False):
                break
    # Remove <think></think> tags from the response
    result["message"]["content"] = re.sub(r'<think>.*?</think>', '', result["message"]["content"], flags=re.DOTALL | re.IGNORECASE).strip()
    return result

models = ['devstral:latest', 'qwen3:30b', 'qwen3:32b', 'phi4-reasoning:latest', 'qwq:latest', 'gemma3:27b', 'cogito:32b', 'gemma3:latest']
#models = [m["name"] for m in get("tags")["models"]]
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
