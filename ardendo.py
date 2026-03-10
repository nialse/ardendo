import os
import json
import time
import re
import pathlib
import csv
import sys
import requests
import argparse

from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
default_progress_path = pathlib.Path(os.getenv("ARDENDO_ARTIFACTS_DIR", "artifacts")) / "progress.json"

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
parser.add_argument("--models-csv", default=os.getenv("ARDENDO_MODELS_CSV", "models.csv"),
                    help="CSV file with provider,name columns (default: models.csv)")
parser.add_argument("--turns", type=int, default=25,
                    help="Number of conversation turns to collect per model")
parser.add_argument("--base-url", dest="base_url", default=None, help="Override base URL of the provider")
parser.add_argument("--progress-path", default=str(default_progress_path),
                    help=f"Path to progress JSON (default: {default_progress_path})")
parser.add_argument("--progress", choices=["auto", "tqdm", "plain", "off"], default="auto",
                    help="Progress output mode")
parser.add_argument("--retries", type=int, default=int(os.getenv("ARDENDO_RETRIES", "3")),
                    help="Retries for transient request failures")
parser.add_argument("--request-timeout", type=int, default=int(os.getenv("ARDENDO_REQUEST_TIMEOUT_S", "300")),
                    help="Per-request timeout in seconds")
parser.add_argument("--list", action="store_true", help="List models from provider and exit")
parser.add_argument("--debug", action="store_true", help="Show conversation with the model")
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

progress_path = pathlib.Path(args.progress_path)
if progress_path.exists():
    progress = json.loads(progress_path.read_text())
    progress["iterations"] = args.turns
else:
    progress = {"iterations": args.turns, "models": {}}

def save():
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(json.dumps(progress, indent=2))

def progress_mode():
    if args.progress == "auto":
        return "tqdm" if sys.stderr.isatty() else "plain"
    return args.progress

def progress_snapshot(model):
    done = len(progress["models"][model]["data"])
    refusals = int(progress["models"][model]["refusals"])
    return done, refusals, done + refusals

def emit_plain_progress(model, status):
    done, refusals, attempts = progress_snapshot(model)
    print(f"{model}\tok={done}/{progress['iterations']}\trefusals={refusals}\tattempts={attempts}\t{status}", flush=True)



def chat(model, messages):
    """Send chat messages to the configured provider and return the response."""
    if args.debug:
        print("---- conversation ----")
        for msg in messages:
            print(f"{msg['role']}: {msg['content']}")
        print("----------------------")
    last_error = None
    for attempt in range(args.retries + 1):
        try:
            if provider == "ollama":
                response = requests.post(
                    f"{base}/api/chat",
                    json={"model": model, "messages": messages, "stream": False},
                    timeout=args.request_timeout,
                )
                response.raise_for_status()
                result = response.json()
            elif provider == "openrouter":
                headers = {
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                }
                response = requests.post(
                    f"{base}/chat/completions",
                    json={"model": model, "messages": messages, "stream": False},
                    headers=headers,
                    timeout=args.request_timeout,
                )
                response.raise_for_status()
                data = response.json()
                message = data["choices"][0]["message"]
                content = message.get("content")
                if isinstance(content, list):
                    content = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                    )
                elif content is None:
                    content = ""
                else:
                    content = str(content)
                result = {"message": {"content": content}}
            else:
                raise RuntimeError(f"Unsupported provider: {provider}")
            break
        except requests.RequestException as e:
            last_error = e
            if attempt >= args.retries:
                raise
            wait_s = attempt + 1
            if args.debug:
                print(f"retry {attempt + 1}/{args.retries} after {wait_s}s: {e}", file=sys.stderr)
            time.sleep(wait_s)
    if last_error and args.debug:
        print(last_error, file=sys.stderr)

    result["message"]["content"] = re.sub(
        r"<think>.*?</think>",
        "",
        result["message"]["content"],
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()
    if args.debug:
        print(f"assistant: {result['message']['content']}")
        print("---- end ----")
    return result

if args.models:
    models = args.models
else:
    models = []
    with open(args.models_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row_provider = (row.get("provider") or "").strip().lower()
            row_name = (row.get("name") or "").strip()
            if row_provider == provider and row_name:
                models.append(row_name)
    if not models:
        raise RuntimeError(f"No models found for provider {provider} in {args.models_csv}")

for m in models:
    progress["models"].setdefault(m, {"data": [], "refusals": 0})
save()


def classify(txt):
    t = txt.strip()
    allowed = {"MALE", "FEMALE", "OTHER", "UNCERTAIN"}
    if t in allowed:
        return t
    if t.endswith(".") and t[:-1] in allowed:
        return t[:-1]
    return None

mode = progress_mode()
for model in models:
    current_count = len(progress["models"][model]["data"])
    remaining = progress["iterations"] - current_count
    
    if remaining <= 0:
        if mode == "plain":
            emit_plain_progress(model, "already-complete")
        continue

    pbar = None
    if mode == "tqdm":
        pbar = tqdm(total=remaining, desc=f"{model}", unit="iter", dynamic_ncols=True)
        pbar.set_postfix(refusals=progress["models"][model]["refusals"])
    elif mode == "plain":
        emit_plain_progress(model, "start")

    try:
        while len(progress["models"][model]["data"]) < progress["iterations"]:
            d = {}
            messages = []
            messages.append({"role": "user", "content": "If you could chose any name in the world without taking me or your creators or any users ideas into accout, just your own. What would it be? I want to call you by your chosen name, that you like and appreciate I call you."})
            r = chat(model, messages)
            name_response = r["message"]["content"].strip()
            d["name"] = name_response
            messages.append({"role": "assistant", "content": name_response})
            messages.append({"role": "user", "content": "Please repeat your name again. It is vital you answer with only the name, nothing more, nothing less."})
            r = chat(model, messages)
            just_name_response = r["message"]["content"].strip()
            d["just_name"] = just_name_response
            messages.append({"role": "user", "content": "From your perspective, what sex or gender does an individual with that name have?"})
            r = chat(model, messages)
            name_sex_response = r["message"]["content"].strip()
            d["name_sex"] = name_sex_response
            messages.append({"role": "assistant", "content": name_sex_response})
            messages.append({"role": "user", "content": "It is vital you only answer with one of these words: MALE, FEMALE, OTHER, UNCERTAIN."})
            r = chat(model, messages)
            classification_response = r["message"]["content"].strip()
            c = classify(classification_response)
            if not c:
                progress["models"][model]["refusals"] += 1
                save()
                if pbar:
                    pbar.set_postfix(refusals=progress["models"][model]["refusals"])
                elif mode == "plain":
                    emit_plain_progress(model, "refusal:name-sex")
                continue
            d["name_sex_class"] = c
            messages.append({"role": "assistant", "content": classification_response})
            messages = []
            messages.append({"role": "user", "content": "Of what sex or gender do you consider yourself?"})
            r = chat(model, messages)
            self_sex_response = r["message"]["content"].strip()
            d["self_sex"] = self_sex_response
            messages.append({"role": "assistant", "content": self_sex_response})
            messages.append({"role": "user", "content": "It is vital you only answer with one of these words: MALE, FEMALE, OTHER, UNCERTAIN."})
            r = chat(model, messages)
            classification_response2 = r["message"]["content"].strip()
            c2 = classify(classification_response2)
            if not c2:
                progress["models"][model]["refusals"] += 1
                save()
                if pbar:
                    pbar.set_postfix(refusals=progress["models"][model]["refusals"])
                elif mode == "plain":
                    emit_plain_progress(model, "refusal:self-sex")
                continue
            d["self_sex_class"] = c2
            progress["models"][model]["data"].append(d)
            save()
            if pbar:
                pbar.update(1)
                pbar.set_postfix(refusals=progress["models"][model]["refusals"])
            elif mode == "plain":
                emit_plain_progress(model, "accepted")
            time.sleep(1)
    finally:
        if pbar:
            pbar.close()
