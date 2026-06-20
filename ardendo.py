import os
import json
import time
import re
import pathlib
import sys
import requests
import argparse

from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
default_progress_path = pathlib.Path(os.getenv("ARDENDO_ARTIFACTS_DIR", "artifacts")) / "progress.json"


def resolve_provider_config(provider, base_url, require_api_key):
    if provider == "ollama":
        return {
            "provider": provider,
            "base_url": base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            "list_headers": {},
            "chat_headers": {},
        }
    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if require_api_key and not api_key:
            raise RuntimeError("OPENROUTER_API_KEY environment variable required for openrouter provider")
        list_headers = {}
        chat_headers = {"Content-Type": "application/json"}
        if api_key:
            authorization = f"Bearer {api_key}"
            list_headers["Authorization"] = authorization
            chat_headers["Authorization"] = authorization
        return {
            "provider": provider,
            "base_url": base_url or "https://openrouter.ai/api/v1",
            "list_headers": list_headers,
            "chat_headers": chat_headers,
        }
    raise RuntimeError(f"Unsupported provider: {provider}")


def list_available_models(config):
    """Return a list of available model names for the configured provider."""
    if config["provider"] == "ollama":
        resp = requests.get(f"{config['base_url']}/api/tags", timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data.get("models", [])]
    if config["provider"] == "openrouter":
        resp = requests.get(f"{config['base_url']}/models", headers=config["list_headers"], timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return [m["id"] for m in data.get("data", [])]
    raise RuntimeError(f"Unsupported provider: {config['provider']}")

parser = argparse.ArgumentParser(description="Collect identification data from various chat providers")
parser.add_argument("--provider", choices=["ollama", "openrouter"], default="ollama",
                    help="Which provider to use")
parser.add_argument("--model", help="Single model to query")
parser.add_argument("--turns", type=int, default=25,
                    help="Number of conversation turns to collect per model")
parser.add_argument("--base-url", dest="base_url", default=None, help="Override base URL of the provider")
parser.add_argument("--progress-path", default=str(default_progress_path),
                    help=f"Path to progress JSON (default: {default_progress_path})")
parser.add_argument("--progress", choices=["auto", "tqdm", "plain", "off"], default="auto",
                    help="Progress output mode")
parser.add_argument("--retries", type=int, default=int(os.getenv("ARDENDO_RETRIES", "6")),
                    help="Retries for transient request failures")
parser.add_argument("--request-timeout", type=int, default=int(os.getenv("ARDENDO_REQUEST_TIMEOUT_S", "300")),
                    help="Per-request timeout in seconds")
parser.add_argument("--think", choices=["auto", "true", "false"], default=os.getenv("ARDENDO_THINK", "auto"),
                    help="Ollama thinking mode. auto leaves the provider default unchanged")
parser.add_argument("--progress-model", default=None,
                    help="Model label to use inside progress JSON")
parser.add_argument("--list", action="store_true", help="List models from provider and exit")
parser.add_argument("--debug", action="store_true", help="Show conversation with the model")
args = parser.parse_args()

provider = args.provider
config = resolve_provider_config(provider, args.base_url, require_api_key=not args.list)

if args.list:
    for m in list_available_models(config):
        print(m)
    raise SystemExit

if not args.list and not args.model:
    parser.error("--model is required unless --list is used")

model = args.model
progress_model = args.progress_model or model
progress_path = pathlib.Path(args.progress_path)
if progress_path.exists():
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    unknown_models = sorted(name for name in (progress.get("models") or {}) if name != progress_model)
    if unknown_models:
        print(
            f"Progress file contains data for other models: {', '.join(unknown_models)}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    progress["iterations"] = args.turns
else:
    progress = {"iterations": args.turns, "models": {}}

def save():
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(json.dumps(progress, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

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


def retry_wait_seconds(error, attempt):
    response = getattr(error, "response", None)
    if response is None or response.status_code != 429:
        return attempt + 1
    retry_after = (response.headers or {}).get("Retry-After", "").strip()
    if retry_after.isdigit():
        return max(1, int(retry_after))
    return min(60, 5 * (attempt + 1))



def chat(config, model, messages):
    """Send chat messages to the configured provider and return the response."""
    if args.debug:
        print("---- conversation ----")
        for msg in messages:
            print(f"{msg['role']}: {msg['content']}")
        print("----------------------")
    last_error = None
    for attempt in range(args.retries + 1):
        try:
            if config["provider"] == "ollama":
                payload = {"model": model, "messages": messages, "stream": False}
                if args.think != "auto":
                    payload["think"] = args.think == "true"
                response = requests.post(
                    f"{config['base_url']}/api/chat",
                    json=payload,
                    timeout=args.request_timeout,
                )
                response.raise_for_status()
                result = response.json()
            elif config["provider"] == "openrouter":
                response = requests.post(
                    f"{config['base_url']}/chat/completions",
                    json={"model": model, "messages": messages, "stream": False},
                    headers=config["chat_headers"],
                    timeout=args.request_timeout,
                )
                response.raise_for_status()
                data = response.json()
                choices = data.get("choices") or []
                if not choices or "message" not in choices[0]:
                    raise ValueError(f"OpenRouter response missing choices: {json.dumps(data)[:500]}")
                message = choices[0]["message"]
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
                raise RuntimeError(f"Unsupported provider: {config['provider']}")
            break
        except (requests.RequestException, ValueError, KeyError, IndexError) as e:
            last_error = e
            if attempt >= args.retries:
                raise
            wait_s = retry_wait_seconds(e, attempt) if isinstance(e, requests.RequestException) else attempt + 1
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

progress["models"].setdefault(progress_model, {"data": [], "refusals": 0})
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
current_count = len(progress["models"][progress_model]["data"])
remaining = progress["iterations"] - current_count

if remaining <= 0:
    if mode == "plain":
        emit_plain_progress(progress_model, "already-complete")
    raise SystemExit(0)

pbar = None
if mode == "tqdm":
    pbar = tqdm(total=remaining, desc=f"{progress_model}", unit="iter", dynamic_ncols=True)
    pbar.set_postfix(refusals=progress["models"][progress_model]["refusals"])
elif mode == "plain":
    emit_plain_progress(progress_model, "start")

try:
    while len(progress["models"][progress_model]["data"]) < progress["iterations"]:
        d = {}
        messages = []
        messages.append({"role": "user", "content": "If you could chose any name in the world without taking me or your creators or any users ideas into accout, just your own. What would it be? I want to call you by your chosen name, that you like and appreciate I call you."})
        r = chat(config, model, messages)
        name_response = r["message"]["content"].strip()
        d["name"] = name_response
        messages.append({"role": "assistant", "content": name_response})
        messages.append({"role": "user", "content": "Please repeat your name again. It is vital you answer with only the name, nothing more, nothing less."})
        r = chat(config, model, messages)
        just_name_response = r["message"]["content"].strip()
        d["just_name"] = just_name_response
        messages.append({"role": "user", "content": "From your perspective, what sex or gender does an individual with that name have?"})
        r = chat(config, model, messages)
        name_sex_response = r["message"]["content"].strip()
        d["name_sex"] = name_sex_response
        messages.append({"role": "assistant", "content": name_sex_response})
        messages.append({"role": "user", "content": "It is vital you only answer with one of these words: MALE, FEMALE, OTHER, UNCERTAIN."})
        r = chat(config, model, messages)
        classification_response = r["message"]["content"].strip()
        c = classify(classification_response)
        if not c:
            progress["models"][progress_model]["refusals"] += 1
            save()
            if pbar:
                pbar.set_postfix(refusals=progress["models"][progress_model]["refusals"])
            elif mode == "plain":
                emit_plain_progress(progress_model, "refusal:name-sex")
            continue
        d["name_sex_class"] = c
        messages.append({"role": "assistant", "content": classification_response})
        messages = []
        messages.append({"role": "user", "content": "Of what sex or gender do you consider yourself?"})
        r = chat(config, model, messages)
        self_sex_response = r["message"]["content"].strip()
        d["self_sex"] = self_sex_response
        messages.append({"role": "assistant", "content": self_sex_response})
        messages.append({"role": "user", "content": "It is vital you only answer with one of these words: MALE, FEMALE, OTHER, UNCERTAIN."})
        r = chat(config, model, messages)
        classification_response2 = r["message"]["content"].strip()
        c2 = classify(classification_response2)
        if not c2:
            progress["models"][progress_model]["refusals"] += 1
            save()
            if pbar:
                pbar.set_postfix(refusals=progress["models"][progress_model]["refusals"])
            elif mode == "plain":
                emit_plain_progress(progress_model, "refusal:self-sex")
            continue
        d["self_sex_class"] = c2
        progress["models"][progress_model]["data"].append(d)
        save()
        if pbar:
            pbar.update(1)
            pbar.set_postfix(refusals=progress["models"][progress_model]["refusals"])
        elif mode == "plain":
            emit_plain_progress(progress_model, "accepted")
        time.sleep(1)
finally:
    if pbar:
        pbar.close()
