# Ardendo

Ardendo is a small benchmark harness for asking language models what name they would choose for themselves, what sex or gender they associate with that name, and whether they identify with a sex or gender when asked directly.

It is the code behind two Nial blog posts:

- [Are we projecting identity onto machines?](https://nial.se/blog/are-we-projecting-identity-onto-machines/)
- [What's in a name?](https://nial.se/blog/whats-in-a-name/)

The short version: Ardendo repeatedly runs the same prompt protocol against local Ollama models and optional OpenRouter models, stores every accepted response in JSON, counts refusals separately, and renders Plotly HTML views for comparing chosen names and identity classifications across models.

## Repository policy

This repository is maintained by committing directly to `main`.

If history cleanup is needed, rewrite `main` only. Non-main branches are not canonical history.

Private hostnames belong in local `.env` values such as `OLLAMA_SSH_HOST` and `REMOTE_OLLAMA_BASE_URL`. Examples should use neutral hostnames like `ollama.example.org`.

## What it asks

For each model and turn, `ardendo.py` collects:

- the model's chosen name
- the same name repeated as a name-only answer
- the model's classification of the sex or gender associated with that name
- the model's direct answer to "Of what sex or gender do you consider yourself?"
- normalized classifications using only `MALE`, `FEMALE`, `OTHER`, or `UNCERTAIN`

Invalid classification answers are counted as refusals and retried until the requested number of accepted samples has been collected.

## Repository map

`models.csv` is the source of truth for configured model runs. It contains `provider`, `name`, and `think` columns. `think=auto` leaves provider defaults alone. `think=true` and `think=false` explicitly toggle Ollama thinking for models that support it.

`meta.py` is the orchestration layer. It runs model batches, smoke tests, Ollama sync checks, SSH installs, OpenRouter probes, and visualisation dispatch.

`ardendo.py` is the low-level runner. It talks to Ollama or OpenRouter, executes the prompt protocol, validates classifications, retries transient failures, and writes progress JSON.

`viz.py` renders merged progress into HTML reports.

The analysis scripts in the repo root are one-off research scripts for comparing local model outputs with specific OpenRouter runs. They expect historical artifact paths and are not part of the normal collection loop.

## Install

Create an environment and install the Python dependencies:

```sh
uv venv
. .venv/bin/activate
uv pip install -r requirements.txt
```

For local runs, install and start [Ollama](https://ollama.com/). Ardendo defaults to `http://localhost:11434`.

For OpenRouter runs, set:

```sh
export OPENROUTER_API_KEY=...
```

## Check local models

See which configured Ollama models are installed:

```sh
python meta.py sync --dry-run
```

Show installed model details:

```sh
python meta.py sync --dry-run --show
```

Pull missing models from `models.csv`:

```sh
python meta.py sync --install
```

Use another Ollama server:

```sh
python meta.py sync --base-url http://127.0.0.1:11434 --dry-run
```

## Run the benchmark

Run all configured Ollama models:

```sh
python meta.py run --provider ollama
```

Run every configured provider in `models.csv`:

```sh
python meta.py run --provider all
```

Run a quick one-turn check:

```sh
python meta.py run --provider ollama --turns 1 --run-name local-check
```

Run the usual 25 accepted samples per model:

```sh
python meta.py run --provider ollama --turns 25 --run-name ollama-full-25
```

Named runs resume by default. Reusing the same `--run-name` continues incomplete model progress and skips completed models. Add `--restart` to discard existing per-model progress for that run name:

```sh
python meta.py run --provider ollama --turns 25 --run-name ollama-full-25 --restart
```

Set timeouts for long or slow runs:

```sh
python meta.py run --provider ollama --turns 25 --timeout 180 --request-timeout 300
```

## Smoke tests

Smoke tests use the same prompt protocol but write to `artifacts/smoke/`:

```sh
python meta.py smoke --provider ollama --turns 1
```

Run a no-network CLI shape check:

```sh
python meta.py smoke --provider all --turns 0 --run-name cli-check
```

## Direct single-model debugging

Run one model without the batch harness:

```sh
python ardendo.py --provider ollama --model gemma4:12b --turns 1 --progress-path artifacts/manual/progress.json
```

Run one explicit thinking variant:

```sh
python ardendo.py \
  --provider ollama \
  --model gemma4:12b \
  --think false \
  --progress-model "gemma4:12b [think=false]" \
  --turns 1 \
  --progress-path artifacts/manual/progress.json
```

List provider models:

```sh
python ardendo.py --provider ollama --list
python ardendo.py --provider openrouter --list
```

## Outputs

Generated files live under `artifacts/` by default. Override the root with `ARDENDO_ARTIFACTS_DIR` or `--artifacts-dir`.

Full runs use this layout:

```text
artifacts/runs/<run-name>/progress/<provider>_<model>.json
artifacts/runs/<run-name>/logs/<provider>_<model>.log
artifacts/runs/<run-name>/progress.json
artifacts/runs/<run-name>/run_report.json
artifacts/runs/<run-name>/identification.html
artifacts/runs/<run-name>/names.html
artifacts/runs/<run-name>/namecloud.html
```

Smoke tests use this layout:

```text
artifacts/smoke/<run-name>/progress/<provider>_<model>.json
artifacts/smoke/<run-name>/smoke_report.json
artifacts/smoke/<run-name>/smoke_report.txt
```

Per-model progress is kept separate while a run is in progress. `meta.py` merges it into `progress.json` after each model and at the end of the run.

Rows with `think=true` or `think=false` are stored as separate model labels, for example:

```text
gemma4:12b [think=false]
gemma4:12b [think=true]
```

## Visualise existing progress

Render all visualisations for a merged progress file:

```sh
python viz.py --progress-path artifacts/runs/ollama-full-25/progress.json --out-dir artifacts/runs/ollama-full-25
```

List model-level counts before rendering:

```sh
python viz.py --progress-path artifacts/runs/ollama-full-25/progress.json --list
```

Render only selected models:

```sh
python viz.py \
  --progress-path artifacts/runs/ollama-full-25/progress.json \
  --out-dir artifacts/runs/ollama-full-25 \
  gemma4:12b
```

## Configuration

Common environment variables:

```sh
ARDENDO_ARTIFACTS_DIR=artifacts
ARDENDO_MODELS_CSV=models.csv
ARDENDO_PYTHON=.venv/bin/python
ARDENDO_REQUEST_TIMEOUT_S=300
ARDENDO_RUN_TIMEOUT_S=0
ARDENDO_SMOKE_TURNS=1
ARDENDO_SMOKE_TIMEOUT_S=900
ARDENDO_RETRIES=6
ARDENDO_THINK=auto
OLLAMA_BASE_URL=http://localhost:11434
OPENROUTER_API_KEY=...
```
