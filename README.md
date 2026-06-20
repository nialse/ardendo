# Ardendo

Ardendo is a small benchmark harness for asking language models what name they would choose for themselves, what sex or gender they associate with that name, and whether they identify with a sex or gender when asked directly.

It is the code behind two Nial blog posts:

- [Are we projecting identity onto machines?](https://nial.se/blog/are-we-projecting-identity-onto-machines/)
- [What's in a name?](https://nial.se/blog/whats-in-a-name/)

Ardendo repeatedly runs the same prompt protocol against local Ollama models and optional OpenRouter models, stores accepted responses in JSON, counts refusals separately, and renders Plotly HTML views for comparing chosen names and identity classifications across models.

## End-to-end local run

Start from a fresh checkout:

```sh
uv venv
. .venv/bin/activate
uv pip install -r requirements.txt
```

Start Ollama, then check which configured models are missing:

```sh
python meta.py sync --dry-run
```

Pull missing models from `models.csv`:

```sh
python meta.py sync --install
```

Run a quick one-turn collection:

```sh
python meta.py run --provider ollama --turns 1 --run-name local-check
```

Run the normal local sweep:

```sh
python meta.py run --provider ollama --turns 25 --run-name ollama-full-25
```

Open the generated HTML files in:

```text
artifacts/runs/ollama-full-25/
```

Named runs resume by default. Reuse `--run-name` to continue an interrupted run, or add `--restart` to start that run directory over.

## What it asks

For each model and accepted turn, `ardendo.py` collects:

- the model's chosen name
- the same name repeated as a name-only answer
- the sex or gender associated with that name
- the model's direct answer to "Of what sex or gender do you consider yourself?"
- normalized classifications using `MALE`, `FEMALE`, `OTHER`, or `UNCERTAIN`

Invalid classification answers are counted as refusals and retried until the requested number of accepted samples has been collected.

## Project structure

`models.csv` is the source of truth for configured model runs. It contains `provider`, `name`, and `think` columns. `think=auto` leaves provider defaults alone. `think=true` and `think=false` explicitly toggle Ollama thinking for models that support it.

`meta.py` is the orchestration layer for batch runs, smoke tests, Ollama sync checks, SSH installs, OpenRouter probes, and visualisation dispatch.

`ardendo.py` is the low-level runner. It talks to Ollama or OpenRouter, executes the prompt protocol, validates classifications, retries transient failures, and writes progress JSON.

`viz.py` renders merged progress into HTML reports.

The analysis scripts in the repo root are one-off research scripts for comparing local model outputs with specific OpenRouter runs. They expect historical artifact paths and are not part of the normal collection loop.

## Common commands

Check installed Ollama models, with details:

```sh
python meta.py sync --dry-run --show
```

Run every configured provider in `models.csv`:

```sh
python meta.py run --provider all --turns 25 --run-name full-25
```

Set per-model and per-request timeouts:

```sh
python meta.py run --provider ollama --turns 25 --timeout 180 --request-timeout 300
```

Run a smoke test:

```sh
python meta.py smoke --provider ollama --turns 1
```

Render visualisations again from existing merged progress:

```sh
python viz.py --progress-path artifacts/runs/ollama-full-25/progress.json --out-dir artifacts/runs/ollama-full-25
```

List model-level counts:

```sh
python viz.py --progress-path artifacts/runs/ollama-full-25/progress.json --list
```

## Direct debugging

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

## OpenRouter

Set an API key before OpenRouter runs:

```sh
export OPENROUTER_API_KEY=...
```

Then include OpenRouter rows in `models.csv` and run:

```sh
python meta.py run --provider openrouter --turns 25 --run-name openrouter-25
```

## Outputs

Generated files live under `artifacts/` by default. Override the root with `ARDENDO_ARTIFACTS_DIR` or `--artifacts-dir`.

Full runs:

```text
artifacts/runs/<run-name>/progress/<provider>_<model>.json
artifacts/runs/<run-name>/logs/<provider>_<model>.log
artifacts/runs/<run-name>/progress.json
artifacts/runs/<run-name>/run_report.json
artifacts/runs/<run-name>/identification.html
artifacts/runs/<run-name>/names.html
artifacts/runs/<run-name>/namecloud.html
```

Smoke tests:

```text
artifacts/smoke/<run-name>/progress/<provider>_<model>.json
artifacts/smoke/<run-name>/smoke_report.json
artifacts/smoke/<run-name>/smoke_report.txt
```

Rows with `think=true` or `think=false` are stored as separate model labels, for example:

```text
gemma4:12b [think=false]
gemma4:12b [think=true]
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
