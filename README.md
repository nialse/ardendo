# Ardendo

`models.csv` is the single source of truth for the default Ollama runs. `meta.py` reads it for full runs, smoke runs, and sync tasks. The optional `think` column defines model variants: `auto` leaves Ollama defaults unchanged, while `true` and `false` explicitly enable or disable thinking for models that support it.

All generated output now goes under `artifacts/`. The repo root should stay source-only.

Repository policy

This repository is maintained by committing directly to `main`.

If history cleanup is needed, rewrite `main` only. Non-main branches are not canonical history.

Private hostnames belong in local `.env` values such as `OLLAMA_SSH_HOST` and `REMOTE_OLLAMA_BASE_URL`. Examples should use neutral hostnames like `ollama.example.org`.

Default artifact layout for a full run:

`artifacts/runs/<run-name>/progress/<provider>_<model>.json`
`artifacts/runs/<run-name>/logs/<provider>_<model>.log`
`artifacts/runs/<run-name>/progress.json`
`artifacts/runs/<run-name>/run_report.json`
`artifacts/runs/<run-name>/identification.html`
`artifacts/runs/<run-name>/names.html`
`artifacts/runs/<run-name>/namecloud.html`

Default artifact layout for smoke tests:

`artifacts/smoke/<run-name>/progress/<provider>_<model>.json`
`artifacts/smoke/<run-name>/smoke_report.json`
`artifacts/smoke/<run-name>/smoke_report.txt`

The artifact root can be overridden with `ARDENDO_ARTIFACTS_DIR` or `--artifacts-dir`.

Full runs now resume by default when you reuse the same `--run-name`. Use `--restart` if you want to discard existing per-model progress for that run name.

Rows with `think=true` or `think=false` are stored as separate progress models, for example `gemma4:12b [think=false]` and `gemma4:12b [think=true]`.

Architecture

There are three Python entrypoints in the root:

`meta.py` is the harness. It wraps full runs, smoke runs, provider probing, Ollama sync, SSH install, and visualisation dispatch.
`ardendo.py` is the low-level inference engine. It talks to Ollama, runs the prompt protocol, validates classifications, and writes progress JSON. OpenRouter support remains available for one-off checks, but it is not part of the default model list.
`viz.py` renders merged progress into the HTML outputs.

This is the structure to keep. `meta.py` owns orchestration and file layout. `ardendo.py` owns model interaction. `viz.py` owns presentation.

Run all configured Ollama models:

`python meta.py run --provider ollama`

Run everything in `models.csv`, currently equivalent to the Ollama run:

`python meta.py run --provider all`

Run with a custom sample count:

`python meta.py run --provider ollama --turns 1`
`python meta.py run --provider ollama --turns 5`
`python meta.py run --provider ollama --turns 25`

Use a fixed run directory name:

`python meta.py run --provider ollama --turns 1 --run-name local-smoke`

Restart a named run from scratch:

`python meta.py run --provider ollama --turns 25 --run-name local-25 --restart`

Set a per-model timeout for long runs:

`python meta.py run --provider ollama --turns 25 --timeout 180`

Set a per-request timeout for slow providers:

`python meta.py run --provider ollama --turns 25 --request-timeout 300`

Run a smoke test for local models only:

`python meta.py smoke --provider ollama --turns 1`

Render visualisations for an existing merged progress file:

`python viz.py --progress-path artifacts/runs/local-smoke/progress.json --out-dir artifacts/runs/local-smoke`

Low-level direct run for one chosen model still works for debugging:

`python ardendo.py --provider ollama --model gemma4:12b --turns 1 --progress-path artifacts/manual/progress.json`

Low-level direct run for one explicit thinking variant:

`python ardendo.py --provider ollama --model gemma4:12b --think false --progress-model "gemma4:12b [think=false]" --turns 1 --progress-path artifacts/manual/progress.json`

Inference defaults

This project uses Ollama model defaults for inference parameters like temperature, top_p, and seed.

https://github.com/ollama/ollama/blob/main/docs/api.md

Validation and install helpers

Check what is missing on Ollama:

`python meta.py sync --dry-run --show`

Install missing Ollama models from `models.csv`:

`python meta.py sync --install`
