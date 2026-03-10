# Ardendo

`models.csv` is the single source of truth. `meta.py` reads it for full runs, smoke runs, probes, and sync tasks.

All generated output now goes under `artifacts/`. The repo root should stay source-only.

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

Architecture

There are three Python entrypoints in the root:

`meta.py` is the harness. It wraps full runs, smoke runs, provider probing, Ollama sync, SSH install, and visualisation dispatch.
`ardendo.py` is the low-level inference engine. It talks to Ollama or OpenRouter, runs the prompt protocol, validates classifications, and writes progress JSON.
`viz.py` renders merged progress into the HTML outputs.

This is the structure to keep. `meta.py` owns orchestration and file layout. `ardendo.py` owns model interaction. `viz.py` owns presentation.

Run all local Ollama models:

`python meta.py run --provider ollama`

Run all OpenRouter models:

`python meta.py run --provider openrouter`

Run everything in `models.csv`:

`python meta.py run --provider all`

Run with a custom sample count:

`python meta.py run --provider ollama --turns 1`
`python meta.py run --provider ollama --turns 5`
`python meta.py run --provider all --turns 25`

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

Run a smoke test for OpenRouter only:

`python meta.py smoke --provider openrouter --turns 1`

Render visualisations for an existing merged progress file:

`python viz.py --progress-path artifacts/runs/local-smoke/progress.json --out-dir artifacts/runs/local-smoke`

Low-level direct run for one provider still works:

`python ardendo.py --provider ollama --turns 1 --progress-path artifacts/manual/progress.json`

Reasoning cost control

This project runs with provider defaults for inference parameters like temperature, top_p, and seed.

OpenRouter defaults temperature to `1.0` when it is omitted:
https://openrouter.ai/docs/api/reference/parameters

Ollama uses model defaults when `options` are omitted:
https://github.com/ollama/ollama/blob/main/docs/api.md

Validation and install helpers

Probe OpenRouter ids in `models.csv`:

`python meta.py probe`

Check what is missing on Ollama:

`python meta.py sync --dry-run --show`

Install missing Ollama models from `models.csv`:

`python meta.py sync --install`
