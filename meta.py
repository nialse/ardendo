#!/usr/bin/env python3

import argparse
import csv
import json
import os
import re
import select
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv


load_dotenv()

root = Path(__file__).resolve().parent
python = Path(os.environ.get("ARDENDO_PYTHON", root / ".venv" / "bin" / "python"))
default_artifacts = Path(os.environ.get("ARDENDO_ARTIFACTS_DIR", root / "artifacts"))
default_models_csv = Path(os.environ.get("ARDENDO_MODELS_CSV", root / "models.csv"))


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def load_models(models_csv, provider):
    rows = []
    with Path(models_csv).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row_provider = (row.get("provider") or "").strip().lower()
            row_name = (row.get("name") or "").strip()
            if not row_provider or not row_name:
                continue
            if provider != "all" and row_provider != provider:
                continue
            rows.append((row_provider, row_name))
    return rows


def load_models_by_provider(models_csv, provider):
    selected = {"ollama": [], "openrouter": []}
    for row_provider, row_name in load_models(models_csv, provider):
        if row_provider in selected:
            selected[row_provider].append(row_name)
    return {name: models for name, models in selected.items() if models}


def model_progress_path(run_dir, provider, model):
    return run_dir / "progress" / f"{provider}_{safe_name(model)}.json"


def model_log_path(run_dir, provider, model):
    return run_dir / "logs" / f"{provider}_{safe_name(model)}.log"


def read_counts(progress_path, model):
    if not progress_path.exists():
        return 0, 0
    data = json.loads(progress_path.read_text(encoding="utf-8"))
    state = (data.get("models") or {}).get(model) or {}
    return len(state.get("data") or []), int(state.get("refusals") or 0)


def merge_progress(run_dir, turns):
    merged = {"iterations": turns, "models": {}}
    for progress_path in sorted((run_dir / "progress").glob("*.json")):
        data = json.loads(progress_path.read_text(encoding="utf-8"))
        merged["iterations"] = data.get("iterations", merged["iterations"])
        for model, model_data in (data.get("models") or {}).items():
            merged["models"][model] = model_data
    merged_path = run_dir / "progress.json"
    merged_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return merged_path


def write_report(run_dir, report, filename="run_report.json"):
    report_path = run_dir / filename
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report_path


def migrate_legacy_progress(run_dir, provider):
    legacy_path = run_dir / f"progress_{provider}.json"
    if not legacy_path.exists():
        return
    if any((run_dir / "progress").glob(f"{provider}_*.json")):
        return
    data = json.loads(legacy_path.read_text(encoding="utf-8"))
    iterations = data.get("iterations")
    for model, model_data in (data.get("models") or {}).items():
        progress_path = model_progress_path(run_dir, provider, model)
        payload = {"iterations": iterations, "models": {model: model_data}}
        progress_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_ardendo(provider, model, turns, progress_path, request_timeout, timeout_s=0, log_path=None):
    cmd = [
        str(python),
        str(root / "ardendo.py"),
        "--provider",
        provider,
        "--models",
        model,
        "--turns",
        str(turns),
        "--progress-path",
        str(progress_path),
        "--progress",
        "plain",
        "--request-timeout",
        str(request_timeout),
    ]

    started = time.time()
    error = None
    exit_code = 0
    timed_out = False
    log_handle = None
    process = None

    try:
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("a", encoding="utf-8")
            log_handle.write(f"\n[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] start {provider} {model}\n")

        process = subprocess.Popen(
            cmd,
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        while True:
            if process.stdout:
                ready, _, _ = select.select([process.stdout], [], [], 1)
                if ready:
                    line = process.stdout.readline()
                    if line:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                        if log_handle:
                            log_handle.write(line)
                            log_handle.flush()
            if process.poll() is not None:
                break
            if timeout_s and (time.time() - started) > timeout_s:
                timed_out = True
                process.kill()
                break

        if process.stdout:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                if log_handle:
                    log_handle.write(line)
        exit_code = process.wait()
    except KeyboardInterrupt:
        if process is not None:
            process.kill()
            exit_code = process.wait()
        raise
    finally:
        if log_handle:
            log_handle.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] end {provider} {model}\n")
            log_handle.close()

    completed_after, refusals_after = read_counts(progress_path, model)
    if timed_out:
        error = f"timeout after {timeout_s}s"
    elif exit_code != 0:
        error = f"exit code {exit_code}"

    return {
        "provider": provider,
        "model": model,
        "progress_path": str(progress_path),
        "log_path": str(log_path) if log_path else None,
        "ok": completed_after >= turns,
        "seconds": round(time.time() - started, 2),
        "refusals": refusals_after,
        "data_points": completed_after,
        "error": error,
    }


parser = argparse.ArgumentParser(description="Ardendo harness")
subparsers = parser.add_subparsers(dest="command")

run_parser = subparsers.add_parser("run")
run_parser.add_argument("--provider", choices=["all", "ollama", "openrouter"], default="all")
run_parser.add_argument("--turns", type=int, default=25)
run_parser.add_argument("--models-csv", default=str(default_models_csv))
run_parser.add_argument("--artifacts-dir", default=str(default_artifacts))
run_parser.add_argument("--run-name", default=None)
run_parser.add_argument("--timeout", type=int, default=int(os.environ.get("ARDENDO_RUN_TIMEOUT_S", "0")))
run_parser.add_argument("--request-timeout", type=int, default=int(os.environ.get("ARDENDO_REQUEST_TIMEOUT_S", "300")))
run_parser.add_argument("--restart", action="store_true")

smoke_parser = subparsers.add_parser("smoke")
smoke_parser.add_argument("--provider", choices=["all", "ollama", "openrouter"], default="all")
smoke_parser.add_argument("--models-csv", "--models", dest="models_csv", default=str(default_models_csv))
smoke_parser.add_argument("--turns", type=int, default=int(os.environ.get("ARDENDO_SMOKE_TURNS", "1")))
smoke_parser.add_argument("--timeout", type=int, default=int(os.environ.get("ARDENDO_SMOKE_TIMEOUT_S", "900")))
smoke_parser.add_argument("--request-timeout", type=int, default=int(os.environ.get("ARDENDO_REQUEST_TIMEOUT_S", "300")))
smoke_parser.add_argument("--artifacts-dir", default=str(default_artifacts))
smoke_parser.add_argument("--run-name", default=None)
smoke_parser.add_argument("--restart", action="store_true")

probe_parser = subparsers.add_parser("probe")
probe_parser.add_argument("--models-csv", default=str(default_models_csv))
probe_parser.add_argument("--out", default=None)

sync_parser = subparsers.add_parser("sync")
sync_parser.add_argument("--models-csv", default=str(default_models_csv))
sync_parser.add_argument("--base-url", default=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
sync_parser.add_argument("--install", action="store_true")
sync_parser.add_argument("--show", action="store_true")
sync_parser.add_argument("--dry-run", action="store_true")

ssh_parser = subparsers.add_parser("install-ssh")
ssh_parser.add_argument("host", nargs="?", default=os.environ.get("OLLAMA_SSH_HOST"))
ssh_parser.add_argument("--models-csv", default=str(default_models_csv))
ssh_parser.add_argument("--remote-base-url", default=os.environ.get("REMOTE_OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
ssh_parser.add_argument("--ssh-args", default=os.environ.get("OLLAMA_SSH_ARGS", ""))

viz_parser = subparsers.add_parser("viz")
viz_parser.add_argument("args", nargs=argparse.REMAINDER)

args = parser.parse_args()

if not args.command:
    parser.print_help()
    raise SystemExit(0)

if not python.exists():
    print(f"Python not found: {python}", file=sys.stderr)
    raise SystemExit(2)

if args.command == "run":
    models_csv = Path(args.models_csv)
    if not models_csv.exists():
        print(f"Models CSV not found: {models_csv}", file=sys.stderr)
        raise SystemExit(2)

    selected = load_models_by_provider(models_csv, args.provider)
    if not selected:
        print("No models selected", file=sys.stderr)
        raise SystemExit(2)

    run_name = args.run_name or time.strftime("run-%Y%m%d-%H%M%S")
    run_dir = Path(args.artifacts_dir) / "runs" / run_name
    (run_dir / "progress").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    for provider in selected:
        migrate_legacy_progress(run_dir, provider)

    report = {
        "models_csv": str(models_csv),
        "provider": args.provider,
        "turns": args.turns,
        "timeout_s": args.timeout,
        "run_dir": str(run_dir),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "results": [],
    }

    failed = []
    for provider, models in selected.items():
        for model in models:
            progress_path = model_progress_path(run_dir, provider, model)
            log_path = model_log_path(run_dir, provider, model)

            if args.restart:
                if progress_path.exists():
                    progress_path.unlink()
                if log_path.exists():
                    log_path.unlink()

            completed_before, refusals_before = read_counts(progress_path, model)
            if completed_before >= args.turns:
                result = {
                    "provider": provider,
                    "model": model,
                    "progress_path": str(progress_path),
                    "log_path": str(log_path),
                    "ok": True,
                    "seconds": 0.0,
                    "refusals": refusals_before,
                    "data_points": completed_before,
                    "error": None,
                    "skipped": True,
                }
                report["results"].append(result)
                print(f"SKIP {provider} {model} ({completed_before}/{args.turns})")
                write_report(run_dir, report)
                continue

            result = run_ardendo(
                provider,
                model,
                args.turns,
                progress_path,
                args.request_timeout,
                timeout_s=args.timeout,
                log_path=log_path,
            )
            result["skipped"] = False
            report["results"].append(result)
            write_report(run_dir, report)
            merge_progress(run_dir, args.turns)
            print(f"{'OK' if result['ok'] else 'FAIL'} {provider} {model} ({result['seconds']}s)")
            if not result["ok"]:
                failed.append(result)

    merged_path = merge_progress(run_dir, args.turns)
    merged = json.loads(merged_path.read_text(encoding="utf-8"))
    if any((model_data.get("data") or []) for model_data in (merged.get("models") or {}).values()):
        subprocess.run(
            [str(python), str(root / "viz.py"), "--progress-path", str(merged_path), "--out-dir", str(run_dir)],
            cwd=str(root),
            check=True,
        )

    print(run_dir)
    raise SystemExit(1 if failed else 0)

if args.command == "smoke":
    models_csv = Path(args.models_csv)
    if not models_csv.exists():
        print(f"Models CSV not found: {models_csv}", file=sys.stderr)
        raise SystemExit(2)

    rows = load_models(models_csv, args.provider)
    if not rows:
        print("No models found in models.csv", file=sys.stderr)
        raise SystemExit(2)

    run_name = args.run_name or time.strftime("smoke-%Y%m%d-%H%M%S")
    run_dir = Path(args.artifacts_dir) / "smoke" / run_name
    progress_dir = run_dir / "progress"
    progress_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "models_csv": str(models_csv),
        "provider": args.provider,
        "turns": args.turns,
        "timeout_s": args.timeout,
        "run_dir": str(run_dir),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "results": [],
    }

    for provider, model in rows:
        progress_path = progress_dir / f"progress_smoke_{provider}_{safe_name(model)}.json"
        if args.restart and progress_path.exists():
            progress_path.unlink()

        data_points, refusals = read_counts(progress_path, model)
        if data_points >= args.turns:
            result = {
                "provider": provider,
                "model": model,
                "progress_path": str(progress_path),
                "ok": True,
                "seconds": 0.0,
                "refusals": refusals,
                "data_points": data_points,
                "error": None,
            }
            report["results"].append(result)
            print(f"SKIP {provider} {model} ({data_points}/{args.turns})", file=sys.stderr)
            continue

        result = run_ardendo(
            provider,
            model,
            args.turns,
            progress_path,
            args.request_timeout,
            timeout_s=args.timeout,
        )
        report["results"].append(result)
        print(f"{'OK' if result['ok'] else 'FAIL'} {provider} {model} ({result['seconds']}s)", file=sys.stderr)

    write_report(run_dir, report, "smoke_report.json")
    lines = []
    for result in report["results"]:
        lines.append(
            f"{'OK' if result['ok'] else 'FAIL'}\t{result['provider']}\t{result['model']}\t"
            f"data={result['data_points']}\trefusals={result['refusals']}\tseconds={result['seconds']}\t{result['error'] or ''}"
        )
    (run_dir / "smoke_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    failed = [result for result in report["results"] if not result["ok"]]
    raise SystemExit(1 if failed else 0)

if args.command == "probe":
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY missing in environment", file=sys.stderr)
        raise SystemExit(2)

    models_csv = Path(args.models_csv)
    if not models_csv.exists():
        print(f"Models CSV not found: {models_csv}", file=sys.stderr)
        raise SystemExit(2)

    selected = [name for provider, name in load_models(models_csv, "openrouter")]
    response = requests.get(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60,
    )
    response.raise_for_status()
    data = response.json().get("data") or []
    by_id = {model.get("id"): model for model in data}

    rows = []
    for model_id in selected:
        model = by_id.get(model_id)
        if not model:
            rows.append({"id": model_id, "present": False})
            continue
        created = int(model.get("created") or 0)
        pricing = model.get("pricing") or {}
        supported = set(model.get("supported_parameters") or [])
        rows.append(
            {
                "id": model_id,
                "present": True,
                "created_utc": datetime.fromtimestamp(created, tz=timezone.utc).strftime("%Y-%m-%d"),
                "context_length": model.get("context_length"),
                "modality": (model.get("architecture") or {}).get("modality"),
                "reasoning_supported": bool(supported & {"reasoning", "include_reasoning", "reasoning_effort"}),
                "prompt_usd_per_1m": round(float(pricing.get("prompt") or 0.0) * 1e6, 6),
                "completion_usd_per_1m": round(float(pricing.get("completion") or 0.0) * 1e6, 6),
            }
        )

    for row in rows:
        if not row.get("present"):
            print(f"MISSING\t{row['id']}")
            continue
        print(
            f"OK\t{row['id']}\tcreated {row['created_utc']}\tctx {row['context_length']}\t"
            f"{row['prompt_usd_per_1m']}/{row['completion_usd_per_1m']} USD per 1M\t"
            f"reasoning_supported {str(row['reasoning_supported']).lower()}\t{row['modality']}"
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    raise SystemExit(0)

if args.command == "sync":
    models_csv = Path(args.models_csv)
    if not models_csv.exists():
        print(f"Models CSV not found: {models_csv}", file=sys.stderr)
        raise SystemExit(2)

    desired = [name for provider, name in load_models(models_csv, "ollama")]
    base_url = args.base_url.rstrip("/")
    response = requests.get(f"{base_url}/api/tags", timeout=30)
    response.raise_for_status()
    installed = {model.get("name") for model in (response.json().get("models") or [])}
    missing = [name for name in desired if name not in installed]

    print(f"Ollama base: {base_url}")
    print(f"Installed: {len(installed)}")
    print(f"Desired: {len(desired)}")
    print(f"Missing: {len(missing)}")

    if args.show:
        for name in desired:
            if name not in installed:
                continue
            response = requests.post(f"{base_url}/api/show", json={"model": name}, timeout=60)
            response.raise_for_status()
            data = response.json()
            details = data.get("details") or {}
            print()
            print(name)
            print(f"family: {details.get('family')}")
            print(f"parameter_size: {details.get('parameter_size')}")
            print(f"quantization_level: {details.get('quantization_level')}")
            print(f"template: {bool(data.get('template'))}")
            print(f"system: {bool(data.get('system'))}")

    if not missing:
        raise SystemExit(0)

    if args.dry_run or not args.install:
        print()
        print("Dry run, not installing. Missing models:")
        for name in missing:
            print(name)
        raise SystemExit(0)

    for name in missing:
        print()
        print(f"Pulling: {name}")
        with requests.post(
            f"{base_url}/api/pull",
            json={"model": name, "stream": True},
            stream=True,
            timeout=3600,
        ) as response:
            response.raise_for_status()
            last = time.time()
            for line in response.iter_lines():
                if not line:
                    continue
                message = json.loads(line.decode("utf-8"))
                status = message.get("status")
                completed = message.get("completed")
                total = message.get("total")
                if status and (time.time() - last) > 0.2:
                    if completed and total:
                        print(f"{status} {(completed / total) * 100:5.1f}%")
                    else:
                        print(status)
                    last = time.time()
                if message.get("error"):
                    raise RuntimeError(message["error"])
    raise SystemExit(0)

if args.command == "install-ssh":
    if not args.host:
        print("Missing host. Usage: python meta.py install-ssh monopol.haj", file=sys.stderr)
        raise SystemExit(2)

    models_csv = Path(args.models_csv)
    if not models_csv.exists():
        print(f"Models CSV not found: {models_csv}", file=sys.stderr)
        raise SystemExit(2)

    desired = [name for provider, name in load_models(models_csv, "ollama")]
    remote_script = (
        "set -euo pipefail; "
        f"base={shlex.quote(args.remote_base_url)}; "
        "echo \"OLLAMA_BASE_URL=$base\"; "
        "curl -fsS \"$base/api/tags\" >/dev/null; "
        "while IFS= read -r model; do "
        "echo; echo \"Pulling $model\"; "
        "curl -fsSL -N -H 'Content-Type: application/json' "
        "\"$base/api/pull\" "
        "-d \"{\\\"model\\\":\\\"$model\\\",\\\"stream\\\":true}\"; "
        "done"
    )
    cmd = ["ssh", *shlex.split(args.ssh_args), args.host, remote_script]
    raise SystemExit(
        subprocess.run(cmd, input="\n".join(desired) + "\n", text=True, cwd=str(root)).returncode
    )

if args.command == "viz":
    raise SystemExit(
        subprocess.run([str(python), str(root / "viz.py"), *args.args], cwd=str(root)).returncode
    )
