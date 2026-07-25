#!/usr/bin/env python3
"""
Per-model smoke test.

Loads one VLM via the project's loader registry, runs a single dummy
inference, and writes a JSON result file with status + any traceback.
Intended to be invoked under `srun` so it runs on a compute node with
a real GPU, but does not require a TTY.

Usage:
    python austin_smoke_test.py --model HF_PATH --output RESULT.json
    python austin_smoke_test.py --model HF_PATH --output RESULT.json --num-frames 8

Exit codes:
    0  success (model loaded and generated without raising)
    1  load stage failed
    2  generate stage failed
    3  unload stage failed
    4  argparse / env / registry issue
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path


def _snapshot_versions() -> dict:
    """Capture torch/transformers/autoawq versions and GPU info for the
    result JSON. None for packages that aren't installed."""
    versions: dict = {}
    for pkg in ("torch", "transformers", "autoawq", "timm", "accelerate"):
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[pkg] = None

    try:
        import torch
        versions["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            versions["cuda_device_name"] = torch.cuda.get_device_name(0)
            versions["cuda_vram_total_mb"] = int(
                torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            )
    except Exception:
        pass
    return versions


def _vram_peak_mb() -> float | None:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 / 1024
    except Exception:
        pass
    return None


def smoke_test_one(model_path: str, num_frames: int) -> dict:
    """Run the load -> generate -> unload sequence and return a result dict."""
    from PIL import Image

    result: dict = {
        "model_path": model_path,
        "num_frames": num_frames,
        "status": "unknown",
        "stage": None,
        "error": None,
        "response_snippet": None,
        "elapsed_seconds": {},
        "vram_peak_mb": None,
        "versions": _snapshot_versions(),
    }

    t0 = time.time()

    # ------------------------------------------------------------------
    # Stage 1: create loader
    # ------------------------------------------------------------------
    result["stage"] = "create_loader"
    try:
        # Add project root so the import path works regardless of cwd.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from prompt_generator.evaluation.model_loader import create_loader  # type: ignore
        loader = create_loader(model_path)
    except Exception:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        return result

    # ------------------------------------------------------------------
    # Stage 2: load (this is the expensive one — weight download + model.eval)
    # ------------------------------------------------------------------
    result["stage"] = "load"
    t_load_start = time.time()
    try:
        loader.load()
    except Exception:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        result["elapsed_seconds"]["load"] = round(time.time() - t_load_start, 2)
        return result
    result["elapsed_seconds"]["load"] = round(time.time() - t_load_start, 2)

    # ------------------------------------------------------------------
    # Stage 3: generate on dummy frames
    # ------------------------------------------------------------------
    result["stage"] = "generate"
    t_gen_start = time.time()
    try:
        frames = [Image.new("RGB", (448, 448), "red") for _ in range(num_frames)]
        response = loader.generate_response(
            frames,
            "Describe what you see in these frames in one sentence.",
        )
        result["response_snippet"] = (str(response) or "")[:300]
    except Exception:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        result["elapsed_seconds"]["generate"] = round(time.time() - t_gen_start, 2)
        try:
            loader.unload()
        except Exception:
            pass
        return result
    result["elapsed_seconds"]["generate"] = round(time.time() - t_gen_start, 2)

    # ------------------------------------------------------------------
    # Stage 4: unload
    # ------------------------------------------------------------------
    result["stage"] = "unload"
    t_unload_start = time.time()
    try:
        loader.unload()
    except Exception:
        # An unload failure still means load + generate worked, so we
        # report success overall but flag the unload error as a note.
        result["unload_error"] = traceback.format_exc()
    result["elapsed_seconds"]["unload"] = round(time.time() - t_unload_start, 2)

    result["vram_peak_mb"] = _vram_peak_mb()
    result["elapsed_seconds"]["total"] = round(time.time() - t0, 2)
    result["status"] = "passed"
    result["stage"] = "done"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Full HuggingFace model path")
    parser.add_argument("--output", required=True, help="Path to write result JSON")
    parser.add_argument("--num-frames", type=int, default=8,
                        help="Dummy frames to feed to the model (default 8)")
    args = parser.parse_args()

    try:
        result = smoke_test_one(args.model, args.num_frames)
    except Exception:  # noqa: BLE001 -- outermost catch, serialise unexpected crashes
        result = {
            "model_path": args.model,
            "status": "failed",
            "stage": "outer",
            "error": traceback.format_exc(),
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[smoke-test] wrote {out_path}", flush=True)
    print(f"[smoke-test] status={result.get('status')} stage={result.get('stage')}",
          flush=True)
    if result.get("response_snippet"):
        print(f"[smoke-test] response_snippet: {result['response_snippet']!r}", flush=True)
    if result.get("error"):
        # Print last 5 lines of traceback for a visible summary in the sbatch log.
        tb_tail = "\n".join(result["error"].splitlines()[-5:])
        print("[smoke-test] error tail:", flush=True)
        print(tb_tail, flush=True)

    # Exit code reflects stage where it failed (for bash visibility).
    mapping = {
        "passed": 0,
        "failed": {
            "create_loader": 4, "load": 1, "generate": 2, "unload": 3,
            "outer": 4,
        }.get(result.get("stage", ""), 1),
    }
    return mapping.get(result.get("status"), 1)


if __name__ == "__main__":
    raise SystemExit(main())
