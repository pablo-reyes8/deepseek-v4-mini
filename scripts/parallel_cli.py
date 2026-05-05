from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import torch

from parallel.data_parallel import get_state_dict_for_save, wrap_ddp_model
from parallel.model_parallel import build_block_device_map, infer_auto_balance, wrap_model_parallel
from parallel.parallel_config import ParallelConfig
from parallel.parallel_utils import cleanup_distributed, setup_distributed
from scripts.inspect_cli import add_model_args, make_model


def print_json(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def _parse_csv_ints(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _parse_csv_strings(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


def cmd_plan(args: argparse.Namespace) -> None:
    devices = _parse_csv_strings(args.devices) or ["cpu"]
    balance = _parse_csv_ints(args.balance)
    block_devices = build_block_device_map(
        n_layers=args.n_layers,
        devices=devices,
        balance=balance,
    )
    print_json(
        {
            "mode": args.mode,
            "n_layers": args.n_layers,
            "devices": devices,
            "balance": balance or infer_auto_balance(args.n_layers, len(devices)),
            "block_device_map": [str(device) for device in block_devices],
            "notes": [
                "model_parallel is layerwise/blockwise placement, not tensor parallelism.",
                "ddp should be launched with torchrun for multi-process runs.",
            ],
        }
    )


def cmd_model_parallel_smoke(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    devices = _parse_csv_strings(args.devices) or ["cpu"]
    balance = _parse_csv_ints(args.balance)

    model = make_model(args)
    parallel_model = wrap_model_parallel(model, devices=devices, balance=balance)
    parallel_model.eval()

    input_ids = torch.randint(1, args.vocab_size, (args.batch_size, args.max_seq_len))
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100

    with torch.no_grad():
        outputs = parallel_model(input_ids=input_ids, labels=labels, return_aux=args.return_aux)

    print_json(
        {
            "status": "ok",
            "mode": "model_parallel",
            "devices": [str(device) for device in parallel_model.devices],
            "block_device_map": [str(device) for device in parallel_model.block_devices],
            "logits_shape": list(outputs["logits"].shape),
            "loss": float(outputs["loss"].item()) if outputs["loss"] is not None else None,
            "logits_finite": bool(torch.isfinite(outputs["logits"]).all().item()),
            "num_parameters": sum(p.numel() for p in parallel_model.parameters()),
        }
    )


def _default_init_method() -> str:
    if os.environ.get("RANK") is not None and os.environ.get("WORLD_SIZE") is not None:
        return "env://"
    tmpdir = tempfile.mkdtemp(prefix="deepseekv4_ddp_")
    return f"file://{os.path.join(tmpdir, 'init')}"


def cmd_ddp_smoke(args: argparse.Namespace) -> None:
    if not torch.distributed.is_available():
        raise SystemExit("torch.distributed is not available in this PyTorch build.")

    torch.manual_seed(args.seed)
    init_method = args.init_method or _default_init_method()
    config = ParallelConfig(
        mode="ddp",
        backend=args.backend,
        init_method=init_method,
        seed=args.seed,
        find_unused_parameters=args.find_unused_parameters,
        gradient_as_bucket_view=not args.no_gradient_as_bucket_view,
        broadcast_buffers=args.broadcast_buffers,
        static_graph=args.static_graph,
    )

    try:
        device = setup_distributed(config)
        model = make_model(args)
        model = wrap_ddp_model(model, config=config, device=device)
        model.train()

        input_ids = torch.randint(
            1,
            args.vocab_size,
            (args.batch_size, args.max_seq_len),
            device=device,
        )
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        outputs = model(input_ids=input_ids, labels=labels)
        outputs["loss"].backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        print_json(
            {
                "status": "ok",
                "mode": "ddp",
                "backend": args.backend,
                "init_method": init_method,
                "device": str(device),
                "world_size": torch.distributed.get_world_size(),
                "rank": torch.distributed.get_rank(),
                "loss": float(outputs["loss"].detach().cpu().item()),
                "state_dict_keys": len(get_state_dict_for_save(model)),
            }
        )
    finally:
        cleanup_distributed()


def cmd_tests(args: argparse.Namespace) -> None:
    cmd = [sys.executable, "-m", "pytest", "tests/parallel"]
    if args.quiet:
        cmd.append("-q")
    env = dict(os.environ)
    env.setdefault("TMPDIR", "/tmp")
    raise SystemExit(subprocess.call(cmd, env=env))


def add_ddp_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=["gloo", "nccl"], default="gloo")
    parser.add_argument("--init-method", default=None)
    parser.add_argument("--find-unused-parameters", action="store_true")
    parser.add_argument("--no-gradient-as-bucket-view", action="store_true")
    parser.add_argument("--broadcast-buffers", action="store_true")
    parser.add_argument("--static-graph", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=3e-4)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DeepSeek-V4 Mini parallelism CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Show a layer placement plan")
    plan.add_argument("--mode", choices=["ddp", "model_parallel", "hybrid"], default="model_parallel")
    plan.add_argument("--n-layers", type=int, default=4)
    plan.add_argument("--devices", default="cpu")
    plan.add_argument("--balance", default=None, help="Comma-separated layer counts, e.g. 2,2")
    plan.set_defaults(func=cmd_plan)

    mp_smoke = subparsers.add_parser("model-parallel-smoke", help="Run a model-parallel forward pass")
    add_model_args(mp_smoke)
    mp_smoke.add_argument("--devices", default="cpu")
    mp_smoke.add_argument("--balance", default=None)
    mp_smoke.add_argument("--seed", type=int, default=42)
    mp_smoke.add_argument("--return-aux", action="store_true")
    mp_smoke.set_defaults(func=cmd_model_parallel_smoke)

    ddp_smoke = subparsers.add_parser("ddp-smoke", help="Run a one-process DDP smoke test")
    add_model_args(ddp_smoke)
    add_ddp_args(ddp_smoke)
    ddp_smoke.add_argument("--seed", type=int, default=42)
    ddp_smoke.set_defaults(func=cmd_ddp_smoke)

    tests = subparsers.add_parser("tests", help="Run CPU-safe parallelism tests")
    tests.add_argument("--quiet", action="store_true")
    tests.set_defaults(func=cmd_tests)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
