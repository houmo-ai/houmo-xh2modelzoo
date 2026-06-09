"""Entry point: python -m hm_eval [--host HOST] [--port PORT] [--cli]"""

from __future__ import annotations

import argparse
import logging


def main() -> None:
    parser = argparse.ArgumentParser(description="HM-Eval 统一评测平台")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--cli", action="store_true", help="Run CLI evaluation (no Gradio)")
    parser.add_argument("--model", type=str, help="[CLI] Model display name or config_id")
    parser.add_argument("--backend", type=str, default="float", help="[CLI] Backend: float or hmonnx")
    parser.add_argument("--datasets", nargs="+", help="[CLI] Dataset names")
    parser.add_argument("--limit", type=int, default=0, help="[CLI] Per-subset sample limit (0=all)")
    parser.add_argument("--max-tokens", type=int, default=512, help="[CLI] Max generation tokens")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.cli:
        _run_cli(args)
    else:
        _run_gradio(args)


def _run_gradio(args) -> None:
    from hm_eval.app import create_app, get_gradio_launch_kwargs
    app = create_app()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        **get_gradio_launch_kwargs(),
    )


def _run_cli(args) -> None:
    """Run a single evaluation from the command line."""
    if not args.model:
        print("Error: --model is required in CLI mode")
        return
    if not args.datasets:
        print("Error: --datasets is required in CLI mode")
        return

    from hm_eval.core.model_registry import ModelRegistry
    from hm_eval.core.dataset_registry import DatasetRegistry
    from hm_eval.core.backends import create_backend
    from hm_eval.core.eval_runner import run_evaluation
    from hm_eval.core.report import generate_report, format_report_text, save_report

    registry = ModelRegistry()
    registry.scan()

    model = registry.get_model(args.model) or registry.get_model_by_display_name(args.model)
    if model is None:
        print(f"Error: Model not found: {args.model}")
        print(f"Available: {[m.display_name for m in registry.list_models()]}")
        return

    print(f"Loading {model.display_name} with {args.backend} backend...")
    backend = create_backend(args.backend, model)

    ds_registry = DatasetRegistry()
    import time
    work_dir = f"hm_eval/outputs/{model.config_id}_{args.backend}_{int(time.time())}"

    print(f"Running evaluation: datasets={args.datasets}, limit={args.limit}")
    results = run_evaluation(
        backend=backend,
        model_display_name=model.display_name,
        datasets=args.datasets,
        work_dir=work_dir,
        dataset_registry=ds_registry,
        limit=args.limit,
        max_tokens=args.max_tokens,
    )

    report = generate_report(results, work_dir)
    report_path = save_report(report, work_dir)
    print(format_report_text(report))
    print(f"\nReport saved to: {report_path}")

    backend.cleanup()


if __name__ == "__main__":
    main()
