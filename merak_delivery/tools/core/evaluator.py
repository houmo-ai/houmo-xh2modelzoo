from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .bindings import get_services


class MerakEvaluator:
    """Run hm_eval and translate its result into the delivery report contract."""

    def __init__(
        self,
        *,
        card: dict[str, Any],
        model_card_path: Path,
        work_dir: Path,
        output: Path,
        eval_work_dir: Path,
        model_id: str,
        backend: str,
        datasets: list[str],
        dataset_hub: str,
        limit: int,
        max_tokens: int,
        hmonnx_meta: Path | None = None,
        vision_hmonnx_meta: Path | None = None,
    ) -> None:
        self.card = card
        self.model_card_path = model_card_path
        self.work_dir = work_dir
        self.output = output
        self.eval_work_dir = eval_work_dir
        self.model_id = model_id
        self.backend = backend
        self.datasets = datasets
        self.dataset_hub = dataset_hub
        self.limit = limit
        self.max_tokens = max_tokens
        self.hmonnx_meta = hmonnx_meta
        self.vision_hmonnx_meta = vision_hmonnx_meta

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "MerakEvaluator":
        services = get_services()
        model_card_path = services.resolve_path(args.model_card)
        card = services.load_model_card(model_card_path)
        work_dir = services.resolve_path(args.work_dir) if args.work_dir else services.default_work_dir(card)
        return cls(
            card=card,
            model_card_path=model_card_path,
            work_dir=work_dir,
            output=(
                services.resolve_path(args.output)
                if args.output
                else services.default_eval_report(work_dir, args.eval_backend)
            ),
            eval_work_dir=(
                services.resolve_path(args.eval_work_dir)
                if args.eval_work_dir
                else services.default_eval_work_dir(work_dir)
            ),
            model_id=services.infer_eval_model_id(card, args.eval_model),
            backend=args.eval_backend,
            datasets=services.infer_eval_datasets(card, args.eval_datasets),
            dataset_hub=getattr(args, "eval_dataset_hub", "modelscope"),
            limit=args.eval_limit,
            max_tokens=args.eval_max_tokens,
            hmonnx_meta=services.resolve_path(args.hmonnx_meta) if args.hmonnx_meta else None,
            vision_hmonnx_meta=(
                services.resolve_path(args.vision_hmonnx_meta) if args.vision_hmonnx_meta else None
            ),
        )

    def execute(self) -> dict[str, Any]:
        services = get_services()
        return services.run_hm_eval(
            model_id=self.model_id,
            backend_type=self.backend,
            datasets=self.datasets,
            dataset_hub=self.dataset_hub,
            work_dir=self.eval_work_dir,
            limit=self.limit,
            max_tokens=self.max_tokens,
            hmonnx_meta=self.hmonnx_meta,
            vision_hmonnx_meta=self.vision_hmonnx_meta,
        )

    def run(self) -> int:
        services = get_services()
        results = self.execute()
        tasks = services.dataset_statuses(results)
        report = {
            "schema_version": 1,
            "model_id": self.card["model"]["id"],
            "version_id": self.card["release"]["version_id"],
            "status": services.derive_eval_report_status(tasks),
            "tasks": tasks,
            "logs": {
                "hm_eval_work_dir": str(self.eval_work_dir),
                "model": self.model_id,
                "backend": self.backend,
                "datasets": self.datasets,
                "dataset_hub": self.dataset_hub,
                "hmonnx_meta": str(self.hmonnx_meta) if self.hmonnx_meta else "",
                "vision_hmonnx_meta": str(self.vision_hmonnx_meta) if self.vision_hmonnx_meta else "",
                "raw_results": results,
            },
        }
        services.write_json(self.output, report)
        print(f"Wrote {services.relative_or_str(self.output)}")
        return 0 if report["status"] == "passed" else 1
