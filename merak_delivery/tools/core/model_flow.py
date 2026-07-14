from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .bindings import get_services
from .delivery_store import MerakDeliveryStore
from .evaluator import MerakEvaluator


class MerakModelFlow:
    """Own the complete quant-to-release lifecycle for one Merak model card."""

    def __init__(
        self,
        args: argparse.Namespace,
        *,
        card: dict[str, Any] | None = None,
        model_card_path: Path | None = None,
    ) -> None:
        services = get_services()
        self.args = args
        self.model_card_path = model_card_path or services.resolve_path(args.model_card)
        self.card = card or services.load_model_card(self.model_card_path)
        self.model_id = self.card["model"]["id"]
        self.version_id = self.card["release"]["version_id"]
        self.work_dir = services.resolve_path(args.work_dir) if args.work_dir else services.default_work_dir(self.card)
        self.quant_dir = (
            services.resolve_path(args.quant_output_dir)
            if args.quant_output_dir
            else services.default_quant_dir(self.work_dir)
        )
        self.export_dir = (
            services.resolve_path(args.export_output_dir)
            if args.export_output_dir
            else services.default_export_dir(self.work_dir)
        )
        self.golden_meta = (
            services.resolve_path(args.golden_meta)
            if args.golden_meta
            else services.find_golden_meta(self.export_dir)
        )
        self.eval_report = services.resolve_path(args.eval_report) if args.eval_report else None
        self.eval_output = (
            services.resolve_path(args.eval_output)
            if args.eval_output
            else services.default_eval_report(self.work_dir, getattr(self.args, "eval_backend", "hmonnx"))
        )
        self.eval_work_dir = (
            services.resolve_path(args.eval_work_dir)
            if args.eval_work_dir
            else services.default_eval_work_dir(self.work_dir)
        )
        self.release_root = (
            services.resolve_path(args.release_root) if args.release_root else services.default_release_root()
        )
        self.catalog_root = (
            services.resolve_path(args.catalog_root) if args.catalog_root else services.default_catalog_root()
        )
        self.store = MerakDeliveryStore()

    def plan(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "model_id": self.model_id,
            "version_id": self.version_id,
            "dry_run": bool(self.args.dry_run),
            "workflow": {
                "class": self.card["workflow"]["class"],
                "config_path": self.card["workflow"]["config_path"],
                "model_dir": self.args.model_dir or self.card["workflow"]["model_dir"],
            },
            "planned_steps": self.card["workflow"]["actions"],
            "outputs": {
                "work_dir": str(self.work_dir),
                "quant_dir": str(self.quant_dir),
                "export_dir": str(self.export_dir),
                "golden_meta": str(self.golden_meta),
                "eval_report": str(self.eval_report or self.eval_output),
                "eval_work_dir": str(self.eval_work_dir),
                "manifest": str(self.work_dir / "delivery_manifest.json"),
                "artifact_check": str(self.work_dir / "artifact_check.json"),
                "release_root": str(self.release_root),
                "catalog_root": str(self.catalog_root),
            },
        }

    def run(self) -> int:
        if self.args.dry_run:
            print(json.dumps(self.plan(), indent=2, ensure_ascii=False))
            return 0

        from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow

        services = get_services()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        workflow = AutoLLMWorkflow.from_config(
            model_dir=self.args.model_dir or self.card["workflow"]["model_dir"],
            config_path=self.card["workflow"]["config_path"],
            seed=self.card["runtime"]["seed"],
            debug=self.args.debug,
        )
        config_overrides: dict[str, Any] = {}
        if self.args.bits is not None:
            config_overrides["quant.bits"] = self.args.bits
        if self.args.skip_quant:
            config_overrides = {"quant": None}

        quant_result = workflow.quant(
            output_dir=str(self.quant_dir),
            device=self.args.device or self.card["runtime"]["device"],
            config_overrides=config_overrides,
        )
        export_result = workflow.export(
            quant_result=quant_result,
            output_dir=str(self.export_dir),
            device=self.args.device or self.card["runtime"]["device"],
        )
        if self.args.dump_golden:
            workflow.dump_golden(
                export_result=export_result,
                device=self.args.device or self.card["runtime"]["device"],
                input_messages={"text": self.args.golden_prompt},
            )

        self.golden_meta = (
            services.resolve_path(self.args.golden_meta)
            if self.args.golden_meta
            else services.find_golden_meta(self.export_dir)
        )
        eval_status = self._run_evaluation() if self.args.run_eval else 0
        self.store.collect_manifest(
            self.model_card_path,
            self.work_dir,
            Path(getattr(export_result, "work_dir", self.export_dir)),
            self.golden_meta,
            self.eval_report,
            services.serialize_result_path(quant_result, self.quant_dir),
        )
        manifest = self.work_dir / "delivery_manifest.json"
        artifact_check = self.work_dir / "artifact_check.json"
        artifact_status = self.store.check_artifact(manifest)
        if artifact_status != 0:
            return artifact_status
        if self.args.register_release:
            self.store.register_release(manifest, artifact_check, self.eval_report, self.release_root)
        if self.args.build_catalog:
            self.store.build_catalog(self.release_root, self.catalog_root)
        if self.args.render_readme:
            self.store.render_readme(self.model_id, self.release_root, self.catalog_root)
        return eval_status

    def _run_evaluation(self) -> int:
        services = get_services()
        self.eval_report = self.eval_output
        return MerakEvaluator(
            card=self.card,
            model_card_path=self.model_card_path,
            work_dir=self.work_dir,
            output=self.eval_output,
            eval_work_dir=self.eval_work_dir,
            model_id=services.infer_eval_model_id(self.card, self.args.eval_model),
            backend=self.args.eval_backend,
            datasets=services.infer_eval_datasets(self.card, self.args.eval_datasets),
            dataset_hub=getattr(self.args, "eval_dataset_hub", "modelscope"),
            limit=self.args.eval_limit,
            max_tokens=self.args.eval_max_tokens,
            hmonnx_meta=(
                services.resolve_path(self.args.hmonnx_meta)
                if self.args.hmonnx_meta
                else self.golden_meta
            ),
            vision_hmonnx_meta=(
                services.resolve_path(self.args.vision_hmonnx_meta)
                if self.args.vision_hmonnx_meta
                else None
            ),
        ).run()
