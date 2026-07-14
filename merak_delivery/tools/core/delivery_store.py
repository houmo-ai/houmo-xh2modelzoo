from __future__ import annotations

from pathlib import Path

from .bindings import get_services


class MerakDeliveryStore:
    """Write and validate delivery metadata produced after model execution."""

    def collect_manifest(
        self,
        model_card: Path,
        work_dir: Path,
        export_dir: Path,
        golden_meta: Path,
        eval_report: Path | None,
        quanted_model_path: str,
    ) -> int:
        return get_services().run_collect_manifest(
            model_card, work_dir, export_dir, golden_meta, eval_report, quanted_model_path
        )

    def check_artifact(self, manifest: Path) -> int:
        return get_services().run_check_artifact(manifest)

    def register_release(
        self,
        manifest: Path,
        artifact_check: Path,
        eval_report: Path | None,
        release_root: Path,
    ) -> int:
        return get_services().run_register_release(
            manifest, artifact_check, eval_report, release_root
        )

    def build_catalog(self, release_root: Path, catalog_root: Path) -> int:
        return get_services().run_build_catalog(release_root, catalog_root)

    def render_readme(self, model_id: str, release_root: Path, catalog_root: Path) -> int:
        return get_services().run_render_readme(model_id, release_root, catalog_root)
