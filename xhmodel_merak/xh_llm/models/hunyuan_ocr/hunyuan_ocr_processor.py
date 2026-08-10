# Copyright 2025 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from PIL import Image

from xhquant.api import get_xhquant_logger

from .resolution_bucket_routing import route_approved_image


class HunyuanOCRMultiBucketProcessor:
    """Apply the approved routing contract before HF preprocessing."""

    def __init__(self, processor: Any, manifest: Mapping[str, Any]) -> None:
        if manifest.get("status") != "approved":
            raise ValueError("HunyuanOCR multi-bucket processor requires an approved manifest")
        self.max_images_per_request = int(manifest.get("input_policy", {}).get("max_images_per_request", 0))
        if self.max_images_per_request not in (1, 3):
            raise ValueError(
                "HunyuanOCR requires max_images_per_request to be 1 or 3, "
                f"got {self.max_images_per_request}"
            )
        self.processor = processor
        self.manifest = copy.deepcopy(dict(manifest))
        self.last_route: dict[str, Any] | None = None
        self.last_routes: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.processor, name)

    def apply_chat_template(self, conversation, chat_template=None, **kwargs):
        routed_conversation, routes = self._route_conversation(conversation)
        kwargs.setdefault("do_resize", False)
        result = self.processor.apply_chat_template(routed_conversation, chat_template=chat_template, **kwargs)
        image_grid_thw = result.get("image_grid_thw") if isinstance(result, Mapping) else None
        if image_grid_thw is not None:
            expected_grid = [self._bucket_by_id(route["bucket_id"])["image_grid_thw"] for route in routes]
            actual_grid = image_grid_thw.tolist()
            if actual_grid != expected_grid:
                mismatch_index = next(
                    (
                        index
                        for index in range(max(len(expected_grid), len(actual_grid)))
                        if index >= len(expected_grid)
                        or index >= len(actual_grid)
                        or actual_grid[index] != expected_grid[index]
                    ),
                    0,
                )
                raise RuntimeError(
                    "HunyuanOCR processor grid does not match selected bucket: "
                    f"image_index={mismatch_index}, expected={expected_grid}, got={actual_grid}"
                )
        self.last_routes = copy.deepcopy(routes)
        self.last_route = copy.deepcopy(routes[0]) if len(routes) == 1 else None
        for route in routes:
            get_xhquant_logger().info(
                "HunyuanOCR visual route: "
                f"image_index={route['image_index']}, source_size={route['source_size']}, "
                f"bucket_id={route['bucket_id']}, bucket_size={route['bucket_size']}, "
                f"geometry={route['geometry']}, visual_token_count={route['image_token_count']}"
            )
        return result

    def _route_conversation(self, conversation) -> tuple[Any, list[dict[str, Any]]]:
        routed = copy.deepcopy(conversation)
        image_items = []
        for message in routed:
            content = message.get("content", []) if isinstance(message, Mapping) else []
            for item in content:
                if isinstance(item, Mapping) and item.get("type") == "image":
                    image_items.append(item)
        if len(image_items) > self.max_images_per_request:
            raise ValueError(
                "HunyuanOCR image count exceeds max_images_per_request: "
                f"images={len(image_items)}, max_images_per_request={self.max_images_per_request}, "
                f"first_rejected_image_index={self.max_images_per_request}"
            )

        routes = []
        for image_index, image_item in enumerate(image_items):
            image = image_item.get("image")
            if isinstance(image, Image.Image):
                source = image.convert("RGB")
            elif isinstance(image, (str, bytes)):
                with Image.open(image) as opened:
                    source = opened.convert("RGB")
            else:
                raise TypeError(
                    f"HunyuanOCR image_index={image_index} must be a PIL image or path, "
                    f"got {type(image).__name__}"
                )

            route = self.route_image(source.width, source.height)
            route["image_index"] = image_index
            if not route["accepted"]:
                raise ValueError(
                    "HunyuanOCR image is unsupported by the approved bucket contract: "
                    f"image_index={image_index}, reason={route['reason']}, "
                    f"source_size={route['source_size']}, candidates={route['candidates']}"
                )
            image_item["image"] = self._letterbox(source, route)
            routes.append(route)
        return routed, routes

    def route_image(self, source_width: int, source_height: int) -> dict[str, Any]:
        return route_approved_image(source_width, source_height, self.manifest)

    def _bucket_by_id(self, bucket_id: str) -> Mapping[str, Any]:
        return next(bucket for bucket in self.manifest["buckets"] if bucket["id"] == bucket_id)

    def _letterbox(self, image: Image.Image, route: Mapping[str, Any]) -> Image.Image:
        geometry = route["geometry"]
        resized_width, resized_height = geometry["resized_size"]
        if image.size != (resized_width, resized_height):
            image = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
        padding_color = tuple(self.manifest["input_policy"]["padding_color_rgb"])
        bucket_width, bucket_height = route["bucket_size"]
        canvas = Image.new("RGB", (bucket_width, bucket_height), padding_color)
        left, top, _, _ = geometry["padding_ltrb"]
        canvas.paste(image, (left, top))
        return canvas


__all__ = ["HunyuanOCRMultiBucketProcessor"]
