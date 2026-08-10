#!/usr/bin/env python3
"""Build the complete persisted FP-versus-HMONNX LIBERO rollout video."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any


TASK_INSTRUCTIONS = {
    0: "Pick up the alphabet soup and place it in the basket",
    1: "Pick up the cream cheese and place it in the basket",
    2: "Pick up the salad dressing and place it in the basket",
    3: "Pick up the BBQ sauce and place it in the basket",
    4: "Pick up the ketchup and place it in the basket",
    5: "Pick up the tomato sauce and place it in the basket",
    6: "Pick up the butter and place it in the basket",
    7: "Pick up the milk and place it in the basket",
    8: "Pick up the chocolate pudding and place it in the basket",
    9: "Pick up the orange juice and place it in the basket",
}

BACKGROUND = "0x08111f"
PANEL = "0x101d2f"
TEXT = "0xedf4ff"
MUTED = "0x9eb0c7"
FP_COLOR = "0x42d3b2"
HMONNX_COLOR = "0xffad66"
SUCCESS_COLOR = "0x61d69b"
FAILURE_COLOR = "0xff6b7a"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _probe_video(path: Path, *, include_hash: bool = False) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,nb_frames,duration",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"{path}: expected exactly one video stream")
    stream = streams[0]
    duration = stream.get("duration") or payload.get("format", {}).get("duration")
    if duration is None:
        raise ValueError(f"{path}: missing duration")
    info: dict[str, Any] = {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "codec_name": stream.get("codec_name"),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "pix_fmt": stream.get("pix_fmt"),
        "r_frame_rate": stream.get("r_frame_rate"),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "nb_frames": int(stream["nb_frames"]) if stream.get("nb_frames") not in {None, "N/A"} else None,
        "duration_s": float(duration),
    }
    if include_hash:
        info["sha256"] = _sha256(path)
    return info


def _load_backend(report: dict[str, Any], backend: str) -> dict[int, dict[str, Any]]:
    tasks: dict[int, dict[str, Any]] = {}
    for source in report["sources"][backend]:
        eval_path = Path(source["path"]).expanduser().resolve()
        if not eval_path.is_file():
            raise FileNotFoundError(eval_path)
        actual_hash = _sha256(eval_path)
        if actual_hash != source["sha256"]:
            raise ValueError(
                f"{eval_path}: SHA-256 changed since the accuracy report ({actual_hash} != {source['sha256']})"
            )
        with eval_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        aggregated = payload.get("aggregated", payload)
        for record in aggregated["per_task"]:
            task_id = int(record["task_id"])
            if task_id in tasks:
                raise ValueError(f"{backend}: duplicate task_id {task_id}")
            metrics = record["metrics"]
            successes = metrics["successes"]
            videos: dict[int, Path] = {}
            for video_path_value in metrics.get("video_paths", []):
                video_path = Path(video_path_value).expanduser().resolve()
                stem = video_path.stem
                prefix = "eval_episode_"
                if not stem.startswith(prefix) or not stem[len(prefix) :].isdigit():
                    raise ValueError(f"{video_path}: cannot determine episode ID")
                episode_id = int(stem[len(prefix) :])
                if episode_id in videos:
                    raise ValueError(f"{backend} task {task_id}: duplicate episode {episode_id}")
                if episode_id >= len(successes):
                    raise ValueError(f"{video_path}: episode ID exceeds success metadata")
                if not video_path.is_file():
                    raise FileNotFoundError(video_path)
                videos[episode_id] = video_path
            tasks[task_id] = {
                "successes": successes,
                "videos": videos,
                "eval_info": str(eval_path),
            }
    return tasks


def _escape_drawtext(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")


def _drawtext(
    text: str,
    *,
    font_file: Path,
    x: str,
    y: str,
    size: int,
    color: str = TEXT,
    align: str | None = None,
) -> str:
    options = [
        f"fontfile='{_escape_drawtext(str(font_file))}'",
        f"text='{_escape_drawtext(text)}'",
        f"x={x}",
        f"y={y}",
        f"fontsize={size}",
        f"fontcolor={color}",
        "expansion=none",
    ]
    if align is not None:
        options.append(f"text_align={align}")
    return "drawtext=" + ":".join(options)


def _encoding_args(args: argparse.Namespace) -> list[str]:
    return [
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        args.preset,
        "-crf",
        str(args.crf),
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(args.fps),
        "-g",
        str(args.fps),
        "-keyint_min",
        str(args.fps),
        "-sc_threshold",
        "0",
    ]


def _reuse_segment(path: Path, expected_frames: int, args: argparse.Namespace) -> bool:
    if not args.reuse_segments or not path.is_file():
        return False
    info = _probe_video(path)
    if (
        info["width"] != args.width
        or info["height"] != args.height
        or Fraction(info["r_frame_rate"]) != args.fps
        or info["nb_frames"] != expected_frames
    ):
        raise ValueError(f"cannot reuse invalid segment: {info}")
    return True


def _write_card(
    path: Path,
    lines: list[tuple[str, int, str]],
    *,
    duration_s: float,
    args: argparse.Namespace,
) -> int:
    frame_count = round(duration_s * args.fps)
    if _reuse_segment(path, frame_count, args):
        return frame_count
    filters = [f"drawbox=x=0:y=0:w={args.width}:h={args.height}:color={BACKGROUND}:t=fill"]
    total_height = sum(size + 18 for _, size, _ in lines) - 18
    current_y = (args.height - total_height) // 2
    for text, size, color in lines:
        filters.append(
            _drawtext(
                text,
                font_file=args.font_file,
                x="(w-text_w)/2",
                y=str(current_y),
                size=size,
                color=color,
            )
        )
        current_y += size + 18
    _run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c={BACKGROUND}:s={args.width}x{args.height}:r={args.fps}",
            "-vf",
            ",".join(filters),
            "-frames:v",
            str(frame_count),
            *_encoding_args(args),
            str(path),
        ]
    )
    return frame_count


def _write_pair(
    path: Path,
    pair: dict[str, Any],
    *,
    args: argparse.Namespace,
) -> int:
    fp_info = pair["fp"]
    hmonnx_info = pair["hmonnx"]
    fp_frames = round(fp_info["duration_s"] * args.fps)
    hmonnx_frames = round(hmonnx_info["duration_s"] * args.fps)
    target_frames = max(fp_frames, hmonnx_frames)
    if _reuse_segment(path, target_frames, args):
        return target_frames
    fp_pad = target_frames - fp_frames
    hmonnx_pad = target_frames - hmonnx_frames
    half_width = args.width // 2
    video_height = args.height - args.header_height
    fp_status = "SUCCESS" if pair["fp_success"] else "FAILURE"
    hmonnx_status = "SUCCESS" if pair["hmonnx_success"] else "FAILURE"
    fp_status_color = SUCCESS_COLOR if pair["fp_success"] else FAILURE_COLOR
    hmonnx_status_color = SUCCESS_COLOR if pair["hmonnx_success"] else FAILURE_COLOR
    filters = [
        (
            f"[0:v]fps={args.fps},"
            f"scale={half_width}:{video_height}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={half_width}:{video_height}:(ow-iw)/2:(oh-ih)/2:color={BACKGROUND},"
            f"setpts=N/({args.fps}*TB),tpad=stop_mode=clone:stop={fp_pad},"
            f"trim=end_frame={target_frames}[fp]"
        ),
        (
            f"[1:v]fps={args.fps},"
            f"scale={half_width}:{video_height}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={half_width}:{video_height}:(ow-iw)/2:(oh-ih)/2:color={BACKGROUND},"
            f"setpts=N/({args.fps}*TB),tpad=stop_mode=clone:stop={hmonnx_pad},"
            f"trim=end_frame={target_frames}[hmonnx]"
        ),
        "[fp][hmonnx]hstack=inputs=2[rollouts]",
        (
            f"[rollouts]pad={args.width}:{args.height}:0:{args.header_height}:color={BACKGROUND},"
            f"drawbox=x=0:y=0:w={args.width}:h={args.header_height}:color={PANEL}:t=fill,"
            f"drawbox=x={half_width - 1}:y={args.header_height}:w=2:h={video_height}:color={PANEL}:t=fill,"
            + _drawtext(
                f"FP | {fp_status}",
                font_file=args.font_file,
                x="20",
                y="16",
                size=26,
                color=fp_status_color,
            )
            + ","
            + _drawtext(
                f"TASK {pair['task_id']} | EPISODE {pair['episode_id']:02d}",
                font_file=args.font_file,
                x="(w-text_w)/2",
                y="17",
                size=24,
                color=TEXT,
            )
            + ","
            + _drawtext(
                f"HMONNX | {hmonnx_status}",
                font_file=args.font_file,
                x="w-text_w-20",
                y="16",
                size=26,
                color=hmonnx_status_color,
            )
            + "[out]"
        ),
    ]
    _run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            fp_info["path"],
            "-i",
            hmonnx_info["path"],
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
            "-frames:v",
            str(target_frames),
            *_encoding_args(args),
            str(path),
        ]
    )
    return target_frames


def _concat_segments(paths: list[Path], output: Path, concat_file: Path) -> None:
    lines = []
    for path in paths:
        escaped = str(path.resolve()).replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    concat_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )


def _score_text(row: dict[str, Any], backend: str) -> str:
    successes = int(row[f"{backend}_successes"])
    episodes = int(row["episodes"])
    rate = float(row[f"{backend}_success_rate_pct"])
    return f"{successes}/{episodes} ({rate:.1f}%)"


def _collect_pairs(
    report: dict[str, Any],
    fp_tasks: dict[int, dict[str, Any]],
    hmonnx_tasks: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expected_task_ids = [int(value) for value in report["protocol"]["task_ids"]]
    if set(fp_tasks) != set(expected_task_ids) or set(hmonnx_tasks) != set(expected_task_ids):
        raise ValueError(
            f"task IDs differ: expected={expected_task_ids}, FP={sorted(fp_tasks)}, HMONNX={sorted(hmonnx_tasks)}"
        )
    score_rows = {int(row["task_id"]): row for row in report["per_task"]}
    pairs: list[dict[str, Any]] = []
    task_coverage: list[dict[str, Any]] = []
    for task_id in expected_task_ids:
        fp_task = fp_tasks[task_id]
        hmonnx_task = hmonnx_tasks[task_id]
        fp_count = sum(bool(value) for value in fp_task["successes"])
        hmonnx_count = sum(bool(value) for value in hmonnx_task["successes"])
        score_row = score_rows[task_id]
        if fp_count != int(score_row["fp_successes"]) or hmonnx_count != int(score_row["hmonnx_successes"]):
            raise ValueError(f"task {task_id}: success metadata differs from the accuracy report")
        fp_episode_ids = set(fp_task["videos"])
        hmonnx_episode_ids = set(hmonnx_task["videos"])
        if fp_episode_ids != hmonnx_episode_ids:
            raise ValueError(
                f"task {task_id}: persisted episodes differ: "
                f"FP={sorted(fp_episode_ids)}, HMONNX={sorted(hmonnx_episode_ids)}"
            )
        if not fp_episode_ids:
            raise ValueError(f"task {task_id}: no persisted rollout videos")
        for episode_id in sorted(fp_episode_ids):
            fp_path = fp_task["videos"][episode_id]
            hmonnx_path = hmonnx_task["videos"][episode_id]
            pairs.append(
                {
                    "task_id": task_id,
                    "episode_id": episode_id,
                    "fp_success": bool(fp_task["successes"][episode_id]),
                    "hmonnx_success": bool(hmonnx_task["successes"][episode_id]),
                    "fp": _probe_video(fp_path, include_hash=True),
                    "hmonnx": _probe_video(hmonnx_path, include_hash=True),
                }
            )
        task_coverage.append(
            {
                "task_id": task_id,
                "instruction": TASK_INSTRUCTIONS[task_id],
                "scored_episodes_per_runtime": int(score_row["episodes"]),
                "fp_score": _score_text(score_row, "fp"),
                "hmonnx_score": _score_text(score_row, "hmonnx"),
                "persisted_paired_episode_ids": sorted(fp_episode_ids),
                "persisted_paired_rollouts": len(fp_episode_ids),
            }
        )
    return pairs, task_coverage


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument(
        "--font-file",
        type=Path,
        default=Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--header-height", type=int, default=64)
    parser.add_argument("--fps", type=int, default=80)
    parser.add_argument("--crf", type=int, default=21)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--keep-work-dir", action="store_true")
    parser.add_argument(
        "--reuse-segments",
        action="store_true",
        help="Reuse existing work-dir segments after validating their shape, FPS, and frame count.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.report_json = args.report_json.expanduser().resolve()
    args.output_video = args.output_video.expanduser().resolve()
    args.output_manifest = (
        args.output_manifest.expanduser().resolve() if args.output_manifest else args.output_video.with_suffix(".json")
    )
    args.work_dir = (
        args.work_dir.expanduser().resolve()
        if args.work_dir
        else args.output_video.parent / f".{args.output_video.stem}_build"
    )
    args.font_file = args.font_file.expanduser().resolve()
    if not args.report_json.is_file():
        raise FileNotFoundError(args.report_json)
    if not args.font_file.is_file():
        raise FileNotFoundError(args.font_file)
    if args.width % 2 or args.height % 2 or args.width < 640 or args.height < 360:
        raise ValueError("width and height must be even and at least 640x360")
    if args.header_height <= 0 or args.header_height >= args.height:
        raise ValueError("header-height must be between zero and height")
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    for output_path in (args.output_video, args.output_manifest):
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"{output_path} exists; pass --overwrite")
        output_path.parent.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    with args.report_json.open(encoding="utf-8") as handle:
        report = json.load(handle)
    fp_tasks = _load_backend(report, "fp")
    hmonnx_tasks = _load_backend(report, "hmonnx")
    pairs, task_coverage = _collect_pairs(report, fp_tasks, hmonnx_tasks)
    expected_pair_count = sum(item["persisted_paired_rollouts"] for item in task_coverage)
    if len(pairs) != expected_pair_count:
        raise AssertionError("internal paired-rollout count mismatch")

    overall = report["overall"]
    segments: list[Path] = []
    total_frames = 0
    intro = args.work_dir / "000_intro.mp4"
    total_frames += _write_card(
        intro,
        [
            ("PI0.5 LIBERO OBJECT", 46, TEXT),
            ("FP REFERENCE VS HMONNX W8A8", 32, HMONNX_COLOR),
            ("COMPLETE PERSISTED ROLLOUT COMPARISON", 27, MUTED),
            (f"{len(pairs)} PAIRED VIDEOS | 10 TASKS | EPISODES 0-9", 25, TEXT),
            (
                f"FP {overall['fp_successes']}/200 ({overall['fp_success_rate_pct']:.1f}%) | "
                f"HMONNX {overall['hmonnx_successes']}/200 ({overall['hmonnx_success_rate_pct']:.1f}%) | "
                f"DELTA {overall['delta_pp']:+.1f} PP",
                24,
                FP_COLOR,
            ),
            ("ACCURACY PROTOCOL: 20 EPISODES PER TASK, 200 PER RUNTIME", 20, MUTED),
        ],
        duration_s=4.0,
        args=args,
    )
    segments.append(intro)

    pair_index = 0
    score_rows = {int(row["task_id"]): row for row in report["per_task"]}
    for task in task_coverage:
        task_id = task["task_id"]
        score_row = score_rows[task_id]
        task_card = args.work_dir / f"task_{task_id:02d}_card.mp4"
        total_frames += _write_card(
            task_card,
            [
                (f"TASK {task_id}", 48, TEXT),
                (task["instruction"].upper(), 28, MUTED),
                ("20-EPISODE CLOSED-LOOP SCORE", 20, MUTED),
                (
                    f"FP {_score_text(score_row, 'fp')} | HMONNX {_score_text(score_row, 'hmonnx')}",
                    30,
                    HMONNX_COLOR,
                ),
                (
                    f"VIDEO COVERAGE: {task['persisted_paired_rollouts']} PAIRED ROLLOUTS "
                    f"(EPISODES {task['persisted_paired_episode_ids'][0]}-"
                    f"{task['persisted_paired_episode_ids'][-1]})",
                    20,
                    MUTED,
                ),
            ],
            duration_s=2.0,
            args=args,
        )
        segments.append(task_card)
        task_pairs = [pair for pair in pairs if pair["task_id"] == task_id]
        for pair in task_pairs:
            pair_index += 1
            print(f"[{pair_index:03d}/{len(pairs):03d}] task {task_id} episode {pair['episode_id']:02d}")
            pair_path = args.work_dir / f"task_{task_id:02d}_episode_{pair['episode_id']:02d}.mp4"
            total_frames += _write_pair(pair_path, pair, args=args)
            segments.append(pair_path)

    outro = args.work_dir / "999_outro.mp4"
    total_frames += _write_card(
        outro,
        [
            ("FINAL CLOSED-LOOP RESULT", 44, TEXT),
            (f"FP {overall['fp_successes']}/200 ({overall['fp_success_rate_pct']:.1f}%)", 34, FP_COLOR),
            (
                f"HMONNX {overall['hmonnx_successes']}/200 ({overall['hmonnx_success_rate_pct']:.1f}%)",
                34,
                HMONNX_COLOR,
            ),
            (f"DELTA {overall['delta_pp']:+.1f} PERCENTAGE POINTS", 28, TEXT),
            (f"ALL {len(pairs)} PERSISTED PAIRED ROLLOUTS INCLUDED", 22, MUTED),
        ],
        duration_s=4.0,
        args=args,
    )
    segments.append(outro)

    concat_file = args.work_dir / "concat.txt"
    _concat_segments(segments, args.output_video, concat_file)
    output_info = _probe_video(args.output_video, include_hash=True)
    nominal_fps = Fraction(output_info["r_frame_rate"])
    average_fps = float(Fraction(output_info["avg_frame_rate"]))
    if output_info["width"] != args.width or output_info["height"] != args.height:
        raise ValueError(f"unexpected output dimensions: {output_info}")
    if nominal_fps != args.fps or abs(average_fps - args.fps) > 0.05:
        raise ValueError(f"unexpected output FPS: nominal={nominal_fps}, average={average_fps:.6f}")
    if output_info["nb_frames"] != total_frames:
        raise ValueError(f"unexpected output frame count: {output_info['nb_frames']} != {total_frames}")

    scored_episodes = int(report["protocol"]["total_episodes_per_runtime"])
    persisted_per_runtime = len(pairs)
    manifest = {
        "manifest_version": 1,
        "source_accuracy_report": {
            "path": str(args.report_json),
            "sha256": _sha256(args.report_json),
        },
        "coverage": {
            "suite": report["protocol"]["suite"],
            "task_ids": report["protocol"]["task_ids"],
            "scored_episodes_per_runtime": scored_episodes,
            "persisted_videos_per_runtime": persisted_per_runtime,
            "paired_persisted_rollouts": len(pairs),
            "source_videos": 2 * len(pairs),
            "unrecorded_scored_episodes_per_runtime": scored_episodes - persisted_per_runtime,
            "all_metadata_video_paths_included": True,
            "note": (
                "The evaluator metadata contains ten persisted videos per task while the "
                "closed-loop score covers twenty episodes per task. This MP4 includes every "
                "persisted FP and HMONNX video from the final matched-partition runs."
            ),
        },
        "accuracy": report["overall"],
        "tasks": task_coverage,
        "pairs": pairs,
        "render": {
            "fps": args.fps,
            "width": args.width,
            "height": args.height,
            "header_height": args.header_height,
            "crf": args.crf,
            "preset": args.preset,
            "intro_seconds": 4.0,
            "task_card_seconds": 2.0,
            "outro_seconds": 4.0,
            "shorter_rollout_alignment": "clone final frame",
            "segment_count": len(segments),
            "expected_total_frames": total_frames,
        },
        "output_video": output_info,
    }
    args.output_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output_video}")
    print(f"Wrote {args.output_manifest}")
    print(json.dumps(manifest["coverage"], indent=2))
    print(json.dumps(output_info, indent=2))

    if not args.keep_work_dir:
        shutil.rmtree(args.work_dir)


if __name__ == "__main__":
    main()
