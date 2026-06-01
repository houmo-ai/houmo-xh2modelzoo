import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import onnx
from loguru import logger
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


DEFAULT_SEQ_LENS = [
    2048,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048576,
    2097152,
    4194304,
]
GIB = 1024**3


@dataclass(frozen=True)
class OnnxProfileContext:
    shape_map: dict[str, list[int | str]]
    base_input_seq_len: int
    base_kv_seq_len: int
    target_seq_len: int


def get_numel(shape: list[int]) -> int:
    numel = 1
    for dim in shape:
        numel *= dim
    return numel


def get_broadcast_shape(lhs_shape: list[int], rhs_shape: list[int]) -> list[int]:
    broadcast_shape_reversed: list[int] = []
    lhs_reversed = list(reversed(lhs_shape))
    rhs_reversed = list(reversed(rhs_shape))
    max_rank = max(len(lhs_reversed), len(rhs_reversed))

    for index in range(max_rank):
        lhs_dim = lhs_reversed[index] if index < len(lhs_reversed) else 1
        rhs_dim = rhs_reversed[index] if index < len(rhs_reversed) else 1
        max_dim = max(lhs_dim, rhs_dim)
        min_dim = min(lhs_dim, rhs_dim)
        if max_dim % min_dim != 0:
            raise ValueError(f"Shapes {lhs_shape} and {rhs_shape} are not broadcastable")
        broadcast_shape_reversed.append(max_dim)

    return list(reversed(broadcast_shape_reversed))


def get_matmul_flops(lhs_shape: list[int], rhs_shape: list[int]) -> int | None:
    if not lhs_shape or not rhs_shape:
        return None

    if len(lhs_shape) == 1 and len(rhs_shape) == 1:
        if lhs_shape[0] != rhs_shape[0]:
            return None
        return 2 * lhs_shape[0]

    if len(lhs_shape) == 1:
        k = lhs_shape[0]
        if rhs_shape[-2] != k:
            return None
        n = rhs_shape[-1]
        batch_shape = rhs_shape[:-2]
        return 2 * get_numel(batch_shape or [1]) * k * n

    if len(rhs_shape) == 1:
        k = rhs_shape[0]
        if lhs_shape[-1] != k:
            return None
        m = lhs_shape[-2]
        batch_shape = lhs_shape[:-2]
        return 2 * get_numel(batch_shape or [1]) * m * k

    if lhs_shape[-1] != rhs_shape[-2]:
        return None

    try:
        batch_shape = get_broadcast_shape(lhs_shape[:-2], rhs_shape[:-2])
    except ValueError:
        return None

    m = lhs_shape[-2]
    k = lhs_shape[-1]
    n = rhs_shape[-1]
    return 2 * get_numel(batch_shape or [1]) * m * k * n


def build_shape_map(graph: onnx.GraphProto) -> dict[str, list[int | str]]:
    shape_map: dict[str, list[int | str]] = {}
    for value_info in list(graph.value_info) + list(graph.input) + list(graph.output):
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue

        dims: list[int | str] = []
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                dims.append(dim.dim_value)
            elif dim.HasField("dim_param"):
                dims.append(dim.dim_param)
            else:
                dims.append("?")
        shape_map[value_info.name] = dims

    return shape_map


def get_static_shape(
    shape_map: dict[str, list[int | str]],
    tensor_name: str,
) -> list[int] | None:
    shape = shape_map.get(tensor_name)
    if not shape or not all(isinstance(dim, int) for dim in shape):
        return None
    return [int(dim) for dim in shape]


def infer_base_input_seq_len(shape_map: dict[str, list[int | str]]) -> int:
    input_shape = get_static_shape(shape_map, "input_1")
    if input_shape is None or len(input_shape) < 2:
        raise ValueError("input_1 shape is required to infer dynamic seq_len")
    return input_shape[1]


def infer_base_kv_seq_len(
    graph: onnx.GraphProto,
    shape_map: dict[str, list[int | str]],
) -> int:
    for node in graph.node:
        if node.op_type != "KVcache" or not node.output:
            continue
        output_shape = get_static_shape(shape_map, node.output[0])
        if output_shape is not None and len(output_shape) >= 3:
            return output_shape[2]
    raise ValueError("KVcache output shape is required to infer KV cache seq_len")


def get_dynamic_linear_shapes(
    node: onnx.NodeProto,
    context: OnnxProfileContext,
) -> tuple[list[int], list[int]] | None:
    input_shape = get_static_shape(context.shape_map, node.input[0])
    output_shape = get_static_shape(context.shape_map, node.output[0])
    if input_shape is None or output_shape is None or len(input_shape) < 3 or len(output_shape) < 2:
        return None

    dynamic_input_shape = input_shape.copy()
    dynamic_output_shape = output_shape.copy()
    dynamic_input_shape[1] = context.target_seq_len
    dynamic_output_shape[1] = context.target_seq_len
    return dynamic_input_shape, dynamic_output_shape


def get_dynamic_matmul_shapes(
    node: onnx.NodeProto,
    context: OnnxProfileContext,
) -> tuple[list[int], list[int]] | None:
    lhs_shape = get_static_shape(context.shape_map, node.input[0])
    rhs_shape = get_static_shape(context.shape_map, node.input[1])
    if lhs_shape is None or rhs_shape is None:
        return None

    output_shape = None
    if node.output:
        output_shape = get_static_shape(context.shape_map, node.output[0])

    dynamic_lhs_shape = lhs_shape.copy()
    dynamic_rhs_shape = rhs_shape.copy()
    seq_len_refs = {context.base_input_seq_len, context.base_kv_seq_len}

    # M dimension typically tracks the input token length.
    if len(dynamic_lhs_shape) >= 2 and dynamic_lhs_shape[-2] in seq_len_refs:
        dynamic_lhs_shape[-2] = context.target_seq_len

    # N dimension tracks the context length for attention score matmuls.
    if (
        len(dynamic_rhs_shape) > 2
        and output_shape is not None
        and output_shape[-1] in seq_len_refs
        and dynamic_rhs_shape[-1] == output_shape[-1]
    ):
        dynamic_rhs_shape[-1] = context.target_seq_len

    # K dimension tracks the context length for attention value matmuls.
    if (
        len(dynamic_rhs_shape) > 2
        and dynamic_lhs_shape[-1] == dynamic_rhs_shape[-2]
        and dynamic_lhs_shape[-1] in seq_len_refs
    ):
        dynamic_lhs_shape[-1] = context.target_seq_len
        dynamic_rhs_shape[-2] = context.target_seq_len

    return dynamic_lhs_shape, dynamic_rhs_shape


def get_dynamic_kvcache_output_shape(
    node: onnx.NodeProto,
    context: OnnxProfileContext,
) -> list[int] | None:
    if not node.output:
        return None

    output_shape = get_static_shape(context.shape_map, node.output[0])
    if output_shape is None or len(output_shape) < 3:
        return None

    dynamic_output_shape = output_shape.copy()
    dynamic_output_shape[2] = context.target_seq_len
    return dynamic_output_shape


def get_linear_node_flops(
    node: onnx.NodeProto,
    context: OnnxProfileContext,
) -> int | None:
    dynamic_shapes = get_dynamic_linear_shapes(node, context)
    if dynamic_shapes is None:
        return None

    input_shape, output_shape = dynamic_shapes
    batch, seq_len, in_features = input_shape[0], input_shape[1], input_shape[2]
    out_features = output_shape[-1]
    return 2 * batch * seq_len * in_features * out_features


def get_matmul_node_flops(
    node: onnx.NodeProto,
    context: OnnxProfileContext,
) -> int | None:
    dynamic_shapes = get_dynamic_matmul_shapes(node, context)
    if dynamic_shapes is None:
        return None

    lhs_shape, rhs_shape = dynamic_shapes
    return get_matmul_flops(lhs_shape, rhs_shape)


def get_group_matmul_node_flops(
    node: onnx.NodeProto,
    context: OnnxProfileContext,
) -> int | None:
    return get_matmul_node_flops(node, context)


def profile_flops(
    graph: onnx.GraphProto,
    context: OnnxProfileContext,
) -> tuple[dict[str, dict[str, int]], list[str]]:
    flops_stats: dict[str, dict[str, int]] = {op_type: {"flops": 0, "count": 0} for op_type in FLOPS_CALCULATORS}
    skipped_nodes: list[str] = []

    for node in graph.node:
        flops_calculator = FLOPS_CALCULATORS.get(node.op_type)
        if flops_calculator is None:
            continue

        flops = flops_calculator(node, context)
        if flops is None:
            skipped_nodes.append(f"{node.op_type}:{node.name}")
            continue

        flops_stats[node.op_type]["flops"] += flops
        flops_stats[node.op_type]["count"] += 1

    return flops_stats, skipped_nodes


def get_total_kv_cache_bytes(
    graph: onnx.GraphProto,
    context: OnnxProfileContext,
    kv_bits: int,
) -> tuple[float, list[str]]:
    total_kv_cache_elements = 0
    skipped_nodes: list[str] = []

    for node in graph.node:
        if node.op_type != "KVcache":
            continue

        output_shape = get_dynamic_kvcache_output_shape(node, context)
        if output_shape is None:
            skipped_nodes.append(f"KVcache:{node.name}")
            continue

        total_kv_cache_elements += get_numel(output_shape)

    return total_kv_cache_elements * kv_bits / 8, skipped_nodes


def format_token_count(token_count: int) -> str:
    if token_count >= 1024**2 and token_count % (1024**2) == 0:
        return f"{token_count // (1024**2)}M"
    if token_count >= 1024 and token_count % 1024 == 0:
        return f"{token_count // 1024}K"
    return str(token_count)


def format_token_count_in_k(token_count: int) -> str:
    token_count_in_k = token_count / 1024
    if token_count_in_k.is_integer():
        return str(int(token_count_in_k))
    return f"{token_count_in_k:.3f}".rstrip("0").rstrip(".")


def infer_model_name(model_dir: str) -> str:
    model_base_name = Path(model_dir).name.rstrip("/")
    parts = [part for part in model_base_name.split("-") if part]
    if len(parts) >= 2:
        size_part = next((part for part in parts if part.upper().endswith("B")), None)
        if size_part is not None:
            return f"{'-'.join(parts[:2])}-{size_part}"
    return model_base_name


def autosize_worksheet_columns(worksheet) -> None:
    for column_cells in worksheet.columns:
        max_length = 0
        column_letter = get_column_letter(column_cells[0].column)
        for cell in column_cells:
            if cell.value is None:
                continue
            max_length = max(max_length, len(str(cell.value)))
        worksheet.column_dimensions[column_letter].width = min(max_length + 2, 40)


def get_model_parameters(model: onnx.GraphProto) -> int:
    total_params = 0
    name2init = {init.name: init for init in model.graph.initializer}
    # name2tensor = {value_info.name: value_info for value_info in model.graph.value_info}
    for node in model.graph.node:
        if node.op_type in ["Linear"]:
            in_features = 0
            out_features = 0
            for attr in node.attribute:
                if attr.name == "in_features":
                    in_features = attr.i
                elif attr.name == "out_features":
                    out_features = attr.i
            total_params += in_features * out_features
            have_bias = False
            for attr in node.attribute:
                if attr.name == "bias":
                    have_bias = bool(attr.i)
                    break
            if have_bias:
                total_params += out_features

        elif node.op_type in ["RMSNorm"]:
            w_name = node.input[1]
            w_init = name2init[w_name]
            total_params += get_numel(list(w_init.dims))

    return total_params


def write_excel_report(
    excel_file: str,
    report_title: str,
    chip_tflops: float,
    chip_bandwidth_gbs: float,
    compute_utilization: float,
    bandwidth_utilization: float,
    c2c_ratio: float,
    summary_rows: list[tuple[str, object]],
    profile_rows: list[dict[str, object]],
    skipped_by_seq_len: dict[int, list[str]],
) -> None:
    workbook = Workbook()
    header_font = Font(bold=True)

    report_sheet = workbook.active
    report_sheet.title = "profile_report"

    thin_side = Side(style="thin", color="D9D9D9")
    border = Border(left=thin_side, right=thin_side, top=thin_side, bottom=thin_side)
    font_title = Font(bold=True, size=16, color="FFFFFF")
    font_header = Font(bold=True, size=11)
    font_value = Font(bold=False, size=11)
    align_center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    align_right = Alignment(horizontal="right", vertical="center")
    fill_title = PatternFill("solid", fgColor="4F81BD")
    fill_info = PatternFill("solid", fgColor="D9EAF7")
    fill_group = PatternFill("solid", fgColor="FCE4D6")
    fill_header = PatternFill("solid", fgColor="EDEDED")
    fill_util = PatternFill("solid", fgColor="E2F0D9")
    fill_body = PatternFill("solid", fgColor="FFFFFF")

    def style(cell, *, font=None, fill=None, alignment=None, number_format: str | None = None) -> None:
        cell.border = border
        if font is not None:
            cell.font = font
        if fill is not None:
            cell.fill = fill
        if alignment is not None:
            cell.alignment = alignment
        if number_format is not None:
            cell.number_format = number_format

    def merged_value(start: str, end: str, value: Any, *, font, fill, alignment) -> None:
        report_sheet.merge_cells(f"{start}:{end}")
        cell = report_sheet[start]
        cell.value = value
        style(cell, font=font, fill=fill, alignment=alignment)
        start_col = report_sheet[start].column
        end_col = report_sheet[end].column
        start_row = report_sheet[start].row
        end_row = report_sheet[end].row
        for row in range(start_row, end_row + 1):
            for col in range(start_col, end_col + 1):
                style(report_sheet.cell(row=row, column=col), fill=fill, alignment=alignment)

    report_sheet.merge_cells("A1:I1")
    report_sheet["A1"] = report_title
    style(report_sheet["A1"], font=font_title, fill=fill_title, alignment=align_center)

    report_sheet["A2"] = "算力（T-FLOPS）："
    report_sheet["B2"] = float(chip_tflops)
    report_sheet["A3"] = "带宽（GB/s）："
    report_sheet["B3"] = float(chip_bandwidth_gbs)
    for ref in ("A2", "A3"):
        style(report_sheet[ref], font=font_header, fill=fill_info, alignment=align_center)
    for ref in ("B2", "B3"):
        style(report_sheet[ref], font=font_value, fill=fill_info, alignment=align_center, number_format="0.##")

    merged_value("C2", "C3", "算力需求\n（T-FLOPS）", font=font_header, fill=fill_group, alignment=align_center)
    merged_value("D2", "F3", "显存占用\n（GiB）", font=font_header, fill=fill_group, alignment=align_center)
    report_sheet["G2"] = "算力利用率"
    report_sheet["H2"] = "带宽利用率"
    report_sheet["I2"] = "C2C倍率"
    for ref in ("G2", "H2", "I2"):
        style(report_sheet[ref], font=font_header, fill=fill_util, alignment=align_center)
    report_sheet["G3"] = float(compute_utilization)
    report_sheet["H3"] = float(bandwidth_utilization)
    report_sheet["I3"] = float(c2c_ratio)
    for ref in ("G3", "H3", "I3"):
        style(report_sheet[ref], font=font_value, fill=fill_util, alignment=align_center, number_format="0.###")

    merged_value("A4", "B4", "Context Length (K)", font=font_header, fill=fill_header, alignment=align_center)
    report_headers = {
        "C4": "Prefill",
        "D4": "Weights",
        "E4": "KV-Cache",
        "F4": "Weight+KV",
    }
    for ref, value in report_headers.items():
        report_sheet[ref] = value
        style(report_sheet[ref], font=font_header, fill=fill_header, alignment=align_center)
    for ref in ("G4", "H4", "I4"):
        style(report_sheet[ref], font=font_header, fill=fill_header, alignment=align_center)

    report_row_idx = 5
    for row in profile_rows:
        merged_value(
            f"A{report_row_idx}",
            f"B{report_row_idx}",
            row["seq_len_label_k"],
            font=font_value,
            fill=fill_body,
            alignment=align_center,
        )
        data_cells = [
            ("C", row["total_tflops"], "0.000000"),
            ("D", row["weights_gib"], "0.000000"),
            ("E", row["kv_cache_gib"], "0.000000"),
            ("F", row["weight_plus_kv_gib"], "0.000000"),
        ]
        for col, value, fmt in data_cells:
            cell = report_sheet[f"{col}{report_row_idx}"]
            cell.value = float(value)
            style(cell, font=font_value, fill=fill_body, alignment=align_right, number_format=fmt)
        for ref in (f"G{report_row_idx}", f"H{report_row_idx}", f"I{report_row_idx}"):
            style(report_sheet[ref], font=font_value, fill=fill_body, alignment=align_center)
        report_row_idx += 1

    for row_index in range(1, report_row_idx):
        report_sheet.row_dimensions[row_index].height = 28 if row_index > 1 else 32
    report_sheet.freeze_panes = "A5"

    summary_sheet = workbook.create_sheet("summary")
    summary_sheet.title = "summary"
    summary_sheet.append(["key", "value"])
    for cell in summary_sheet[1]:
        cell.font = header_font
    for key, value in summary_rows:
        summary_sheet.append([key, value])
    summary_sheet.freeze_panes = "A2"

    if skipped_by_seq_len:
        skipped_sheet = workbook.create_sheet("skipped_nodes")
        skipped_sheet.append(["seq_len", "node"])
        for cell in skipped_sheet[1]:
            cell.font = header_font
        for seq_len, nodes in skipped_by_seq_len.items():
            for node in nodes:
                skipped_sheet.append([seq_len, node])
        skipped_sheet.freeze_panes = "A2"
        skipped_sheet.auto_filter.ref = skipped_sheet.dimensions

    for worksheet in workbook.worksheets:
        autosize_worksheet_columns(worksheet)

    excel_path = Path(excel_file)
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(excel_path)


FLOPS_CALCULATORS: dict[
    str,
    Callable[[onnx.NodeProto, OnnxProfileContext], int | None],
] = {
    "Linear": get_linear_node_flops,
    "MatMul": get_matmul_node_flops,
    "GroupMatMul": get_group_matmul_node_flops,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test Qwen3 TTS model")
    parser.add_argument(
        "--hmonnx-file",
        type=str,
        default=(
            "work_dirs/qwen3_tts_12hz_1_7B_voicedesign_talker_2k_xh2a/prefill_onnx/"
            "qwen3_tts_12hz_1_7B_voicedesign_talker_2k_xh2a_prefill.onnx"
        ),
        help="Prefill ONNX file path",
    )
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=DEFAULT_SEQ_LENS,
        help="Input token lengths used for dynamic FLOPs / weights / KV cache profiling",
    )
    parser.add_argument(
        "--kv-bits",
        type=int,
        default=8,
        help="KV cache bit-width per element",
    )
    parser.add_argument(
        "--wparam-bits",
        type=int,
        default=8,
        help="Weight bit-width per parameter",
    )
    parser.add_argument(
        "--excel-file",
        type=str,
        default=None,
        help="Path to the output Excel report (.xlsx)",
    )
    parser.add_argument(
        "--compute",
        type=float,
        default=200,
        help="算力（T-FLOPS） for Excel report header",
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=272,
        help="带宽（GB/s） for Excel report header",
    )
    parser.add_argument(
        "--compute-util",
        type=float,
        default=0.5,
        help="算力利用率 for Excel report header",
    )
    parser.add_argument(
        "--bandwidth-util",
        type=float,
        default=0.7,
        help="带宽利用率 for Excel report header",
    )
    parser.add_argument(
        "--c2c-ratio",
        type=float,
        default=1.0,
        help="C2C倍率 for Excel report header",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name shown in the Excel report title",
    )
    args = parser.parse_args()

    logger.info(f"Prefill ONNX file: {args.hmonnx_file}")
    onnx_model = onnx.load(args.hmonnx_file, load_external_data=False)
    graph = onnx_model.graph
    shape_map = build_shape_map(graph)
    base_input_seq_len = infer_base_input_seq_len(shape_map)
    base_kv_seq_len = infer_base_kv_seq_len(graph, shape_map)
    op_counts = Counter(node.op_type for node in graph.node)
    model_dir = str(Path(args.hmonnx_file).parent)

    total_params = get_model_parameters(onnx_model)

    total_params_b = total_params / 1e9
    bytes_per_param = args.wparam_bits / 8
    total_params_gib = total_params * bytes_per_param / GIB

    embeding_params = 317456384
    embeding_params_b = embeding_params / 1e9
    embeding_params_gib = embeding_params * 1 / GIB
    total_params_gib += embeding_params_gib

    logger.info(f"input_1_seq_len_ref={base_input_seq_len}")
    logger.info(f"kv_cache_seq_len_ref={base_kv_seq_len}")
    logger.info(
        "op_counts="
        f"Linear:{op_counts.get('Linear', 0)}, "
        f"MatMul:{op_counts.get('MatMul', 0)}, "
        f"GroupMatMul:{op_counts.get('GroupMatMul', 0)}, "
        f"KVcache:{op_counts.get('KVcache', 0)}"
    )

    logger.info("\n--- ONNX Dynamic Profile ---")
    logger.info(
        f"{'Context Length(K)':>10}  {'Prefill(TF)':>12}  {'Weights(GiB)':>14}  {'KVCache(GiB)':>14}  {'Weight+KV(GiB)':>16}"
    )

    profile_rows: list[dict[str, object]] = []
    skipped_by_seq_len: dict[int, list[str]] = {}
    for seq_len in args.seq_lens:
        context = OnnxProfileContext(
            shape_map=shape_map,
            base_input_seq_len=base_input_seq_len,
            base_kv_seq_len=base_kv_seq_len,
            target_seq_len=seq_len,
        )
        flops_stats, skipped_nodes = profile_flops(graph, context)
        kv_cache_bytes, skipped_kv_nodes = get_total_kv_cache_bytes(graph, context, args.kv_bits)
        total_flops = sum(stats["flops"] for stats in flops_stats.values())
        kv_cache_gib = kv_cache_bytes / GIB
        weight_plus_kv_gib = total_params_gib + kv_cache_gib
        seq_len_in_k = seq_len / 1024
        seq_len_label = format_token_count_in_k(seq_len)

        profile_rows.append(
            {
                "seq_len": seq_len,
                "seq_len_k": seq_len_in_k,
                "seq_len_label_k": seq_len_label,
                "total_flops": total_flops,
                "total_tflops": total_flops / 1e12,
                "weights_gib": total_params_gib,
                "kv_cache_gib": kv_cache_gib,
                "weight_plus_kv_gib": weight_plus_kv_gib,
            }
        )

        logger.info(
            f"{seq_len_label:>10}  "
            f"{total_flops / 1e12:>12.6f}  "
            f"{total_params_gib:>14.6f}  "
            f"{kv_cache_gib:>14.6f}  "
            f"{weight_plus_kv_gib:>16.6f}"
        )

        skipped_items = skipped_nodes + skipped_kv_nodes
        if skipped_items:
            skipped_by_seq_len[seq_len] = skipped_items

    if skipped_by_seq_len:
        logger.info("\nSkipped nodes:")
        for seq_len, skipped_items in skipped_by_seq_len.items():
            logger.info(f"  seq_len={seq_len}: {skipped_items}")

    default_excel_path = Path(args.hmonnx_file).with_name(f"{Path(args.hmonnx_file).stem}_profile.xlsx")
    excel_file = args.excel_file or str(default_excel_path)
    summary_rows = [
        ("model_dir", model_dir),
        ("onnx_file", args.hmonnx_file),
        ("model_name", args.model_name or infer_model_name(model_dir)),
        ("wparam_bits", args.wparam_bits),
        ("kv_bits", args.kv_bits),
        ("compute_tflops", args.compute),
        ("bandwidth_gbs", args.bandwidth),
        ("compute_utilization", args.compute_util),
        ("bandwidth_utilization", args.bandwidth_util),
        ("c2c_ratio", args.c2c_ratio),
        ("total_params", total_params),
        ("total_params_b", total_params_b),
        ("total_params_gib", total_params_gib),
        ("input_1_seq_len_ref", base_input_seq_len),
        ("kv_cache_seq_len_ref", base_kv_seq_len),
        ("linear_op_count", op_counts.get("Linear", 0)),
        ("matmul_op_count", op_counts.get("MatMul", 0)),
        ("group_matmul_op_count", op_counts.get("GroupMatMul", 0)),
        ("kvcache_op_count", op_counts.get("KVcache", 0)),
    ]
    report_title = f"{args.model_name or infer_model_name(model_dir)} (w{args.wparam_bits}, a{args.kv_bits})"
    write_excel_report(
        excel_file,
        report_title,
        args.compute,
        args.bandwidth,
        args.compute_util,
        args.bandwidth_util,
        args.c2c_ratio,
        summary_rows,
        profile_rows,
        skipped_by_seq_len,
    )
    logger.info(f"Excel report saved to: {excel_file}")
