"""从 QAT 训练 log 中提取 loss 并画图

用法:
    python plot_training_loss.py <log_path> [--smooth 50] [--out loss.png]

支持实时跟踪: log 增长后重新运行即可更新图表。
"""

import argparse
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ----------------------------------------------------------------
#  解析 log
# ----------------------------------------------------------------

_PATTERN = re.compile(
    r"step=(\d+)\s+lr=[\d.e+-]+\s+loss=([\d.]+)\s+grad=[\d.]+\s+"
    r"(?:llm_loss|llm)=([\d.]+)\s+(?:flow_loss|flow)=([\d.]+)"
)


def parse_log(path):
    steps, losses, llm_losses, flow_losses = [], [], [], []
    with open(path) as f:
        for line in f:
            m = _PATTERN.search(line)
            if not m:
                continue
            steps.append(int(m.group(1)))
            losses.append(float(m.group(2)))
            llm_losses.append(float(m.group(3)))
            flow_losses.append(float(m.group(4)))
    return (
        np.array(steps, dtype=np.int64),
        np.array(losses, dtype=np.float64),
        np.array(llm_losses, dtype=np.float64),
        np.array(flow_losses, dtype=np.float64),
    )


# ----------------------------------------------------------------
#  指数移动平均平滑
# ----------------------------------------------------------------

def ema_smooth(y, window):
    if window <= 1 or len(y) < 2:
        return y
    alpha = 2.0 / (window + 1)
    out = np.empty_like(y)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1 - alpha) * out[i - 1]
    return out


# ----------------------------------------------------------------
#  画图
# ----------------------------------------------------------------

def plot(steps, loss, llm_loss, flow_loss, smooth_w, out_path):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                             gridspec_kw={"hspace": 0.08})

    series = [
        (loss,      "Total Loss",       "#2563eb", axes[0]),
        (llm_loss,  "LLM Loss",         "#dc2626", axes[1]),
        (flow_loss, "Flow Loss",        "#16a34a", axes[2]),
    ]

    for y, label, color, ax in series:
        # 半透明散点做背景，不喧宾夺主
        ax.scatter(steps, y, s=0.3, alpha=0.06, color=color, rasterized=True)
        # EMA 主线
        if smooth_w > 1 and len(y) > smooth_w:
            y_smooth = ema_smooth(y, smooth_w)
            ax.plot(steps, y_smooth, linewidth=1.5, color=color,
                    label=f"{label} (EMA {smooth_w})")
            # 用 EMA 值定 y 轴，避免离群点撑高
            lo, hi = np.percentile(y_smooth, 0.5), np.percentile(y_smooth, 99.8)
            margin = max(0.1, (hi - lo) * 0.1)
            ax.set_ylim(lo - margin, hi + margin)
        else:
            ax.plot(steps, y, linewidth=0.5, alpha=0.6, color=color, label=label)
        ax.set_ylabel(label, fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[2].set_xlabel("Step", fontsize=11)
    axes[0].set_title(
        f"QAT Training Loss  (steps 0-{steps[-1]}, {len(steps)} points)",
        fontsize=13, fontweight="bold",
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved: {out_path}")


# ----------------------------------------------------------------
#  入口
# ----------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Plot QAT training loss")
    p.add_argument("log", help="训练 log 文件路径")
    p.add_argument("--smooth", type=int, default=50, help="EMA 平滑窗口 (默认 50)")
    p.add_argument("--out", type=str, default="training_loss.png", help="输出图片路径")
    args = p.parse_args()

    steps, loss, llm_loss, flow_loss = parse_log(args.log)
    if len(steps) == 0:
        print("[ERROR] 未找到 loss 记录")
        sys.exit(1)

    print(f"parsed: {len(steps)} steps  (step {steps[0]} ~ {steps[-1]})")
    plot(steps, loss, llm_loss, flow_loss, args.smooth, args.out)


if __name__ == "__main__":
    main()
