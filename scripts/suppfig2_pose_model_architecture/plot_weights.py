from sppaper.common.plot import setup_matplotlib_params

setup_matplotlib_params()

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sppaper.common.resources import get_outputs_dir

###########################################################################
## THESE NUMBERS ARE TAKEN FROM THE MODEL OVERVIEWS AT THE TOP OF MODEL  ##
## TRAINING/INFERENCE LOGS.                                              ##
###########################################################################
nparam_encoder = 11_176_512
nparam_contrastive_projhead = 393_984

nparam_kpt3d_total = 16_136_992
nparam_kpt3d_xyhead = 18_464  # Conv2d-99
nparam_kpt3d_depthhead = 73_728 + 256 + 264_192  # Conv2d-101, GroupNorm-102, Linear-105
nparam_kpt3d_upsampler = (
    nparam_kpt3d_total - nparam_encoder - nparam_kpt3d_xyhead - nparam_kpt3d_depthhead
)

nparam_bodyseg_total = 15_789_597
nparam_bodyseg_seghead = nparam_bodyseg_total - nparam_encoder

bar_totals = [
    nparam_encoder + nparam_contrastive_projhead,
    nparam_kpt3d_total,
    nparam_bodyseg_total,
]

output_path = get_outputs_dir() / "weights_distribution/weights_distribution.pdf"

COLOR_ENCODER = "#546a76"
COLOR_ENCODER_REUSED = "#bccad1"
COLOR_PROJHEAD = "#000000"
COLOR_UPSAMPLER = "#689829"
COLOR_XYHEAD = "#20A8DE"
COLOR_DEPTHHEAD = "#fc7a1e"
COLOR_SEGHEAD = "#a23e48"

MM_TO_IN = 1 / 25.4

bar_labels = [
    "Contrastive pretraining",
    "3D keypoints model",
    "Body segmentation model",
]

fig, ax = plt.subplots(figsize=(180 * MM_TO_IN, 70 * MM_TO_IN), tight_layout=True)

y_positions = [2, 1, 0]
bar_height = 0.5
max_total = max(bar_totals)

INSIDE_THRESHOLD = 0.04  # fraction of axis width below which label goes outside


def draw_segment(ax, y, left, width, color, total, arrow_dir=1):
    ax.barh(
        y,
        width,
        left=left,
        height=bar_height,
        color=color,
        edgecolor="white",
        linewidth=0,
    )

    pct = width / total * 100
    pct_str = f"{round(pct, 1):.1f}%"
    cx = left + width / 2
    frac = width / max_total

    if frac >= INSIDE_THRESHOLD:
        ax.text(
            cx,
            y,
            pct_str,
            ha="center",
            va="center",
            color="white",
            # fontweight="bold",
        )
    else:
        y_tip = y + (bar_height / 2) * arrow_dir
        y_text = y + (bar_height / 2 + 0.28) * arrow_dir
        ax.annotate(
            pct_str,
            xy=(cx, y_tip),
            xytext=(cx, y_text),
            ha="center",
            va="bottom" if arrow_dir > 0 else "top",
            color=color,
            # fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=color, lw=1.0),
        )


# Bar 0: Contrastive
total0 = bar_totals[0]
draw_segment(ax, y_positions[0], 0, nparam_encoder, COLOR_ENCODER, total0)
draw_segment(
    ax,
    y_positions[0],
    nparam_encoder,
    nparam_contrastive_projhead,
    COLOR_PROJHEAD,
    total0,
)

# Bar 1: Kpt3D
total1 = bar_totals[1]
left = 0
draw_segment(ax, y_positions[1], left, nparam_encoder, COLOR_ENCODER_REUSED, total1)
left += nparam_encoder
draw_segment(ax, y_positions[1], left, nparam_kpt3d_upsampler, COLOR_UPSAMPLER, total1)
left += nparam_kpt3d_upsampler
draw_segment(
    ax, y_positions[1], left, nparam_kpt3d_xyhead, COLOR_XYHEAD, total1, arrow_dir=-1
)
left += nparam_kpt3d_xyhead
draw_segment(
    ax,
    y_positions[1],
    left,
    nparam_kpt3d_depthhead,
    COLOR_DEPTHHEAD,
    total1,
    arrow_dir=1,
)


# Bar 2: BodySeg
total2 = bar_totals[2]
draw_segment(ax, y_positions[2], 0, nparam_encoder, COLOR_ENCODER_REUSED, total2)
draw_segment(
    ax, y_positions[2], nparam_encoder, nparam_bodyseg_seghead, COLOR_SEGHEAD, total2
)

ax.set_yticks(y_positions)
ax.set_yticklabels(bar_labels, fontsize=6)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
ax.set_xlabel("Trainable parameters")
ax.set_title("Parameter breakdown", fontsize=7)
ax.set_xlim(0, max_total * 1.02)
ax.set_ylim(-0.65, 2.8)
ax.spines[["top", "right"]].set_visible(False)

legend_items = [
    mpatches.Patch(color=COLOR_ENCODER, label="Shared encoder"),
    mpatches.Patch(color=COLOR_ENCODER, label="Shared encoder (reused weights)"),
    mpatches.Patch(color=COLOR_PROJHEAD, label="Contrastive proj. head"),
    mpatches.Patch(color=COLOR_UPSAMPLER, label="3D keypoints shared upsampler"),
    mpatches.Patch(color=COLOR_XYHEAD, label="3D keypoints xy head"),
    mpatches.Patch(color=COLOR_DEPTHHEAD, label="3D keypoints depth head"),
    mpatches.Patch(color=COLOR_SEGHEAD, label="Body segmentation head"),
]
ax.legend(
    handles=legend_items, loc="upper left", bbox_to_anchor=(1.04, 1), frameon=False
)

output_path.parent.mkdir(exist_ok=True, parents=True)
fig.savefig(output_path)
print(f"Figure saved to {output_path.absolute()}")
