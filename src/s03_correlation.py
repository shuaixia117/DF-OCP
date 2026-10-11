"""核心变量 Pearson 相关性分析。"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from config import FIG_DIR
from plot_style import setup
from utils import stars

setup()

VARS = ["ZOP", "ZDFU", "ZFL", "ZSFL", "ZOFL"]
NAMES = ["过度消费倾向", "数字金融使用", "综合金融素养", "主观金融素养", "客观金融素养"]


def corr_table(df):
    """表6-1 Pearson 相关系数（下三角，带显著性）。"""
    rows = []
    for i, a in enumerate(VARS):
        row = [a]
        for j, b in enumerate(VARS):
            if j < i:
                r, p = stats.pearsonr(df[a], df[b])
                row.append(f"{r:.3f}{stars(p)}")
            elif j == i:
                row.append("1.000")
            else:
                row.append("")
        rows.append(row)
    return pd.DataFrame(rows, columns=[""] + VARS)


def corr_heatmap(df):
    """图6-1 相关系数热力图。"""
    R = df[VARS].corr().values
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(R, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(VARS)), NAMES, rotation=45, ha="right")
    ax.set_yticks(range(len(VARS)), NAMES)
    for i in range(len(VARS)):
        for j in range(len(VARS)):
            ax.text(j, i, f"{R[i, j]:.3f}", ha="center", va="center",
                    color="white" if abs(R[i, j]) > 0.7 else "black")
    ax.set_xticks(np.arange(-0.5, len(VARS)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(VARS)), minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.tick_params(which="minor", length=0)
    fig.colorbar(im, ax=ax, label="Pearson相关系数")
    ax.set_title("图6-1 核心变量Pearson相关性热力图")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "图6-1_相关系数热力图.png", dpi=200)
    return fig


def run(df):
    return {"corr": corr_table(df), "fig": corr_heatmap(df)}
