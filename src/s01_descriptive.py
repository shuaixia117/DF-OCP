"""描述性统计：样本特征图、消费特征图、核心变量描述性统计。"""
import matplotlib.pyplot as plt
import pandas as pd
from config import FIG_DIR, VALUE_LABELS, CONTROL_NAMES
from plot_style import setup

setup()

CORE_VARS = {
    "OP": "过度消费倾向", "ZOP": "标准化OP", "DFU": "数字金融使用", "ZDFU": "标准化DFU",
    "FL": "金融素养", "ZFL": "标准化FL", "OFL": "客观金融素养", "SFL": "主观金融素养",
    "ZOFL": "标准化OFL", "ZSFL": "标准化SFL", "parentedu": "父母平均受教育程度",
}


def core_stats(df):
    """表5-1 核心变量描述性统计。"""
    t = df[list(CORE_VARS)].agg(["count", "mean", "std", "min", "max"]).T
    t["mean"] = t["mean"].where(t["mean"].abs() > 5e-4, 0.0)  # 标准化变量均值显示为 0.000
    out = pd.DataFrame({
        "Variable": list(CORE_VARS),
        "含义": list(CORE_VARS.values()),
        "Obs": t["count"].astype(int).values,
        "Mean": [f"{x:.3f}" for x in t["mean"]],
        "Std.": [f"{x:.3f}" for x in t["std"]],
        "Min": [f"{x:.3f}" for x in t["min"]],
        "Max": [f"{x:.3f}" for x in t["max"]],
    })
    return out


def freq_table(df):
    """分类控制变量频数分布。"""
    rows = []
    for var, labs in VALUE_LABELS.items():
        vc = df[var].value_counts().sort_index()
        for k, n in vc.items():
            rows.append([CONTROL_NAMES[var], labs[int(k)], int(n), f"{n / len(df) * 100:.2f}%"])
    return pd.DataFrame(rows, columns=["控制变量", "类别", "频数", "比例"])


def count_multi(series, options):
    """多选题（以“┋”分隔）各选项的选择人数。"""
    s = series.fillna("")
    return pd.Series({k: s.str.contains(v, regex=False).sum() for k, v in options.items()})


def fig_sample(df):
    """图5-1 样本人口学特征。"""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    colors = ["#4472C4", "#ED7D31", "#A5A5A5", "#FFC000", "#5B9BD5"]

    g = df["gender"].map(VALUE_LABELS["gender"]).value_counts().reindex(["男", "女"])
    axes[0, 0].pie(g, labels=g.index, autopct="%.2f%%", colors=colors, startangle=90,
                   wedgeprops=dict(width=0.45), pctdistance=0.78)
    axes[0, 0].set_title("(a) 样本性别比例")

    m = df["major"].map(VALUE_LABELS["major"]).value_counts().reindex(VALUE_LABELS["major"].values())
    axes[0, 1].pie(m, labels=m.index, autopct="%.2f%%", colors=colors, startangle=90)
    axes[0, 1].set_title("(b) 样本专业分布")

    le = df["le"].map(VALUE_LABELS["le"]).value_counts().reindex(VALUE_LABELS["le"].values())
    bars = axes[1, 0].bar(le.index, le.values, color="#4472C4", width=0.55)
    axes[1, 0].bar_label(bars)
    axes[1, 0].set_ylabel("人数")
    axes[1, 0].set_title("(c) 样本生活费分布")

    gr = df["grade"].map(VALUE_LABELS["grade"]).value_counts().reindex(VALUE_LABELS["grade"].values())
    axes[1, 1].pie(gr, labels=gr.index, autopct="%.2f%%", colors=colors, startangle=90,
                   wedgeprops=dict(width=0.45), pctdistance=0.78)
    axes[1, 1].set_title("(d) 样本年级比例")

    fig.suptitle("图5-1 样本人口学特征", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "图5-1_样本人口学特征.png", dpi=200)
    return fig


def bar_line(counts, n, title, fname, ylabel="人数"):
    """柱状图（人数）+ 折线（占比）。"""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(counts.index, counts.values, color="#4472C4", width=0.45, label=ylabel)
    ax.bar_label(bars, padding=2)
    ax.set_ylim(0, counts.max() * 1.2)
    ax2 = ax.twinx()
    share = counts / n * 100
    ax2.plot(counts.index, share.values, color="#C00000", lw=2.5, marker="o", label="占比(%)")
    for x, v in zip(counts.index, share.values):
        ax2.annotate(f"{v:.2f}%", (x, v), textcoords="offset points", xytext=(18, 6))
    ax2.set_ylim(0, 120)
    ax2.set_ylabel("占比(%)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=200)
    return fig


def fig_consumption(df):
    """图5-2~5-4 消费特征。"""
    n = len(df)
    src = count_multi(df["income_source"], {"家庭提供": "家庭提供", "奖/助学金": "奖",
                                            "兼职/实习收入": "兼职", "其他": "其他"})
    f1 = bar_line(src, n, "图5-2 生活费来源分布", "图5-2_生活费来源.png")

    svc = count_multi(df["df_services"], {"日常支付与生活缴费": "日常支付", "线上理财与投资": "线上理财",
                                          "信用消费与分期": "信用消费", "线上借贷": "线上借贷"})
    f2 = bar_line(svc, n, "图5-3 数字金融服务使用情况", "图5-3_数字金融服务使用.png", ylabel="频数")

    users = df["credit_use"].notna().sum()
    use = count_multi(df["credit_use"], {"大额消费": "大额消费", "日常生活消费": "日常生活",
                                         "医疗/突发支出": "医疗", "学习培训/考试": "学习培训",
                                         "其他": "其他"})
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(use.index[::-1], (use / users * 100).values[::-1], color="#5B9BD5")
    ax.bar_label(bars, fmt="%.2f%%", padding=3)
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"占使用数字信贷受访者的比例(%)，n={users}")
    ax.set_title("图5-4 数字信贷主要用途（多选）")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "图5-4_数字信贷用途.png", dpi=200)
    return [f1, f2, fig]


def run(df):
    return {"core": core_stats(df), "freq": freq_table(df),
            "figs": [fig_sample(df)] + fig_consumption(df)}
