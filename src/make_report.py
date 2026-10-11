"""将全部表格与图形整理为 结果.pdf，并导出 全部结果表格.xlsx。"""
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from config import PDF_FILE, TAB_DIR, NOTE
from plot_style import setup

setup()
A4 = (8.27, 11.69)


def draw_table(pdf, title, df, note=NOTE, fontsize=None):
    """三线表样式绘制一页表格。"""
    n_rows, n_cols = len(df) + 1, len(df.columns)
    fig = plt.figure(figsize=A4)
    fig.text(0.5, 0.95, title, ha="center", va="top", fontsize=13, weight="bold")

    left, right, top = 0.07, 0.93, 0.91
    header_lines = max(str(c).count("\n") + 1 for c in df.columns)
    row_h = min(0.024, 0.80 / (n_rows + header_lines))
    head_h = row_h * header_lines
    fontsize = fontsize or min(9.5, row_h * A4[1] * 72 * 0.6)
    first_w = 0.28 if n_cols <= 6 else 0.17
    col_w = (right - left - first_w) / (n_cols - 1)
    xs = [left + first_w / 2] + [left + first_w + col_w * (i + 0.5) for i in range(n_cols - 1)]

    def hline(y, lw):
        fig.add_artist(plt.Line2D([left, right], [y, y], lw=lw, color="black"))

    hline(top, 1.5)
    for x, c in zip(xs, df.columns):
        fig.text(x, top - head_h / 2, str(c), ha="center", va="center", fontsize=fontsize)
    y = top - head_h
    hline(y, 0.8)
    for _, row in df.iterrows():
        y -= row_h
        for j, (x, v) in enumerate(zip(xs, row.values)):
            text = "" if pd.isna(v) else str(v)
            if j == 0:
                fig.text(left + 0.005, y + row_h / 2, text, ha="left", va="center", fontsize=fontsize)
            else:
                fig.text(x, y + row_h / 2, text, ha="center", va="center", fontsize=fontsize)
    hline(y, 1.5)
    if note:
        fig.text(left, y - 0.02, note, ha="left", va="top", fontsize=8.5, wrap=True)
    pdf.savefig(fig)
    plt.close(fig)


def cover(pdf, sections):
    fig = plt.figure(figsize=A4)
    fig.text(0.5, 0.80, "数字金融使用与大学生过度消费倾向", ha="center", fontsize=20, weight="bold")
    fig.text(0.5, 0.75, "实证分析结果（图表汇编）", ha="center", fontsize=15)
    fig.text(0.15, 0.66, "目录", fontsize=13, weight="bold")
    for i, s in enumerate(sections):
        fig.text(0.17, 0.62 - i * 0.026, s, fontsize=10.5)
    pdf.savefig(fig)
    plt.close(fig)


def build(tables, figures):
    """tables: [(标题, DataFrame, 注释)]；figures: [(标题, Figure)]"""
    with PdfPages(PDF_FILE) as pdf:
        cover(pdf, [t[0] for t in tables] + [f[0] for f in figures])
        for title, df, note in tables:
            draw_table(pdf, title, df, note)
        for _, fig in figures:
            fig.set_size_inches(A4[0], A4[0] * fig.get_figheight() / fig.get_figwidth())
            pdf.savefig(fig)
            plt.close(fig)

    with pd.ExcelWriter(TAB_DIR / "全部结果表格.xlsx") as writer:
        for title, df, _ in tables:
            df.to_excel(writer, sheet_name=title.split(" ")[0][:31], index=False)
