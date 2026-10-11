"""绘图字体与样式。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from config import FONTS


def setup():
    installed = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in FONTS if f in installed), "sans-serif")
    plt.rcParams.update({
        "font.family": font,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
    })
    return font
