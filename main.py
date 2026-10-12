import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import TAB_DIR, FIG_DIR, NOTE
from data_prep import build_dataset
import s01_descriptive, s02_scale, s03_correlation, s04_baseline
import s05_robustness, s06_mechanism, s07_heterogeneity
import make_report

TAB_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = build_dataset()

desc = s01_descriptive.run(df)
scale = s02_scale.run(df)
corr = s03_correlation.run(df)
base = s04_baseline.run(df)
robust = s05_robustness.run(df, base["m4"])
mech = s06_mechanism.run(df)
het = s07_heterogeneity.run(df)

BASE_NOTE = NOTE + "变量名称后括号中的类别为相应分类变量的基准组。"
CTRL_NOTE = NOTE + "控制变量包括性别、年级、专业、学校城市层级、生源地、独生子女、父母平均受教育程度和月均生活费。"
STAR_NOTE = "注：***、**、*分别表示在1%、5%、10%水平上显著。"

tables = [
    ("表4-1 Cronbach's α系数表", scale["alpha"], ""),
    ("表4-2 结构效度检验结果", scale["kmo"], ""),
    ("表4-3 标准化因子载荷结果", scale["loading"], "注：采用主成分法提取公因子。"),
    ("表5-1 核心变量描述性统计表", desc["core"], "注：ZOP、ZDFU、ZFL、ZSFL和ZOFL均为Z-score标准化变量；"
                                          "ParentEdu数值越大表示父母平均受教育程度越低。"),
    ("表5-2 分类控制变量频数分布", desc["freq"], ""),
    ("表6-1 Pearson相关系数表", corr["corr"], STAR_NOTE),
    ("表6-2 基本回归结果", base["table"], BASE_NOTE),
    ("表6-2附 模型(4)控制变量联合显著性检验", base["tests"], "注：基于稳健协方差矩阵的Wald F检验。"),
    ("表6-3 模型替换稳健性检验", robust["ologit"], CTRL_NOTE + "Ordered Logit被解释变量为五个题项总分（5—25）。"),
    ("表6-4 维度拆分稳健性检验", robust["dims"], CTRL_NOTE + "各维度变量均已标准化。"),
    ("表6-5 替换被解释变量检验", robust["items"], CTRL_NOTE + "各分项被解释变量均已标准化。"),
    ("表6-6 多重共线性检验（VIF）", robust["vif"], "注：基于模型(4)的解释变量计算。"),
    ("表7-1 机制分析回归结果", mech["table"], BASE_NOTE),
    ("表7-2 异质性检验表(1)", het["group1"], NOTE + "各组回归均剔除对应的分组变量。"),
    ("表7-3 异质性检验表(2)", het["group2"], NOTE + "父母平均教育水平按样本均值分组。"),
    ("表7-4 异质性交互项检验", het["interaction"], NOTE + "边际效应为ZDFU对ZOP的边际影响；"
                                               "交互项p值用于判断组间差异是否显著。"),
]

figures = [("图5-1 样本人口学特征", desc["figs"][0]),
           ("图5-2 生活费来源分布", desc["figs"][1]),
           ("图5-3 数字金融服务使用情况", desc["figs"][2]),
           ("图5-4 数字信贷主要用途", desc["figs"][3]),
           ("图6-1 核心变量Pearson相关性热力图", corr["fig"])]

make_report.build(tables, figures)
print("分析完成：output/结果.pdf")
