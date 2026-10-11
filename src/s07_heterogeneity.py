"""异质性检验：分组回归与交互项检验。"""
import pandas as pd
from scipy import stats
from config import CATS_M4
from utils import ols, coef_cell, stars

# (维度, 分组变量, 从控制变量中剔除的变量, [(取值, 组名), ...], 交互项模型中的比较组取值)
GROUPS = [
    ("性别", "gender", "gender", [(1, "男"), (2, "女")], 2),
    ("专业", "econmajor", "major", [(1, "经管类"), (0, "非经管类")], 1),
    ("生源地", "urbanorigin", "origin", [(1, "城市"), (0, "非城市")], 1),
    ("学校城市", "highcity", "city", [(1, "新一线及以上"), (0, "其他")], 1),
    ("独生子女", "single", "single", [(1, "是"), (0, "否")], 1),
    ("父母平均教育水平", "highpedu", "parentedu", [(1, "高"), (0, "低")], 1),
    ("生活费水平", "highle", "le", [(0, "2000元及以下"), (1, "高于2000元")], 1),
]


def controls(drop):
    cont = [] if drop == "parentedu" else ["parentedu"]
    return cont, [c for c in CATS_M4 if c != drop]


def subgroup_table(df, groups):
    """分组回归：各组分别估计 ZDFU 的系数。"""
    cols, coef, se, n, adj = [], [], [], [], []
    for dim, g, drop, levels, _ in groups:
        cont, cats = controls(drop)
        for val, lab in levels:
            m = ols(df[df[g] == val], "ZOP", ["ZDFU", *cont], cats)
            c, s = coef_cell(m, "ZDFU")
            cols.append(f"{dim}\n{lab}")
            coef.append(c)
            se.append(s)
            n.append(f"{int(m.nobs)}")
            adj.append(f"{m.rsquared_adj:.3f}")
    rows = [["ZDFU"] + coef, [""] + se, ["N"] + n,
            ["Control Variable"] + ["Yes"] * len(cols), ["Adj R²"] + adj]
    return pd.DataFrame(rows, columns=[""] + cols)


def interaction_table(df):
    """全样本交互项模型：ZOP = ZDFU + G + ZDFU×G + 控制变量，检验组间差异。"""
    rows = []
    for dim, g, drop, levels, cmp_val in GROUPS:
        cont, cats = controls(drop)
        names = dict(levels)
        cmp_lab = names[cmp_val]
        ref_lab = next(lab for val, lab in levels if val != cmp_val)
        d = df.assign(G=(df[g] == cmp_val).astype(int))
        d["ZDFU_G"] = d["ZDFU"] * d["G"]
        m = ols(d, "ZOP", ["ZDFU", "G", "ZDFU_G", *cont], cats)
        b, V = m.params, m.cov_params()
        eff_cmp = b["ZDFU"] + b["ZDFU_G"]
        se_cmp = (V.loc["ZDFU", "ZDFU"] + V.loc["ZDFU_G", "ZDFU_G"] + 2 * V.loc["ZDFU", "ZDFU_G"]) ** 0.5
        p_cmp = 2 * stats.t.sf(abs(eff_cmp / se_cmp), m.df_resid)
        rows.append([dim, ref_lab, cmp_lab,
                     f"{b['ZDFU']:.3f}{stars(m.pvalues['ZDFU'])}",
                     f"{eff_cmp:.3f}{stars(p_cmp)}",
                     f"{b['ZDFU_G']:.3f}", f"({m.bse['ZDFU_G']:.3f})",
                     f"{m.pvalues['ZDFU_G']:.3f}", int(m.nobs)])
    return pd.DataFrame(rows, columns=["维度", "参照组", "比较组", "参照组边际效应", "比较组边际效应",
                                       "交互项系数", "标准误", "p值", "N"])


def run(df):
    return {"group1": subgroup_table(df, GROUPS[:4]), "group2": subgroup_table(df, GROUPS[4:]),
            "interaction": interaction_table(df)}
