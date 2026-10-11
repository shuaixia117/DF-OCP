"""回归估计与表格整理的通用函数。"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from config import BASE, VALUE_LABELS, VAR_TITLES, CATS_M4


def design(d, cont=(), cats=(), const=True):
    """构造解释变量矩阵：分类变量生成 k-1 个虚拟变量，剔除空列与共线列。"""
    parts = [d[list(cont)].astype(float)]
    for var in cats:
        for lv in sorted(d[var].unique()):
            if lv != BASE[var]:
                parts.append((d[var] == lv).astype(float).rename(f"{var}_{int(lv)}"))
    X = pd.concat(parts, axis=1)
    X = X.loc[:, X.abs().sum() > 0]
    if const:
        X.insert(0, "const", 1.0)
    keep = []
    for c in X.columns:
        if np.linalg.matrix_rank(X[keep + [c]].values) > len(keep):
            keep.append(c)
    return X[keep]


def ols(df, y, cont=(), cats=()):
    """OLS 回归，异方差稳健标准误（HC1）。"""
    d = df[[y, *cont, *cats]].dropna()
    X = design(d, cont, cats)
    return sm.OLS(d[y], X).fit(cov_type="HC1", use_t=True)


def wald_test(res, cols):
    """稳健 Wald 检验：F = b'(V)^-1 b / q，约束线性相关时按秩处理。"""
    b = res.params[cols].values
    V = res.cov_params().loc[cols, cols].values
    r = np.linalg.matrix_rank(V)
    F = float(b @ np.linalg.pinv(V) @ b) / r
    return F, r, stats.f.sf(F, r, res.df_resid)


def model_f(res):
    """模型整体 F 统计量；稳健协方差矩阵不满秩时无法计算，记为缺失。"""
    cols = [c for c in res.params.index if c != "const"]
    V = res.cov_params().loc[cols, cols].values
    if np.linalg.matrix_rank(V) < len(cols):
        return np.nan
    return wald_test(res, cols)[0]


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def coef_cell(res, var):
    """返回 (系数+星号, (标准误))。"""
    if var not in res.params.index:
        return "", ""
    b, se, p = res.params[var], res.bse[var], res.pvalues[var]
    b = 0.0 if abs(b) < 5e-4 else b
    return f"{b:.3f}{stars(p)}", f"({se:.3f})"


def fmt_f(x):
    return "." if pd.isna(x) else f"{x:.2f}"


def regression_table(models, keep, labels, show_controls=False, stats_rows=("N", "R2", "AdjR2", "F")):
    """整理回归结果：系数行 + 标准误行 + 统计量行。"""
    rows = []
    for v in keep:
        cells = [coef_cell(m, v) for m in models.values()]
        rows.append([labels.get(v, v)] + [c[0] for c in cells])
        rows.append([""] + [c[1] for c in cells])
    if show_controls:
        rows.append(["控制变量"] + ["Yes"] * len(models))
    stat_fun = {
        "N": ("N", lambda m: f"{int(m.nobs)}"),
        "R2": ("R²", lambda m: f"{m.rsquared:.3f}"),
        "AdjR2": ("Adj R²", lambda m: f"{m.rsquared_adj:.3f}"),
        "F": ("F", lambda m: fmt_f(model_f(m))),
    }
    for s in stats_rows:
        name, f = stat_fun[s]
        rows.append([name] + [f(m) for m in models.values()])
    return pd.DataFrame(rows, columns=["变量"] + list(models))


def full_table(models, core, cats=CATS_M4, cont_extra=("parentedu",), labels=None):
    """带全部控制变量系数的回归表，分类变量下列出各类别（括号内为基准组）。"""
    labels = labels or {}
    rows = []

    def add(name, var):
        cells = [coef_cell(m, var) for m in models.values()]
        rows.append([name] + [c[0] for c in cells])
        rows.append([""] + [c[1] for c in cells])

    for v in core:
        add(labels.get(v, v), v)
    for var in cats:
        if var == "le":
            for v in cont_extra:
                add("ParentEdu", v)
        base_lab = VALUE_LABELS[var][BASE[var]]
        rows.append([f"{VAR_TITLES[var]}（{base_lab}）"] + [""] * len(models))
        for lv, lab in VALUE_LABELS[var].items():
            if lv != BASE[var]:
                add(f"  {lab}", f"{var}_{lv}")
    add("常数项", "const")
    rows.append(["N"] + [f"{int(m.nobs)}" for m in models.values()])
    rows.append(["R²"] + [f"{m.rsquared:.3f}" for m in models.values()])
    rows.append(["Adj R²"] + [f"{m.rsquared_adj:.3f}" for m in models.values()])
    rows.append(["F"] + [fmt_f(model_f(m)) for m in models.values()])
    return pd.DataFrame(rows, columns=["变量"] + list(models))
