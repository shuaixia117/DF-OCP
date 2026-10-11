"""基准回归：逐步加入控制变量的 OLS 模型。"""
import pandas as pd
from config import CATS_M2, CATS_M3, CATS_M4, CONTROL_NAMES
from utils import ols, full_table, wald_test

SPECS = {
    "(1) ZOP": ([], []),
    "(2) ZOP": ([], CATS_M2),
    "(3) ZOP": ([], CATS_M3),
    "(4) ZOP": (["parentedu"], CATS_M4),
}


def fit_models(df):
    return {k: ols(df, "ZOP", ["ZDFU", *cont], cats) for k, (cont, cats) in SPECS.items()}


def control_tests(model):
    """模型(4)中各控制变量的联合显著性检验。"""
    rows = []
    for var in CATS_M4 + ["parentedu"]:
        cols = [c for c in model.params.index if c == var or c.startswith(f"{var}_")]
        F, q, p = wald_test(model, cols)
        rows.append([CONTROL_NAMES[var], f"{F:.3f}", q, int(model.df_resid), f"{p:.4f}"])
    return pd.DataFrame(rows, columns=["控制变量", "F统计量", "分子自由度", "分母自由度", "p值"])


def run(df):
    models = fit_models(df)
    return {"table": full_table(models, ["ZDFU"]), "tests": control_tests(models["(4) ZOP"]),
            "m4": models["(4) ZOP"]}
