"""稳健性检验：Ordered Logit、分维度回归、替换被解释变量、多重共线性。"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.miscmodels.ordinal_model import OrderedModel
from config import CATS_M4
from utils import ols, design, regression_table, stars

CONT = ["parentedu"]


def ordered_logit(df, y="OPsum", x="ZDFU"):
    """有序 Logit，稳健标准误（三明治估计，n/(n-1) 小样本调整）。"""
    d = df[[y, x, *CONT, *CATS_M4]].dropna()
    X = design(d, [x, *CONT], CATS_M4, const=False)
    endog = pd.Series(pd.Categorical(d[y], ordered=True), index=d.index)
    res = OrderedModel(endog, X, distr="logit").fit(
        method="bfgs", maxiter=5000, gtol=1e-8, disp=False, cov_type="HC0")
    n, k = len(d), X.shape[1]
    b = res.params[X.columns].values
    Vb = res.cov_params().loc[X.columns, X.columns].values * n / (n - 1)
    se = np.sqrt(np.diag(Vb))

    counts = d[y].value_counts()
    ll0 = float((counts * np.log(counts / n)).sum())     # 仅含切点的零模型
    wald = float(b @ np.linalg.solve(Vb, b))
    i = list(X.columns).index(x)
    p = 2 * stats.norm.sf(abs(b[i] / se[i]))
    return {"coef": b[i], "se": se[i], "p": p, "OR": np.exp(b[i]), "llf": res.llf,
            "pseudo_r2": 1 - res.llf / ll0, "wald": wald, "wald_df": k,
            "wald_p": stats.chi2.sf(wald, k), "n": n}


def model_table(df, base_model):
    """表6-3 OLS 与 Ordered Logit 对比。"""
    r = ordered_logit(df)
    b_ols, se_ols, p_ols = base_model.params["ZDFU"], base_model.bse["ZDFU"], base_model.pvalues["ZDFU"]
    rows = [
        ["ZDFU", f"{b_ols:.3f}{stars(p_ols)}", f"{r['coef']:.3f}{stars(r['p'])}"],
        ["", f"({se_ols:.3f})", f"({r['se']:.3f})"],
        ["Odds Ratio", "", f"{r['OR']:.3f}"],
        ["Control Variable", "Yes", "Yes"],
        ["N", f"{int(base_model.nobs)}", f"{r['n']}"],
        ["Adj R² / Pseudo R²", f"{base_model.rsquared_adj:.3f}", f"{r['pseudo_r2']:.3f}"],
        ["Log-Likelihood", "", f"{r['llf']:.3f}"],
        [f"Wald χ²({r['wald_df']})", "", f"{r['wald']:.2f}"],
    ]
    return pd.DataFrame(rows, columns=["Variable", "OLS", "Ordered Logit"])


def dimension_table(df):
    """表6-4 数字金融使用分维度回归。"""
    dims = {"zpay": "支付维度", "zcredit": "信贷维度", "zinvest": "理财维度", "zfreq": "强度维度"}
    models = {name: ols(df, "ZOP", [v, *CONT], CATS_M4) for v, name in dims.items()}
    models["联合回归"] = ols(df, "ZOP", [*dims, *CONT], CATS_M4)
    return regression_table(models, list(dims), dims, show_controls=True)


def item_table(df, base_model):
    """表6-5 以五个分项题目替换被解释变量。"""
    names = ["非必要消费", "冲动下单", "超计划消费", "即时拥有", "经济压力"]
    models = {"(1) 综合指标": base_model}
    for i, name in enumerate(names, 1):
        models[f"({i + 1}) {name}"] = ols(df, f"zy{i}", ["ZDFU", *CONT], CATS_M4)
    return regression_table(models, ["ZDFU"], {}, show_controls=True)


def vif_table(df):
    """基准模型(4)解释变量的方差膨胀因子。"""
    d = df[["ZOP", "ZDFU", *CONT, *CATS_M4]].dropna()
    X = design(d, ["ZDFU", *CONT], CATS_M4)
    rows = []
    for c in X.columns.drop("const"):
        r2 = sm.OLS(X[c], X.drop(columns=c)).fit().rsquared
        rows.append([c, 1 / (1 - r2), 1 - r2])
    t = pd.DataFrame(rows, columns=["变量", "VIF", "1/VIF"]).sort_values("VIF", ascending=False)
    t.loc[len(t)] = ["Mean VIF", t["VIF"].mean(), np.nan]
    t["VIF"] = t["VIF"].map("{:.2f}".format)
    t["1/VIF"] = t["1/VIF"].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    return t


def run(df, base_model):
    return {"ologit": model_table(df, base_model), "dims": dimension_table(df),
            "items": item_table(df, base_model), "vif": vif_table(df)}
