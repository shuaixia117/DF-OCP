"""过度消费倾向量表的信度与效度检验。"""
import numpy as np
import pandas as pd
from scipy import stats

ITEMS = ["y1", "y2", "y3", "y4", "y5"]
ITEM_NAMES = ["第一问", "第二问", "第三问", "第四问", "第五问"]


def cronbach_alpha(x):
    k = x.shape[1]
    return k / (k - 1) * (1 - x.var().sum() / x.sum(axis=1).var())


def alpha_table(df):
    """表4-1 Cronbach's α 系数（整体及逐项剔除）。"""
    x = df[ITEMS]
    rows = [["整体", f"{cronbach_alpha(x):.3f}"]]
    for item, name in zip(ITEMS, ITEM_NAMES):
        rows.append([f"剔除{name}", f"{cronbach_alpha(x.drop(columns=item)):.3f}"])
    return pd.DataFrame(rows, columns=["选取范围", "Cronbach's α系数"])


def kmo(R):
    inv = np.linalg.inv(R)
    d = np.sqrt(np.outer(np.diag(inv), np.diag(inv)))
    partial = -inv / d
    off = ~np.eye(len(R), dtype=bool)
    r2, p2 = (R[off] ** 2).sum(), (partial[off] ** 2).sum()
    return r2 / (r2 + p2)


def bartlett(R, n):
    p = len(R)
    chi2 = -(n - 1 - (2 * p + 5) / 6) * np.log(np.linalg.det(R))
    dof = p * (p - 1) // 2
    return chi2, dof, stats.chi2.sf(chi2, dof)


def validity_tables(df):
    """表4-2 KMO与巴特利特球形检验；表4-3 第一主成分因子载荷。"""
    x = df[ITEMS]
    R = x.corr().values
    chi2, dof, p = bartlett(R, len(x))
    eigval, eigvec = np.linalg.eigh(R)
    order = eigval.argsort()[::-1]
    eigval, eigvec = eigval[order], eigvec[:, order]
    loading = np.abs(eigvec[:, 0]) * np.sqrt(eigval[0])

    t42 = pd.DataFrame([
        ["KMO", f"{kmo(R):.4f}"],
        ["巴特利特球形检验 近似卡方", f"{chi2:.3f}"],
        ["自由度", f"{dof}"],
        ["显著性", "<0.001" if p < 0.001 else f"{p:.3f}"],
        ["特征根大于1的因子数", f"{(eigval > 1).sum()}"],
        ["第一因子特征根", f"{eigval[0]:.4f}"],
        ["第一因子解释方差", f"{eigval[0] / len(ITEMS) * 100:.2f}%"],
    ], columns=["检验项目", "结果"])
    t43 = pd.DataFrame({"题项": ITEM_NAMES,
                        "第一因子载荷": [f"{v:.3f}" for v in loading],
                        "共同度": [f"{v ** 2:.3f}" for v in loading]})
    return t42, t43


def run(df):
    t42, t43 = validity_tables(df)
    return {"alpha": alpha_table(df), "kmo": t42, "loading": t43}
