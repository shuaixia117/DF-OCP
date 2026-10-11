"""读取分析数据并构造变量。"""
import pandas as pd
from config import DATA_FILE, SURVEY_FILE, SURVEY_COLUMNS


def zscore(s):
    """Z-score 标准化（样本标准差）。"""
    return (s - s.mean()) / s.std()


def load_data():
    df = pd.read_excel(DATA_FILE, sheet_name="数据")
    survey = pd.read_excel(SURVEY_FILE).iloc[:, list(SURVEY_COLUMNS)]
    survey.columns = list(SURVEY_COLUMNS.values())
    return df.merge(survey, on="id", how="left").sort_values("id").reset_index(drop=True)


def build_dataset():
    df = load_data()
    items = ["y1", "y2", "y3", "y4", "y5"]

    # 被解释变量：过度消费倾向
    df["OP"] = df[items].mean(axis=1)
    df["ZOP"] = zscore(df["OP"])
    df["OPsum"] = df[items].sum(axis=1)
    for i, y in enumerate(items, 1):
        df[f"zy{i}"] = zscore(df[y])

    # 核心解释变量：数字金融使用指数（四维度等权平均）
    df["pay_norm"] = (df["pay_share"] - 1) / 3
    df["creditfreq_norm"] = df["credit_freq"] / 2
    dims = ["pay_norm", "creditpay", "creditfreq_norm", "invest"]
    df["DFU"] = df[dims].mean(axis=1)
    df["ZDFU"] = zscore(df["DFU"])
    for new, old in zip(["zpay", "zcredit", "zfreq", "zinvest"], dims):
        df[new] = zscore(df[old])

    # 机制变量：金融素养
    df["SFL"] = df[["sfl_stock", "sfl_funds", "sfl_bonds", "sfl_credits"]].mean(axis=1)
    df["OFL"] = df[["ofl1", "ofl2", "ofl3", "ofl4"]].mean(axis=1)
    df["ZSFL"] = zscore(df["SFL"])
    df["ZOFL"] = zscore(df["OFL"])
    df["FL"] = (df["ZSFL"] + df["ZOFL"]) / 2
    df["ZFL"] = zscore(df["FL"])

    # 父母平均受教育程度（数值越大学历越低）
    df["parentedu"] = df[["fatheredu", "motheredu"]].mean(axis=1)

    # 异质性分组变量
    df["econmajor"] = (df["major"] == 2).astype(int)
    df["urbanorigin"] = df["origin"].isin([1, 2]).astype(int)
    df["highcity"] = df["city"].isin([1, 2]).astype(int)
    df["highpedu"] = (df["parentedu"] <= df["parentedu"].mean()).astype(int)
    df["highle"] = df["le"].isin([3, 4]).astype(int)
    return df
