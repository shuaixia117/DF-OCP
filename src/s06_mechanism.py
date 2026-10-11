"""机制分析：数字金融使用与金融素养。"""
from config import CATS_M4
from utils import ols, full_table


def run(df):
    models = {y: ols(df, y, ["ZDFU", "parentedu"], CATS_M4) for y in ["ZFL", "ZSFL", "ZOFL"]}
    return {"table": full_table(models, ["ZDFU"])}
