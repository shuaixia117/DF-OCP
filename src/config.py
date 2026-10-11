"""项目路径、变量编码与标签。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "thesis_analysis_data.xlsx"   # 分析数据
SURVEY_FILE = ROOT / "data" / "处理版.xlsx"               # 问卷原始导出（多选题）
TAB_DIR = ROOT / "output" / "tables"
FIG_DIR = ROOT / "output" / "figures"
PDF_FILE = ROOT / "output" / "结果.pdf"

# 中文字体候选（按顺序选用已安装的第一个）
FONTS = ["SimHei", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC",
         "WenQuanYi Zen Hei", "Noto Sans CJK JP"]

# 问卷中的多选题（仅用于样本消费特征描述）：列位置 → 变量名
SURVEY_COLUMNS = {0: "id", 10: "income_source", 11: "df_services", 17: "credit_use"}

# 分类变量取值标签
VALUE_LABELS = {
    "gender": {1: "男", 2: "女"},
    "grade": {1: "大一", 2: "大二", 3: "大三", 4: "大四/大五", 5: "研究生"},
    "major": {1: "理工类", 2: "经管类", 3: "文史类", 4: "医学类"},
    "city": {1: "一线", 2: "新一线", 3: "二线", 4: "地级市"},
    "origin": {1: "省会及以上", 2: "地级市", 3: "县城", 4: "农村"},
    "single": {0: "否", 1: "是"},
    "le": {1: "1000元以下", 2: "1000-2000元", 3: "2000-3000元", 4: "3000元以上"},
}
VAR_TITLES = {"gender": "Gender", "grade": "Grade", "major": "Major", "city": "City",
              "origin": "Origin", "single": "OnlyChild", "le": "LC"}
CONTROL_NAMES = {"gender": "性别", "grade": "年级", "major": "专业", "city": "学校城市层级",
                 "origin": "生源地", "single": "独生子女", "le": "月均生活费",
                 "parentedu": "父母平均受教育程度"}

# 回归中各分类变量的基准组
BASE = {"gender": 1, "grade": 4, "major": 1, "city": 2, "origin": 1, "single": 0, "le": 3}

# 逐步加入的分类控制变量
CATS_M2 = ["gender", "grade", "major"]
CATS_M3 = CATS_M2 + ["city", "origin"]
CATS_M4 = CATS_M3 + ["single", "le"]

NOTE = "注：括号内为稳健标准误；***、**、*分别表示在1%、5%、10%水平上显著。"
