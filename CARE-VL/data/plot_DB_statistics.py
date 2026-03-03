import os
import matplotlib.pyplot as plt
import numpy as np

# 데이터 경로
SITE_A_DIR = "/path/to/SIIC_data/SNUBH"  # Update to your Site A data path
SITE_B_DIR = "/path/to/SIIC_data/PNU"    # Update to your Site B data path

# 5개의 social indicators (폴더명 기준)
INDICATOR_MAP = {
    "03": "Response to Name",
    "04": "Eye Contact",
    "06": "Imitation Behavior",
    "07": "Social Smiling",
    "08": "Pointing"
}

# View 개수 설정
NUM_VIEWS_A = 5  # Site A: 5개 View
NUM_VIEWS_B = 4  # Site B: 4개 View


# 색상 설정 (논문용 고대비 컬러)
COLOR_SITE_A_POS = "#4C72B0"  # 진한 블루 (Positive Site A)
COLOR_SITE_A_NEG = "#92C6FF"  # 연한 블루 (Negative Site A)
COLOR_SITE_B_POS = "#DD8452"  # 진한 오렌지 (Positive Site B)
COLOR_SITE_B_NEG = "#FFB482"  # 연한 오렌지 (Negative Site B)


# 결과 저장용 변수
site_data = {
    "Site A": {"clip_counts": {key: 0 for key in INDICATOR_MAP.values()},
               "positive_counts": {key: 0 for key in INDICATOR_MAP.values()},
               "negative_counts": {key: 0 for key in INDICATOR_MAP.values()}},
    "Site B": {"clip_counts": {key: 0 for key in INDICATOR_MAP.values()},
               "positive_counts": {key: 0 for key in INDICATOR_MAP.values()},
               "negative_counts": {key: 0 for key in INDICATOR_MAP.values()}}
}

def parse_data(data_dir, site_name, num_views):
    """ Site A 또는 Site B 데이터 파싱 """
    for subject_folder in os.listdir(data_dir):
        subject_path = os.path.join(data_dir, subject_folder)
        if not os.path.isdir(subject_path):
            continue  # 폴더가 아닐 경우 스킵

        for indicator_code, indicator_name in INDICATOR_MAP.items():
            indicator_path = os.path.join(subject_path, indicator_code, "01")  # 01 폴더 접근
            if not os.path.isdir(indicator_path):
                continue

            for trial_folder in os.listdir(indicator_path):  # Trial 폴더 (000, 001, 002)
                tagging_path = os.path.join(indicator_path, trial_folder, "event_info.txt")
                if not os.path.exists(tagging_path):
                    continue  # event_info.txt가 없으면 스킵

                # 하나의 Trial에 대해 num_views 만큼 Clip이 생성됨
                site_data[site_name]["clip_counts"][indicator_name] += num_views

                # event_info.txt 읽기
                with open(tagging_path, "r") as file:
                    tagging_data = file.readlines()
                value = [line.split("\t")[0].strip() for line in tagging_data if "# EventTaggingInfo - Value" in line]

                if not value or value[0].strip() == '':
                    site_data[site_name]["negative_counts"][indicator_name] += num_views
                    continue

                if int(value[0]) == 0:
                    site_data[site_name]["positive_counts"][indicator_name] += num_views
                else:
                    site_data[site_name]["negative_counts"][indicator_name] += num_views

# Site A & Site B 데이터 파싱 (각 View 개수 적용)
parse_data(SITE_A_DIR, "Site A", NUM_VIEWS_A)
parse_data(SITE_B_DIR, "Site B", NUM_VIEWS_B)

# 📊 **Site A vs. Site B: 지표별 Positive vs. Negative 비율 비교 (Stacked Bar Chart)**
plt.figure(figsize=(8, 5))
bar_width = 0.35
ind = np.arange(len(INDICATOR_MAP))

# Site A Positive/Negative
plt.bar(ind - bar_width/2, site_data["Site A"]["positive_counts"].values(), width=bar_width, label="Positive (Site A)", color=COLOR_SITE_A_POS)
plt.bar(ind - bar_width/2, site_data["Site A"]["negative_counts"].values(), width=bar_width, bottom=list(site_data["Site A"]["positive_counts"].values()), label="Negative (Site A)", color=COLOR_SITE_A_NEG)

# Site B Positive/Negative
plt.bar(ind + bar_width/2, site_data["Site B"]["positive_counts"].values(), width=bar_width, label="Positive (Site B)", color=COLOR_SITE_B_POS)
plt.bar(ind + bar_width/2, site_data["Site B"]["negative_counts"].values(), width=bar_width, bottom=list(site_data["Site B"]["positive_counts"].values()), label="Negative (Site B)", color=COLOR_SITE_B_NEG)

plt.xticks(ind, list(INDICATOR_MAP.values()), rotation=30, ha="right", fontsize=10)
plt.ylabel("Number of Responses", fontsize=11)
plt.title("Positive vs. Negative Responses per Indicator (Site A vs. Site B)", fontsize=12, fontweight="bold")
plt.legend(fontsize=9)

plt.tight_layout()
plt.savefig('Instruction_DB_stats.png', bbox_inches='tight')
plt.show()
