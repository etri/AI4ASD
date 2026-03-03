import os
import glob

BASE_DIR = "/path/to/SIIC_data/SNUBH"  # Update to your data path
INDICATOR_LIST = ["03", "04", "06", "07", "08"]  # 호명반응, 눈맞춤 등

def parse_event_info(txt_path):
    """
    event_info.txt 파일을 열어,
    'EventTaggingInfo - Value : X' 값을 찾아서 X 반환
    (없으면 None 반환)
    """
    value_found = None
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 예: "EventTaggingInfo - Value : 0"
            if line.startswith("EventTaggingInfo - Value"):
                # 뒤쪽에 ':'가 있을 것이므로 파싱
                # 보통 "EventTaggingInfo - Value : 0" 형태라고 가정
                # split(':')하면 ["EventTaggingInfo - Value ", " 0"] 정도로 나올 것
                parts = line.split(':')
                if len(parts) >= 2:
                    value_found = parts[-1].strip()  # "0" (문자열)
                break
    return value_found

def load_tagging_info(tagging_path):
    """Load tagging information from the event_info.txt file."""
    with open(tagging_path, 'r') as file:
        tagging_data = file.readlines()
    event_tagging_info = [line.split("\t")[0].strip() for line in tagging_data if "# EventTaggingInfo - Value" in line]
    return event_tagging_info


def tagging_value_to_response_mapping(tagging_values, index):
    """Process tagging values based on the indicator mapping."""
    response_mapping = {
        "03": {0: "positive response", 1: "negative response"}, ## name-calling
        "04": {0: "positive response", 1: "negative response", 2: "negative response"}, ## eye-contact
        "06": {0: "positive response", 1: "negative response"}, ## imitation-behavior
        "07": {0: "positive response", 1: "negative response", 2: "negative response"}, ## social-smiling
        "08": {0: "positive response", 2: "negative response", 3: "negative response"} ## pointing
    }
    # 빈 리스트 확인
    if not tagging_values or tagging_values[0].strip() == '':
        print(f"Empty or invalid tagging value for index {index}. Skipping...")
        #return "unknown"
        return "negative response"

    # 정상적인 값 처리
    #responses = response_mapping[index].get(int(tagging_values[0]), "unknown")
    responses = response_mapping[index].get(int(tagging_values[0]), "negative response")

    return responses

def main():
    # 결과 저장할 딕셔너리 (예시)
    # subject_summary[subject_id] = {
    #   "03": {"000": "정반응", "001": "비반응", ...},
    #   "04": {...},
    #   ...
    # }
    subject_summary = {}

    # (1) Subject 폴더 순회
    subjects = sorted(os.listdir(BASE_DIR))
    for subj_id in subjects:
        subj_path = os.path.join(BASE_DIR, subj_id)
        if not os.path.isdir(subj_path):
            continue  # 폴더 아닐 경우 스킵

        subject_summary[subj_id] = {}

        # (2) Indicator 폴더 순회
        for indicator in INDICATOR_LIST:
            indicator_path = os.path.join(subj_path, indicator)
            if not os.path.isdir(indicator_path):
                continue

            # 일반적으로 indicator_path 안에 "01" 폴더가 존재한다고 가정
            folder_01 = os.path.join(indicator_path, "01")
            if not os.path.isdir(folder_01):
                continue

            # (3) trial 폴더(000, 001, 002 등) 순회
            # 어떤 경우는 3개 trial만 있을 수도 있고, 혹은 더 있을 수도 있으니 glob으로 찾기
            trial_dirs = sorted(os.listdir(folder_01))
            indicator_results = {}

            for trial_dir in trial_dirs:
                trial_path = os.path.join(folder_01, trial_dir)
                if not os.path.isdir(trial_path):
                    continue

                event_info_path = os.path.join(trial_path, "event_info.txt")
                if not os.path.exists(event_info_path):
                    continue

                tagging_info = load_tagging_info(event_info_path)
                processed_responses = tagging_value_to_response_mapping(tagging_info, indicator)
                binary_answer = "Yes" if processed_responses == "positive response" else "No"

                indicator_results[trial_dir] = binary_answer

            subject_summary[subj_id][indicator] = indicator_results

   # (5) 결과 예시 출력
    for subj_id in sorted(subject_summary.keys()):
        print(f"\n===== Subject: {subj_id} =====")
        for indicator in INDICATOR_LIST:
            if indicator not in subject_summary[subj_id]:
                continue
            trials_info = subject_summary[subj_id][indicator]
            if not trials_info:
                continue

            # 예시: "Indicator 03 - 호명반응"
            # 실제로는 indicator 코드별로 의미를 매핑해서 출력 가능
            print(f"  Indicator {indicator}:")
            for trial_id, tag_result in trials_info.items():
                print(f"    Trial {trial_id} -> {tag_result}")
    
    for subj_id, indicator_dict in subject_summary.items():
        total_checks = 0
        correct_count = 0
        for indicator, trials_dict in indicator_dict.items():
            for trial_id, reaction in trials_dict.items():
                total_checks += 1
                if reaction == "Yes":
                    correct_count += 1
        # 비율 계산
        ratio_correct = correct_count / total_checks if total_checks > 0 else 0
        # 간단히 예시:
        if ratio_correct > 0.8:
            print(f"{subj_id}: 정반응 비율 {ratio_correct:.2f} -> 거의 TD에 가깝다?")
        elif ratio_correct < 0.2:
            print(f"{subj_id}: 정반응 비율 {ratio_correct:.2f} -> 거의 ASD에 가깝다?")
        else:
            print(f"{subj_id}: 정반응 비율 {ratio_correct:.2f} -> 중간 케이스")

    # (6) 필요하다면 CSV/엑셀로 저장
    # 예: pandas DataFrame으로 만들거나, CSV에 각 subject/indicator/trial/value를 기록
    # (사용자가 원하는 형식에 맞춰 저장)
    # ...

if __name__ == "__main__":
    main()
