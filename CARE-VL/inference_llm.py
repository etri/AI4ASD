"""
CARE-VL: Stage 2 - Few-shot LLM Classification
Aggregates clip-level VLM outputs (captions + MC-QA) per subject and uses
an LLM with few-shot exemplars and DSM-5 criteria to classify ASD vs TD
at the subject level.

Pipeline:
  1. Load VLM inference results (caption + MC-QA JSONs) for test data
  2. Load few-shot exemplar data (caption + MC-QA JSONs) from source site
  3. Filter to front-view camera clips only
  4. Build few-shot prompt with DSM-5 criteria + exemplar subjects
  5. Run LLM inference per test subject
  6. Evaluate ASD/TD classification performance
"""

import argparse
import os
import re
import json
import time
from collections import defaultdict

import torch
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support as score


# ============================================================================
# DSM-5 Criteria
# ============================================================================

DSM_5_CRITERIA = """
DSM-5 ASD Diagnostic Criteria:
A. Persistent deficits in social communication and social interaction.
A.1. Deficits in social-emotional reciprocity, ranging, for example, from abnormal social approach and failure of normal back-and-forth conversation; to reduced sharing of interests, emotions, or affect; to failure to initiate or respond to social interactions.
A.2. Deficits in nonverbal communicative behaviors used for social interaction, ranging, for example, from poorly integrated verbal and nonverbal communication; to abnormalities in eye contact and body language or deficits in understanding and use of gestures; to a total lack of facial expressions and nonverbal communication.
A.3. Deficits in developing, maintaining, and understanding relationships, ranging, for example, from difficulties adjusting behavior to suit various social contexts; to difficulties in sharing imaginative play or in making friends; to absence of interest in peers.
"""


# ============================================================================
# Data Loading & Parsing
# ============================================================================

def load_subject_label(xlsx_path):
    """
    Load subject labels (ASD/TD) from Excel file.
    Column 0: subject id, Column 1: ASD/TD/OD
    Returns: dict { subject_id -> 'ASD'/'TD' }
    """
    df = pd.read_excel(xlsx_path)
    subject_label = {}
    for _, row in df.iterrows():
        sbj_id = str(row.iloc[0]).strip()
        label_str = str(row.iloc[1]).strip().upper()
        if label_str in ["ASD", "TD"]:
            subject_label[sbj_id] = label_str
    return subject_label


def load_json_data(caption_path, mcqa_path):
    """Load caption and MCQA JSON files."""
    with open(caption_path, "r", encoding="utf-8") as f:
        caption_data = json.load(f)
    with open(mcqa_path, "r", encoding="utf-8") as f:
        mcqa_data = json.load(f)
    return caption_data, mcqa_data


def parse_caption_json(caption_list):
    """Returns dict: { clip_id -> detailed_caption_text }"""
    caption_dict = {}
    for item in caption_list:
        clip_id = item["id"]
        for conv in item["conversations"]:
            if conv["from"] == "gpt":
                caption_dict[clip_id] = conv["value"]
                break
    return caption_dict


def parse_mcqa_json(mcqa_list):
    """Returns dict: { clip_id -> { "question": str, "answer": str } }"""
    mcqa_dict = {}
    for item in mcqa_list:
        clip_id = item["id"]
        question_str = None
        answer_str = None
        for conv in item["conversations"]:
            if conv["from"] == "human":
                question_str = conv["value"].strip()
            elif conv["from"] == "gpt":
                answer_str = conv["value"].strip()
        if question_str and answer_str:
            mcqa_dict[clip_id] = {"question": question_str, "answer": answer_str}
    return mcqa_dict


def get_subject_id(clip_id):
    """Extract subject ID from clip ID."""
    return clip_id.split("/")[0]


def build_subject_clips_only_front_view(caption_dict, mcqa_dict, target_data):
    """
    Build subject clips mapping, filtering to front-view camera only.

    Args:
        caption_dict: { clip_id -> caption_text }
        mcqa_dict: { clip_id -> { question, answer } }
        target_data: 'SNUBH' or 'PNU' to apply site-specific camera filtering

    Returns: { subject_id -> list of {clip_id, caption, mcqa} }
    """
    subject_clips = defaultdict(list)
    for clip_id in caption_dict:
        if clip_id not in mcqa_dict:
            continue

        # Front-view camera filtering by site
        clip_basename = clip_id.split("/")[-1]
        if target_data == 'SNUBH':
            if not clip_basename.startswith(("000_001047292912", "001_001047292912", "002_001047292912")):
                continue
        elif target_data == 'PNU':
            if not (clip_basename.startswith(("000_000219201812", "001_000219201812", "002_000219201812")) or
                    clip_basename.startswith(("000_000494494512", "001_000494494512", "002_000494494512"))):
                continue

        sbj_id = get_subject_id(clip_id)
        subject_clips[sbj_id].append({
            "clip_id": clip_id,
            "caption": caption_dict[clip_id],
            "mcqa": mcqa_dict[clip_id]
        })
    return subject_clips


# ============================================================================
# LLM Inference
# ============================================================================

def parse_llm_output(text: str) -> str:
    """
    Parse LLM output to classify ASD vs TD.
    A. => TD, B. => ASD. Fallback to 'ASD' if ambiguous.
    """
    text_lower = text.lower()
    found_a = "a." in text_lower
    found_b = "b." in text_lower

    if found_a and not found_b:
        return "TD"
    elif found_b and not found_a:
        return "ASD"
    else:
        return "ASD"  # fallback


def run_inference_with_fixed_few_shot(subject_clips, subject_label, model_dir,
                                       asd_examples, td_examples):
    """
    Perform subject-level ASD/TD classification using few-shot LLM inference.

    The prompt includes:
    - System message with task description
    - DSM-5 criteria for ASD
    - Few-shot exemplars (ASD and TD subjects with their clip data)
    - Test subject's clip data

    Args:
        subject_clips: { subject_id -> list of clips }
        subject_label: { subject_id -> 'ASD'/'TD' }
        model_dir: HuggingFace model ID or path
        asd_examples: list of subject IDs to use as ASD exemplars
        td_examples: list of subject IDs to use as TD exemplars

    Returns: dict with evaluation metrics
    """
    few_shot_ids = set(asd_examples + td_examples)

    # Build pipeline
    print(f"Loading pipeline with model: {model_dir}")
    pipe = pipeline(
        "text-generation",
        model=model_dir,
        tokenizer=model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_new_tokens=512
    )

    y_true, y_pred = [], []

    for sbj_id, clips in tqdm(subject_clips.items(), desc="Classifying subjects"):
        if sbj_id not in subject_label:
            continue

        # Skip subjects used as few-shot exemplars
        if sbj_id in few_shot_ids:
            continue

        gt_label = subject_label[sbj_id]

        # Build messages with DSM-5 criteria and few-shot examples
        messages = [
            {"role": "system", "content": (
                "You are a helpful assistant for ASD screening. "
                "Given Q-R clip observations, you will decide if the child is "
                "ASD(Autism Spectrum Disorder) or TD(Typical development)."
            )},
            {"role": "system", "content": DSM_5_CRITERIA}
        ]

        # Add ASD exemplars
        for idx, ex_id in enumerate(asd_examples):
            if ex_id in subject_clips and ex_id in subject_label:
                ex_clips = subject_clips[ex_id]
                ex_user_text = f"Subject ID: {idx}\n"
                for i, clip in enumerate(ex_clips, start=1):
                    ex_user_text += (
                        f"\n[Clip {i}]\nQuestion: {clip['mcqa']['question']}\n"
                        f"Answer: {clip['mcqa']['answer']}\nCaption: {clip['caption']}\n"
                    )
                ex_user_text += (
                    "\nBased on the observations above, determine whether the child "
                    "is more likely to have ASD or TD.\n"
                    "A. TD\nB. ASD\n"
                    "Please provide your answer by stating the letter followed by the "
                    "full option and explain why did you make that decision.\n"
                    "Please ensure that the conclusion aligns with the detailed "
                    "observations provided.\n"
                )
                messages.append({"role": "user", "content": ex_user_text})
                messages.append({"role": "assistant", "content": "B. ASD"})

        # Add TD exemplars
        for idx, ex_id in enumerate(td_examples):
            if ex_id in subject_clips and ex_id in subject_label:
                ex_clips = subject_clips[ex_id]
                ex_user_text = f"Subject ID: {idx}\n"
                for i, clip in enumerate(ex_clips, start=1):
                    ex_user_text += (
                        f"\n[Clip {i}]\nQuestion: {clip['mcqa']['question']}\n"
                        f"Answer: {clip['mcqa']['answer']}\nCaption: {clip['caption']}\n"
                    )
                ex_user_text += (
                    "\nBased on the observations above, determine whether the child "
                    "is more likely to have ASD or TD.\n"
                    "A. TD\nB. ASD\n"
                    "Please provide your answer by stating the letter followed by the "
                    "full option and explain why did you make that decision.\n"
                    "Please ensure that the conclusion aligns with the detailed "
                    "observations provided.\n"
                )
                messages.append({"role": "user", "content": ex_user_text})
                messages.append({"role": "assistant", "content": "A. TD"})

        # Add test subject data
        main_user_content = f"Subject ID: {sbj_id}\n"
        for i, clip in enumerate(clips, start=1):
            main_user_content += (
                f"\n[Clip {i}]\nQuestion: {clip['mcqa']['question']}\n"
                f"Answer: {clip['mcqa']['answer']}\nCaption: {clip['caption']}\n"
            )
        main_user_content += (
            "\nBased on the observations above, determine whether the child "
            "is more likely to have ASD or TD.\n"
            "A. TD\nB. ASD\n"
            "Please provide your answer by stating the letter followed by the "
            "full option and explain why did you make that decision.\n"
            "Please ensure that the conclusion aligns with the detailed "
            "observations provided.\n"
        )
        messages.append({"role": "user", "content": main_user_content})

        # Run inference
        start_time = time.time()
        outputs = pipe(
            messages,
            max_new_tokens=512,
            do_sample=False,
            temperature=0
        )
        elapsed_time = time.time() - start_time
        print(f"\nInference time: {elapsed_time:.2f}s")

        generated_text = outputs[0]["generated_text"][-1]['content']
        pred_label = parse_llm_output(generated_text)

        print(f"Subject: {sbj_id}, Predicted: {pred_label}, GT: {gt_label}")
        print(f"LLM output: {generated_text}")

        y_true.append(gt_label)
        y_pred.append(pred_label)

    # Evaluate
    labels = ["ASD", "TD"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cr = classification_report(y_true, y_pred, labels=labels)
    precision, recall, fscore, _ = score(y_true, y_pred, average="binary", pos_label="ASD")
    accuracy = sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true) if y_true else 0

    print("\n===== Evaluation Results =====")
    print(f"Subjects tested: {len(y_true)}")
    print(f"Accuracy:        {accuracy:.3f}")
    print(f"Recall(ASD):     {recall:.3f}")
    print(f"Precision(ASD):  {precision:.3f}")
    print(f"F1-score(ASD):   {fscore:.3f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Classification Report:\n{cr}")

    return {
        "accuracy": accuracy,
        "precision_asd": precision,
        "recall_asd": recall,
        "f1_asd": fscore,
        "confusion_matrix": cm.tolist(),
        "n_tested": len(y_true)
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="CARE-VL Stage 2: Few-shot LLM Classification")

    # Test data (e.g., PNU)
    parser.add_argument("--test_caption_json", type=str, required=True,
                        help="Path to test set caption inference results JSON")
    parser.add_argument("--test_mcqa_json", type=str, required=True,
                        help="Path to test set MCQA inference results JSON")
    parser.add_argument("--test_tagging_xlsx", type=str, required=True,
                        help="Path to test set subject label Excel file")
    parser.add_argument("--test_site", type=str, default="PNU", choices=["PNU", "SNUBH"],
                        help="Test site name for camera filtering")

    # Few-shot exemplar data (e.g., SNUBH)
    parser.add_argument("--fewshot_caption_json", type=str, required=True,
                        help="Path to few-shot exemplar caption JSON")
    parser.add_argument("--fewshot_mcqa_json", type=str, required=True,
                        help="Path to few-shot exemplar MCQA JSON")
    parser.add_argument("--fewshot_tagging_xlsx", type=str, required=True,
                        help="Path to few-shot exemplar label Excel file")
    parser.add_argument("--fewshot_site", type=str, default="SNUBH", choices=["PNU", "SNUBH"],
                        help="Few-shot site name for camera filtering")

    # Few-shot exemplar subject IDs
    parser.add_argument("--asd_examples", type=str, nargs='+', required=True,
                        help="Subject IDs for ASD exemplars (e.g., AI-153-03 AI-200-03)")
    parser.add_argument("--td_examples", type=str, nargs='+', required=True,
                        help="Subject IDs for TD exemplars (e.g., AI-232-03 AI-240-04)")

    # Model
    parser.add_argument("--llm_model", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                        help="HuggingFace model ID for LLM classification")

    args = parser.parse_args()

    # 1) Load test data
    test_caption_data, test_mcqa_data = load_json_data(args.test_caption_json, args.test_mcqa_json)
    test_caption_dict = parse_caption_json(test_caption_data)
    test_mcqa_dict = parse_mcqa_json(test_mcqa_data)
    test_subject_clips = build_subject_clips_only_front_view(test_caption_dict, test_mcqa_dict, args.test_site)
    test_label_dict = load_subject_label(args.test_tagging_xlsx)

    # 2) Load few-shot exemplar data
    fs_caption_data, fs_mcqa_data = load_json_data(args.fewshot_caption_json, args.fewshot_mcqa_json)
    fs_caption_dict = parse_caption_json(fs_caption_data)
    fs_mcqa_dict = parse_mcqa_json(fs_mcqa_data)
    fs_subject_clips = build_subject_clips_only_front_view(fs_caption_dict, fs_mcqa_dict, args.fewshot_site)
    fs_label_dict = load_subject_label(args.fewshot_tagging_xlsx)

    # 3) Filter few-shot exemplars
    few_shot_ids = args.asd_examples + args.td_examples
    fs_few_shot_clips = {sbj_id: clips for sbj_id, clips in fs_subject_clips.items() if sbj_id in few_shot_ids}
    fs_few_shot_labels = {sbj_id: label for sbj_id, label in fs_label_dict.items() if sbj_id in few_shot_ids}

    # 4) Combine test + few-shot data
    subject_clips_combined = dict(test_subject_clips)
    subject_clips_combined.update(fs_few_shot_clips)
    label_dict_combined = dict(test_label_dict)
    label_dict_combined.update(fs_few_shot_labels)

    # 5) Run few-shot inference
    run_inference_with_fixed_few_shot(
        subject_clips_combined, label_dict_combined, args.llm_model,
        args.asd_examples, args.td_examples
    )


if __name__ == "__main__":
    main()
