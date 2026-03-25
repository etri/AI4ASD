"""
CARE-VL: Stage 1 - VLM Inference Script
Performs clip-level inference on SIIC video data using the fine-tuned CARE-VL model.
Outputs MC-QA predictions and detailed captions for each clip.
"""

import argparse
import os
import json
import time
import copy
import warnings

import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from decord import VideoReader, cpu
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support as score

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

warnings.filterwarnings("ignore")


# ============================================================================
# Utility Functions
# ============================================================================

def load_tagging_info(tagging_path):
    """Load tagging information from the event_info.txt file."""
    with open(tagging_path, 'r') as file:
        tagging_data = file.readlines()
    event_tagging_info = [line.split("\t")[0].strip() for line in tagging_data if "# EventTaggingInfo - Value" in line]
    return event_tagging_info


def tagging_value_to_response_mapping(tagging_values, index):
    """Map tagging values to positive/negative response labels."""
    response_mapping = {
        "03": {0: "positive response", 1: "negative response"},             # name-calling
        "04": {0: "positive response", 1: "negative response", 2: "negative response"},  # eye-contact
        "06": {0: "positive response", 1: "negative response"},             # imitation-behavior
        "07": {0: "positive response", 1: "negative response", 2: "negative response"},  # social-smiling
        "08": {0: "positive response", 2: "negative response", 3: "negative response"},  # pointing
    }
    if not tagging_values or tagging_values[0].strip() == '':
        return "negative response"

    responses = response_mapping[index].get(int(tagging_values[0]), "negative response")
    return responses


def index_to_indicator_mapping(indicator):
    """Map indicator index to indicator name and explanation."""
    indicator_mapping = {
        "03": "name-calling",
        "04": "eye-contact",
        "06": "imitation-behavior",
        "07": "social-smiling",
        "08": "pointing"
    }
    explanations = {
        "03": "'Name-calling' involves the child responding to their name when called. This tests their auditory attention and recognition of their own name.",
        "04": "'Eye-contact' measures whether the child can maintain or initiate eye contact, which is critical for social interaction and communication.",
        "06": "'Imitation-behavior' assesses the child's ability to mimic actions or gestures, reflecting their observational learning skills.",
        "07": "'Social-smiling' evaluates the child's ability to smile in response to social stimuli, indicating their emotional and social engagement.",
        "08": "'Pointing' tests whether the child can use gestures to indicate objects or events, which is an important non-verbal communication skill."
    }
    return indicator_mapping[indicator], explanations[indicator]


def load_video(video_path, max_frames_num):
    """Load and uniformly sample frames from a video."""
    if isinstance(video_path, str):
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames


# ============================================================================
# Main Inference
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="CARE-VL Stage 1: VLM Inference")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the fine-tuned CARE-VL model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory of video data (e.g., /path/to/SIIC_data/PNU)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save results (default: same as model_path)")
    parser.add_argument("--max_frames", type=int, default=16,
                        help="Maximum number of frames to sample per video")
    parser.add_argument("--question_type", type=str, nargs='+', default=['MCQA', 'DC'],
                        choices=['MCQA', 'DC'],
                        help="Types of questions to evaluate: MCQA (multiple-choice) and/or DC (detailed caption)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.model_path

    # Load model
    model_name = "llava_qwen"
    device = "cuda"
    device_map = "auto"

    print(f"Loading model from: {args.model_path}")
    overwrite_config = {
        "tie_word_embeddings": False,
        "use_cache": True,
        "vocab_size": 152064,
    }
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        args.model_path, None, model_name, device_map=device_map,
        attn_implementation="sdpa", overwrite_config=overwrite_config
    )
    model.eval()

    # List participants
    base_path = args.data_dir
    participants = [p for p in os.listdir(base_path)
                    if not p.startswith("PNU") and not p.startswith(".")]

    # Initialize metrics
    total_videos = 0
    correct_answers = 0
    indicator_results = {
        "03": {'pred': [], 'gt': []},  # name-calling
        "04": {'pred': [], 'gt': []},  # eye-contact
        "06": {'pred': [], 'gt': []},  # imitation-behavior
        "07": {'pred': [], 'gt': []},  # social-smiling
        "08": {'pred': [], 'gt': []},  # pointing
    }
    label_pred = []
    label_gt = []
    mcqa_list = []
    dc_list = []

    # Run inference
    for participant in tqdm(participants, desc="Processing participants"):
        total_time = 0
        total_clips = 0

        for index in ["03", "04", "06", "07", "08"]:
            for subfolder in ["000", "001", "002"]:
                video_dir = f"{base_path}/{participant}/{index}/rec"
                tagging_path = f"{base_path}/{participant}/{index}/01/{subfolder}/event_info.txt"

                if not os.path.exists(video_dir) or not os.path.exists(tagging_path):
                    continue

                tagging_info = load_tagging_info(tagging_path)
                processed_responses = tagging_value_to_response_mapping(tagging_info, index)
                ground_truth = "Yes" if processed_responses == "positive response" else "No"

                for video_file in os.listdir(video_dir):
                    if video_file.startswith(f"{subfolder}_"):
                        video_path = os.path.join(video_dir, video_file)
                        print(f'Video file: {video_path}')

                        # Load video frames
                        video_frames = load_video(video_path, args.max_frames)
                        frames = image_processor.preprocess(video_frames, return_tensors="pt")[
                            "pixel_values"].half().cuda()
                        image_tensors = [frames]

                        # Prepare clip ID
                        base_name = os.path.splitext(video_file)[0]
                        clip_id = f"{participant}/{index}/rec/{base_name}"
                        video_field = f"{participant}/{index}/rec/{video_file}"

                        for i in range(len(args.question_type)):
                            indicator, explanation = index_to_indicator_mapping(index)

                            if args.question_type[i] == 'MCQA':
                                question = (
                                    f"<image>\n Did the child respond appropriately during "
                                    f"the stimulus-response interval of {indicator}?\nYes\nNo"
                                )
                            elif args.question_type[i] == 'DC':
                                question = (
                                    "The input video is part of a Social Interaction-Inducing "
                                    "Content(SIIC)-based test that tests whether the child responds "
                                    "appropriately to certain indicators based on stimuli displayed "
                                    "on the monitor.\n"
                                    "Your description should prioritize the child's responses, and "
                                    "interactions with the stimuli presented during the test.\n"
                                    "Avoid describing irrelevant details, and clearly state whether "
                                    "the child responded appropriately to the stimuli.\n"
                                    f"Here is an explanation of the indicator: {explanation}\n"
                                    f"<image>\n Generate video descriptions in detail, focusing "
                                    f"specifically on the child's '{indicator}' indicator being "
                                    f"currently tested.\n"
                                )

                            conv_template = "qwen_1_5"
                            conv = copy.deepcopy(conv_templates[conv_template])
                            conv.append_message(conv.roles[0], question)
                            conv.append_message(conv.roles[1], None)

                            prompt_question = conv.get_prompt()
                            input_ids = tokenizer_image_token(
                                prompt_question, tokenizer, IMAGE_TOKEN_INDEX,
                                return_tensors="pt"
                            ).unsqueeze(0).to(device)
                            image_sizes = [frame.size for frame in video_frames]

                            start_time = time.time()
                            cont = model.generate(
                                input_ids,
                                images=image_tensors,
                                image_sizes=image_sizes,
                                do_sample=False,
                                temperature=0,
                                max_new_tokens=4096,
                                modalities=["video"],
                            )
                            elapsed_time = time.time() - start_time
                            total_time += elapsed_time
                            total_clips += 1

                            text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
                            gpt_response = text_outputs[0]
                            print(f"Model output: {gpt_response}")

                            if args.question_type[i] == 'MCQA':
                                print(f"Ground truth: {ground_truth}")
                                model_answer = text_outputs[0].strip()
                                total_videos += 1

                                pred_value = 1 if model_answer.startswith('Yes') else 0
                                gt_value = 1 if ground_truth.startswith('Yes') else 0

                                label_pred.append(pred_value)
                                label_gt.append(gt_value)

                                if model_answer.startswith(ground_truth):
                                    correct_answers += 1

                                indicator_results[index]['pred'].append(pred_value)
                                indicator_results[index]['gt'].append(gt_value)

                            # Save in JSON format
                            conv_item = {
                                "id": clip_id,
                                "video": video_field,
                                "conversations": [
                                    {"from": "human", "value": question},
                                    {"from": "gpt",   "value": gpt_response}
                                ]
                            }
                            if args.question_type[i] == 'MCQA':
                                mcqa_list.append(conv_item)
                            else:
                                dc_list.append(conv_item)

        if total_clips > 0:
            avg_time = total_time / total_clips
            print(f"\nTotal clips: {total_clips}, Avg inference time: {avg_time:.2f}s")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    if mcqa_list:
        mcqa_path = os.path.join(args.output_dir, "mcqa_inference_results.json")
        with open(mcqa_path, "w", encoding="utf-8") as f:
            json.dump(mcqa_list, f, ensure_ascii=False, indent=2)
        print(f"\nMCQA results saved to {mcqa_path}")

    if dc_list:
        dc_path = os.path.join(args.output_dir, "caption_inference_results.json")
        with open(dc_path, "w", encoding="utf-8") as f:
            json.dump(dc_list, f, ensure_ascii=False, indent=2)
        print(f"Caption results saved to {dc_path}")

    # Print evaluation metrics
    if 'MCQA' in args.question_type and total_videos > 0:
        precision, recall, fscore, _ = score(label_gt, label_pred, average="binary")
        acc = correct_answers / total_videos
        cm = confusion_matrix(label_gt, label_pred)
        cr = classification_report(label_gt, label_pred, target_names=['No', 'Yes'])

        print('\n====== Final Accuracy on Binary Classification ======')
        print(f"Accuracy:  {acc:.3f}")
        print(f"Recall:    {recall:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"F-score:   {fscore:.3f}")
        print(f"\nConfusion Matrix:\n{cm}")
        print(f"\nClassification Report:\n{cr}")

        # Save results to file
        results_path = os.path.join(args.output_dir, 'results_test.txt')
        with open(results_path, 'a') as fp:
            fp.write(f'\ntest_Accuracy: {acc:.3f}, recall: {recall:.3f}, precision: {precision:.3f}, f_score: {fscore:.3f}\n')
            fp.write(f"\nConfusion matrix:\n{cm}")
            fp.write(f"\n\nClassification Report:\n{cr}")

            # Per-indicator results
            for ind in ["03", "04", "06", "07", "08"]:
                pred_list = indicator_results[ind]['pred']
                gt_list = indicator_results[ind]['gt']

                if len(pred_list) == 0:
                    fp.write(f"\nIndicator {ind}: No videos found.\n")
                    continue

                precision_ind, recall_ind, fscore_ind, _ = score(gt_list, pred_list, average="binary")
                acc_ind = sum(1 for p, g in zip(pred_list, gt_list) if p == g) / len(pred_list)
                cm_ind = confusion_matrix(gt_list, pred_list)
                cr_ind = classification_report(gt_list, pred_list, target_names=['No', 'Yes'])

                fp.write(f"\n===== Indicator {ind} results =====\n")
                fp.write(f"Number of videos: {len(pred_list)}\n")
                fp.write(f"Accuracy:  {acc_ind:.3f}\n")
                fp.write(f"Precision: {precision_ind:.3f}\n")
                fp.write(f"Recall:    {recall_ind:.3f}\n")
                fp.write(f"F-score:   {fscore_ind:.3f}\n")
                fp.write(f"Confusion Matrix:\n{cm_ind}\n")
                fp.write(f"Classification Report:\n{cr_ind}\n")

        print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
