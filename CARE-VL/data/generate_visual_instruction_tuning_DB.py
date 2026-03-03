"""
CARE-VL: Instruction-Tuning Dataset Generation
Generates caption and MC-QA instruction-tuning datasets from SIIC video data.
Uses a general-purpose VLM to create detailed captions with label-guided reasoning.
"""

import argparse
import os
import json
import gc
import copy
import warnings

import torch
import numpy as np
import cv2
from tqdm import tqdm
from decord import VideoReader, cpu

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.conversation import conv_templates
from llava.constants import IMAGE_TOKEN_INDEX


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
        "03": {0: "positive response", 1: "negative response"},
        "04": {0: "positive response", 1: "negative response", 2: "negative response"},
        "06": {0: "positive response", 1: "negative response"},
        "07": {0: "positive response", 1: "negative response", 2: "negative response"},
        "08": {0: "positive response", 2: "negative response", 3: "negative response"},
    }
    if not tagging_values or tagging_values[0].strip() == '':
        return "negative response"
    return response_mapping[index].get(int(tagging_values[0]), "negative response")


def index_to_indicator_mapping(indicator):
    """Map indicator index to name and explanation."""
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


def load_video(video_path, max_frames_num=16):
    """Load video frames using decord."""
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    if total_frame_num == 0:
        print(f"Empty video file: {video_path}")
        return None
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    return vr.get_batch(frame_idx).asnumpy()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate SIIC instruction-tuning dataset")
    parser.add_argument("--model_path", type=str, default="lmms-lab/llava-onevision-qwen2-7b-ov",
                        help="Path to the VLM model for caption generation")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory of SIIC video data")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory to save the generated JSON datasets")
    parser.add_argument("--output_prefix", type=str, default="SIIC_visual_instruct",
                        help="Prefix for output filenames")
    parser.add_argument("--max_frames", type=int, default=16,
                        help="Maximum frames to sample per video")
    args = parser.parse_args()

    gc.collect()
    torch.cuda.empty_cache()

    # Load model
    model_name = "llava_qwen"
    device = "cuda"
    device_map = "auto"

    print(f"Loading model: {args.model_path}")
    overwrite_config = {'tie_word_embeddings': False, 'use_cache': True, "vocab_size": 152064}
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, None, model_name, device_map=device_map,
        attn_implementation="sdpa", overwrite_config=overwrite_config
    )
    model.eval()

    # List participants
    participants = [p for p in os.listdir(args.data_dir)
                    if not p.startswith("PNU") and not p.startswith(".")]

    output_data_dc = []
    output_data_mcqa = []

    for participant in tqdm(participants, desc="Processing participants"):
        for index in ["03", "04", "06", "07", "08"]:
            for subfolder in ["000", "001", "002"]:
                video_dir = f"{args.data_dir}/{participant}/{index}/rec"
                tagging_path = f"{args.data_dir}/{participant}/{index}/01/{subfolder}/event_info.txt"

                if not os.path.exists(video_dir) or not os.path.exists(tagging_path):
                    continue

                tagging_info = load_tagging_info(tagging_path)
                processed_responses = tagging_value_to_response_mapping(tagging_info, index)
                binary_answer = "Yes" if processed_responses == "positive response" else "No"

                for video_file in os.listdir(video_dir):
                    gc.collect()
                    torch.cuda.empty_cache()

                    if not video_file.startswith(f"{subfolder}_"):
                        continue

                    video_path = os.path.join(video_dir, video_file)
                    print(f'Video file: {video_path}')

                    # Load video frames
                    video_frames = load_video(video_path, args.max_frames)
                    if video_frames is None:
                        continue

                    frames = image_processor.preprocess(
                        video_frames, return_tensors="pt"
                    )["pixel_values"].half().cuda()
                    image_tensors = [frames]

                    # Prepare prompts
                    indicator, explanation = index_to_indicator_mapping(index)

                    question_default = (
                        "The input video is part of a Social Interaction-Inducing Content"
                        "(SIIC)-based test that tests whether the child responds "
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

                    # Label-guided reasoning: hint with ground truth for better captions
                    question_aug = (
                        question_default +
                        f"\nHint: The child is now showing '{processed_responses}' "
                        f"to '{indicator}' in the corresponding input video."
                    )

                    conv_template = "qwen_1_5"
                    conv = copy.deepcopy(conv_templates[conv_template])
                    conv.append_message(conv.roles[0], question_aug)
                    conv.append_message(conv.roles[1], None)

                    prompt_question = conv.get_prompt()
                    input_ids = tokenizer_image_token(
                        prompt_question, tokenizer, IMAGE_TOKEN_INDEX,
                        return_tensors="pt"
                    ).unsqueeze(0).to(device)
                    image_sizes = [frame.size for frame in video_frames]

                    cont = model.generate(
                        input_ids, images=image_tensors, image_sizes=image_sizes,
                        do_sample=False, temperature=0, max_new_tokens=4096,
                        modalities=["video"],
                    )

                    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
                    gpt_response = text_outputs[0]
                    print(gpt_response)

                    # Build clip ID
                    tmp = video_dir.split('/')
                    clip_id = f"{tmp[-3]}/{tmp[-2]}/{tmp[-1]}/{video_file.split('.')[0]}"
                    video_field = f"{tmp[-3]}/{tmp[-2]}/{tmp[-1]}/{video_file}"

                    # Caption entry (without hint)
                    output_data_dc.append({
                        "id": clip_id,
                        "video": video_field,
                        "conversations": [
                            {"from": "human", "value": question_default},
                            {"from": "gpt", "value": gpt_response}
                        ]
                    })

                    # MC-QA entry
                    output_data_mcqa.append({
                        "id": clip_id,
                        "video": video_field,
                        "conversations": [
                            {"from": "human",
                             "value": f"<image>\n Did the child respond appropriately during the stimulus-response interval of {indicator}?\nYes\nNo"},
                            {"from": "gpt", "value": binary_answer}
                        ]
                    })

    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)

    dc_path = os.path.join(args.output_dir, f"{args.output_prefix}_caption.json")
    with open(dc_path, "w", encoding="utf-8") as f:
        json.dump(output_data_dc, f, indent=4, ensure_ascii=False)

    mcqa_path = os.path.join(args.output_dir, f"{args.output_prefix}_mcqa.json")
    with open(mcqa_path, "w", encoding="utf-8") as f:
        json.dump(output_data_mcqa, f, indent=4, ensure_ascii=False)

    print(f"Caption dataset saved: {dc_path} ({len(output_data_dc)} entries)")
    print(f"MC-QA dataset saved:   {mcqa_path} ({len(output_data_mcqa)} entries)")


if __name__ == "__main__":
    main()
