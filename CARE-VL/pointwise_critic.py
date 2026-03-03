"""
CARE-VL: Pointwise Evaluation with LLaVA-Critic
Evaluates the quality of CARE-VL's generated captions using LLaVA-Critic
by scoring each caption against ground-truth annotations.
"""

import argparse
import json
import os
import re
import copy
import warnings
import logging

import torch
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
from decord import VideoReader, cpu

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
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


def parse_ground_truth(video_file, base_path, indicator):
    """
    Extract ground truth (positive/negative response) from the video file path.

    Args:
        video_file: Video file path (e.g., 'A1-1003-4/03/rec/000_000036592912.mp4')
        base_path: Root data directory
        indicator: Indicator code (e.g., '03', '04')

    Returns:
        str: 'positive response' or 'negative response'
    """
    try:
        subject_id_match = re.match(r'(.+?/rec/)', video_file)
        if not subject_id_match:
            return None
        subject_id = subject_id_match.group(1).strip("rec/")

        trial_match = re.search(r'_(\d{3})', video_file)
        if not trial_match:
            return None
        trial_id = trial_match.group(1)

        gt_file_path = '/' + os.path.join(subject_id, "01", trial_id, "event_info.txt")
        if not os.path.exists(gt_file_path):
            return None

        tagging_info = load_tagging_info(gt_file_path)
        return tagging_value_to_response_mapping(tagging_info, indicator)

    except Exception as e:
        logger.error(f"Error parsing ground truth for {video_file}: {e}")
        return None


# ============================================================================
# Evaluator
# ============================================================================

@dataclass
class EvaluatorConfig:
    llava_model: str = "lmms-lab/llava-critic-7b"
    max_frames: int = 16
    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    max_new_tokens: int = 512
    temperature: float = 0.0


class VideoLLaVACritic:
    def __init__(self, json_path, base_path, config: EvaluatorConfig):
        self.config = config
        self.base_path = base_path
        logger.info(f"Loading critic model: {config.llava_model}")

        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            config.llava_model, None, model_name="llava_qwen", device_map="auto",
        )
        self.json_path = json_path
        self.model.eval()

    def load_json(self):
        with open(self.json_path) as f:
            return json.load(f)

    def extract_frames(self, video_path, num_frames):
        """Extract uniformly sampled frames from video."""
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            indices = np.linspace(0, len(vr) - 1, num_frames, dtype=int)
            return vr.get_batch(indices).asnumpy()
        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {e}")
            raise

    def process_frames(self, frames):
        processed = self.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
        return processed.to(device=self.config.device, dtype=self.config.dtype)

    def find_indicator(self, video_path):
        """Extract indicator name and explanation from video path."""
        match = re.search(r'/([^/]+)/rec/', video_path)
        if match:
            indicator_code = match.group(1)
            explanation_dict = {
                '03': ("name-calling", "'Name-calling' involves the child responding to their name when called."),
                '04': ("eye-contact", "'Eye-contact' measures whether the child can maintain or initiate eye contact."),
                '06': ("imitation-behavior", "'Imitation-behavior' assesses the child's ability to mimic actions or gestures."),
                '07': ("social-smiling", "'Social-smiling' evaluates the child's ability to smile in response to social stimuli."),
                '08': ("pointing", "'Pointing' tests whether the child can use gestures to indicate objects or events."),
            }
            if indicator_code in explanation_dict:
                indicator, explanation = explanation_dict[indicator_code]
            else:
                indicator, explanation = '', ''
            return indicator, explanation, indicator_code
        return '', '', ''

    def evaluate_single_video(self, video_path, answer):
        """Evaluate a single video caption with LLaVA-Critic."""
        try:
            self.model.eval()
            frames = self.extract_frames(video_path, self.config.max_frames)
            processed_frames = self.process_frames(frames)
            indicator, explanation, indicator_code = self.find_indicator(video_path)

            ground_truth = parse_ground_truth(video_path, self.base_path, indicator_code)

            prompt = f"""
Given a video and a corresponding question, evaluate if the LMM response describes the child's behavior accurately.
Score the response out of 100 based on the criteria below and explain your reasoning.

Question: [The input video is part of a SIIC-based test that tests whether the child responds appropriately to {indicator} based on stimuli displayed on the monitor.]
The LMM response: [{answer}]

**[EXPERT ANNOTATION - Ground Truth]**
The expert-verified response label for this video is: **[{ground_truth}]**
(PR = Positive Response, NR = No Response)

[EVALUATION CRITERIA]
- Does the description explicitly evaluate the child's response to '{indicator}'?
- Does the description match the expert-labeled response of {ground_truth}?
- Does the response focus on clinically relevant aspects of the child's behavior?

[EVALUATION FORMAT]
Score: <number>
Explanation: <detailed justification>
"""

            conv = conv_templates["qwen_1_5"].copy()
            full_prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
            conv.append_message(conv.roles[0], full_prompt)
            conv.append_message(conv.roles[1], None)

            input_ids = tokenizer_image_token(
                conv.get_prompt(), self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt",
            ).unsqueeze(0).to(self.config.device)

            with torch.inference_mode():
                outputs = self.model.generate(
                    input_ids, images=[processed_frames], modalities=["video"],
                    do_sample=False, temperature=self.config.temperature,
                    max_new_tokens=self.config.max_new_tokens,
                )

            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            match = re.search(r"Score: (\d+)", result)
            score_val = int(match.group(1)) if match else None

            print(f"Score: {score_val}")
            return result, score_val

        except Exception as e:
            logger.error(f"Error evaluating video {video_path}: {e}")
            return None, None

    def evaluate_pointwise(self, output_path, max_example=None):
        """Evaluate all videos in the JSON file."""
        json_data = self.load_json()
        results = []
        running_sum = 0
        count = 0

        if max_example:
            json_data = json_data[:max_example]

        for item in tqdm(json_data, desc="Evaluating videos"):
            video_file = item['video']
            caption = next(conv["value"] for conv in item["conversations"] if conv["from"] == "gpt")
            video_path = os.path.join(self.base_path, video_file)

            result, score_val = self.evaluate_single_video(video_path, caption)
            if score_val is not None:
                results.append({
                    'video_path': video_path,
                    'generated_caption': caption,
                    'llava_critic_result': result,
                    'Score': score_val
                })
                running_sum += score_val
                count += 1

            if count > 0 and (count % 10 == 0 or count == len(json_data)):
                print(f"Processed {count}/{len(json_data)} - Avg Score: {running_sum/count:.2f}")

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

        return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="CARE-VL: Pointwise Critic Evaluation")
    parser.add_argument("--json_path", type=str, required=True,
                        help="Path to caption inference results JSON")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save critic evaluation results")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory of video data")
    parser.add_argument("--max_example", type=int, default=None,
                        help="Maximum number of examples to evaluate")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    if not os.path.exists(args.output_path):
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=4)

    critic = VideoLLaVACritic(args.json_path, args.data_dir, config=EvaluatorConfig())
    critic.evaluate_pointwise(args.output_path, args.max_example)

    # Print average score
    with open(args.output_path, 'r') as f:
        data = json.load(f)
    scores = [item["Score"] for item in data if "Score" in item]
    avg_score = round(sum(scores) / len(scores) if scores else 0, 3)

    print(f'\nAverage Score: {avg_score}')
    print('======== FINISH EVALUATING ========')


if __name__ == "__main__":
    main()
