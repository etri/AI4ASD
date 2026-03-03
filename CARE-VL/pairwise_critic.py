"""
CARE-VL: Pairwise Evaluation with LLaVA-Critic
Compares CARE-VL (expert) captions against general VLM captions
using LLaVA-Critic to determine which is better.
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
    def __init__(self, json_path, config: EvaluatorConfig):
        self.config = config
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
                return explanation_dict[indicator_code]
        return '', ''

    def evaluate_two_video(self, video_path, general_answer, expert_answer):
        """Compare two captions (expert vs general) for the same video."""
        try:
            self.model.eval()
            frames = self.extract_frames(video_path, self.config.max_frames)
            processed_frames = self.process_frames(frames)
            indicator, explanation = self.find_indicator(video_path)

            prompt = f"""
Given a video and a corresponding question, evaluate if the LMM response describes the child's behavior accurately.
Determine WHICH ANSWER IS BETTER and explain your reasoning with specific details.

Question: [The input video is part of a SIIC-based test that tests whether the child responds appropriately to {indicator} based on stimuli displayed on the monitor.]

Caption 1: {expert_answer}
Caption 2: {general_answer}

[EVALUATION CRITERIA]
- Does it provide meaningful insights into the child's response that could be useful for evaluation?
- Does the caption prioritize whether the child responded negatively or positively to {indicator}?

[EVALUATION FORMAT]
<answer> is better than <answer>
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
            print(result)
            return result

        except Exception as e:
            logger.error(f"Error evaluating video {video_path}: {e}")
            return None

    def evaluate_pairwise(self, output_path, max_example=None):
        """Evaluate all video pairs in the JSON file."""
        json_data = self.load_json()
        results = []

        if max_example:
            json_data = json_data[:max_example]

        for item in tqdm(json_data, desc="Evaluating videos"):
            video_path = item['video_id']
            expert_caption = item['expert_caption']
            general_caption = item['general_caption']

            result = self.evaluate_two_video(video_path, general_caption, expert_caption)
            if result:
                results.append({
                    'video_path': video_path,
                    'general_answer': general_caption,
                    'expert_answer': expert_caption,
                    'llava_critic_result': result
                })

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

        return results


# ============================================================================
# Result Analysis
# ============================================================================

def analyze_results(output_path):
    """Analyze pairwise comparison results."""
    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pattern_expert = re.compile(r'(answer 1|caption 1) is better', re.IGNORECASE)
    pattern_general = re.compile(r'(answer 2|caption 2) is better', re.IGNORECASE)
    pattern_tie = re.compile(r'both caption', re.IGNORECASE)

    count_expert = 0
    count_general = 0
    count_tie = 0

    for item in data:
        critic_result = item.get("llava_critic_result", "")

        if pattern_expert.search(critic_result):
            count_expert += 1
        elif pattern_general.search(critic_result):
            count_general += 1
        elif pattern_tie.search(critic_result):
            count_tie += 1

    total = len(data)
    other = total - count_expert - count_general - count_tie

    print(f"\n========== Pairwise Comparison Results ==========")
    print(f"Expert VLM wins:  {count_expert} ({count_expert/total*100:.1f}%)")
    print(f"General VLM wins: {count_general} ({count_general/total*100:.1f}%)")
    print(f"Tie:              {count_tie} ({count_tie/total*100:.1f}%)")
    print(f"Unclassified:     {other} ({other/total*100:.1f}%)")
    print(f"Total:            {total}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="CARE-VL: Pairwise Critic Evaluation")
    parser.add_argument("--json_path", type=str, required=True,
                        help="Path to pairwise comparison JSON (with expert_caption and general_caption)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save critic evaluation results")
    parser.add_argument("--max_example", type=int, default=None,
                        help="Maximum number of examples to evaluate")
    parser.add_argument("--analyze_only", action="store_true",
                        help="Only analyze existing results without running evaluation")
    args = parser.parse_args()

    if args.analyze_only:
        analyze_results(args.output_path)
        return

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    if not os.path.exists(args.output_path):
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=4)

    critic = VideoLLaVACritic(args.json_path, config=EvaluatorConfig())
    critic.evaluate_pairwise(args.output_path, args.max_example)

    # Analyze results
    analyze_results(args.output_path)


if __name__ == "__main__":
    main()
