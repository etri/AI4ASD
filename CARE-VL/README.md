# CARE-VL: A Domain-Specialized Vision-Language Model for Early ASD Screening - Official-Pytorch-Implementation

<img src="fig_architecture.png" width="1000">




> CARE-VL: A Domain-Specialized Vision-Language Model for Early ASD Screening
>
> Cheol-Hwan Yoo*, Jang-Hee Yoo, Jae-Yoon Jang
>
> MICCAI 2025 (accepted)
>
>
We propose an autism spectrum disorder (ASD) screening framework that integrates an expert vision-language model (VLM), CARE-VL, with a large language model (LLM)-based aggregation module to assess children's social interactions and derive subject-level ASD/typical development (TD) classifications.  
Our framework processes video data collected using social interaction-inducing content, where medical experts annotated predefined query-response (Q-R) intervals based on key social indicators—such as response to name, eye contact, imitation behavior, social smiling, and pointing—by marking correct responses and assigning subject-level ASD/TD classifications. 
To adapt the general-purpose VLM to the ASD screening domain, we constructed a synthetic instruction-tuning dataset using a label-guided reasoning method on these clinical tags, fine-tuning the model to generate detailed captions and multiple-choice question-answer (MC-QA) pairs, capturing children's critical social behaviors.  
CARE-VL processes Q-R intervals to produce clip-level MC-QA results and descriptive captions, which are then aggregated by an LLM to derive final ASD/TD classification and clinical reasoning.  
Our end-to-end framework combines visual understanding and linguistic reasoning, achieving 84.6\% accuracy for clip-level response prediction and 75.8\% accuracy for subject-level ASD/TD classification. These results demonstrate the potential of our framework as a practical and interpretable tool for early ASD screening and behavioral assessment.
---

## Updates
22/04/2024: Project page built
>
15/11/2024: Project v1.2 update

All code and datasets related to this work will be made available. 

## Get Started
- Clone this repo and install dependencies:
```bash
install  Python>=3.8 environment with PyTorch>=1.8
git clone this repository
cd pbr4RRB
pip install -r requirements.txt
```

## Training
- Firstly, download video file (SSBD, ESBD, Countix, ...).
- To download and parse datasets, run the command below:
```python
python data/download_video_from_URL_SSBD_ESBD.py
python data/parse_SSBD_dataset.py
python data/parse_ESBD_dataset.py
python data/download_video_from_URL_countix.py
python data/parse_countix_dataset.py
```

- To train code, run the command below:
```python
python main_classifier.py
python main_repdetector.py
```

## Test

- To test code, run the command below:
```python
python demo.py --data_choice 'ESBD' or 'SSBD'
```


## Model

We provide our pre-trained models. 
You can test our network by putting pre-trained models on checkpoints folder.
- RRB_LA_Net_tr_countix.checkpoint : Repetition detector trained on Counitx.
https://drive.google.com/file/d/17KD7lQ5xm9lV9zsAqtL0sz9UO4aynJ7m/view?usp=drive_link
- RRB_RA_Net_tr_ESBD_parsing.checkpoint : Action classifier (VST) trained on parsed ESBD.
https://drive.google.com/file/d/1scUBv1v3HSenmo4ix4LKNfVMyOJQpVP8/view?usp=drive_link
- RRB_RA_Net_tr_SSBD_parsing.checkpoint : Action classifier (VST) trained on parsed SSBD.
https://drive.google.com/file/d/19Vamf353wwHyerQPjOkQbJhRqX7YObhs/view?usp=drive_link



## Experimental Results

Examples of result images on the *SSBD* and *ESBD* dataset. 
The green and red colors denote whether a frame belongs to a repetitive segment or not, respectively.

<img src="fig_result.png" width="1000">



## LICENSE
Please see [LICENSE.md](../LICENSE.md).

## Contact
If you have any question or comment, please email <ch.yoo@etri.re.kr>.
