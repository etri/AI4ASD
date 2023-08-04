# RRBNet: Offcial Pytorch Implementation



This repository provides the official PyTorch implementation of the following paper:
<img src="fig_architecture.png" width="1000">

> Video-based Stereotyped Behavior Detection for Early Diagnosis of Autism Spectrum Disorder
>
> Abstract: -

---

## Updates
08/04/2023: Project page built

All code related to this work will be made available. 

## Get Started
- Clone this repo and install dependencies:
```bash
install  Python>=3.8 environment with PyTorch>=1.8
git clone this repository
cd pbr4RRB
pip install -r requirements.txt
```

## Test
- Prepare download video file (SSBD and ESBD).
After preparing data, the data folder should be like the format below:

```
data
├─ video
│ ├─ ESBD    
│ │ ├─ Dataset
│ │ | ├─ ArmFlapping
│ │ | ├─ ArmFlapping_per_frame_periodicity
│ │ | ├─ ......
│ ├─ SSBD    
│ │ ├─ Dataset
│ │ | ├─ ArmFlapping
│ │ | ├─ ArmFlapping_per_frame_periodicity
│ │ | ├─ ......

```

- To test code, run the command below:
```python
python demo_test.py 
```

## Training
- To train code, run the command below:
```python
python main.py 
```

## Model

We provide our pre-trained models. 
You can test our network by putting pre-trained models on checkpoints folder.
- RRB_LA_Net_tr_countix.checkpoint : Repetition detector trained on Counitx.
- RRB_RA_Net_tr_ESBD_parsing.checkpoint : Action classifier (VST) trained on parsed ESBD.
- RRB_RA_Net_tr_SSBD_parsing.checkpoint : Action classifier (VST) trained on parsed SSBD.



## Experimental Results

Examples of result images on the *SSBD* and *ESBD* dataset. 
The green and red colors denote whether a frame belongs to a repetitive segment or not, respectively.

<img src="fig_result.png" width="1000">

## LICENSE
Please see [LICENSE.md](../LICENSE.md).

## Contact
If you have any question or comment, please email <ch.yoo@etri.re.kr>.
