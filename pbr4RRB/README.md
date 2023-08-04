# RRBNet: Offcial Pytorch Implementation



This repository provides the official PyTorch implementation of the following paper:
<img src="fig_architecture.png" width="800">

> Video-based Stereotyped Behavior Detection for Early Diagnosis of Autism Spectrum Disorder_표절검사결과서_copykiller
>
> Abstract: The ability to point to objects for sharing social purpose or attention is known as one of the key indicators in distinguishing children with typical development (TD) from those with autism spectrum disorder (ASD). However, there is a lack of datasets specifically tailored for children’s pointing gestures. This lack of training data from the target domain becomes a major factor in the performance degradation of conventional supervised CNNs due to domain shift. Toward an effective and practical solution, we propose an end-to-end learning scheme for domain generalized pointing gesture recognition adopting self-supervised regularization (SSR). To prove the effectiveness of our method in real-world situations, we designed a Social Interaction-Inducing Content (SIIC)-based ASD diagnostic system and collected an ASD-Pointing dataset consisting of 40 TD and ASD children. Through extensive experiments on our collected datasets, we achieved an ASD screening accuracy of 72.5%, showing that pointing ability can play a vital role as an indicator in distinguishing between ASD and TD.

---

## Updates
08/04/2023: Project page built

All code related to this work will be made available. 

## Get Started
- Clone this repo and install dependencies:
```bash
git clone this repository
cd pbr4RRB
install Pytorch (1.8)
pip install -r requirements.txt
```

## Test
- Prepare recorded video file in .mkv format.
After preparing data, the data folder should be like the format below:

```
data
├─ video
│ ├─ ESBD    
│ │ ├─ subject_nmae
│ │ | ├─ xxxx.mkv
│ │ ├─ ......
│ │
│ ├─ SSBD    
│ │ ├─ subject_nmae
│ │ | ├─ xxxx.mkv
│ │ ├─ ......
│ │

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
You can test our network by putting pre-trained models on checkpoints/logs/resnet50_ntu or resnet50_ntu_SimSiam/model_best.checkpoint

- ResNet-50 (baseline): https://drive.google.com/file/d/
- Proposed<sub>SimSiam</sub>: https://drive.google.com/file/d/



## Experimental Results

Examples of result images on the *ASD-Pointing* dataset. The green and red colors represent test cases where pointing is performed
and not performed, respectively. The videos were captured with four Azure Kinect cameras in three living lab spaces.

<!--<img src="fig_result.png" width="1000">-->

## LICENSE
Please see [LICENSE.md](../LICENSE.md).

## Contact
If you have any question or comment, please email <hw_kim@etri.re.kr>.
