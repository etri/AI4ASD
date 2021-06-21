## **Prerequisites**
- Facial expression dataset (AffectNet, RAF, etc.)
- Metafile for the dataset
  - Example
  ```
  File_Name Emotion_Label Dataset_Label
  /AffectNet/basic/Image/bb/train_04757.jpg 0 0
  ```

## **Quick Start**
Command to run our module:
```
python ESP_predict_test.py --model resnet50 --<path to model>
```

## **Train a model**
To train a model(resnet-50), use this sample command:
```
python libFER_train_test.py --model resnet50 --root_path <path_to_dataset? --train_list <path_to_train_metafile> --val_list <path_to_validation_metafile> --save_path <path_to_result_dir> 
```

## **LICENSE**
Please see [LICENSE.md](../LICENSE.md).

## Contact
If you have any question or comment, please email <byungok.han@etri.re.kr>.

## Citation
If you use this code for your research, please cite our paper:

```
@ARTICLE{9174732,
  author={Han, ByungOk and Yun, Woo-Han and Yoo, Jang-Hee and Kim, Won Hwa},
  journal={IEEE Access}, 
  title={Toward Unbiased Facial Expression Recognition in the Wild via Cross-Dataset Adaptation}, 
  year={2020},
  volume={8},
  number={},
  pages={159172-159181},
  doi={10.1109/ACCESS.2020.3018738}}
```

