import os
import cv2
import numpy as np
import pickle
from sklearn.utils import shuffle
from PIL import Image
import torch
import torch.nn.functional as F
import os
import numpy as np
import random
import cv2
import torchvision.transforms as transforms
import itertools
import random
from sklearn.utils import shuffle
from utility import *
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvideotransforms import video_transforms, volume_transforms
from sklearn.utils.class_weight import compute_class_weight


def get_train_test_list_SSBD(args):

    ## SSBD
    random.seed('2023-07-03')
    action_cls =['ArmFlapping_parsed', 'Spinning_parsed', 'HeadBanging_parsed']

    folder_dir = '/home/ych/data/SSBD/Dataset_revised_ych/'
    train_list = []
    train_labels = []
    test_list = []
    test_labels = []
    train_test_ratio = 0.8
    k_fold_idx = 1

    for i in range(len(action_cls)):
        file_dir = folder_dir + action_cls[i]
        input_list = []
        input_list += [os.path.join(file_dir, f) for f in sorted(os.listdir(file_dir))
                                            if f.split(".")[-1] == "mp4" or f.split(".")[-1] == "avi"]
        #shuffle
        random.shuffle(input_list)

        ## original dataset split ##
        train_list.append(input_list[:int(len(input_list)*train_test_ratio)])
        labels = [i for j in range(len(train_list[i]))]
        train_labels.append(labels)

        test_list.append(input_list[int(len(input_list)*train_test_ratio):])
        labels = [i for j in range(len(test_list[i]))]
        test_labels.append(labels)

        # ## k-fold cross validation ##
        # for k in range(5):
        #     if k == k_fold_idx - 1:
        #
        #         test_list.append(input_list[int(len(input_list) * (k) * 0.2): int(len(input_list) * (k+1) * 0.2)])
        #         labels = [i for j in range(len(test_list[-1]))]
        #         test_labels.append(labels)
        #     else:
        #
        #         train_list.append(input_list[int(len(input_list) * (k) * 0.2) : int(len(input_list) * (k+1) * 0.2)])
        #         labels = [i for j in range(len(train_list[-1]))]
        #         train_labels.append(labels)

    train_list = list(itertools.chain.from_iterable(train_list))
    train_labels = list(itertools.chain.from_iterable(train_labels))

    test_list = list(itertools.chain.from_iterable(test_list))
    test_labels = list(itertools.chain.from_iterable(test_labels))

    train_test_list = {'train_list': train_list, 'train_labels': train_labels,
                       'test_list': test_list, 'test_labels': test_labels}

    return train_test_list


def get_train_test_list_ESBD(args):
    ## SSBD
    random.seed('2023-07-03')

    action_cls = ['ArmFlapping_parsed', 'Spinning_parsed', 'HeadBanging_parsed', 'HandAction_parsed']

    folder_dir = '/home/ych/data/ESBD/Dataset/'
    train_list = []
    train_labels = []
    test_list = []
    test_labels = []
    train_test_ratio = 0.8

    for i in range(len(action_cls)):
        file_dir = folder_dir + action_cls[i]
        input_list = []
        input_list += [os.path.join(file_dir, f) for f in sorted(os.listdir(file_dir))
                       if f.split(".")[-1] == "mp4" or f.split(".")[-1] == "avi"]
        # shuffle
        random.shuffle(input_list)
        train_list.append(input_list[:int(len(input_list) * train_test_ratio)])

        labels = []
        for j in range(len(train_list[i])):
            if i == 3: #HandAction
                labels.append(0)
            else:
                labels.append(i)
        train_labels.append(labels)

        test_list.append(input_list[int(len(input_list) * train_test_ratio):])

        labels = []
        for j in range(len(test_list[i])):
            if i == 3:  # HandAction
                labels.append(0)
            else:
                labels.append(i)

        test_labels.append(labels)

    train_list = list(itertools.chain.from_iterable(train_list))
    train_labels = list(itertools.chain.from_iterable(train_labels))

    test_list = list(itertools.chain.from_iterable(test_list))
    test_labels = list(itertools.chain.from_iterable(test_labels))

    train_test_list = {'train_list': train_list, 'train_labels': train_labels,
                       'test_list': test_list, 'test_labels': test_labels}

    return train_test_list




def sample_frame_indices(clip_len, frame_sample_rate, seg_len, stage):

    '''
    Sample a given number of frame indices from the video.

    Args:

        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.

    Returns:

        indices (`List[int]`): List of sampled frame indices
    '''
    if stage == 'train':
        converted_len = int(clip_len * frame_sample_rate)

        # Ensure converted_len does not exceed seq_len
        if converted_len > seg_len:
            converted_len = seg_len

        # Adjust end_idx and start_idx if converted_len is greater than seg_len
        if converted_len == seg_len:
            start_idx = 0
            end_idx = seg_len
        else:
            end_idx = np.random.randint(converted_len, seg_len)
            start_idx = end_idx - converted_len

        indices = np.linspace(start_idx, end_idx, num=clip_len)
        indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)

    else:
        # Calculate the start and end indices for center dense sampling
        mid_frame_index = seg_len // 2  # Index of the middle frame
        start_idx = max(0, mid_frame_index - clip_len // 2)  # Start index
        end_idx = min(seg_len, start_idx + clip_len)  # End index

        # Extract the indices for center dense sampling
        indices = np.arange(start_idx, end_idx)

        # If the number of frames to extract is less than clip_len,
        # pad the indices with the first or last frame index as needed
        if len(indices) < clip_len:
            pad_front = (clip_len - len(indices)) // 2
            pad_back = clip_len - len(indices) - pad_front
            indices = np.pad(indices, (pad_front, pad_back), mode='edge')

        # Ensure that the indices are within the valid range
        indices = np.clip(indices, 0, seg_len - 1).astype(np.int64)

    return indices


def load_video_and_sampling(path, vid_sample_len=64, stage='train', args=None):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_f = 0
    end_f = num_frames

    if stage == 'train':
        width = 256
        height = 256
        if num_frames > vid_sample_len:
            start_f = random.randint(0, num_frames - vid_sample_len)
            end_f = random.randint(start_f + vid_sample_len, num_frames)
    else:
        width = 224
        height = 224

    if args.backbone == 'VideoMAE' or args.backbone == 'InternVideo2':
        taken = sample_frame_indices(clip_len=16, frame_sample_rate=1, seg_len=num_frames, stage=stage)
        video = np.zeros((16, width, height, 3)).astype(np.float32)

    elif args.backbone == 'videomamba':
        taken = sample_frame_indices(clip_len=32, frame_sample_rate=1, seg_len=num_frames, stage=stage)
        video = np.zeros((32, width, height, 3)).astype(np.float32)

    else:
        # Init the numpy array
        video = np.zeros((vid_sample_len, width, height, 3)).astype(np.float32)
        #taken = np.linspace(0, num_frames, vid_sample_len).astype(int)
        taken = np.linspace(start_f, end_f, vid_sample_len).astype(int)

    np_idx = 0
    for fr_idx in range(num_frames):
        ret, frame = cap.read()

        if cap.isOpened() and fr_idx in taken:

            if args.crop_ROI == 'body_grounding_dino':

                TEXT_PROMPT = args.text_prompt
                BOX_TRESHOLD = args.box_th
                TEXT_TRESHOLD = args.text_th

                with torch.no_grad():
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(img_rgb)
                    image_transformed, _ = transform(pil_img, None)

                    boxes, logits, phrases = predict(
                        model=model_bodycrop,
                        image=image_transformed,
                        caption=TEXT_PROMPT,
                        box_threshold=BOX_TRESHOLD,
                        text_threshold=TEXT_TRESHOLD
                    )

                if len(phrases) != 0:

                    ## most baby-like person in the image
                    max_idx = torch.argmax(logits).numpy()
                    h, w, _ = img_rgb.shape

                    box = boxes[max_idx] * torch.Tensor([w, h, w, h])
                    xyxy = box_convert(boxes=box, in_fmt="cxcywh", out_fmt="xyxy").numpy()

                    box_width = xyxy[2] - xyxy[0]
                    box_height = xyxy[3] - xyxy[1]
                    bbox_margin_w = int(box_width / 4.0) #add margin to box x1.5
                    bbox_margin_h = int(box_height / 4.0) #add margin to box x1.5

                    xmin = max(int(xyxy[0] - bbox_margin_w), 0)
                    ymin = max(int(xyxy[1] - bbox_margin_h), 0)
                    xmax = min(int(xyxy[2] + bbox_margin_w), w - 1)
                    ymax = min(int(xyxy[3] + bbox_margin_h), h - 1)
                    frame = frame[ymin:ymax, xmin:xmax, :]
                else:
                    pass

            elif args.crop_ROI == 'body_yolo_world' and stage == 'test' and 'ADOS_BeDevel' in args.test_DB:

                if fr_idx % args.skip_frame_stride == 0: ## skip frame, use detectionr results of previous frame

                    height_ori, width_ori, channel = frame.shape

                    if fr_idx == taken[0]:
                        ymin = 0
                        ymax = height_ori
                        xmin = 0
                        xmax = width_ori

                    # 이미지 리사이징 비율 계산
                    scale = 640.0 / max(width_ori, height_ori)
                    new_width = int(width_ori * scale)
                    new_height = int(height_ori * scale)

                    # 이미지 리사이징
                    resized_image = cv2.resize(frame, (new_width, new_height))

                    # Zero padding 추가
                    if new_width > new_height:
                        top = 0
                        bottom = (640 - new_height)
                        left = right = 0
                    else:
                        top = bottom = 0
                        #left = right = (640 - new_width) // 2
                        left = 0
                        right = (640 - new_width)

                    # 이미지에 padding 추가
                    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                                      value=[0, 0, 0])
                    input_data = padded_image.astype('float32') / 255

                    input_data = np.expand_dims(input_data, axis=0)
                    input_data = np.transpose(input_data, (0, 3, 1, 2))

                    ########################################################################
                    ############################# Inference ################################
                    ########################################################################
                    input_name = model_bodycrop.get_inputs()[0].name
                    output_names = [output.name for output in model_bodycrop.get_outputs()]

                    result = model_bodycrop.run(output_names, {input_name: input_data})
                    ## box coords rescale
                    result[1][0][:-1][:, [0, 2]] = result[1][0][:-1][:, [0, 2]] / (scale * width_ori)
                    result[1][0][:-1][:, [1, 3]] = result[1][0][:-1][:, [1, 3]] / (scale * height_ori)

                    boxes = torch.Tensor(result[1][0][:-1])
                    logits = torch.Tensor(result[2][0][:-1])
                    phrases = torch.Tensor(result[3][0][:-1])

                    ########################################################################
                    ###################### Non-Maximum Suppression #########################
                    ########################################################################
                    IOU_THRESHOLD = args.iou_threshold
                    nms_idx = nms(torch.cat((boxes[:, :2], boxes[:, 2:]), dim=1), logits,
                                  IOU_THRESHOLD).numpy().tolist()  ## yolo-world ver.
                    boxes = boxes[nms_idx]
                    logits = logits[nms_idx]
                    phrases = phrases[nms_idx]

                    ########################################################################
                    ####################### Confidence Thresholding ########################
                    ########################################################################
                    CONFIDENCE_THRESHOLD = args.confidence_threshold
                    boxes = boxes[logits > CONFIDENCE_THRESHOLD]
                    phrases = phrases[logits > CONFIDENCE_THRESHOLD]
                    logits = logits[logits > CONFIDENCE_THRESHOLD]

                    boxes = boxes * torch.Tensor([width_ori, height_ori, width_ori, height_ori])

                    # Filter logits based on phrases
                    logits_filtered = logits[phrases == 0] ## 0: child, 1: adult

                if logits_filtered.numel() > 0:
                    # Find the index of the maximum value in the filtered logits
                    max_idx = torch.argmax(logits_filtered)

                    # Retrieve the corresponding box using the index
                    max_box = boxes[phrases == 0][max_idx]
                    pt1 = (int(max_box[0]), int(max_box[1]))
                    pt2 = (int(max_box[2]), int(max_box[3]))

                    box_width = pt2[0] - pt1[0]
                    box_height = pt2[1] - pt1[1]

                    # bbox_margin_w = int(box_width / 4.0)  # add margin to box x1.5 (4.0)
                    # bbox_margin_h = int(box_height / 4.0)  # add margin to box x1.5 (4.0)

                    bbox_margin_w = int(box_width / 8.0) #1.25
                    bbox_margin_h = 0

                    xmin = max(int(pt1[0] - bbox_margin_w), 0)
                    ymin = max(int(pt1[1] - bbox_margin_h), 0)
                    xmax = min(int(pt2[0] + bbox_margin_w), width_ori - 1)
                    ymax = min(int(pt2[1] + bbox_margin_h), height_ori - 1)
                    frame = frame[ymin:ymax, xmin:xmax, :]

                else:
                    #pass ## whole image
                    frame = frame[ymin:ymax, xmin:xmax, :] ##bbox of previous frame

            frame = cv2.resize(frame, dsize=(width, height))
            video[np_idx, :, :, :] = frame.astype(np.float32)
            # np_idx += 1
            # print(np_idx)
            # save_name = 'imgs/' + str(fr_idx) + '.png'
            # cv2.imwrite(save_name, frame)

    cap.release()

    return video


def get_dataloaders(args):

    train_test_list_SSBD = get_train_test_list_SSBD(args)
    train_test_list_ESBD = get_train_test_list_ESBD(args)

    train_test_list = {'SSBD': train_test_list_SSBD, 'ESBD': train_test_list_ESBD}

    transformer_tra = transforms.Compose([video_transforms.RandomRotation(30),
                                          #video_transforms.RandomResizedCrop(224),
                                          video_transforms.RandomCrop((224, 224)),
                                          video_transforms.RandomHorizontalFlip(),
                                          volume_transforms.ClipToTensor(div_255=False)])

    transformer_val = transforms.Compose([volume_transforms.ClipToTensor(div_255=False)])


    dataset_training = RepetitionDataset(transform=transformer_tra, stage='train', args=args, train_test_list=train_test_list)
    #dataset_val     = RepetitionDataset(transform=transformer_val, stage='val', args=args)
    dataset_test     = RepetitionDataset(transform=transformer_val, stage='test', args=args, train_test_list=train_test_list)

    #datasets = {'train': dataset_training, 'val': dataset_val, 'test': dataset_test}
    datasets = {'train': dataset_training, 'test': dataset_test}

    # For handling imbalacned dataset
    #class_weights = calculate_class_weights(datasets['train'].labels)
    # Create WeightedRandomSampler
    #sampler = WeightedRandomSampler(weights=class_weights, num_samples=len(datasets['train']))

    dataloaders = { 'train': DataLoader(datasets['train'], batch_size=args.batchsize, shuffle=True,# sampler=sampler, #shuffle=True
                                        num_workers=args.num_workers, drop_last=True, pin_memory=True),
                    # 'val': DataLoader(datasets['val'], batch_size=1, shuffle=False,
                    #                     num_workers=args.num_workers, drop_last=False, pin_memory=True),
                    'test': DataLoader(datasets['test'], batch_size=1, shuffle=False,# sampler=sampler,
                                      num_workers=args.num_workers, drop_last=False, pin_memory=True),
                  }

    return dataloaders


class RepetitionDataset(Dataset):

    def __init__(self,
                 transform=None,
                 stage='train',
                 args=None,
                 train_test_list=None):

        self.args = args
        self.transform = transform
        self.rgb_list = []
        self.labels = []
        self.stage = stage
        self.train_DB_list = args.train_DB
        self.test_DB_list = args.test_DB

        if stage == 'train':
            if 'ESBD' in self.train_DB_list:
                self.rgb_list = train_test_list['ESBD']['train_list']
                self.labels = train_test_list['ESBD']['train_labels']
                ## merge train/test DB for cross-dataset evaluation
                self.rgb_list = self.rgb_list + train_test_list['ESBD']['test_list']
                self.labels = self.labels + train_test_list['ESBD']['test_labels']

            if 'SSBD' in self.train_DB_list:
                self.rgb_list = train_test_list['SSBD']['train_list']
                self.labels = train_test_list['SSBD']['train_labels']
                ## merge train/test DB for cross-dataset evaluation
                self.rgb_list = self.rgb_list + train_test_list['SSBD']['test_list']
                self.labels = self.labels + train_test_list['SSBD']['test_labels']

            self.rgb_list, self.labels = shuffle(self.rgb_list, self.labels)


        elif stage == 'test':
            if 'ESBD' in self.test_DB_list:
                self.rgb_list = train_test_list['ESBD']['test_list']
                self.labels = train_test_list['ESBD']['test_labels']
                ## merge train/test DB for cross-dataset evaluation
                self.rgb_list = self.rgb_list + train_test_list['ESBD']['train_list']
                self.labels = self.labels + train_test_list['ESBD']['train_labels']

            if 'SSBD' in self.test_DB_list:
                self.rgb_list = train_test_list['SSBD']['test_list']
                self.labels = train_test_list['SSBD']['test_labels']
                ## merge train/test DB for cross-dataset evaluation
                self.rgb_list = self.rgb_list + train_test_list['SSBD']['train_list']
                self.labels = self.labels + train_test_list['SSBD']['train_labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        rgbpath = self.rgb_list[idx]
        label = self.labels[idx]

        video = load_video_and_sampling(rgbpath, self.args.vid_sample_len, self.stage, self.args)
        video = video_transform(video)

        if self.transform:
            video = self.transform(video)

        # append rgb_path to sample dictionary
        sample = {'rgb': video, 'label': label, 'rgbpath': rgbpath}

        return sample




def video_transform(np_clip): #from imagent

    # Div by 255
    np_clip /= 255.

    # Normalization
    np_clip -= np.asarray([0.485, 0.456, 0.406]).reshape(1, 1, 3)  # mean
    np_clip /= np.asarray([0.229, 0.224, 0.225]).reshape(1, 1, 3)  # std

    return np_clip


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        rgb, label = sample['rgb'], sample['label']
        return {'rgb': torch.from_numpy(rgb.astype(np.float32)),
                'label': torch.from_numpy(np.asarray(label))}


class NormalizeLen(object):
    """ Return a normalized number of frames. """

    def __init__(self, vid_len=64):
        self.vid_len = vid_len

    def __call__(self, sample):
        rgb, label = sample['rgb'], sample['label']
        if rgb.shape[0] != 1:
            num_frames_rgb = len(rgb)
            indices_rgb = np.linspace(0, num_frames_rgb - 1, self.vid_len).astype(int)
            rgb = rgb[indices_rgb, :, :, :]

        return {'rgb': rgb,
                'label': label}


class AugCrop(object):
    """ Return a temporal crop of given sequences """

    def __init__(self, p_interval=0.5):
        self.p_interval = p_interval

    def __call__(self, sample):
        rgb, label = sample['rgb'], sample['label']
        ratio = (1.0 - self.p_interval * np.random.rand())
        if rgb.shape[0] != 1:
            num_frames_rgb = len(rgb)
            begin_rgb = (num_frames_rgb - int(num_frames_rgb * ratio)) // 2
            rgb = rgb[begin_rgb:(num_frames_rgb - begin_rgb), :, :, :]

        return {'rgb': rgb,
                'label': label}


