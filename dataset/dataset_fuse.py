"""
Definition of PyTorch "Dataset" that iterates through compressed videos
and return compressed representations (I-frames, motion vectors, 
or residuals) for training or testing.
"""

import os
import os.path
import random

import numpy as np
import torch
import torch.utils.data as data

from coviar import get_num_frames
from coviar import load
from models import transforms


import torchvision
import pdb
import pandas as pd
from tqdm import tqdm, trange
from sklearn.utils import shuffle


GOP_SIZE = 12


def clip_and_scale(img, size):
    return (img * (127.5 / size)).astype(np.int32)


def get_seg_range(n, num_segments, seg, representation):
    if representation in ['residual', 'mv']:
        n -= 1

    seg_size = float(n - 1) / num_segments
    seg_begin = int(np.round(seg_size * seg))
    seg_end = int(np.round(seg_size * (seg+1)))
    if seg_end == seg_begin:
        seg_end = seg_begin + 1

    if representation in ['residual', 'mv']:
        return seg_begin + 1, seg_end + 1

    return seg_begin, seg_end


def get_gop_pos(frame_idx, representation):
    gop_index = frame_idx // GOP_SIZE
    gop_pos = frame_idx % GOP_SIZE
    if representation in ['residual', 'mv']:
        gop_pos = random.randint(1, GOP_SIZE-1)
    else:
        gop_pos = 0
    return gop_index, gop_pos


class CoviarDataSet_inference(data.Dataset):
    def __init__(self, data_root, data_name,
                 video_list,
                 frame_transform,
                 motion_transform,
                 residual_transform,
                 num_segments,
                 accumulate,
                 test_crops,
                 num_clips):

        self._data_root = data_root
        self._data_name = data_name
        self._num_segments = num_segments
        self.frame_transform = frame_transform
        self.motion_transform = motion_transform
        self.residual_transform = residual_transform
        self._accumulate = accumulate
        self._num_clips = num_clips
        if test_crops == 10:
            self._crop_groups = 5
        elif test_crops == 3:
            self._crop_groups = 3
        elif test_crops == 1:
            self._crop_groups = 1

        self._input_mean = torch.from_numpy(
            np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))).float()
        self._input_std = torch.from_numpy(
            np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))).float()

        if data_name == 'hmdb51':
            self._load_list_hmdb51(video_list)
        elif data_name == 'ucf101':
            self._load_list_ucf101(video_list)
        elif data_name == 'kinetic400':
            self._load_list_kinetic400(video_list)



    def _load_list_hmdb51(self, video_list):
        self._video_list = []
        f = pd.read_csv(video_list, header=None)
        f = f.iloc[1:]
        class_names = sorted(os.listdir(self._data_root))
        label_map = {k: v for k, v in enumerate(class_names)}
        f_cs = pd.read_csv(video_list[:-4]+'_cs.csv')
        print('loading {} videos'.format(len(f)))
        for line in trange(f_cs.shape[0]):
            video, label = f_cs.iloc[line][0], f_cs.iloc[line][1]
            class_name = label_map[int(label)]
            video_path = os.path.join(self._data_root, class_name, video + '.mp4')
            self._video_list.append((
                video_path,
                int(label),
                get_num_frames(video_path)
                ))
        max_val = 0
        min_val = float('inf')
        count_min = 0
        all_len = 0
        for i in range(len(self._video_list)):    
            max_val = max(max_val, self._video_list[i][2])
            min_val = min(min_val, self._video_list[i][2])
            all_len += self._video_list[i][2]
        print('{} good videos loaded!'.format(len(self._video_list)))
        print('max len {}, min len {}, avg len {}'.format(max_val, min_val, all_len/len(self._video_list)))



    def _load_list_ucf101(self, video_list):
        self._video_list = []
        f = pd.read_csv(video_list, header=None)
        f = f.iloc[1:]
        print('loading {} videos'.format(len(f)))
        for line in range(f.shape[0]):
            video, label = f.iloc[line][1], f.iloc[line][2]
            class_name = video.split('_')[1]
            if class_name == 'HandStandPushups':
                class_name = 'HandstandPushups'
            video_path = os.path.join(self._data_root, class_name, video + '.mp4')
            if get_num_frames(video_path) > 3:
                self._video_list.append((
                    video_path,
                    int(label),
                    get_num_frames(video_path)))
        max_val = 0
        min_val = float('inf')
        count_min = 0
        all_len = 0
        for i in range(len(self._video_list)):
            max_val = max(max_val, self._video_list[i][2])
            min_val = min(min_val, self._video_list[i][2])
            all_len += self._video_list[i][2]
        print('{} good videos loaded!'.format(len(self._video_list)))
        print('max len {}, min len {}, avg len {}'.format(max_val, min_val, all_len/len(self._video_list)))



    def _load_list_kinetic400(self, video_list):
        self._video_list = []
        f = pd.read_csv(video_list, header=None)
        f = f.iloc[1:]
        f_cs = pd.read_csv(video_list[:-4]+'_cs.csv')
        print('loading {} videos'.format(len(f)))
        max_val = 0
        min_val = float('inf')
        count_min = 0
        all_len = 0
        for i in trange(len(f_cs)):
            self._video_list.append((f_cs.iloc[i][0],  f_cs.iloc[i][1], f_cs.iloc[i][2]))
            max_val = max(max_val, self._video_list[i][2])
            min_val = min(min_val, self._video_list[i][2])
            all_len += self._video_list[i][2]
        print('{} good videos loaded!'.format(len(self._video_list)))
        print('max len {}, min len {}, avg len {}'.format(max_val, min_val, all_len/len(self._video_list)))


    def _get_train_frame_index(self, num_frames, seg, representation):
        # Compute the range of the segment.
        seg_begin, seg_end = get_seg_range(num_frames, self._num_segments, seg, 
                                           representation=representation)
        # Sample one frame from the segment.
        v_frame_idx = random.randint(seg_begin, seg_end - 1)
        return v_frame_idx

    def _get_test_frame_index(self, num_frames, seg, representation):
        if representation in ['mv', 'residual']:
            num_frames -= 1

        seg_size = float(num_frames - 1) / self._num_segments
        v_frame_idx = int(np.round(seg_size * (seg + 0.5)))

        if representation in ['mv', 'residual']:
            v_frame_idx += 1

        return v_frame_idx 


    def __getitem__(self, index):
        video_path, label, num_frames = self._video_list[index]
        frames = []
        motions = []
        residuals = []
        for clip in range(self._num_clips):
            frames_clip = []
            motions_clip = []
            residuals_clip = []
            for seg in range(self._num_segments):
                v_frame_idx = self._get_train_frame_index(num_frames, seg, representation='iframe')
                frame_gop_index, frame_gop_pos = get_gop_pos(v_frame_idx, representation = 'iframe')
                # for tracking frame_gop_index + 1 is in the seg_begin and seg_end
                seg_begin, seg_end = get_seg_range(num_frames, self._num_segments, seg, 
                                                    representation='iframe')
                p_gop_index = frame_gop_index
                # p_gop_pos = random.randint(1, GOP_SIZE-1)
                p_gop_pos = (GOP_SIZE - 1) // 2

                img = load(video_path, frame_gop_index, 0,
                           0, self._accumulate)
                mv = load(video_path, p_gop_index, p_gop_pos,
                          1, self._accumulate)
                residual = load(video_path, p_gop_index, p_gop_pos,
                            2, self._accumulate)

                if img is None:
                    print('Error: loading video %s failed.' % video_path)
                else:
                    # formulate I-frame
                    img = img[..., ::-1]

                    # formulate motion
                    mv = clip_and_scale(mv, 20)
                    mv += 128
                    mv = (np.minimum(np.maximum(mv, 0), 255)).astype(np.uint8)

                    # formulate residuals
                    residual += 128
                    residual = (np.minimum(np.maximum(residual, 0), 255)).astype(np.uint8)

                    frames_clip.append(img)
                    motions_clip.append(mv)
                    residuals_clip.append(residual)


            frames_crop = self.frame_transform(frames_clip)
            motions_crop = self.frame_transform(motions_clip)
            residuals_crop = self.frame_transform(residuals_clip)


            frames_crop = np.array(frames_crop)
            frames_crop = np.transpose(frames_crop, (0, 3, 1, 2))
            frame_input = torch.from_numpy(frames_crop).float() / 255.0

            motions_crop = np.array(motions_crop) # crop*alpha, c, h, w
            motions_crop = np.transpose(motions_crop, (0, 3, 1, 2))
            motion_input = torch.from_numpy(motions_crop).float() / 255.0

            residuals_crop = np.array(residuals_crop)
            residuals_crop = np.transpose(residuals_crop, (0, 3, 1, 2))
            residual_input = torch.from_numpy(residuals_crop).float() / 255.0

            frame_input = (frame_input - self._input_mean) / self._input_std
            residual_input = (residual_input - 0.5) / self._input_std
            motion_input = (motion_input - 0.5)
      
            frames.append(frame_input)
            motions.append(motion_input)
            residuals.append(residual_input)

        frames = torch.stack(frames)
        motions = torch.stack(motions)
        residuals = torch.stack(residuals)

        frames = frames.view((-1,self._num_segments)+frames.size()[2:]) # clips*crops, t, c, h, w
        motions = motions.view((-1,self._num_segments)+motions.size()[2:]) # clips*crops, t*alpha, c, h, w
        residuals = residuals.view((-1,self._num_segments)+residuals.size()[2:])
        return [frames, motions, residuals], label


    def __len__(self):
        return len(self._video_list)



class CoviarDataSet(data.Dataset):
    def __init__(self, data_root, data_name,
                 video_list,
                 frame_transform,
                 motion_transform,
                 residual_transform,
                 num_segments,
                 is_train,
                 accumulate):

        self._data_root = data_root
        self._data_name = data_name
        self._num_segments = num_segments
        self.frame_transform = frame_transform
        self.motion_transform = motion_transform
        self.residual_transform = residual_transform
        self._is_train = is_train
        self._accumulate = accumulate

        self._input_mean = torch.from_numpy(
            np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))).float()
        self._input_std = torch.from_numpy(
            np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))).float()

        if data_name == 'hmdb51':
            self._load_list_hmdb51(video_list)
        elif data_name == 'ucf101':
            self._load_list_ucf101(video_list)
        elif data_name == 'kinetic400':
            self._load_list_kinetic400(video_list)


    def _load_list_hmdb51(self, video_list):
        self._video_list = []
        f = pd.read_csv(video_list, header=None)
        f = f.iloc[1:]
        class_names = sorted(os.listdir(self._data_root))
        label_map = {k: v for k, v in enumerate(class_names)}
        f_cs = pd.read_csv(video_list[:-4]+'_cs.csv')
        print('loading {} videos'.format(len(f)))
        for line in trange(f_cs.shape[0]):
            video, label = f_cs.iloc[line][0], f_cs.iloc[line][1]
            class_name = label_map[int(label)]
            video_path = os.path.join(self._data_root, class_name, video + '.mp4')
            self._video_list.append((
                video_path,
                int(label),
                get_num_frames(video_path)
                ))
        max_val = 0
        min_val = float('inf')
        count_min = 0
        all_len = 0
        for i in range(len(self._video_list)):    
            max_val = max(max_val, self._video_list[i][2])
            min_val = min(min_val, self._video_list[i][2])
            all_len += self._video_list[i][2]
        print('{} good videos loaded!'.format(len(self._video_list)))
        print('max len {}, min len {}, avg len {}'.format(max_val, min_val, all_len/len(self._video_list)))

    def _load_list_ucf101(self, video_list):
        self._video_list = []
        f = pd.read_csv(video_list, header=None)
        f = f.iloc[1:]
        for line in range(f.shape[0]):
            video, label = f.iloc[line][1], f.iloc[line][2]
            class_name = video.split('_')[1]
            if class_name == 'HandStandPushups':
                class_name = 'HandstandPushups'
            video_path = os.path.join(self._data_root, class_name, video + '.mp4')
            self._video_list.append((
                video_path,
                int(label),
                get_num_frames(video_path)))


    def _load_list_kinetic400(self, video_list):
        self._video_list = []
        f = pd.read_csv(video_list, header=None)
        f = f.iloc[1:]
        f_cs = pd.read_csv(video_list[:-4]+'_cs.csv')
        print('loading {} videos'.format(len(f_cs)))
        max_val = 0
        min_val = float('inf')
        count_min = 0
        all_len = 0
        for i in trange(len(f_cs)):
            self._video_list.append((f_cs.iloc[i][0],  f_cs.iloc[i][1], f_cs.iloc[i][2]))
            max_val = max(max_val, self._video_list[i][2])
            min_val = min(min_val, self._video_list[i][2])
            all_len += self._video_list[i][2]
        print('{} good videos loaded!'.format(len(self._video_list)))
        print('max len {}, min len {}, avg len {}'.format(max_val, min_val, all_len/len(self._video_list)))




    def _get_train_frame_index(self, num_frames, seg, representation):
        # Compute the range of the segment.
        seg_begin, seg_end = get_seg_range(num_frames, self._num_segments, seg, 
                                           representation=representation)
        # Sample one frame from the segment.
        v_frame_idx = random.randint(seg_begin, seg_end - 1)
        return v_frame_idx

    def _get_test_frame_index(self, num_frames, seg, representation):
        if representation in ['mv', 'residual']:
            num_frames -= 1

        seg_size = float(num_frames - 1) / self._num_segments
        v_frame_idx = int(np.round(seg_size * (seg + 0.5)))

        if representation in ['mv', 'residual']:
            v_frame_idx += 1

        return v_frame_idx 

    def __getitem__(self, index):
        if self._is_train:
            video_path, label, num_frames = self._video_list[index]
        else:
            video_path, label, num_frames = self._video_list[index]

        frames = []
        motions = []
        residuals = []
        for seg in range(self._num_segments):
            if self._is_train:
                v_frame_idx = self._get_train_frame_index(num_frames, seg, representation='iframe')
                frame_gop_index, frame_gop_pos = get_gop_pos(v_frame_idx, representation = 'iframe')
                # for tracking frame_gop_index + 1 is in the seg_begin and seg_end
                seg_begin, seg_end = get_seg_range(num_frames, self._num_segments, seg, 
                                                   representation='iframe')
                p_gop_index = frame_gop_index
                p_gop_pos = random.randint(1, GOP_SIZE-1)

            else:
                v_frame_idx = self._get_test_frame_index(num_frames, seg, representation = 'iframe')
                 # for tracking frame_gop_index + 1 is in the seg_begin and seg_end
                frame_gop_index, frame_gop_pos = get_gop_pos(v_frame_idx, representation = 'iframe')
                seg_begin, seg_end = get_seg_range(num_frames, self._num_segments, seg, 
                                                   representation='iframe')
                p_gop_index = frame_gop_index
                p_gop_pos = (GOP_SIZE - 1) // 2

            img = load(video_path, frame_gop_index, 0,
                       0, self._accumulate)
            mv = load(video_path, p_gop_index, p_gop_pos,
                        1, self._accumulate)
            residual = load(video_path, p_gop_index, p_gop_pos,
                        2, self._accumulate)

            if img is None:
                print('Error: loading video %s failed.' % video_path)
            else:
                # formulate I-frame
                img = img[..., ::-1]

                # formulate motion
                mv = clip_and_scale(mv, 20)
                mv += 128
                mv = (np.minimum(np.maximum(mv, 0), 255)).astype(np.uint8)

                # formulate residuals
                residual += 128
                residual = (np.minimum(np.maximum(residual, 0), 255)).astype(np.uint8)

            frames.append(img)
            motions.append(mv)
            residuals.append(residual)

        frames = self.frame_transform(frames)
        motions = self.motion_transform(motions)
        residuals = self.residual_transform(residuals)



        frames = np.array(frames)
        frames = np.transpose(frames, (0, 3, 1, 2))
        frame_input = torch.from_numpy(frames).float() / 255.0

        motions = np.array(motions)
        motions = np.transpose(motions, (0, 3, 1, 2))
        motion_input = torch.from_numpy(motions).float() / 255.0

        residuals = np.array(residuals)
        residuals = np.transpose(residuals, (0, 3, 1, 2))
        residual_input = torch.from_numpy(residuals).float() / 255.0


        frame_input = (frame_input - self._input_mean) / self._input_std
        residual_input = (residual_input - 0.5) / self._input_std
        motion_input = (motion_input - 0.5)


        return [frame_input, motion_input, residual_input], label

    def __len__(self):
        return len(self._video_list)


       

