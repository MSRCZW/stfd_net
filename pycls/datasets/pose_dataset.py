# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
import pickle
#from ..utils import get_root_logger
from .base import BaseDataset


class PoseDataset(BaseDataset):
    """Pose dataset for action recognition.

    The dataset loads pose and apply specified transforms to return a
    dict containing pose information.

    The ann_file is a pickle file, the json file contains a list of
    annotations, the fields of an annotation include frame_dir(video_id),
    total_frames, label, kp, kpscore.

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        split (str | None): The dataset split used. For UCF101 and HMDB51, allowed choices are 'train1', 'test1',
            'train2', 'test2', 'train3', 'test3'. For NTURGB+D, allowed choices are 'xsub_train', 'xsub_val',
            'xview_train', 'xview_val'. For NTURGB+D 120, allowed choices are 'xsub_train', 'xsub_val', 'xset_train',
            'xset_val'. For FineGYM, allowed choices are 'train', 'val'. Default: None.
        valid_ratio (float | None): The valid_ratio for videos in KineticsPose. For a video with n frames, it is a
            valid training sample only if n * valid_ratio frames have human pose. None means not applicable (only
            applicable to Kinetics Pose). Default: None.
        box_thr (float): The threshold for human proposals. Only boxes with confidence score larger than `box_thr` is
            kept. None means not applicable (only applicable to Kinetics). Allowed choices are 0.5, 0.6, 0.7, 0.8, 0.9.
            Default: 0.5.
        class_prob (list | None): The class-specific multiplier, which should be a list of length 'num_classes', each
            element >= 1. The goal is to resample some rare classes to improve the overall performance. None means no
            resampling performed. Default: None.
        memcached (bool): Whether keypoint is cached in memcached. If set as True, will use 'frame_dir' as the key to
            fetch 'keypoint' from memcached. Default: False.
        mc_cfg (tuple): The config for memcached client, only applicable if `memcached==True`.
            Default: ('localhost', 22077).
        **kwargs: Keyword arguments for 'BaseDataset'.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 split=None,
                 valid_ratio=None,
                 box_thr=0.5,
                 class_prob=None,
                 memcached=False,
                 mc_cfg=('localhost', 22077),
                 **kwargs):
        modality = 'Pose'
        self.split = split

        super().__init__(
            ann_file, pipeline, start_index=0, modality=modality, memcached=memcached, mc_cfg=mc_cfg, **kwargs)

        # box_thr, which should be a string
        self.box_thr = box_thr
        self.class_prob = class_prob
        if self.box_thr is not None:
            assert box_thr in [.5, .6, .7, .8, .9]

        # Thresholding Training Examples
        self.valid_ratio = valid_ratio
        if self.valid_ratio is not None:
            assert isinstance(self.valid_ratio, float)
            self.video_infos = [
                x for x in self.video_infos
                if x['valid'][self.box_thr] / x['total_frames'] >= valid_ratio
            ]
            for item in self.video_infos:
                anno_inds = (item['box_score'] >= self.box_thr)
                item['anno_inds'] = anno_inds
        for item in self.video_infos:
            item.pop('valid', None)
            item.pop('box_score', None)
            if self.memcached:
                item['key'] = item['frame_dir']

    def load_annotations(self):
        """Load annotation file to get video information."""
        assert self.ann_file.endswith('.pkl')
        return self.load_pkl_annotations()

    def load_pkl_annotations(self):
        data = mmcv.load(self.ann_file)


        if self.split:
            split, data = data['split'], data['annotations']
            identifier = 'filename' if 'filename' in data[0] else 'frame_dir'
            split = set(split[self.split])
            data = [x for x in data if x[identifier] in split]

        for item in data:
            # Sometimes we may need to load anno from the file
            if 'filename' in item:
                item['filename'] = osp.join(self.data_prefix, item['filename'])
            if 'frame_dir' in item:
                item['frame_dir'] = osp.join(self.data_prefix, item['frame_dir'])
        return data


ntu_edge = ((0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5),
            (7, 6), (8, 20), (9, 8), (10, 9), (11, 10), (12, 0),
            (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), (18, 17),
            (19, 18), (21, 22), (22, 7), (23, 24), (24, 11))

edge = ((0, 1), (1, 2), (2, 3), (2, 4), (11, 2),  # stem
        (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (7, 10),  # left arm
        (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (14, 17),  # right arm
        (0, 22), (22, 23), (23, 24), (24, 25),  # right leg
        (0, 18), (18, 19), (19, 20), (20, 21),  # left leg
        (3, 26), (29, 28), (30, 31), (27, 28), (26, 27), (27, 30)  # head
        )
class AnubisDataset(BaseDataset):
    def __init__(self,
                 ann_file,
                 pipeline,
                 split=None,
                 valid_ratio=None,
                 box_thr=0.5,
                 class_prob=None,
                 memcached=False,
                 mc_cfg=('localhost', 22077),
                 **kwargs):
        modality = 'Pose'
        self.split = split
        self.edge = ntu_edge
        self.debug = False
        self.label_path = '/home/cv-ar/datasets/ANUBIS/train_data/trn_label_all_action_front.pkl'
        self.data_path = '/home/cv-ar/datasets/ANUBIS/train_data/trn_data_all_action_front.npy'
        self.mode = 'train'
        self.separator =0.7
        super().__init__(
            ann_file, pipeline, start_index=0, modality=modality, memcached=memcached, mc_cfg=mc_cfg, **kwargs)


        # box_thr, which should be a string
        self.box_thr = box_thr
        self.class_prob = class_prob

        for item in self.video_infos:
            item.pop('valid', None)
            item.pop('box_score', None)
            if self.memcached:
                item['key'] = item['frame_dir']


    def load_annotations(self): #读取文件 其中split为video_name   annotation为经过去除无效骨架后的原始数据
        self.load_data()
        result = []
        for i in range(self.data.shape[0]):
            result.append(
                {'frame_dir': '', 'label': self.label[i], 'keypoint': np.copy(self.data[i, :, :, :, :].transpose((3, 1, 2, 0))),
                 'total_frames': 300})

        return result

    def load_data(self):
        if self.mode in ["train", "val"]:
            with open(self.label_path, 'rb') as f:
                self.label = pickle.load(f)[1]  # 33501
                # 102 classes
        else:
            self.label = [i for i in range(32731)]
        data_all = np.load(self.data_path, mmap_mode='r')
        data_num = min(data_all.shape[0], len(self.label))
        data_all = data_all[:data_num, :, :, :, :]
        self.label = self.label[:data_num]
        sp = int(self.separator * data_num)
        if self.mode == "train":
            self.data = data_all[:, :, :, :, :]
        elif self.mode == "val":
            self.data = data_all[sp:, :, :, :, :]
            self.label = self.label[sp:]
        elif self.mode == "test":
            self.data = data_all
        else:
            raise Exception
        self.format_ntu()

    def format_ntu(self):
        ntu_index = [0, 1, 3, 26, 5, 6, 7, 8, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25, 2, 9, 10, 16, 17]
        self.data = self.data[:, :, :, ntu_index, :]
        self.edge = ntu_edge

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)