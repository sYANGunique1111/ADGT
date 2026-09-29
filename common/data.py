"""Human3.6M data loading: 3D poses in camera space, normalised 2D detections, batching."""
from functools import reduce
from os import path

import numpy as np
import torch
from torch.utils.data import Dataset

from common.camera import normalize_screen_coordinates, world_to_camera
from common.h36m_dataset import Human36mDataset, TEST_SUBJECTS, TRAIN_SUBJECTS

# The 17-joint Human3.6M skeleton -> the 16 joints used by ADGT (drops the "Nose").
SELECTED_KEYPOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16]


def load_h36m(data_path, keypoints_name):
    """Returns (dataset with camera-space `positions_3d`, normalised 2D keypoints)."""
    dataset = Human36mDataset(path.join(data_path, 'data_3d_h36m.npz'))
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]
            positions_3d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                pos_3d[:, :] -= pos_3d[:, :1]  # root-relative
                positions_3d.append(pos_3d)
            anim['positions_3d'] = positions_3d

    keypoints = np.load(path.join(data_path, 'data_2d_h36m_{}.npz'.format(keypoints_name)), allow_pickle=True)
    keypoints = keypoints['positions_2d'].item()
    for subject in keypoints:
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                cam = dataset.cameras()[subject][cam_idx]
                if kps.shape[1] == 17:
                    kps = kps[:, SELECTED_KEYPOINTS]
                elif kps.shape[1] != 16:
                    raise ValueError('Unexpected number of joints: {}'.format(kps.shape[1]))
                keypoints[subject][action][cam_idx] = normalize_screen_coordinates(
                    kps[..., :2], w=cam['res_w'], h=cam['res_h'])
    return dataset, keypoints


def fetch(subjects, dataset, keypoints, action_filter=None):
    """Collects (3D poses, 2D poses, action labels) over all cameras of the given subjects/actions."""
    out_3d, out_2d, out_actions = [], [], []
    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None and action.split(' ')[0] not in action_filter:
                continue
            poses_2d = keypoints[subject][action]
            poses_3d = dataset[subject][action]['positions_3d']
            assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
            for cam_id in range(len(poses_3d)):
                length = poses_3d[cam_id].shape[0]
                assert poses_2d[cam_id].shape[0] >= length
                out_3d.append(poses_3d[cam_id])
                out_2d.append(poses_2d[cam_id][:length])
                out_actions.append([action.split(' ')[0]] * length)
    return out_3d, out_2d, out_actions


class PoseGenerator(Dataset):
    """Single-frame (2D -> 3D) pose pairs."""

    def __init__(self, poses_3d, poses_2d, actions):
        self._poses_3d = np.concatenate(poses_3d)
        self._poses_2d = np.concatenate(poses_2d)
        self._actions = reduce(lambda x, y: x + y, actions)
        assert self._poses_3d.shape[0] == self._poses_2d.shape[0] == len(self._actions)
        print('Generating {} poses...'.format(len(self._actions)))

    def __getitem__(self, index):
        return (torch.from_numpy(self._poses_3d[index]).float(),
                torch.from_numpy(self._poses_2d[index]).float(),
                self._actions[index])

    def __len__(self):
        return len(self._actions)
