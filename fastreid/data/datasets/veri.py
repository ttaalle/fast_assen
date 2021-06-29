# encoding: utf-8
"""
@author:  Jinkai Zheng
@contact: 1315673509@qq.com
"""

import glob
import os.path as osp
import re

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY
import json
import numpy as np
import pdb

@DATASET_REGISTRY.register()
class VeRi(ImageDataset):
    """VeRi.

    Reference:
        Xinchen Liu et al. A Deep Learning based Approach for Progressive Vehicle Re-Identification. ECCV 2016.
        Xinchen Liu et al. PROVID: Progressive and Multimodal Vehicle Reidentification for Large-Scale Urban Surveillance. IEEE TMM 2018.

    URL: `<https://vehiclereid.github.io/VeRi/>`_

    Dataset statistics:
        - identities: 775.
        - images: 37778 (train) + 1678 (query) + 11579 (gallery).
    """
    dataset_dir = "/home/lhc/ours0/data/veri"
    dataset_name = "veri"

    def __init__(self, root='datasets', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'test')

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir)
        query = self.process_dir(self.query_dir, is_train=False)
        gallery = self.process_dir(self.gallery_dir, is_train=False)

        super(VeRi, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([\d]+)_c(\d\d\d)')

        if is_train: 
            #train_view_dir = osp.join(dir_path[0:-19], 'image_view.json')
            train_view_dir = osp.join(dir_path[0:-5], 'train_view_label.txt')
            fr = open(train_view_dir,'r')
            img_dic = {}
            keys = [] 
            for line in fr.readlines():
                line = line.strip()
                k = line.split(' ')[0]
                v = line.split(' ')[1]
                img_dic[k] = v
            fr.close()
            #with open(train_view_dir,'r', encoding='UTF-8') as f:
                  #img_dic = f.read()
        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 776  # pid == 0 means background
            assert 0 <= camid <= 20
            camid -= 1  # index starts from 0
            vid = -1
            coid = -1
            tyid = -1
            tid = -1
            moid = -1
            if is_train: 
                pid = pid2label[pid]
                vid = int(img_dic[img_path[-37:]])-1
                coid = int(img_path[-9:-7])-1
                tyid = int(img_path[-5:-4])-1
            dataset.append((img_path, pid, camid, vid, tid, moid, coid, tyid))
        return dataset


"""
        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 776
            assert 1 <= camid <= 20
            camid -= 1  # index starts from 0
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid))

        return data
"""