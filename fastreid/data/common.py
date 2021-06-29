# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import Dataset

from .data_utils import read_image


class CommDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=True):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel

        pid_set = set()
        cam_set = set()
        view_set = set()
        time_set = set()
        model_set = set()
        color_set = set()
        type_set = set()
        for i in img_items:
            pid_set.add(i[1])
            cam_set.add(i[2])
            view_set.add(i[3])
            time_set.add(i[4])
            model_set.add(i[5])
            color_set.add(i[6])
            type_set.add(i[7])

        self.pids = sorted(list(pid_set))
        self.cams = sorted(list(cam_set))
        self.views = sorted(list(view_set))
        self.times = sorted(list(time_set))
        self.mods = sorted(list(model_set))
        self.colors = sorted(list(color_set))
        self.types = sorted(list(type_set))

        if relabel:
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
            self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])
            self.view_dict = dict([(p, i) for i, p in enumerate(self.views)])
            self.time_dict = dict([(p, i) for i, p in enumerate(self.times)])
            self.mod_dict = dict([(p, i) for i, p in enumerate(self.mods)])
            self.color_dict = dict([(p, i) for i, p in enumerate(self.colors)])
            self.type_dict = dict([(p, i) for i, p in enumerate(self.types)])

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_item = self.img_items[index]
        img_path = img_item[0]
        pid = img_item[1]
        camid = img_item[2]
        viewid = img_item[3]
        timeid = img_item[4]
        modid = img_item[5]
        colorid = img_item[6]
        typeid = img_item[7]
        img = read_image(img_path)
        if self.transform is not None: img = self.transform(img)
        if self.relabel:
            pid = self.pid_dict[pid]
            camid = self.cam_dict[camid]
            viewid = self.view_dict[viewid]
            timeid = self.time_dict[timeid]
            modid = self.mod_dict[modid]
            colorid = self.color_dict[colorid]
            typeid = self.type_dict[typeid]
        return {
            "images": img,
            "targets": pid,
            "camids": camid,
            "viewids": viewid,
            "timeids": timeid,
            "modids": modid,
            "colorids": colorid,
            "typeids": typeid,
            "img_paths": img_path,

        }

    @property
    def num_classes(self):
        return len(self.pids)

    @property
    def num_cameras(self):
        return len(self.cams)
