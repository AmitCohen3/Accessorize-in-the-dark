import torch.utils.data as data
import os
import os.path
import torch
from itertools import groupby
import random
import utils

def default_list_reader(fileList, folder_prefix=""):
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            split_line = line.strip().split('\\')
            split_line[1] = folder_prefix + split_line[1]
            img_path = os.path.sep.join(split_line)
            label = img_path.split(os.path.sep)[-2]
            img_path = os.path.normpath(img_path)
            imgList.append((img_path, int(label)))
    return imgList

class NirVisDataset(data.Dataset):
    def __init__(self, root, file_list, transform=None):
        self.root      = root
        self.img_list   = default_list_reader(file_list)
        self.transform = transform
        self.loader    = utils.nir_vis_loader
        name_label_dict = {}
        for (name, label) in self.img_list:
            name_label_dict.setdefault(label, []).append(name)
        self.labels_list = list(name_label_dict.keys())

    def __getitem__(self, index):
        imgPath, label = self.img_list[index]
        img = self.loader(os.path.join(self.root, imgPath))

        target_label = random.choice([x for x in self.labels_list if x != label])

        if self.transform:
            img = self.transform(img)

        pose_transformation_matrix = utils.find_perspective_transform_matrix(img)

        return img, label, target_label, pose_transformation_matrix

    def __len__(self):
        return len(self.img_list)


def subject_list_reader(fileList, folder_prefix=""):
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            split_line = line.strip().split('\\')
            split_line[1] = folder_prefix + split_line[1]
            img_path = os.path.sep.join(split_line)
            label = img_path.split(os.path.sep)[-2]
            img_path = os.path.normpath(img_path)
            imgList.append((img_path, int(label)))
    
    images_per_label = []
    # Group by the images label
    for key, group in groupby(imgList,key=lambda x:x[1]):
        images_per_label.append(list(group))

    return images_per_label


class NirVisDatasetPerSubject(data.Dataset):
    def __init__(self, root, file_list, transform=None):
        self.root = root
        self.images_per_label = subject_list_reader(file_list)
        self.transform = transform
        self.loader = utils.nir_vis_loader

    def __getitem__(self, index):
        imgs=[]
        pose_transformation_matrices = []
        allowed_values = list(range(len(self.images_per_label)))
        allowed_values.remove(index)
        _, target_label = self.images_per_label[random.choice(allowed_values)][0]
        for imgPath, label in self.images_per_label[index]:
            img = self.loader(os.path.join(self.root, imgPath))

            if self.transform:
                img = self.transform(img)
            pose_transformation_matrices.append(utils.find_perspective_transform_matrix(img))
            imgs.append(img)

        return torch.stack(imgs), torch.stack([torch.tensor(label)]*len(imgs)), torch.stack([torch.tensor(target_label)]*len(imgs)), torch.stack(pose_transformation_matrices)

    def __len__(self):
        return len(self.images_per_label)