# Custom dataset
from PIL import Image
import torch.utils.data as data
import os
import random
import numpy as np
import torch


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, real_folder="real_imgs", label_folder="label_imgs", test_folder="test_imgs", transform=None, is_train=False):
        super(DatasetFromFolder, self).__init__()
        self.input_path = os.path.join(image_dir, real_folder)
        self.label_path = os.path.join(image_dir, label_folder)
        self.test_path = os.path.join(image_dir, test_folder)
        self.image_filenames = [x for x in sorted(os.listdir(self.input_path))]
        self.label_filenames = [x for x in sorted(os.listdir(self.label_path))]
        self.test_filenames = [x for x in sorted(os.listdir(self.test_path))]
        self.transform = transform
        self.is_train = is_train

    def __getitem__(self, index):
        # Load Image
        if self.is_train:
            img_fn = os.path.join(self.input_path, self.image_filenames[index])
            label_img_fn = os.path.join(self.label_path, self.label_filenames[index])
            img = Image.open(img_fn).convert("RGB")
            label_img = Image.open(label_img_fn).convert("RGB")
            # preprocessing
            # print(np.array(img).shape)
            input_img = self.transform(img)
            # print(input_img.shape)
            label_img = self.transform(label_img)
            return input_img, label_img
        else:
            tets_fn = os.path.join(self.test_path, self.test_filenames[index])
            img = Image.open(tets_fn)
            test_img = self.transform(img)
            return test_img


    def __len__(self):
        return len(self.image_filenames)





class TrainDataset(data.Dataset):
    def __init__(self, image_dir, real_folder="real_imgs", label_folder="label_imgs", transform=None):
        super(TrainDataset, self).__init__()
        self.input_path = os.path.join(image_dir, real_folder)
        self.label_path = os.path.join(image_dir, label_folder)
        self.image_filenames = [x for x in sorted(os.listdir(self.input_path))]
        self.label_filenames = [x for x in sorted(os.listdir(self.label_path))]
        self.transform = transform

    def __getitem__(self, index):
        # Load Image
        # print(index)
        img_fn = os.path.join(self.input_path, self.image_filenames[index])
        label_img_fn = os.path.join(self.label_path, self.label_filenames[index])
        img = Image.open(img_fn).convert("RGB")
        label_img = Image.open(label_img_fn).convert("RGB")
        dic_class = torch.tensor(int(self.image_filenames[index].split("_")[-2]))
        river_class = torch.tensor(int(self.image_filenames[index].split("_")[-1].split(".")[0]))
        # print(img_fn, label_img_fn)
        # print(dic_class, river_class)
        # exit()
        # class_tensor = onehot(real_class)


        input_img = self.transform(img)
        # print(input_img.shape)
        label_img = self.transform(label_img)
        # print("......")
        return input_img, label_img, dic_class, river_class

    def __len__(self):
        return len(self.image_filenames)
class TrainDataset_dic(data.Dataset):
    def __init__(self, image_dir, real_folder="real_imgs", label_folder="label_imgs", transform=None):
        super(TrainDataset_dic, self).__init__()
        self.input_path = os.path.join(image_dir, real_folder)
        self.label_path = os.path.join(image_dir, label_folder)
        self.image_filenames = [x for x in sorted(os.listdir(self.input_path))]
        self.label_filenames = [x for x in sorted(os.listdir(self.label_path))]
        self.transform = transform

    def __getitem__(self, index):
        # Load Image
        # print(index)
        img_fn = os.path.join(self.input_path, self.image_filenames[index])
        label_img_fn = os.path.join(self.label_path, self.label_filenames[index])
        img = Image.open(img_fn).convert("RGB")
        label_img = Image.open(label_img_fn).convert("RGB")
        dic_class = torch.tensor(int(self.image_filenames[index].split("_")[-2]))
        # river_class = torch.tensor(int(self.image_filenames[index].split("_")[-1].split(".")[0]))
        # class_tensor = onehot(real_class)
        input_img = self.transform(img)
        label_img = self.transform(label_img)
        return input_img, label_img, dic_class

    def __len__(self):
        return len(self.image_filenames)
class TestDataset(data.Dataset):
    def __init__(self, image_dir, test_folder="test_imgs", transform=None):
        super(TestDataset, self).__init__()
        self.test_path = os.path.join(image_dir, test_folder)
        self.test_filenames = [x for x in sorted(os.listdir(self.test_path))]
        self.transform = transform

    def __getitem__(self, index):
        # Load Image
        tets_fn = os.path.join(self.test_path, self.test_filenames[index])
        img = Image.open(tets_fn)
        test_img = self.transform(img)
        return test_img, self.test_filenames[index]


    def __len__(self):
        return len(self.test_filenames)

