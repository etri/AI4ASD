# -*- coding: utf-8 -*- 

""" 
   * Source: libFER.data_loader.py
   * License: PBR License (Dual License)
   * Modified by ByungOk Han <byungok.han@etri.re.kr>
   * Date: 13 Mar 2021, ETRI

"""

import torch.utils.data as data

from PIL import Image
import pandas as pd
import os
import os.path


def default_loader(path):

    """default_loader function

    Note: image loader

    Arguments: 
        path (str): path for image file 

    Returns:
        img (PIL.image): PIL image

    """

    img = Image.open(path)
    return img


def default_list_reader(datalist_file, include_ds=True):

    """default_list_reader function

    Note: data list reader to load dataset pathes and labels

    Arguments: 
        datalist_file (str): path for data list file 

    Returns:
        datalist (Pandas.Dataframe): dataframe for dataset pathes and labels

    """

    dataframe = pd.read_csv(datalist_file, delimiter = ' ', header = 0)  
    if include_ds:
        datalist = [tuple(x) for x in dataframe[['File_Name', 'Emotion_Label', 'Dataset_Label']].values]
    else:
        datalist = [tuple(x) for x in dataframe[['File_Name', 'Emotion_Label']].values]

    return datalist


class ImageList(data.Dataset):

    """ImageList class

    Note: class for ImageList

    """

    def __init__(self,
                 root,
                 datalist_filename, 
                 include_ds=True,
                 transform=None,
                 list_reader = default_list_reader,
                 loader = default_loader):

        """__init__ function

        Note: function for __init__

        """

        self.root = root
        self.datalist = list_reader(datalist_filename, include_ds)
        self.include_ds = include_ds
        self.transform = transform
        self.loader = loader

    def _getitem_impl(self, index):

        """_getitem_impl function

        Note: function for _getitem_impl

        """

        if self.include_ds:
            img_path, label, ds_label = self.datalist[index]
        else:
            img_path, label = self.datalist[index]
        img = self.loader(self.root + img_path)
        if self.transform is not None:
            img = self.transform(img)

        if self.include_ds:
            return img, label, ds_label
        return img, label

    def __getitem__(self, index):

        """__getitem__ function

        Note: function for __getitem__

        """

        return self._getitem_impl(index)
           
    def __len__(self):

        """__len__ function

        Note: function for __len__

        """

        return len(self.datalist)