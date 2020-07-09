#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import os
import gzip
import struct
import numpy as np
import hashlib
import sys
import shutil
import requests
import six

import paddle.dataset.common
import paddle
from paddle.fluid.io import Dataset

# from .utils import _check_exists_and_download

__all__ = ["MNIST"]

URL_PREFIX = 'https://dataset.bj.bcebos.com/mnist/'
TEST_IMAGE_URL = URL_PREFIX + 't10k-images-idx3-ubyte.gz'
TEST_IMAGE_MD5 = '9fb629c4189551a2d022fa330f9573f3'
TEST_LABEL_URL = URL_PREFIX + 't10k-labels-idx1-ubyte.gz'
TEST_LABEL_MD5 = 'ec29112dd5afa0611ce80d1b7f02629c'
TRAIN_IMAGE_URL = URL_PREFIX + 'train-images-idx3-ubyte.gz'
TRAIN_IMAGE_MD5 = 'f68b3c2dcbeaaa9fbdd348bbdeb94873'
TRAIN_LABEL_URL = URL_PREFIX + 'train-labels-idx1-ubyte.gz'
TRAIN_LABEL_MD5 = 'd53e105ee54ea40749a09fcbcd1e9432'


def _check_exists_and_download(path, url, md5, module_name, download=True):
    if path is None:
        path = os.path.expanduser('~') + '/paddlepaddle/dataset'
    if download:
        return download_file(url, path, module_name, md5)
    else:
        raise ValueError('{} not exists and auto download disabled'.format(
            path))


def md5file(fname):
    hash_md5 = hashlib.md5()
    f = open(fname, "rb")
    for chunk in iter(lambda: f.read(4096), b""):
        hash_md5.update(chunk)
    f.close()
    return hash_md5.hexdigest()


def download_file(url, path, module_name, md5sum, save_name=None):
    dirname = os.path.join(path, module_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    filename = os.path.join(dirname,
                            url.split('/')[-1]
                            if save_name is None else save_name)

    if os.path.exists(filename) and md5file(filename) == md5sum:
        return filename

    retry = 0
    retry_limit = 3
    while not (os.path.exists(filename) and md5file(filename) == md5sum):
        if os.path.exists(filename):
            sys.stderr.write("file %s  md5 %s\n" % (md5file(filename), md5sum))
        if retry < retry_limit:
            retry += 1
        else:
            raise RuntimeError("Cannot download {0} within retry limit {1}".
                               format(url, retry_limit))
        sys.stderr.write("Cache file %s not found, downloading %s \n" %
                         (filename, url))
        sys.stderr.write("Begin to download\n")
        r = requests.get(url, stream=True)
        total_length = r.headers.get('content-length')

        if total_length is None:
            with open(filename, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        else:
            with open(filename, 'wb') as f:
                chunk_size = 4096
                total_length = int(total_length)
                total_iter = total_length / chunk_size + 1
                log_interval = total_iter / 20 if total_iter > 20 else 1
                log_index = 0
                for data in r.iter_content(chunk_size=chunk_size):
                    if six.PY2:
                        data = six.b(data)
                    f.write(data)
                    log_index += 1
                    if log_index % log_interval == 0:
                        sys.stderr.write(".")
                    sys.stdout.flush()
    sys.stderr.write("\nDownload finished\n")
    sys.stdout.flush()
    return filename


class MNIST(Dataset):
    """
    Implement of MNIST dataset

    Args:
        image_path(str): path to image file, can be set None if
            :attr:`download` is True. Default None
        label_path(str): path to label file, can be set None if
            :attr:`download` is True. Default None
        chw_format(bool): If set True, the output shape is [1, 28, 28],
            otherwise, output shape is [1, 784]. Default True.
        mode(str): 'train' or 'test' mode. Default 'train'.
        download(bool): whether auto download mnist dataset if
            :attr:`image_path`/:attr:`label_path` unset. Default
            True

    Returns:
        Dataset: MNIST Dataset.

    Examples:

        .. code-block:: python

            from paddle.incubate.hapi.datasets import MNIST

            mnist = MNIST(mode='test')

            for i in range(len(mnist)):
                sample = mnist[i]
                print(sample[0].shape, sample[1])

    """

    def __init__(self,
                 root=None,
                 mode='train',
                 transform=None,
                 download=True):
        assert mode.lower() in ['train', 'test'], \
            "mode should be 'train' or 'test', but got {}".format(mode)
        self.mode = mode.lower()
        self.root = root

        if self.root is None:
            self.root = os.path.expanduser('~') + '/paddlepaddle/dataset'

        if download:
            image_url = TRAIN_IMAGE_URL if mode == 'train' else TEST_IMAGE_URL
            image_md5 = TRAIN_IMAGE_MD5 if mode == 'train' else TEST_IMAGE_MD5

            self.image_path = _check_exists_and_download(
                self.root, image_url, image_md5, 'mnist', download)

            label_url = TRAIN_LABEL_URL if mode == 'train' else TEST_LABEL_URL
            label_md5 = TRAIN_LABEL_MD5 if mode == 'train' else TEST_LABEL_MD5
            self.label_path = _check_exists_and_download(
                self.root, label_url, label_md5, 'mnist', download)

        # if self.image_path is None:
        #     assert download, "image_path not set and auto download disabled"
        #     image_url = TRAIN_IMAGE_URL if mode == 'train' else TEST_IMAGE_URL
        #     image_md5 = TRAIN_IMAGE_MD5 if mode == 'train' else TEST_IMAGE_MD5
        #     self.image_path = _check_exists_and_download(
        #         image_path, image_url, image_md5, 'mnist', download)
        #
        # self.label_path = label_path
        # if self.label_path is None:
        #     assert download, "label_path not set and auto download disabled"
        #     label_url = TRAIN_LABEL_URL if mode == 'train' else TEST_LABEL_URL
        #     label_md5 = TRAIN_LABEL_MD5 if mode == 'train' else TEST_LABEL_MD5
        #     self.label_path = _check_exists_and_download(
        #         label_path, label_url, label_md5, 'mnist', download)

        self.transform = transform

        # read dataset into memory
        self._parse_dataset()

    def _parse_dataset(self, buffer_size=100):
        self.images = []
        self.labels = []
        # print(self.image_path)
        with gzip.GzipFile(self.image_path, 'rb') as image_file:
            img_buf = image_file.read()
            with gzip.GzipFile(self.label_path, 'rb') as label_file:
                lab_buf = label_file.read()

                step_label = 0
                offset_img = 0
                # read from Big-endian
                # get file info from magic byte
                # image file : 16B
                magic_byte_img = '>IIII'
                magic_img, image_num, rows, cols = struct.unpack_from(
                    magic_byte_img, img_buf, offset_img)
                offset_img += struct.calcsize(magic_byte_img)

                offset_lab = 0
                # label file : 8B
                magic_byte_lab = '>II'
                magic_lab, label_num = struct.unpack_from(magic_byte_lab,
                                                          lab_buf, offset_lab)
                offset_lab += struct.calcsize(magic_byte_lab)

                while True:
                    if step_label >= label_num:
                        break
                    fmt_label = '>' + str(buffer_size) + 'B'
                    labels = struct.unpack_from(fmt_label, lab_buf, offset_lab)
                    offset_lab += struct.calcsize(fmt_label)
                    step_label += buffer_size

                    fmt_images = '>' + str(buffer_size * rows * cols) + 'B'
                    images_temp = struct.unpack_from(fmt_images, img_buf,
                                                     offset_img)
                    images = np.reshape(images_temp, (buffer_size, rows *
                                                      cols))
                    offset_img += struct.calcsize(fmt_images)

                    # images = images / 255.0
                    # images = images * 2.0
                    # images = images - 1.0

                    for i in range(buffer_size):
                        self.images.append(images[i, :])
                        self.labels.append(
                            np.array([labels[i]]).astype('int64'))

    def __getitem__(self, idx):
        image, label = self.images[idx].astype('uint8'), self.labels[idx]


        image = np.reshape(image, [28, 28, 1])
        if self.transform is not None:
            image = self.transform(image)
        # print(image.shape, label)
        return image, label

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    dataset = MNIST(root='./test', chw_format=True,mode='train')
    from paddle import fluid
    from paddle.fluid.io import DataLoader

    place = fluid.CPUPlace()
    fluid.enable_dygraph(place)
    loader = DataLoader(dataset, places=place, return_list=True, batch_size=256, shuffle=True, num_workers=0)
    print(len(loader))
    itr = iter(loader)
    images, labels = next(itr)
    print(images.numpy().max(), images.numpy().min(), images.shape)
