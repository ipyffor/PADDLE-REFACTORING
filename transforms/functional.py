# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import collections
import random

import cv2
import numpy as np
from PIL import Image
from paddle.fluid import dygraph, core_avx

try:
    import accimage
except ImportError:
    accimage = None

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable

__all__ = ['flip', 'resize']


def flip(image, code):
    """
    Accordding to the code (the type of flip), flip the input image

    Args:
        image: Input image, with (H, W, C) shape
        code: Code that indicates the type of flip.
            -1 : Flip horizontally and vertically
            0 : Flip vertically
            1 : Flip horizontally

    Examples:
        .. code-block:: python

            import numpy as np
            from paddle.incubate.hapi.vision.transforms import functional as F

            fake_img = np.random.rand(224, 224, 3)

            # flip horizontally and vertically
            F.flip(fake_img, -1)

            # flip vertically
            F.flip(fake_img, 0)

            # flip horizontally
            F.flip(fake_img, 1)
    """
    return cv2.flip(image, flipCode=code)



def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _is_numpy(img):
    return isinstance(img, np.ndarray)


def _is_numpy_image(img):
    return img.ndim in {2, 3}

def to_tensor(pic):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    See ``ToTensor`` for more details.

    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """

    if not(_is_pil_image(pic) or _is_numpy(pic)):
        raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

    if _is_numpy(pic) and not _is_numpy_image(pic):
        raise ValueError('pic should be 2/3 dimensional. Got {} dimensions.'.format(pic.ndim))

    if isinstance(pic, np.ndarray):
        # handle numpy array
        if pic.ndim == 2:
            pic = pic[:, :, None]

        img = pic.transpose((2, 0, 1)).astype('float32')/255
        # backward compatibility
        # print('ddd',pic.shape)
        # img = img.astype(dtype='float32')
        # print(pic.shape)
        return img
    else:
        return pic
        # else:
        #     return img

    # # handle PIL Image
    # if pic.mode == 'I':
    #     img = dygraph.to_variable(np.array(pic, np.int32, copy=False))
    # elif pic.mode == 'I;16':
    #     img = dygraph.to_variable(np.array(pic, np.int16, copy=False))
    # elif pic.mode == 'F':
    #     img = dygraph.to_variable(np.array(pic, np.float32, copy=False))
    # elif pic.mode == '1':
    #     img = 255 * dygraph.to_variable(np.array(pic, np.uint8, copy=False))
    # img = img.reshape(pic.shape[1], pic.shape[0], len(pic.getbands()))
    # # put it from HWC to CHW format
    # img = img.permute((2, 0, 1)).contiguous()
    # if isinstance(img, torch.ByteTensor):
    #     return img.float().div(255)
    # else:
    #     return img


def resize(img, size, interpolation=cv2.INTER_LINEAR):
    """
    resize the input data to given size

    Args:
        input: Input data, could be image or masks, with (H, W, C) shape
        size: Target size of input data, with (height, width) shape.
        interpolation: Interpolation method.

    Examples:
        .. code-block:: python

            import numpy as np
            from paddle.incubate.hapi.vision.transforms import functional as F

            fake_img = np.random.rand(256, 256, 3)

            F.resize(fake_img, 224)

            F.resize(fake_img, (200, 150))
    """

    if isinstance(interpolation, Sequence):
        interpolation = random.choice(interpolation)
    if isinstance(size, int):
        h, w = img.shape[:2]
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return cv2.resize(img, (ow, oh), interpolation=interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return cv2.resize(img, (ow, oh), interpolation=interpolation)
    else:
        return cv2.resize(img, size[::-1], interpolation=interpolation)
