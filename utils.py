import paddle
from paddle import fluid
from paddle.fluid import layers as F
from paddle.fluid import dygraph, core_avx

import math
irange = range
import cv2

def make_grid(tensor, nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """

    if len(tensor.shape) == 2:  # single image H x W
        tensor = F.unsqueeze(tensor, 0)
    elif len(tensor.shape) == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = F.concat([tensor,tensor, tensor], 0)
        tensor = F.unsqueeze(tensor, 0)

    if len(tensor.shape) == 4 and tensor.shape[1] == 1:  # single-channel images
        tensor = F.concat([tensor, tensor, tensor], 1)

    if normalize is True:
        tensor = tensor.detach()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"
        def norm_ip(img, min, max):
            img = F.clamp(img, min=min, max = max)

            img = (img -min)/(max-min + 1e-5)
            # img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(F.reduce_min(t)), float(F.reduce_max(t)))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)
    if tensor.shape[0] == 1:
        return F.squeeze(tensor,0)
    # make the mini-batch of images into a grid
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] + padding)
    num_channels = tensor.shape[1]
    grid = F.full(shape=(num_channels, height * ymaps + padding, width * xmaps + padding), fill_value=pad_value, dtype=tensor.dtype)
    # grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    # dygraph.to_variable().tra
    grid = grid.numpy()
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            t = narraw(grid, 1, y * height + padding, height - padding)
            t = narraw(t, 2, x * width + padding, width - padding)
            t[:, :, :] = tensor[k].detach().numpy()[:,:,:]
            # grid.narrow(1, y * height + padding, height - padding)\
            #     .narrow(2, x * width + padding, width - padding)\
            #     .copy_()
            k = k + 1
    grid = dygraph.to_variable(grid)
    return grid
def narraw(input, dim, start, length):
    assert 0<=dim<=3
    if dim == 0:
        return input[start:start+length]
    elif dim == 1:
        return input[:, start:start + length]
    elif dim == 2:
        return input[:,:, start:start + length]
    elif dim == 3:
        return input[:,:,:, start:start + length]

def save_image(tensor, fp, nrow=8, padding=4,
               normalize=False, range=None, scale_each=False, pad_value=0, format=None):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr =F.clamp((grid*255+0.5),0,255)
    ndarr = F.transpose(ndarr, [1, 2, 0]).numpy().astype('uint8')
    # ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    # cv2.imshow('img', ndarr)
    # cv2.waitKey(2000)
    # im = Image.fromarray(ndarr)
    cv2.imwrite(fp, ndarr)