from collections.abc import Sequence

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
import cv2

from ..registry import PIPELINES


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).float()
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError('type {} cannot be converted to tensor.'.format(
            type(data)))


@PIPELINES.register_module
class ToTensor(object):

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            results[key] = to_tensor(results[key])
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(keys={})'.format(self.keys)


@PIPELINES.register_module
class ImageToTensor(object):
    '''
    function that imread img and transpose channel and normalize the value by div 255
    '''
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys: 
            results[key] = to_tensor(results[key].transpose(2, 0, 1))/255.            
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(keys={})'.format(self.keys)


@PIPELINES.register_module
class Transpose(object):

    def __init__(self, keys, order):
        self.keys = keys
        self.order = order

    def __call__(self, results):
        for key in self.keys:
            results[key] = results[key].transpose(self.order)
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(keys={}, order={})'.format(
            self.keys, self.order)


@PIPELINES.register_module
class Collect(object):
    """Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "gt_labels".

    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:

        - "img_shape": shape of the image input to the network as a tuple
                (h, w, c).  Note that images may be zero padded on the
                bottom/right, if the batch tensor is larger than this shape.

        - "scale_factor": a float indicating the preprocessing scale

        - "flip": a boolean indicating if image flip transform was used

        - "filename": path to the image file

        - "ori_shape": original shape of the image as a tuple (h, w, c)

        - "pad_shape": image shape after padding

        - "img_norm_cfg": a dict of normalization information:
            - mean - per channel mean subtraction
            - std - per channel std divisor
            - to_rgb - bool indicating if bgr was converted to rgb
    """

    def __init__(self,
                 keys,
                 meta_keys=('video_infos', 'ori_shape', 'img_shape',
                            'pad_shape', 'flip_direction', 'img_norm_cfg')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        data = {}
        img_meta = {}
        for key in self.meta_keys:
            img_meta[key] = results[key]
        # data['img_meta'] = DC(img_meta, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + '(keys={}, meta_keys={})'.format(
            self.keys, self.meta_keys)


@PIPELINES.register_module
class FormatShape(object):
    """Format final imgs shape to the given input_format

    Required keys are "imgs", "num_clips" and "clip_len",
    added or modified keys are "imgs" and "input_shape".

    Attributes:
        input_format (str): define the final imgs format.

    """

    def __init__(self, input_format='NCTHW'):
        self.input_format = input_format

    def __call__(self, results):
        imgs = results['imgs']
        # [M x C x H x W]
        # M = 1 * N_oversample * N_clips * L
        if self.input_format == 'NCTHW':
            num_clips = results['num_clips']
            clip_len = results['clip_len']

            imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
            # N_over x N_clips x L x C x H x W
            imgs = np.transpose(imgs, (0, 1, 3, 2, 4, 5))
            # N_over x N_clips x C x L x H x W
            imgs = imgs.reshape(() + imgs.shape[2:])
            # M' x C x L x H x W
            # M' = N_over x N_clips
        results['imgs'] = imgs
        results['input_shape'] = imgs.shape
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(input_format={})'.format(self.input_format)
