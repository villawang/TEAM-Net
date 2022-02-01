"""Functions for data augmentation and related preprocessing."""

import random
import numpy as np
import torch
import cv2
from PIL import Image
import torchvision
import pdb


class IdentityTransform(object):
    def __call__(self, data):
        return data

def color_aug(img, random_h=36, random_l=50, random_s=50):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(float)

    h = (random.random() * 2 - 1.0) * random_h
    l = (random.random() * 2 - 1.0) * random_l
    s = (random.random() * 2 - 1.0) * random_s

    img[..., 0] += h
    img[..., 0] = np.minimum(img[..., 0], 180)

    img[..., 1] += l
    img[..., 1] = np.minimum(img[..., 1], 255)

    img[..., 2] += s
    img[..., 2] = np.minimum(img[..., 2], 255)

    img = np.maximum(img, 0)
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_HLS2BGR)
    return img



class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):

        w, h = img_group[0].shape
        th, tw = self.size

        out_images = list()

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert(img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))
        return out_images


class GroupCenterCrop(object):
    def __init__(self, size):
        self._size = size

    def __call__(self, img_group):
        h, w, _ = img_group[0].shape
        hs = (h - self._size) // 2
        ws = (w - self._size) // 2
        return [img[hs:hs+self._size, ws:ws+self._size] for img in img_group]


class GroupRandomHorizontalFlip(object):
    def __init__(self, is_mv=False):
        self._is_mv = is_mv

    def __call__(self, img_group, is_mv=False):
        if random.random() < 0.5:
            ret = [img[:, ::-1, :].astype(np.int32) for img in img_group]
            if self._is_mv:
                for i in range(len(ret)):
                    ret[i] -= 128
                    ret[i][..., 0] *= (-1)
                    ret[i] += 128
            return ret
        else:
            return img_group

# Group scale shorter side
# class GroupScale(object):
#     def __init__(self, size):
#         self._size = size

#     def __call__(self, img_group):
#         if img_group[0].shape[2] == 3:
#             return [self._shorter_side_scale(img) for img in img_group]
#         elif img_group[0].shape[2] == 2:
#             return [resize_mv(img, self._size, cv2.INTER_LINEAR) for img in img_group]
#         else:
#             assert False

#     def _shorter_side_scale(self, img):
#         h, w = img.shape[:2]
#         if h <= w:
#             ratio = w/h
#             new_size = (int(self._size*ratio), self._size) # width, height
#         elif h > w:
#             ratio = h/w
#             new_size = (self._size, int(self._size*ratio)) # width, height
#         return cv2.resize(img, new_size, cv2.INTER_LINEAR)


# def resize_mv(img, shape, interpolation):
#     h, w = img.shape[:2]
#     if h <= w:
#         ratio = w/h
#         new_size = (int(shape*ratio), shape) # width, height
#     elif h > w:
#         ratio = h/w
#         new_size = (shape, int(shape*ratio)) # width, height
#     return np.stack([cv2.resize(img[..., i], new_size, interpolation) for i in range(2)], axis=2)


# class GroupScale_w_h(object):
#     def __init__(self, size):
#         self._size = (size[0], size[1]) # width, height


#     def __call__(self, img_group):
#         if img_group[0].shape[2] == 3:
#             return [cv2.resize(img, self._size, cv2.INTER_LINEAR) for img in img_group]
#         elif img_group[0].shape[2] == 2:
#             return [resize_mv_w_h(img, self._size, cv2.INTER_LINEAR) for img in img_group]
#         else:
#             assert False
 

# def resize_mv_w_h(img, shape, interpolation):
#     return np.stack([cv2.resize(img[..., i], shape, interpolation)
#                      for i in range(2)], axis=2)




class GroupScale(object):
    def __init__(self, size):
        self._size = size

    def __call__(self, img_group):
        imgs = []
        for img in img_group:
            if img.shape[2] == 3:
                imgs.append(self._shorter_side_scale(img))
            elif img.shape[2] == 2:
                imgs.append(resize_mv(img, self._size, cv2.INTER_LINEAR))
            else:
                assert False
        return imgs

    def _shorter_side_scale(self, img):
        h, w = img.shape[:2]
        if h <= w:
            ratio = w/h
            new_size = (int(self._size*ratio), self._size) # width, height
        elif h > w:
            ratio = h/w
            new_size = (self._size, int(self._size*ratio)) # width, height
        return cv2.resize(img, new_size, cv2.INTER_LINEAR)


def resize_mv(img, shape, interpolation):
    if isinstance(shape, int):
        h, w = img.shape[:2]
        if h <= w:
            ratio = w/h
            new_size = (int(shape*ratio), shape) # width, height
        elif h > w:
            ratio = h/w
            new_size = (shape, int(shape*ratio)) # width, height
        return np.stack([cv2.resize(img[..., i], new_size, interpolation) for i in range(2)], axis=2)
    else:
        return np.stack([cv2.resize(img[..., i], shape, interpolation) for i in range(2)], axis=2)


class GroupScale_w_h(object):
    def __init__(self, size):
        self._size = (size[0], size[1]) # width, height


    def __call__(self, img_group):
        imgs = []
        for img in img_group:
            if img.shape[2] == 3:
                imgs.append(cv2.resize(img, self._size, cv2.INTER_LINEAR))
            elif img.shape[2] == 2:
                imgs.append(resize_mv_w_h(img, self._size, cv2.INTER_LINEAR))
            else:
                assert False
        return imgs
    

def resize_mv_w_h(img, shape, interpolation):
    return np.stack([cv2.resize(img[..., i], shape, interpolation)
                     for i in range(2)], axis=2)


class GroupFullResSample(object):
    def __init__(self, crop_size, scale_size=None, flip=True):
        self.crop_size = crop_size if not isinstance(crop_size, int) else (crop_size, crop_size)

        if scale_size is not None:
            self.scale_worker = GroupScale(scale_size)
            # self.scale_worker2 = GroupScale([224,224])
        else:
            self.scale_worker = None
        self.flip = flip

    def __call__(self, img_group):
        if self.scale_worker is not None:
            img_group = self.scale_worker(img_group)

        image_w, image_h, _ = img_group[0].shape
        crop_w, crop_h = self.crop_size

        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        offsets = list()
        offsets.append((0 * w_step, 2 * h_step))  # left
        offsets.append((4 * w_step, 2 * h_step))  # right
        offsets.append((2 * w_step, 2 * h_step))  # center

        # crops * T
        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = img[o_w:o_w + crop_w, o_h:o_h + crop_h]
                # crop = img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h))
                normal_group.append(crop)
                if self.flip:
                    flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)

                    if img.mode == 'L' and i % 2 == 0:
                        flip_group.append(ImageOps.invert(flip_crop))
                    else:
                        flip_group.append(flip_crop)
            ######################################
            # normal_group = self.scale_worker2(normal_group)
            ######################################
            oversample_group.extend(normal_group)
            oversample_group.extend(flip_group)
        return oversample_group



class GroupOverSample(object):
    def __init__(self, crop_size, scale_size=None, is_mv=False):
        self.crop_size = crop_size if not isinstance(crop_size, int) else (crop_size, crop_size)

        if scale_size is not None:
            self.scale_worker = GroupScale(scale_size)
        else:
            self.scale_worker = None
        self._is_mv = is_mv

    def __call__(self, img_group):

        if self.scale_worker is not None:
            img_group = self.scale_worker(img_group)

        image_w, image_h, _ = img_group[0].shape
        crop_w, crop_h = self.crop_size

        offsets = GroupMultiScaleCrop.fill_fix_offset(False, image_w, image_h, crop_w, crop_h)
        oversample_group = list()

        for o_w, o_h in offsets:
            for img in img_group:

                crop = img[o_w:o_w+crop_w, o_h:o_h+crop_h]
                oversample_group.append(crop)

                # flip_crop = crop[:, ::-1, :].astype(np.int32)
                # if self._is_mv:
                #     assert flip_crop.shape[2] == 2, flip_crop.shape
                #     flip_crop -= 128
                #     flip_crop[..., 0] *= (-1)
                #     flip_crop += 128
                # oversample_group.append(flip_crop)

        return oversample_group


class GroupMultiScaleCrop(object):
    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]

    def __call__(self, img_group):

        im_size = img_group[0].shape

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)

        crop_img_group = [img[offset_w:offset_w + crop_w, offset_h:offset_h + crop_h] for img in img_group]

        # if crop_img_group[0].shape[2] == 3:
        #     ret_img_group = [cv2.resize(img, (self.input_size[0], self.input_size[1]),
        #                                 cv2.INTER_LINEAR)
        #                      for img in crop_img_group]
        # elif crop_img_group[0].shape[2] == 2:
        #     ret_img_group = [resize_mv(img, (self.input_size[0], self.input_size[1]), cv2.INTER_LINEAR)
        #                      for img in crop_img_group]

        '''
        this is for fusion data of [iframe, mv, residual]
        '''
        ret_img_group = []
        for img in crop_img_group:
            if img.shape[2] == 3:
                ret_img_group.append(cv2.resize(img, (self.input_size[0], self.input_size[1]),
                                                cv2.INTER_LINEAR))
            elif img.shape[2] == 2:
                ret_img_group.append(resize_mv(img, (self.input_size[0], self.input_size[1]), 
                                               cv2.INTER_LINEAR))               
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret


if __name__ == "__main__":
    trans = torchvision.transforms.Compose([
        GroupScale(256)]
    )

    # im = Image.open('../tensorflow-model-zoo.torch/lena_299.png')

    im = np.array(Image.open('/home/zhengwei/test_image.jpg').convert("RGB"))
    color_group = [im.transpose(1,0,2)] * 3
    rst = trans(color_group)
    pdb.set_trace()
    Image.fromarray(rst[0]).save('/home/zhengwei/test_image_trans.jpg') 
    # cv2.imwrite('/home/zhengwei/test_image_trans.jpg', rst[0]) 

    # gray_group = [im.convert('L')] * 9
    # gray_rst = trans(gray_group)

    # trans2 = torchvision.transforms.Compose([
    #     GroupRandomSizedCrop(256),
    #     Stack(),
    #     ToTorchFormatTensor(),
    #     GroupNormalize(
    #         mean=[.485, .456, .406],
    #         std=[.229, .224, .225])
    # ])
    # print(trans2(color_group))








'''
---------------------------------------------------------------------------------------------------------
Modules from TSM
'''

# class ToTorchFormatTensor(object):
#     """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
#     to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
#     def __init__(self, div=True):
#         self.div = div

#     def __call__(self, pic):
#         if isinstance(pic, np.ndarray):
#             # handle numpy array
#             img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
#         else:
#             # handle PIL Image
#             img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
#             img = img.view(pic.size[1], pic.size[0], len(pic.mode))
#             # put it from HWC to CHW format
#             # yikes, this transpose takes 80% of the loading time/CPU
#             img = img.transpose(0, 1).transpose(0, 2).contiguous()
#         return img.float().div(255) if self.div else img.float()

# class Stack(object):
#     def __init__(self, roll=False):
#         self.roll = roll

#     def __call__(self, img_group):
#         if img_group[0].mode == 'L':
#             return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
#         elif img_group[0].mode == 'RGB':
#             if self.roll:
#                 return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
#             else:
#                 return np.concatenate(img_group, axis=2)

# class GroupNormalize(object):
#     def __init__(self, mean, std):
#         self.mean = mean
#         self.std = std

#     def __call__(self, tensor):
#         # pdb.set_trace()
#         rep_mean = self.mean * (tensor.size()[0]//len(self.mean))
#         rep_std = self.std * (tensor.size()[0]//len(self.std))

#         # TODO: make efficient
#         for t, m, s in zip(tensor, rep_mean, rep_std):
#             # pdb.set_trace()
#             t.sub_(m).div_(s)
#         return tensor