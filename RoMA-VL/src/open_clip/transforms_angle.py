# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'

import pywt
import torch
import math

from skimage.feature import ORB

from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np

class RandomCropWithPosition(transforms.RandomCrop):
    def __init__(self, coord, size):
        self.coord = coord
        self.size = size

    @staticmethod
    def get_params(img, output_size, *args, **kwargs):  # img: 224 output_size: 96
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image or Tensor): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        h, w = img.shape[1], img.shape[2]
        th, tw = output_size
        if h + 1 < th or w + 1 < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger then input image size {(h, w)}")
        if w == tw and h == th:
            return 0, 0, h, w

        coord = args[0]
        idx_i = torch.randint(0, coord.shape[0], size=(1,))
        idx_j = torch.randint(0, coord.shape[0], size=(1,))
        i = coord[idx_i].item()
        j = coord[idx_j].item()

        return i, j, th, tw

    def __call__(self, img):
        i, j, h, w = self.get_params(img, (self.size, self.size), self.coord)

        return F.crop(img, i, j, h, w), i, j, h, w


class ScalingCenterCrop(object):
    def __init__(self, input_size, crop_size, nums_crop, r_range):
        self.nums_crop = nums_crop
        self.inp = input_size
        self.crop = 16
        self.angles = r_range
        self.bounding = self.crop * (2 ** 0.5)
        self.patch_size = 16
        coord_init = torch.arange(0, self.inp - self.crop, self.patch_size) - ((self.bounding - self.crop) // 2)
        self.coord = coord_init.int()
        #self.hog = HOGLayerC(nbins=9, pool=8, norm_out=True, in_channels=3)

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(self.inp, scale=(0.2, 1.0), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.crop_bounding = RandomCropWithPosition(self.coord, size=math.ceil(self.bounding))
        self.rotated_crop_t_96 = transforms.Compose([
            transforms.RandomRotation(self.angles),
            transforms.CenterCrop(size=96)
        ])
        self.rotated_crop_t_64 = transforms.Compose([
            transforms.RandomRotation(self.angles),
            transforms.CenterCrop(size=64)
        ])
        self.rotated_crop_t_32 = transforms.Compose([
            transforms.RandomRotation(self.angles),
            transforms.CenterCrop(size=32)
        ])
    def generate_coords(self):
        top_init = torch.arange(0, self.inp, self.patch_size) - ((self.bounding - self.crop) // 2)
        top_init = top_init[torch.gt(top_init, 0)].int()
        left_init = torch.arange(0, self.inp, self.patch_size) - ((self.bounding - self.crop) // 2)
        left_init = left_init[torch.gt(left_init, 0)].int()
        grid_top, grid_left = torch.meshgrid(top_init, left_init)
        coords = torch.stack((grid_top, grid_left), dim=-1).view(-1, 2)
        return coords

    def extract_patches(self,img, patch_size):
        C, H, W = img.shape
        pad_h = (patch_size - H % patch_size) % patch_size
        pad_w = (patch_size - W % patch_size) % patch_size
        img_padded = np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
        patches = np.lib.stride_tricks.as_strided(
            img_padded,
            shape=(C, (H + pad_h) // patch_size, patch_size, (W + pad_w) // patch_size, patch_size),
            strides=(img_padded.strides[0], patch_size * img_padded.strides[1], img_padded.strides[1],
                     patch_size * img_padded.strides[2], img_padded.strides[2])
        )
        patches = patches.transpose(1, 3, 2, 4, 0).reshape(-1, C, patch_size, patch_size)

        patch_list = []
        for idx, patch in enumerate(patches):
            i = (idx % ((W + pad_w) // patch_size)) * patch_size
            j = (idx // ((W + pad_w) // patch_size)) * patch_size
            patch_list.append((patch, i, j))
        return patch_list

    def compute_hog_for_patch(self, patch):

        if patch.shape[0] == 1:
            gray_patch = patch[0]
            fd = hog(gray_patch, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
            return np.sum(fd)
        elif patch.shape[0] == 3:

            fd_r = hog(patch[0], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
            fd_g = hog(patch[1], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
            fd_b = hog(patch[2], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
            fd_all = np.concatenate((fd_r, fd_g, fd_b))
            return np.sum(fd_all)
        else:
            raise ValueError(f"Invalid patch shape: {patch.shape}. Expected 1 or 3 channels.")

    def compute_wavelet_for_patch(self,patch):

        coeffs = pywt.dwt2(patch, 'haar')
        LL, (LH, HL, HH) = coeffs

        wavelet_score = np.linalg.norm(LH) + np.linalg.norm(HL) + np.linalg.norm(HH)
        return wavelet_score

    def compute_lbp(self, patch):
        gray_patch = np.mean(patch, axis=2)
        height, width = gray_patch.shape
        lbp_values = np.zeros_like(gray_patch, dtype=int)
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                center = gray_patch[i, j]
                binary_values = []
                for dx, dy in [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]:
                    neighbor = gray_patch[i + dx, j + dy]
                    binary_values.append(1 if neighbor >= center else 0)
                lbp_values[i, j] = sum([bit * (2 ** idx) for idx, bit in enumerate(binary_values)])

        hist, _ = np.histogram(lbp_values.flatten(), bins=np.arange(0, 256), density=True)
        lbp_norm = np.linalg.norm(hist)
        return lbp_norm

    def __call__(self, image):
        img = image
        img_ori = img.clone()

        top_start = []
        left_start = []
        crop_size = []

        for _ in range(self.nums_crop):
            patches = self.extract_patches(img, 16)
            patch_scores = []
            hog_dict = {}

            for patch, i, j in patches:
                hog_values = self.compute_lbp(patch)
                patch_scores.append((hog_values, i, j, patch))
                hog_dict[(i, j)] = hog_values

            avg_hog_value = sum(score[0] for score in patch_scores) / len(patch_scores)
            patch_scores.sort(reverse=True, key=lambda x: x[0])
            top_patch = patch_scores[0]
            top_i, top_j = top_patch[1], top_patch[2]

            candidate_patches_hog = []
            for offset_i in range(-80, 81, 16):
                for offset_j in range(-80, 81, 16):
                    start_i = top_i + offset_i
                    start_j = top_j + offset_j
                    if start_i < 0 or start_j < 0 or start_i + 96 > img.shape[1] or start_j + 96 > img.shape[2]:
                        continue
                    large_hog_value = 0
                    for i_offset in range(0, 96, 16):
                        for j_offset in range(0, 96, 16):
                            sub_i = start_i + i_offset
                            sub_j = start_j + j_offset
                            if (sub_i, sub_j) in hog_dict:
                                large_hog_value += hog_dict[(sub_i, sub_j)]
                    large_patch = img[:, start_i:start_i + 96, start_j:start_j + 96]
                    candidate_patches_hog.append((large_hog_value, start_i, start_j, large_patch))
            candidate_patches_hog.sort(reverse=True, key=lambda x: x[0])
            best_large_patch = candidate_patches_hog[0]
            best_hog_value, best_top, best_left, best_patch = best_large_patch

            if best_hog_value / 36 >=  avg_hog_value:
                rotated_patch_96 = self.rotated_crop_t_96(torch.from_numpy(best_patch.numpy()))
                img[:, best_top:best_top + 96, best_left:best_left + 96] = rotated_patch_96
                top_start.append(best_top // self.patch_size)
                left_start.append(best_left // self.patch_size)
                crop_size.append(96)
            else:
                candidate_patches_hog_64 = []
                for offset_i in range(-48, 49, 16):
                    for offset_j in range(-48, 49, 16):
                        start_i_64 = top_i + offset_i
                        start_j_64 = top_j + offset_j
                        if start_i_64 < 0 or start_j_64 < 0 or start_i_64 + 64 > img.shape[1] or start_j_64 + 64 > img.shape[2]:
                            continue
                        large_hog_value_64 = 0
                        for i_offset in range(0, 64, 16):
                            for j_offset in range(0, 64, 16):
                                sub_i_64 = start_i_64 + i_offset
                                sub_j_64 = start_j_64 + j_offset
                                if (sub_i_64, sub_j_64) in hog_dict:
                                    large_hog_value_64 += hog_dict[(sub_i_64, sub_j_64)]
                        large_patch_64 = img[:, start_i_64:start_i_64 + 64, start_j_64:start_j_64 + 64]
                        candidate_patches_hog_64.append((large_hog_value_64, start_i_64, start_j_64, large_patch_64))

                candidate_patches_hog_64.sort(reverse=True, key=lambda x: x[0])
                best_large_patch_64 = candidate_patches_hog_64[0]
                best_hog_value_64, best_top_64, best_left_64, best_patch_64 = best_large_patch_64

                if best_hog_value_64 / 16 >= avg_hog_value:
                    rotated_patch_64 = self.rotated_crop_t_64(torch.from_numpy(best_patch_64.numpy()))
                    img[:, best_top_64:best_top_64 + 64, best_left_64:best_left_64 + 64] = rotated_patch_64
                    rotated_patch=rotated_patch_64
                    top_start.append(best_top_64 // self.patch_size)
                    left_start.append(best_left_64 // self.patch_size)
                    crop_size.append(64)

                else:
                    candidate_patches_hog_32 = []
                    for offset_i in range(-16, 17, 16):
                        for offset_j in range(-16, 17, 16):
                            start_i_32 = top_i + offset_i
                            start_j_32 = top_j + offset_j

                            if start_i_32 < 0 or start_j_32 < 0 or start_i_32 + 32 > img.shape[1] or start_j_32 + 32 > img.shape[2]:
                                continue

                            large_hog_value_32 = 0
                            for i_offset in range(0, 32, 16):
                                for j_offset in range(0, 32, 16):
                                    sub_i_32 = start_i_32 + i_offset
                                    sub_j_32 = start_j_32 + j_offset
                                    if (sub_i_32, sub_j_32) in hog_dict:
                                        large_hog_value_32 += hog_dict[(sub_i_32, sub_j_32)]

                            large_patch_32 = img[:, start_i_32:start_i_32 + 32, start_j_32:start_j_32 + 32]
                            candidate_patches_hog_32.append((large_hog_value_32, start_i_32, start_j_32, large_patch_32))

                    candidate_patches_hog_32.sort(reverse=True, key=lambda x: x[0])
                    best_large_patch_32 = candidate_patches_hog_32[0]

                    best_hog_value_32, best_top_32, best_left_32, best_patch_32 = best_large_patch_32
                    rotated_patch_32 = self.rotated_crop_t_32(torch.from_numpy(best_patch_32.numpy()))

                    img[:, best_top_32:best_top_32 + 32, best_left_32:best_left_32 + 32] = rotated_patch_32
                    rotated_patch=rotated_patch_32
                    top_start.append(best_top_32 // self.patch_size)
                    left_start.append(best_left_32 // self.patch_size)
                    crop_size.append(32)

        return img, img_ori, torch.tensor(top_start), torch.tensor(left_start), torch.tensor(crop_size)
    





