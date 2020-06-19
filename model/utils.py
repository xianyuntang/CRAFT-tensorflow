import glob
import math
import os
import numpy as np
import cv2
import random
from imgaug import augmenters as iaa


def load_train_data(dataset_dir: list, batch_size):
    images = []
    affinity_maps = []
    region_maps = []
    for dataset_path in dataset_dir:
        images += glob.glob(os.path.join(dataset_path, 'image', '*'))
        affinity_maps += glob.glob(os.path.join(dataset_path, 'affinity_map', '*'))
        region_maps += glob.glob(os.path.join(dataset_path, 'region_map', '*'))
    images.sort()
    affinity_maps.sort()
    region_maps.sort()
    while True:
        batches = random.sample(range(0, len(images)), k=batch_size)
        batch_image = []
        batch_affinity = []
        batch_region = []

        for i in batches:
            image = cv2.imread(images[i], 0)
            _, image = cv2.threshold(image, 127, 255, cv2.THRESH_OTSU)
            image = image / 255
            image = image.astype(np.float32)
            image = np.expand_dims(image, -1)
            batch_image.append(image)

            affinity_map = cv2.imread(affinity_maps[i], 0)
            affinity_map = affinity_map / 255
            affinity_map = affinity_map.astype(np.float32)

            batch_affinity.append(affinity_map)

            region_map = cv2.imread(region_maps[i], 0)
            region_map = region_map / 255
            region_map = region_map.astype(np.float32)

            batch_region.append(region_map)

        batch_image = np.array(batch_image)
        batch_affinity = np.array(batch_affinity)
        batch_region = np.array(batch_region)

        yield batch_image, batch_region, batch_affinity


def load_val_data(dataset_dir: list, batch_size=3):
    images = []
    affinity_maps = []
    region_maps = []
    for dataset_path in dataset_dir:
        images += glob.glob(os.path.join(dataset_path, 'image', '*'))

        affinity_maps += glob.glob(os.path.join(dataset_path, 'affinity_map', '*'))
        region_maps += glob.glob(os.path.join(dataset_path, 'region_map', '*'))
    images.sort()
    affinity_maps.sort()
    region_maps.sort()
    for i in range(len(images)):
        batch_image = []
        batch_affinity = []
        batch_region = []
        for j in range(batch_size):
            image = cv2.imread(images[i], 0)
            _, image = cv2.threshold(image, 127, 255, cv2.THRESH_OTSU)
            image = image / 255
            image = image.astype(np.float32)
            image = np.expand_dims(image, -1)
            batch_image.append(image)

            affinity_map = cv2.imread(affinity_maps[i], 0)
            affinity_map = affinity_map / 255
            affinity_map = affinity_map.astype(np.float32)

            batch_affinity.append(affinity_map)

            region_map = cv2.imread(region_maps[i], 0)
            region_map = region_map / 255
            region_map = region_map.astype(np.float32)

            batch_region.append(region_map)

        batch_image = np.array(batch_image)
        batch_affinity = np.array(batch_affinity)
        batch_region = np.array(batch_region)

        yield batch_image, batch_region, batch_affinity


def img_aug_seq():
    # first we need switch height and width

    def sometimes(aug):
        return iaa.Sometimes(0.5, aug)

    seq = iaa.Sequential(
        [iaa.AdditiveGaussianNoise(scale=(0, 20)),
         iaa.SaltAndPepper(p=0.05),
         sometimes(iaa.Invert(p=1)),
         ]
    )
    return seq


def getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text):
    # prepare data
    linkmap = np.squeeze(linkmap)
    textmap = np.squeeze(textmap)
    linkmap = linkmap.copy()
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    """ labeling method """
    ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
    ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)

    text_score_comb = np.clip(text_score + link_score, 0, 1)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8),
                                                                         connectivity=4)

    det = []
    mapper = []
    for k in range(1, nLabels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10: continue

        # thresholding
        # if np.max(textmap[labels == k]) < text_threshold: continue

        # make segmentation map
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels == k] = 255
        segmap[np.logical_and(link_score == 1, text_score == 0)] = 0  # remove link area
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        # boundary check
        if sx < 0: sx = 0
        if sy < 0: sy = 0
        if ex >= img_w: ex = img_w
        if ey >= img_h: ey = img_h
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        # make box
        np_contours = np.roll(np.array(np.where(segmap != 0)), 1, axis=0).transpose().reshape(-1, 2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        # align diamond-shape
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
            t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        box = np.array(box)

        det.append(box)
        mapper.append(k)

    return det, labels, mapper


def getDetBoxes(textmap, linkmap, text_threshold, link_threshold, low_text):
    boxes, labels, mapper = getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text)

    return boxes


def adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net=2):
    if len(polys) > 0:
        polys = np.array(polys)
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    return polys


def get_box(region_score, affinity_map, min_thresh=25, max_thresh=100000):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    region_score = np.squeeze((region_score * 255).astype(np.uint8))
    affinity_map = np.squeeze((affinity_map * 255).astype(np.uint8))
    _, region_score = cv2.threshold(region_score, 100, 255, 0)
    _, affinity_map = cv2.threshold(affinity_map, 100, 255, 0)
    text_map = region_score + affinity_map
    text_map = np.clip(text_map, 0, 255)
    cv2.imshow('origin', text_map)
    # cv2.waitKey(0)
    mix = cv2.morphologyEx(text_map, cv2.MORPH_CLOSE, kernel)
    # mix = cv2.dilate(text_map,kernel)
    cv2.imshow('morphology', mix)
    cv2.waitKey(0)
    output, _, cord, _ = cv2.connectedComponentsWithStats(mix, 4, cv2.CV_32S)
    boxes = []
    for i in range(output):
        if min_thresh <= cord[i][4] <= max_thresh:
            boxes.append([cord[i][0] * 3, cord[i][1] * 2.8125, (cord[i][0] + cord[i][2]) * 3,
                          (cord[i][1] + cord[i][3]) * 2.8125])
    print(boxes)
    return boxes


if __name__ == '__main__':
    next(test_data())
