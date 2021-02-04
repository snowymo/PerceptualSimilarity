import os
import ffmpeg
import cv2
import imageio
import numpy as np
import json
import glob
import sys

def load_centers(data_desc_file):
    with open(data_desc_file, 'r', encoding='utf-8') as file:
        data_desc = json.loads(file.read())
        return data_desc['gaze_centers']


hint = cv2.imread('fovea_hint.png', cv2.IMREAD_UNCHANGED)


def add_hint(img, center):
    fovea_origin = (
        int(center[0]) + 1440 // 2 - hint.shape[1] // 2,
        int(center[1]) + 1600 // 2 - hint.shape[0] // 2
    )
    fovea_region = (
        slice(fovea_origin[1], fovea_origin[1] + hint.shape[0]),
        slice(fovea_origin[0], fovea_origin[0] + hint.shape[1]),
        ...
    )
    img[fovea_region] = (img[fovea_region] * (1 - hint[..., 3:] / 255.0) +
                         hint[..., :3] * (hint[..., 3:] / 255.0)).astype(np.uint8)
    return img

# print(data_desc.shape)
for folder in glob.glob('hint/NeRF/*_demoonly_*'):  # assuming folder
    print(folder)
    i = 0
    if("gallery" in folder):
        data_desc = load_centers("scene5_left.json")
    else:
        data_desc = load_centers("scene1-4_left.json")
    if("bedroom" in folder):
        continue
    for filename in glob.glob(folder + '/*.png'):  # assuming png
        if i % 100 == 0:
            print(filename)
        im = cv2.imread(filename)
        # print(im.shape)
        im2 = add_hint(im, data_desc[i])
        i = i+1
        hintFile = filename.replace("only","hint")
        # print(hintFile)
        cv2.imwrite(hintFile, im2)
# imageio.mimwrite(os.path.join('hint/NeRF/bedroom_demohint_580001/', 'video.mp4'), to8b(rgbs), fps=30, quality=8)
# ffmpeg.input('hint/NeRF/bedroom_demohint_580001/*.png', pattern_type='glob', framerate=30).output('movie.mp4')