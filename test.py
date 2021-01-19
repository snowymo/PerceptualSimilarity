import torch
import lpips
from IPython import embed
import cv2
import numpy as np

use_gpu = False  # Whether to use GPU
spatial = True  # Return a spatial map of perceptual distance.

# Linearly calibrated models (LPIPS)
loss_fn = lpips.LPIPS(net='alex', spatial=spatial)  # Can also set net = 'squeeze' or 'vgg'
# loss_fn = lpips.LPIPS(net='alex', spatial=spatial, lpips=False) # Can also set net = 'squeeze' or 'vgg'

if (use_gpu):
    loss_fn.cuda()

## Example usage with dummy tensors
dummy_im0 = torch.zeros(1, 3, 64, 64)  # image should be RGB, normalized to [-1,1]
dummy_im1 = torch.zeros(1, 3, 64, 64)
if (use_gpu):
    dummy_im0 = dummy_im0.cuda()
    dummy_im1 = dummy_im1.cuda()
dist = loss_fn.forward(dummy_im0, dummy_im1)

## Example usage with images
# replace with ours
# read image and load them
scenes=["gas", "mc", "studyroom", "bedroom"]

result = np.empty(shape=(8 * 21 * len(scenes) + 1, 5))
result[0, 0] = 0 #scene id
result[0, 1] = 368 #fov
result[0, 2] = 687 #our
result[0, 3] = 6373 #nerf
result[0, 4] = 36832 #fovea

for sceneID, scene in enumerate(scenes):
    for imgID in range(8):
        img_gt = cv2.imread('./imgs/gt_' + scene + '/view_000' + str(imgID) + '.png')
        img_our = cv2.imread('./imgs/eval_'+scene+'/view000' + str(imgID) + '.png')
        img_nerf = cv2.imread('./imgs/NeRF_'+scene+'/00' + str(imgID) + '.png')
        img_gtfova = cv2.imread('./imgs/gt_'+scene+'/view_000' + str(imgID) + '_RT_k3.0.png')
        print('./imgs/gt_'+scene+'/view_000' + str(imgID) + '_RT_k3.png')
        print(img_gtfova.shape)
        height, width = img_gt.shape[:2]
        for fov in range(5,110,5):
            rect_top = height / 2 - float(fov) / 110.0 * height / 2
            rect_top = int(rect_top)
            rect_btm = height / 2 + float(fov) / 110.0 * height / 2
            rect_btm = int(rect_btm)
            rect_left = width / 2 - float(fov) / 110.0 * width / 2
            rect_left = int(rect_left)
            rect_right = width / 2 + float(fov) / 110.0 * width / 2
            rect_right = int(rect_right)
            crop_img_gt = img_gt[rect_top:rect_btm, rect_left:rect_right]
            ex_ref = lpips.im2tensor(crop_img_gt[:,:,::-1])
            crop_img_our = img_our[rect_top:rect_btm, rect_left:rect_right]
            ex_p0 = lpips.im2tensor(crop_img_our[:,:,::-1])
            crop_img_nerf = img_nerf[rect_top:rect_btm, rect_left:rect_right]
            ex_p1 = lpips.im2tensor(crop_img_nerf[:,:,::-1])
            crop_img_gt_fova = img_gtfova[rect_top:rect_btm, rect_left:rect_right]
            ex_p2 = lpips.im2tensor(crop_img_gt_fova[:, :, ::-1])
            if (use_gpu):
                ex_ref = ex_ref.cuda()
                ex_p0 = ex_p0.cuda()
                ex_p1 = ex_p1.cuda()
                ex_p2 = ex_p2.cuda()

            ex_d0 = loss_fn.forward(ex_ref, ex_p0)
            ex_d1 = loss_fn.forward(ex_ref, ex_p1)
            ex_d2 = loss_fn.forward(ex_ref, ex_p2)

            if not spatial:
                print('fov %d Distances: OUR %.3f, NeRF %.3f, FOVA %.3f' % (fov, ex_d0, ex_d1, ex_d2))
                result[sceneID*21*8 + imgID * 21 + int(fov / 5 - 1) + 1, 0] = sceneID # scene id
                result[sceneID*21*8 + imgID * 21 + int(fov / 5 - 1) + 1, 1] = fov
                result[sceneID*21*8 + imgID * 21 + int(fov / 5 - 1) + 1, 2] = ex_d0
                result[sceneID*21*8 + imgID * 21 + int(fov / 5 - 1) + 1, 3] = ex_d1
                result[sceneID * 21 * 8 + imgID * 21 + int(fov / 5 - 1) + 1, 4] = ex_d2
            else:
                print('fov %d Distances: OUR %.3f, NeRF %.3f, FOVA %.3f' % (
                    fov, ex_d0.mean(), ex_d1.mean() , ex_d2.mean()))  # The mean distance is approximately the same as the non-spatial distance
                result[sceneID*21*8 + imgID * 21 + int(fov / 5 - 1) + 1, 0] = sceneID  # scene id
                result[sceneID*21*8 + imgID * 21 + int(fov / 5 - 1) + 1, 1] = fov
                result[sceneID*21*8 + imgID * 21 + int(fov / 5 - 1) + 1, 2] = ex_d0.mean()
                result[sceneID*21*8 + imgID * 21 + int(fov / 5 - 1) + 1, 3] = ex_d1.mean()
                result[sceneID * 21 * 8 + imgID * 21 + int(fov / 5 - 1) + 1, 4] = ex_d2.mean()

                # Visualize a spatially-varying distance map between ex_p0 and ex_ref
                # import pylab

                # pylab.imshow(ex_d0[0, 0, ...].data.cpu().numpy())
                # pylab.show()

np.savetxt('lpips_result_fova.csv', result, delimiter=',')
# crop_img = img[y:y+h, x:x+w]
# ex_ref = lpips.im2tensor(lpips.load_image('./imgs/ex_ref.png'))
# ex_p0 = lpips.im2tensor(lpips.load_image('./imgs/ex_p0.png'))
# ex_p1 = lpips.im2tensor(lpips.load_image('./imgs/ex_p1.png'))
