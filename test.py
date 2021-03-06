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
scenes=["bedroom", "gas", "lobby", "mc", "gallery"]
# scenes=["mc", "gallery"]
# scenes=["lobby"]

imgCount = 20
step = 5
fovCount = int((110-5)/step)
result = np.empty(shape=(fovCount * len(scenes) + 2, 2 + imgCount * 4))
result[0, 0] = 0 #scene id
result[0, 1] = 368 #fov
# result[0, 2] = 687 #our
# result[0, 3] = 6373 #nerf
# result[0, 4] = 36832 #fovea
result4curve = np.empty(shape = (imgCount*fovCount*len(scenes), 7))

# f=open('lpips_result_fova_anova.csv','a')
anova = np.empty(shape=(len(scenes)*imgCount+1, 2 + 4 * fovCount))
# anova[0,0] = "scene"
# anova[0,1] = "imgid"
# for i in range(5,110,5):
#     anova[0, 2 + 3 * (int(i / 5) - 1)] = "our-" + str(i)
#     anova[0, 2 + 3 * (int(i / 5) - 1) + 1] = "nerf-" + str(i)
#     anova[0, 2 + 3 * (int(i / 5) - 1) + 2] = "fovea-" + str(i)

for sceneID, scene in enumerate(scenes):
    if scene == "lobby":
        continue
    # # if scene == "mc":
    # #     continue
    # if sceneID < 4:
    #     continue
    for imgID in range(imgCount):
        anova[sceneID * imgCount + imgID + 1, 0] = sceneID
        anova[sceneID * imgCount + imgID + 1, 1] = imgID

        img_gt = cv2.imread('./imgs/gt_' + scene + '/view_' + f'{imgID:04d}' + '.png')
        img_our_left = cv2.imread('./imgs/eval_mono/ref_as_left_eye/'+scene+'/view' + f'{imgID:04d}' + '_blended.png')
        img_our_right = cv2.imread(
            './imgs/eval_mono/ref_as_right_eye/' + scene + '/view' + f'{imgID:04d}' + '_blended.png')
        img_nerf = cv2.imread('./imgs/NeRF_'+scene+'/' + f'{imgID:03d}' + '.png')
        img_gtfova = cv2.imread('./imgs/gt_'+scene+'/view_' + f'{imgID:04d}' + '_RT_k3.0.png')
        print('./imgs/gt_' + scene + '/view_' + f'{imgID:04d}' + '.png')
        print(img_our_right.shape)
        # print(img_gtfova.shape)
        height, width = img_gt.shape[:2]
        for fov in range(5,110,step):
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
            crop_img_our_left = img_our_left[rect_top:rect_btm, rect_left:rect_right]
            ex_p0l = lpips.im2tensor(crop_img_our_left[:,:,::-1])
            crop_img_our_right = img_our_right[rect_top:rect_btm, rect_left:rect_right]
            ex_p0r = lpips.im2tensor(crop_img_our_right[:, :, ::-1])

            crop_img_nerf = img_nerf[rect_top:rect_btm, rect_left:rect_right]
            ex_p1 = lpips.im2tensor(crop_img_nerf[:,:,::-1])
            crop_img_gt_fova = img_gtfova[rect_top:rect_btm, rect_left:rect_right]
            ex_p2 = lpips.im2tensor(crop_img_gt_fova[:, :, ::-1])
            if (use_gpu):
                ex_ref = ex_ref.cuda()
                ex_p0l = ex_p0l.cuda()
                ex_p0r = ex_p0r.cuda()
                ex_p1 = ex_p1.cuda()
                ex_p2 = ex_p2.cuda()

            ex_d0l = loss_fn.forward(ex_ref, ex_p0l)
            ex_d0r = loss_fn.forward(ex_ref, ex_p0r)
            ex_d1 = loss_fn.forward(ex_ref, ex_p1)
            ex_d2 = loss_fn.forward(ex_ref, ex_p2)

            if not spatial:
                print('SPATIAL fov %d Distances: OUR %.3f, FOVA %.3f' % (fov, ex_d0, ex_d2))
                # print('fov %d Distances: OUR %.3f, NeRF %.3f, FOVA %.3f' % (fov, ex_d0, ex_d1, ex_d2))
                result[sceneID*fovCount + int(fov / 5 - 1) + 1, 0] = sceneID # scene id
                result[sceneID*fovCount + int(fov / 5 - 1) + 1, 1] = fov
                result[sceneID*fovCount  + int(fov / 5 - 1) + 1, 0 * imgCount + 2+imgID] = ex_d0
                result[sceneID*fovCount + imgID * 21 + int(fov / 5 - 1) + 1, 1 * imgCount + 2+imgID] = ex_d1
                result[sceneID * fovCount + int(fov / 5 - 1) + 1, 2 * imgCount + 2+imgID] = ex_d2

                anova[sceneID * imgCount + imgID + 1, 2 + 3 * int(fov / 5 - 1)] = ex_d0
                anova[sceneID * imgCount + imgID + 1, 2 + 3 * int(fov / 5 - 1)+1] = ex_d1
                anova[sceneID * imgCount + imgID + 1, 2 + 3 * int(fov / 5 - 1)+2] = ex_d2

                result4curve[sceneID * fovCount * imgCount + imgID * fovCount + int(fov / step - 1)] = [sceneID, fov, ex_d0, ex_d1, ex_d2, imgID]
            else:
                # print('fov %d Distances: OUR %.3f, FOVA %.3f' % (
                #     fov, ex_d0.mean(), ex_d2.mean()))  # The mean distance is approximately the same as the non-spatial distance
                print('fov %d Distances: OUR %.3f, %.3f,, NeRF %.3f, FOVA %.3f' % (
                    fov, ex_d0l.mean(), ex_d0r.mean(), ex_d1.mean(),
                    ex_d2.mean()))  # The mean distance is approximately the same as the non-spatial distance
                result[sceneID * fovCount + int((fov-5) / step) + 1, 0] = sceneID  # scene id
                result[sceneID * fovCount + int((fov-5) / step) + 1, 1] = fov
                result[sceneID * fovCount + int((fov-5) / step) + 1, 0 * imgCount + 2 + imgID] = ex_d0l.mean()
                result[sceneID * fovCount + int((fov - 5) / step) + 1, 1 * imgCount + 2 + imgID] = ex_d0r.mean()
                result[sceneID * fovCount + int((fov-5) / step) + 1, 2 * imgCount + 2 + imgID] = ex_d1.mean()
                result[sceneID * fovCount + int((fov-5) / step) + 1, 3 * imgCount + 2+imgID] = ex_d2.mean()

                fovidx = int((fov-5) / step)
                anova[sceneID * imgCount + imgID + 1, 2 +fovidx ] = ex_d0l.mean()
                anova[sceneID * imgCount + imgID + 1, 2 + fovCount + fovidx] = ex_d0r.mean()
                anova[sceneID * imgCount + imgID + 1, 2 + fovCount * 2 + fovidx] = ex_d1.mean()
                anova[sceneID * imgCount + imgID + 1, 2 + fovCount * 3 + fovidx] = ex_d2.mean()

                result4curve[sceneID * fovCount * imgCount + imgID * fovCount + int((fov-5) / step)] = [sceneID, fov,
                                                                                                     ex_d0l.mean(), ex_d0r.mean(),ex_d1.mean(),
                                                                                                     ex_d2.mean(), imgID]

                # Visualize a spatially-varying distance map between ex_p0 and ex_ref
                # import pylab

                # pylab.imshow(ex_d0[0, 0, ...].data.cpu().numpy())
                # pylab.show()
    # np.savetxt(f, anova[(sceneID) * 8+1:sceneID * 8+9], delimiter=',')
    np.savetxt('lpips_curve_125_' + scene + '.csv', result4curve, delimiter=',')
    np.savetxt('lpips_fova_125_' + scene+'.csv', result, delimiter=',')
    np.savetxt('lpips_anova_125_' + scene+'.csv', anova, delimiter=',')


# crop_img = img[y:y+h, x:x+w]
# ex_ref = lpips.im2tensor(lpips.load_image('./imgs/ex_ref.png'))
# ex_p0 = lpips.im2tensor(lpips.load_image('./imgs/ex_p0.png'))
# ex_p1 = lpips.im2tensor(lpips.load_image('./imgs/ex_p1.png'))
