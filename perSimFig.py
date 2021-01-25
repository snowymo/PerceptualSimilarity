import cv2


scenes = ["bedroom", "gallery", "gas", "mc"]
scene = "bedroom"
imgID=12
fovs=[20,60,100]

height=1600
width=1440

for scene in scenes:
    for fov in fovs:
        rect_top = int(height / 2 - float(fov) / 110.0 * height / 2)
        rect_btm = int(height / 2 + float(fov) / 110.0 * height / 2)
        rect_left = int(width / 2 - float(fov) / 110.0 * width / 2)
        rect_right = int(width / 2 + float(fov) / 110.0 * width / 2)

        img_our = cv2.imread('./imgs/eval_'+scene+'/view' + f'{imgID:04d}' + '.png')
        crop_img_our = img_our[rect_top:rect_btm, rect_left:rect_right]

        cv2.imwrite(scene+str(imgID)+"fov"+str(fov)+".png", crop_img_our)