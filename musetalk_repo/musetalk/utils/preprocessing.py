import sys
from face_detection import FaceAlignment,LandmarksType
from os import listdir, path
import subprocess
import numpy as np
import cv2
import pickle
import os
import json
import torch
from tqdm import tqdm

# PlayItNews note: the original MuseTalk used dwpose/mmpose (mmcv) to obtain the
# 68 iBUG face landmarks. mmcv has no cu128 wheel and cannot be source-built
# here (no CUDA toolkit), so on Blackwell GPUs we replace it with the pure-torch
# `face-alignment` package, which returns the SAME iBUG-68 ordering that the
# COCO-wholebody face slice [23:91] produced. All downstream index math
# (28/29/30, min/max) is therefore unchanged.
import face_alignment as _fa_pkg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# bundled SFD detector — used only for the coarse face bounding box
fa = FaceAlignment(LandmarksType._2D, flip_input=False, device=str(device))

# 68-point landmark detector (replaces dwpose)
_fa68 = _fa_pkg.FaceAlignment(
    _fa_pkg.LandmarksType.TWO_D, flip_input=False, device=str(device)
)


def _detect_face_landmarks(frame_bgr):
    """Return an (68, 2) int32 array of iBUG-68 face landmarks, or None."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    preds = _fa68.get_landmarks_from_image(rgb)
    if not preds:
        return None
    return preds[0][:68].astype(np.int32)


# maker if the bbox is not sufficient 
coord_placeholder = (0.0,0.0,0.0,0.0)

# Per-frame face detection produces small frame-to-frame coordinate jitter that
# makes the inpainted mouth "shake". For a near-static talking-head avatar we
# can smooth the bbox coordinates with a centered moving average. Window size is
# env-tunable (MUSETALK_BBOX_SMOOTH, odd number; <=1 disables). Changing it
# affects avatar PREP, so the avatar cache must be cleared after a change.
_BBOX_SMOOTH = int(os.environ.get("MUSETALK_BBOX_SMOOTH", "5"))


def _smooth_coords(coords_list, window):
    """Centered moving average over valid bbox coords; skips placeholders."""
    if window is None or window <= 1 or len(coords_list) < 2:
        return coords_list
    valid = [c != coord_placeholder for c in coords_list]
    half = window // 2
    out = list(coords_list)
    n = len(coords_list)
    for i in range(n):
        if not valid[i]:
            continue
        seg = [coords_list[j] for j in range(max(0, i - half), min(n, i + half + 1)) if valid[j]]
        if not seg:
            continue
        m = np.mean(np.asarray(seg, dtype=float), axis=0)
        out[i] = (int(round(m[0])), int(round(m[1])), int(round(m[2])), int(round(m[3])))
    return out


def resize_landmark(landmark, w, h, new_w, new_h):
    w_ratio = new_w / w
    h_ratio = new_h / h
    landmark_norm = landmark / [w, h]
    landmark_resized = landmark_norm * [new_w, new_h]
    return landmark_resized

def read_imgs(img_list):
    frames = []
    print('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

def get_bbox_range(img_list,upperbondrange =0):
    frames = read_imgs(img_list)
    batch_size_fa = 1
    batches = [frames[i:i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
    coords_list = []
    landmarks = []
    if upperbondrange != 0:
        print('get key_landmark and face bounding boxes with the bbox_shift:',upperbondrange)
    else:
        print('get key_landmark and face bounding boxes with the default value')
    average_range_minus = []
    average_range_plus = []
    for fb in tqdm(batches):
        face_land_mark = _detect_face_landmarks(np.asarray(fb)[0])

        # get bounding boxes by face detetion
        bbox = fa.get_detections_for_batch(np.asarray(fb))
        
        # adjust the bounding box refer to landmark
        # Add the bounding box to a tuple and append it to the coordinates list
        for j, f in enumerate(bbox):
            if f is None or face_land_mark is None: # no face in the image
                coords_list += [coord_placeholder]
                continue
            
            half_face_coord =  face_land_mark[29]#np.mean([face_land_mark[28], face_land_mark[29]], axis=0)
            range_minus = (face_land_mark[30]- face_land_mark[29])[1]
            range_plus = (face_land_mark[29]- face_land_mark[28])[1]
            average_range_minus.append(range_minus)
            average_range_plus.append(range_plus)
            if upperbondrange != 0:
                half_face_coord[1] = upperbondrange+half_face_coord[1] #手动调整  + 向下（偏29）  - 向上（偏28）

    text_range=f"Total frame:「{len(frames)}」 Manually adjust range : [ -{int(sum(average_range_minus) / len(average_range_minus))}~{int(sum(average_range_plus) / len(average_range_plus))} ] , the current value: {upperbondrange}"
    return text_range
    

def get_landmark_and_bbox(img_list,upperbondrange =0):
    frames = read_imgs(img_list)
    batch_size_fa = 1
    batches = [frames[i:i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
    coords_list = []
    landmarks = []
    if upperbondrange != 0:
        print('get key_landmark and face bounding boxes with the bbox_shift:',upperbondrange)
    else:
        print('get key_landmark and face bounding boxes with the default value')
    average_range_minus = []
    average_range_plus = []
    for fb in tqdm(batches):
        face_land_mark = _detect_face_landmarks(np.asarray(fb)[0])

        # get bounding boxes by face detetion
        bbox = fa.get_detections_for_batch(np.asarray(fb))
        
        # adjust the bounding box refer to landmark
        # Add the bounding box to a tuple and append it to the coordinates list
        for j, f in enumerate(bbox):
            if f is None or face_land_mark is None: # no face in the image
                coords_list += [coord_placeholder]
                continue
            
            half_face_coord =  face_land_mark[29]#np.mean([face_land_mark[28], face_land_mark[29]], axis=0)
            range_minus = (face_land_mark[30]- face_land_mark[29])[1]
            range_plus = (face_land_mark[29]- face_land_mark[28])[1]
            average_range_minus.append(range_minus)
            average_range_plus.append(range_plus)
            if upperbondrange != 0:
                half_face_coord[1] = upperbondrange+half_face_coord[1] #手动调整  + 向下（偏29）  - 向上（偏28）
            half_face_dist = np.max(face_land_mark[:,1]) - half_face_coord[1]
            min_upper_bond = 0
            upper_bond = max(min_upper_bond, half_face_coord[1] - half_face_dist)
            
            f_landmark = (np.min(face_land_mark[:, 0]),int(upper_bond),np.max(face_land_mark[:, 0]),np.max(face_land_mark[:,1]))
            x1, y1, x2, y2 = f_landmark
            
            if y2-y1<=0 or x2-x1<=0 or x1<0: # if the landmark bbox is not suitable, reuse the bbox
                coords_list += [f]
                w,h = f[2]-f[0], f[3]-f[1]
                print("error bbox:",f)
            else:
                coords_list += [f_landmark]
    
    print("********************************************bbox_shift parameter adjustment**********************************************************")
    print(f"Total frame:「{len(frames)}」 Manually adjust range : [ -{int(sum(average_range_minus) / len(average_range_minus))}~{int(sum(average_range_plus) / len(average_range_plus))} ] , the current value: {upperbondrange}")
    print("*************************************************************************************************************************************")
    if _BBOX_SMOOTH > 1:
        coords_list = _smooth_coords(coords_list, _BBOX_SMOOTH)
        print(f"Applied bbox temporal smoothing (window={_BBOX_SMOOTH}) to reduce jitter")
    return coords_list,frames
    

if __name__ == "__main__":
    img_list = ["./results/lyria/00000.png","./results/lyria/00001.png","./results/lyria/00002.png","./results/lyria/00003.png"]
    crop_coord_path = "./coord_face.pkl"
    coords_list,full_frames = get_landmark_and_bbox(img_list)
    with open(crop_coord_path, 'wb') as f:
        pickle.dump(coords_list, f)
        
    for bbox, frame in zip(coords_list,full_frames):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        crop_frame = frame[y1:y2, x1:x2]
        print('Cropped shape', crop_frame.shape)
        
        #cv2.imwrite(path.join(save_dir, '{}.png'.format(i)),full_frames[i][0][y1:y2, x1:x2])
    print(coords_list)
