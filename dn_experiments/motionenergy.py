import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

import cv2

from scipy.ndimage import gaussian_filter1d, gaussian_filter, median_filter

import params

def get_frames_mean(video_file, load_frames=True, me_cam_mean=params.me_cam_mean):
    frames = []
    images_dir = os.path.dirname(video_file)
    mean_frame_file = os.path.join(images_dir, me_cam_mean)
    if load_frames:
        f = cv2.VideoCapture(video_file)
        rval, frame = f.read()
        # Convert rgb to grey scale
        mean_frame = np.zeros_like(frame[:, :, 0], dtype=np.int64)
        count = 0
        while rval:
            mean_frame =  mean_frame + frame[:, :, 0]
            frames.append(frame[:, :, 0])
            rval, frame = f.read()
            count += 1
        f.release()
        mean_frame = mean_frame / count
        mean_frame = mean_frame.astype(np.uint8)
        cv2.imwrite(mean_frame_file, mean_frame)
    else:
        frames = None
        mean_frame = cv2.imread(mean_frame_file)[:,:,0]
    return frames, mean_frame

def get_mask(mean_frame):
    ret2,th2 = cv2.threshold(mean_frame,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask = th2.astype(bool)
    return mask

def detect_front(mean_frame, bottom_of_shin=120, top_of_ball=240, default=640):
    # detect the front of the head
    ret2,th2 = cv2.threshold(mean_frame,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thres_image = th2.astype(bool)
    thres_imgage_detect = thres_image[bottom_of_shin:top_of_ball]  # cut of stage on top and ball on bottom
    fly_line = np.mean(thres_imgage_detect, axis=0) > 0
    x_front = np.argwhere(np.diff(fly_line.astype(int)) == -1).flatten()  # detect the most frontal part of fly
    if not len(x_front):
        return default
    return int(x_front[-1])

def get_front_mask(mean_frame, bottom_of_shin=120, top_of_ball=320, ball_mask=None):
    front = detect_front(mean_frame, top_of_ball=top_of_ball)
    ret2,th2 = cv2.threshold(mean_frame,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thres_image = th2.astype(bool)
    thres_image_front = np.logical_not(thres_image)
    thres_image_front[:,:front] = False
    if ball_mask is not None:
        thres_image_front[ball_mask] = False
    else:
        thres_image_front[top_of_ball:,:] = False
    return thres_image_front

def detect_neck(mean_frame, top_lim=150, right_lim=550, left_lim=400):
    ret2,th2 = cv2.threshold(mean_frame,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thres_image = th2.astype(bool)
    thres_image_detect = thres_image[top_lim:, left_lim:right_lim]
    fly_line = np.mean(thres_image_detect, axis=1) > 0
    y_neck = np.argwhere(np.diff(fly_line.astype(int)) == 1).flatten()
    x_neck = np.argwhere(thres_image_detect[y_neck+1,:]).flatten() if len(y_neck) else [0]
    return int(x_neck[0]+left_lim)

def get_back_mask(mean_frame, top_of_ball=320, ball_mask=None, neck=480):
    if neck is None:
        neck = detect_neck(mean_frame)
    ret2,th2 = cv2.threshold(mean_frame,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thres_image = th2.astype(bool)
    thres_image_back = np.logical_not(thres_image)
    thres_image_back[:,neck:] = False
    if ball_mask is not None:
        thres_image_back[ball_mask] = False
    else:
        thres_image_back[top_of_ball:,:] = False
    return thres_image_back

def get_mid_mask(mean_frame, top_of_ball=320, ball_mask=None, neck=480):
    if neck is None:
        neck = detect_neck(mean_frame)
    front = detect_front(mean_frame, top_of_ball=top_of_ball)
    ret2,th2 = cv2.threshold(mean_frame,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thres_image = th2.astype(bool)
    thres_image_mid = np.logical_not(thres_image)
    thres_image_mid[:,:neck] = False
    thres_image_mid[:,front:] = False
    if ball_mask is not None:
        thres_image_mid[ball_mask] = False
    else:
        thres_image_mid[top_of_ball:,:] = False
    return thres_image_mid

def get_ball_mask(mean_frame, y_min=240, return_top=False, r_adjust=10):
    img = cv2.medianBlur(mean_frame, 5)[y_min:,:]
    black = np.zeros_like(img)
    extended_img = np.concatenate((img,black,black,black,black,black),axis=0)
    circles = cv2.HoughCircles(extended_img, cv2.HOUGH_GRADIENT, 2, minDist=200, param1=20, param2=20, minRadius=500, maxRadius=1200)
    circles = np.round(circles[0, :]).astype(int)
    x_orig, y, r = circles[0]
    y_orig = y + y_min
    r_orig = r + r_adjust
    circle_mask = np.zeros_like(mean_frame, dtype=bool)
    for i_y in range(mean_frame.shape[0]):
        for i_x in range(mean_frame.shape[1]):
            if np.sqrt((y_orig-i_y)**2 + (x_orig-i_x)**2) <= r_orig:
                circle_mask[i_y, i_x] = True
    if return_top:
        return circle_mask, y_orig - r_orig
    return circle_mask

def compute_me(frames, mean_frame, mask_dir=None, mask_name="me_mask.png"):
    mask = get_mask(mean_frame)
    
    n_frames = len(frames) if frames is not None else 0
    motion_energy_off = np.zeros((n_frames))
    motion_energy_front = np.zeros((n_frames))
    motion_energy_back = np.zeros((n_frames))
    motion_energy_mid = np.zeros((n_frames))
    

    not_thres_image = np.logical_not(mask)
    ball_mask, top_of_ball = get_ball_mask(mean_frame, return_top=True)

    thres_image_front = get_front_mask(mean_frame, top_of_ball=top_of_ball, ball_mask=ball_mask)
    thres_image_back = get_back_mask(mean_frame, top_of_ball=top_of_ball, ball_mask=ball_mask)
    thres_image_mid = get_mid_mask(mean_frame, top_of_ball=top_of_ball, ball_mask=ball_mask)

    if mask_dir is not None:
        overall_mask = mask.astype(int) + 2 * ball_mask.astype(int) + 3 * thres_image_front.astype(int) + \
                       4 * thres_image_mid.astype(int) + 5 * thres_image_back.astype(int)
        fig, ax = plt.subplots(1,1,figsize=(6,4))
        ax.imshow(overall_mask)
        if isinstance(mask_dir, list):
            _ = [fig.savefig(os.path.join(this_mask_dir, this_mask_name)) for this_mask_dir, this_mask_name in zip(mask_dir, mask_name)]
        else:
            fig.savefig(os.path.join(mask_dir, mask_name))
        plt.close(fig)
    
    if frames is not None:
        for i_frame in tqdm(range(n_frames-1)):
            motion_energy_off[i_frame] = np.sum(frames[i_frame+1][not_thres_image]**2 - frames[i_frame][not_thres_image]**2)
            motion_energy_front[i_frame] = np.sum(frames[i_frame+1][thres_image_front]**2 - frames[i_frame][thres_image_front]**2)
            motion_energy_back[i_frame] = np.sum(frames[i_frame+1][thres_image_back]**2 - frames[i_frame][thres_image_back]**2)
            motion_energy_mid[i_frame] = np.sum(frames[i_frame+1][thres_image_mid]**2 - frames[i_frame][thres_image_mid]**2)
        
        meo_filt = gaussian_filter1d(median_filter(motion_energy_off, size=5), sigma=10)
        # meo_filt = utils.normalise_quantile(meo_filt)
        mef_filt = gaussian_filter1d(median_filter(motion_energy_front, size=5), sigma=10)
        # mef_filt = utils.normalise_quantile(mef_filt)
        meb_filt = gaussian_filter1d(median_filter(motion_energy_back, size=5), sigma=10)
        # meb_filt = utils.normalise_quantile(meb_filt)
        mem_filt = gaussian_filter1d(median_filter(motion_energy_mid, size=5), sigma=10)
        # meb_filt = utils.normalise_quantile(meb_filt)
    else:
        meo_filt, mef_filt, meb_filt, mem_filt = None, None, None

    return meo_filt, mef_filt, meb_filt, mem_filt

def add_me_to_df(meo_filt, mef_filt, meb_filt, mem_filt, index_df, df_out_dir=None):
    if isinstance(index_df, str) and os.path.isfile(index_df):
        index_df = pd.read_pickle(index_df)
    if index_df is not None:
        assert isinstance (index_df, pd.DataFrame)

    assert len(index_df)  == len(meo_filt)

    index_df["me_front"] = mef_filt
    index_df["me_back"] = meb_filt
    index_df["me_all"] = meo_filt
    index_df["me_mid"] = mem_filt

    if df_out_dir is not None:
        index_df.to_pickle(df_out_dir)
    return index_df

def compute_and_add_me_to_df(trial_dir, beh_df, camera_name=params.me_cam):
    video_file = os.path.join(trial_dir,"behData", "images", "camera_5.mp4")
    if not os.path.isfile(beh_df):
        print("No beh_df.pkl exists. Continuing.")
        return None
    print("get frames for me computation")
    frames, mean_frame = get_frames_mean(video_file)

    mask_dir = [os.path.join(trial_dir, "processed"), os.path.join(params.plot_base_dir, "me_masks")]
    trial_name = "_".join(trial_dir.split("/")[4:])
    mask_name = ["me_mask.png", f"me_mask_{trial_name}.png"]  # {row.fly_id}_{row.date}_{row.fly_number}_{row.trial_number}
    print("compute motion energy")
    meo_filt, mef_filt, meb_filt, mem_filt = compute_me(frames, mean_frame, mask_dir, mask_name)
    
    if frames is not None:
        del frames
        beh_df = add_me_to_df(meo_filt, mef_filt, meb_filt, mem_filt, index_df=beh_df, df_out_dir=beh_df)
    return beh_df

if __name__ == "__main__":
    data_summary_dir = "/mnt/nas2/JB/_data_summary"
    csv_dir = os.path.join(data_summary_dir, "fly_selection_manual_230224.csv")
    df_all = pd.read_csv(csv_dir,)
    df_include = df_all[np.logical_not(df_all.exclude == True)]
    df_xz = df_include[df_include.image_type == "xz"]
    df = df_xz[np.logical_and(df_xz.CO2 == False, [not "co2_puff" in trial_name for trial_name in df_xz.trial_name])]
    print(f"Working on {len(df)} trials")

    for i, (index, row) in enumerate(df.iterrows()):
        # if not "221117" in row.trial_dir:
        #     continue
        trial_dir = row.trial_dir
        print(f"{i}/{len(df)} {trial_dir}")
        video_file = os.path.join(trial_dir,"behData", "images", "camera_5.mp4")
        beh_df = os.path.join(trial_dir, "processed", "beh_df.pkl")
        if not os.path.isfile(beh_df):
            print("No beh_df.pkl exists. Continuing.")
            continue
        print("get frames")
        frames, mean_frame = get_frames_mean(video_file)  # , load_frames=False)

        mask_dir = [os.path.join(trial_dir, "processed"), os.path.join(data_summary_dir, "plots", "me_masks")]
        mask_name = ["me_mask.png", f"me_mask_{row.fly_id}_{row.date}_{row.fly_number}_{row.trial_number}.png"]
        print("compute motion energy")
        meo_filt, mef_filt, meb_filt, mem_filt = compute_me(frames, mean_frame, mask_dir, mask_name)
        
        if frames is not None:
            del frames
            add_me_to_df(meo_filt, mef_filt, meb_filt, mem_filt, index_df=beh_df, df_out_dir=beh_df)

    
