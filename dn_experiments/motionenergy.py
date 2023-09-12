"""
Module to compute pixel motion energy from side view video of the fly (camera 5).
Not used anymore. Use motion energy of SLEAP tracked keypoints instead.
Author: jonas.braun@epfl.ch
"""
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
    """
    Calculate the mean frame from a video file and optionally load individual frames.

    Args:
        video_file (str): Path to the video file.
        load_frames (bool): Whether to load individual frames.
        me_cam_mean (str): Path to save the mean frame image.

    Returns:
        frames (list): List of loaded frames.
        mean_frame (numpy.ndarray): Mean frame of the video.
    """
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
    """
    Generate a mask of the fly based on OTSU thresholding the mean frame.

    Args:
        mean_frame (numpy.ndarray): Mean frame image.

    Returns:
        mask (numpy.ndarray): Binary mask.
    """
    ret2,th2 = cv2.threshold(mean_frame,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask = th2.astype(bool)
    return mask

def detect_front(mean_frame, bottom_of_shin=120, top_of_ball=240, default=640):
    """
    Detect the front of the fly's head (x axis) in the mean frame.

    Args:
        mean_frame (numpy.ndarray): Mean frame image.
        bottom_of_shin (int): Top boundary for detection = bottom of shin.
        top_of_ball (int): Bottom boundary for detection = top of ball.
        default (int): Default position if front is not detected.

    Returns:
        front (int): Detected front position.
    """
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
    """
    Generate a mask for the region around the front of the fly. Can be used to calculate a proxy of frontleg motion energy.

    Args:
        mean_frame (numpy.ndarray): Mean frame image.
        bottom_of_shin (int): Top boundary for front detection.
        top_of_ball (int): Bottom boundary for front detection.
        ball_mask (numpy.ndarray): Binary mask for the ball.

    Returns:
        thres_image_front (numpy.ndarray): Front mask.
    """
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
    """
    Detect the neck position in the mean frame.

    Args:
        mean_frame (numpy.ndarray): Mean frame image.
        top_lim (int): Top boundary for neck detection.
        right_lim (int): Right boundary for neck detection.
        left_lim (int): Left boundary for neck detection.

    Returns:
        neck (int): Detected neck position.
    """
    ret2,th2 = cv2.threshold(mean_frame,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thres_image = th2.astype(bool)
    thres_image_detect = thres_image[top_lim:, left_lim:right_lim]
    fly_line = np.mean(thres_image_detect, axis=1) > 0
    y_neck = np.argwhere(np.diff(fly_line.astype(int)) == 1).flatten()
    x_neck = np.argwhere(thres_image_detect[y_neck+1,:]).flatten() if len(y_neck) else [0]
    return int(x_neck[0]+left_lim)

def get_back_mask(mean_frame, top_of_ball=320, ball_mask=None, neck=480):
    """
    Generate a mask for the back region of the fly. Can be used to compute a proxy of the hindleg motion energy.

    Args:
        mean_frame (numpy.ndarray): Mean frame image.
        top_of_ball (int): Top boundary for back detection.
        ball_mask (numpy.ndarray): Binary mask for the ball.
        neck (int): Neck position. Otherwise detect neck position.

    Returns:
        thres_image_back (numpy.ndarray): Back mask.
    """
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
    """
    Generate a mask for the mid region of the fly. Can be used to compute a proxy of motion energy in the central region of the frame.

    Args:
        mean_frame (numpy.ndarray): Mean frame image.
        top_of_ball (int): Bottom boundary for region detection to exclude the ball.
        ball_mask (numpy.ndarray): Binary mask for the ball.
        neck (int): Detected neck position.

    Returns:
        thres_image_mid (numpy.ndarray): Mid mask.
    """
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
    """
    Generate a mask for the ball. Used to exclude from other masks

    Args:
        mean_frame (numpy.ndarray): Mean frame image.
        y_min (int): Minimum Y coordinate for ball detection.
        return_top (bool): Whether to return the top position.
        r_adjust (int): Radius adjustment. Increases the ball radius to incldue proximal pixels.

    Returns:
        circle_mask (numpy.ndarray): Ball mask.
        top_of_ball (int): Top position of the ball.
    """
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
    """
    Compute motion energy from frames using multiple masks.

    Args:
        frames (list): List of frames.
        mean_frame (numpy.ndarray): Mean frame image.
        mask_dir (str or list): Directory for saving masks.
        mask_name (str or list): Mask file name.

    Returns:
        meo_filt (numpy.ndarray): Motion energy for the whole fly.
        mef_filt (numpy.ndarray): Motion energy for the front.
        meb_filt (numpy.ndarray): Motion energy for the back.
        mem_filt (numpy.ndarray): Motion energy for the mid.
    """
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
    """
    Add motion energy data to a DataFrame.

    Args:
        meo_filt (numpy.ndarray): Motion energy for the whole fly.
        mef_filt (numpy.ndarray): Motion energy for the front.
        meb_filt (numpy.ndarray): Motion energy for the back.
        mem_filt (numpy.ndarray): Motion energy for the mid.
        index_df (pandas.DataFrame or str): DataFrame or path to DataFrame file.
        df_out_dir (str): Output directory for the DataFrame.

    Returns:
        index_df (pandas.DataFrame): Updated DataFrame.
    """
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
    """
    Compute motion energy and add it to a behavior DataFrame.
    High level interface to the external.

    Args:
        trial_dir (str): Path to the trial directory.
        beh_df (pandas.DataFrame): Behavior DataFrame.
        camera_name (str): Name of the camera.

    Returns:
        beh_df (pandas.DataFrame): Updated behavior DataFrame.
    """
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

    
