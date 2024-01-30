"""
This file contains functions used to copy and compress data in order to prepare it for the upload to Harvard Dataverse.
It also contains functions to decompress and re-arrange the data to a consistent folder structure after downloading it from Harvard Dataverse.
Author: jonas.braun@epfl.ch
"""
import os
import shutil
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import tarfile
import pickle
import matplotlib.pyplot as plt
from scipy.signal import medfilt

from twoppp.utils import makedirs_safe, save_stack
from twoppp import rois
from twoppp.pipeline import PreProcessParams
import utils2p
import summarydf, params, loaddata, summarydf


BASE_DIR = "/mnt/nas2/JB/_paper_data"
IMAGING_DIR = os.path.join(BASE_DIR, "imaging")
IMAGING_ZIP_DIR = os.path.join(BASE_DIR, "imaging_zip_novideo")
OPTOGENETICS_DIR = os.path.join(BASE_DIR, "optogenetics")
HEADLESS_DIR = os.path.join(BASE_DIR, "headless")
HEADLESS_ZIP_DIR = os.path.join(BASE_DIR, "headless_zip_novideo")
OTHER_DIR = os.path.join(BASE_DIR, "other")
OTHER_ZIP_DIR = os.path.join(BASE_DIR, "other_zip_novideo")


def make_roi_center_annotation_file(fly_dir, quantile=0.99):
    """
    Create a PDF file with annotated ROI centers on a standard deviation image.

    Parameters:
        fly_dir (str): The directory containing the fly data.
        quantile (float): The quantile for colormap scaling. Defaults to 0.99.
    """
    process_params = PreProcessParams()
    pca_map_file = os.path.join(fly_dir, "processed", process_params.pca_maps)
    roi_centers = rois.read_roi_center_file(os.path.join(fly_dir, "processed", "ROI_centers.txt"))
    with open(pca_map_file, "rb") as f:
        pca_data = pickle.load(f)
    fig, ax = plt.subplots(1,1, figsize=(9.5,5))
    im = medfilt(np.log10(pca_data["green_std"]), kernel_size=5)  # std_img
    title = "std across trials" + "\n" + fly_dir
    this_cmap = plt.cm.get_cmap("viridis")
    this_clim = [np.quantile(im, 1-quantile), np.quantile(im, quantile)]
    ax.imshow(im, clim=this_clim, cmap=this_cmap)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    _ = [ax.plot(pixel[1], pixel[0], 'r+') for pixel in roi_centers]

    for i in range(len(roi_centers)):
        ax.annotate(str(i), np.flip(roi_centers[i]), size=8)
        
    fig.tight_layout()
    fig.savefig(os.path.join(fly_dir, "processed", "roi_center_annotation.pdf"), dpi=300)


def make_background_image(fly_dir):
    """
    Create and save a background image from fly data.
    This defaults to the standard deviation projection image, but can be changed in params.py

    Parameters:
        fly_dir (str): The directory containing the fly data.
    """
    background_image = loaddata.get_background_image(fly_dir)
    save_stack(os.path.join(fly_dir, "processed", "background_image.tif"), background_image)


def convert_trial_dir(trial_dir, imaging=True):
    """
    Convert the trial directory path to a new directory path in the temporary data dir.

    Parameters:
        trial_dir (str): The directory path of the trial to be converted.
        imaging (bool, optional): Whether the trial is an imaging trial. Defaults to True.

    Returns:
        str: The new directory path after conversion.
    """
    fly_dir, trial_name = os.path.split(trial_dir)
    date_genotype_dir, fly_name = os.path.split(fly_dir)
    _, date_genotype_name = os.path.split(date_genotype_dir)
    if imaging:
        new_fly_dir = os.path.join(IMAGING_DIR, date_genotype_name + "_" + fly_name)
    else:
        new_fly_dir = os.path.join(HEADLESS_DIR, date_genotype_name + "_" + fly_name)
    return os.path.join(new_fly_dir, trial_name)


def copy_all_imaging_trials(overwrite=False):
    """
    Copy all imaging trials to an external directory from where they can be uploaded to Harvard dataverse.

    Parameters:
        overwrite (bool): Whether to overwrite existing files. Defaults to False.
    """
    df = summarydf.load_data_summary()
    df = summarydf.filter_data_summary(df, no_co2=False)
    df = summarydf.get_selected_df(df, [{"GCaMP": "Dfd"}])
    ball_df = summarydf.get_selected_df(df, [{"walkon": "ball"}])
    wheel_df = summarydf.get_selected_df(df, [{"walkon": "wheel"}])
    
    fly_ids_DNp09 = [15,18,44,46,61]
    fly_ids_aDN2 = [25,32,60,63,72,74]
    fly_ids_MDN = [3,5,7,20,28,34,42,50,51,62,71]
    fly_ids_PR = [8,10,65,11]  # the last one is only used in stimulation power control.
    fly_ids_ball = fly_ids_DNp09 + fly_ids_aDN2 + fly_ids_MDN + fly_ids_PR
    fly_id_natbeh_wheel = 72
    fly_ids_VNC_cut = [89,90,91] + [95,96,97]  # DNp09 + PR flies

    selected_trials = np.zeros((len(df)), dtype=bool)
    for fly_id in fly_ids_ball:
        new_trials = np.logical_and(df.walkon.values == "ball", df.fly_id.values == fly_id)
        selected_trials = np.logical_or(selected_trials, new_trials)

    new_trials = np.logical_and(df.walkon.values == "wheel", df.fly_id.values == fly_id_natbeh_wheel)
    selected_trials = np.logical_or(selected_trials, new_trials)

    for fly_id in fly_ids_VNC_cut:
        new_trials = np.logical_and(df.walkon.values == "no", df.fly_id.values == fly_id)
        selected_trials = np.logical_or(selected_trials, new_trials)

    df = df[selected_trials]

    nan  = df["exclude"].iloc[0]
    df["neural_quality"] = nan
    df["behaviour_quality"] = nan
    df["stim_response_quality"] = nan
    df["exclude"] = nan
    df["comment"] = nan

    old_df = df.copy()

    for index, row in df.iterrows():
        old_trial_dir = row.trial_dir
        new_trial_dir = convert_trial_dir(old_trial_dir, imaging=True)
        new_fly_dir = os.path.dirname(new_trial_dir)
        df["trial_dir"].iloc[df.index == index] = new_trial_dir
        df["fly_dir"].iloc[df.index == index] = new_fly_dir

    df.to_csv(os.path.join(IMAGING_DIR, "imaging_summary_df.csv"), index=False)
    df = old_df

    for i_fly, (fly_id, fly_df) in enumerate(df.groupby("fly_id")):
        trial_dirs = fly_df.trial_dir.values
        copy_one_fly(trial_dirs, imaging=True, overwrite=overwrite)


def copy_all_headless_trials(overwrite=False):
    """
    Copy all headless trials to an external directory from where they can be uploaded to Harvard dataverse.

    Parameters:
        overwrite (bool): Whether to overwrite existing files. Defaults to False.
    """
    df = summarydf.get_headless_df()

    for index, row in df.iterrows():
        old_trial_dir = row.trial_dir
        new_trial_dir = convert_trial_dir(old_trial_dir, imaging=False)
        new_fly_dir = os.path.dirname(new_trial_dir)
        df["trial_dir"].iloc[df.index == index] = new_trial_dir
        df["fly_dir"].iloc[df.index == index] = new_fly_dir

    nan  = df["exclude"].iloc[0]
    df["comment"] = nan
    df.to_csv(os.path.join(HEADLESS_DIR, "headless_summary_df.csv"), index=False)
    df = summarydf.get_headless_df()  # make sure to use old directory structure to copy

    for i_fly, (fly_id, fly_df) in enumerate(df.groupby("fly_id")):
        trial_dirs = list(fly_df.trial_dir.values)
        if any(fly_df.plot_appendix == "imaging"):  # copy all files needed for imaging, but put it in the headless data nonetheless
            copy_one_fly(trial_dirs, imaging=True, overwrite=overwrite, base_dir=HEADLESS_DIR)
        else:
            for trial_dir in trial_dirs:
                if "noball" in trial_dir and not "nohead" in trial_dir:
                    trial_dirs.remove(trial_dir)
            copy_one_fly(trial_dirs, imaging=False, overwrite=overwrite)


def copy_all_predictions_trials(genotypes=["DNa01", "DNa02", "DNb02", "aDN1", "DNg14", "mute"], overwrite=False):
    """
    Copy all prediction trials for specified genotypes to an external directory from where they can be uploaded to Harvard dataverse.

    Parameters:
        genotypes (list): List of genotypes to copy. Defaults to ["DNa01", "DNa02", "DNb02", "aDN1", "DNg14", "mute"].
        overwrite (bool): Whether to overwrite existing files. Defaults to False.
    """
    df = summarydf.get_predictions_df()
    df = summarydf.get_selected_df(df, select_dicts=[{"CsChrimson": genotype} for genotype in genotypes])
    df_old = df.copy()

    for index, row in df.iterrows():
        old_trial_dir = row.trial_dir
        new_trial_dir = convert_trial_dir(old_trial_dir, imaging=False)
        new_fly_dir = os.path.dirname(new_trial_dir)
        df["trial_dir"].iloc[df.index == index] = new_trial_dir
        df["fly_dir"].iloc[df.index == index] = new_fly_dir

    nan  = df["exclude"].iloc[0]
    df["comment"] = nan
    df.to_csv(os.path.join(HEADLESS_DIR, "predictions_summary_df.csv"), index=False)
    df = df_old

    for i_fly, (fly_id, fly_df) in enumerate(df.groupby("fly_id")):
        trial_dirs = list(fly_df.trial_dir.values)
        for trial_dir in trial_dirs:
            if "noball" in trial_dir and not "nohead" in trial_dir:
                trial_dirs.remove(trial_dir)
        copy_one_fly(trial_dirs, imaging=False, overwrite=overwrite)


def copy_stim_control_trials(overwrite=False):
    """
    Copy stimulation control trials to an external directory from where they can be uploaded to Harvard dataverse.

    Parameters:
        overwrite (bool): Whether to overwrite existing files. Defaults to False.
    """
    fly_dirs = [
        # "/mnt/nas2/JB/220816_MDN3xCsChrimson/Fly5",
        # "/mnt/nas2/JB/230125_BPNxCsChrimson/Fly1",
        "/mnt/nas2/JB/230914_MDN3xCsChrimson/Fly1",
        "/mnt/nas2/JB/230914_MDN3xCsChrimson/Fly2",
        "/mnt/nas2/JB/230914_BPNxCsChrimson/Fly3",
        "/mnt/nas2/JB/230914_BPNxCsChrimson/Fly4",
        "/mnt/nas2/JB/230915_MDN3xCsChrimson/Fly11",
        "/mnt/nas2/JB/230921_BPNxCsChrimson/Fly5",
        "/mnt/nas2/JB/230921_BPNxCsChrimson/Fly6",
        "/mnt/nas2/JB/231115_DfdxGtACR1/Fly1",
        "/mnt/nas2/JB/231115_DfdxGtACR1/Fly2",
        "/mnt/nas2/JB/231115_DfdxGtACR1/Fly3"
    ]
    for i_fly, fly_dir in enumerate(fly_dirs):
        trial_dirs = [os.path.join(fly_dir, folder) for folder in os.listdir(fly_dir) \
            if folder != "processed" and os.path.isdir(os.path.join(fly_dir, folder))]
        for trial_dir in trial_dirs:
            if "led" in trial_dir:
                trial_dirs.remove(trial_dir)
        copy_one_fly(trial_dirs, base_dir=OTHER_DIR, overwrite=overwrite, imaging=False)

def copy_leg_cutting_trials(overwrite : bool =False, genotypes = ['MDN', 'DNp09', 'CantonS']):
    """
    Copy leg cutting trials to an external directory from where they can be uploaded to Harvard dataverse.
    """
    df = summarydf.get_predictions_df()
    df = summarydf.get_selected_df(df, select_dicts=[{"CsChrimson": genotype} for genotype in genotypes])
    # intact flies
    df_intact = summarydf.get_selected_df(df, select_dicts=[{"leg_amp": 'FALSE', "head": True}])
    # need to filter for flies that have had specific amputations, but keep the controls
    df_target = summarydf.get_selected_df(
        df, select_dicts=[{"joint_amp": 'TiTa'}]
    )
    df_target = summarydf.get_selected_df(
        df_target, select_dicts=[{"leg_amp": leg_} for leg_ in ['FL', 'ML', 'HL']]
    )
    df = df[df.trial_dir.isin(df_target.trial_dir.unique()) | df.trial_dir.isin(df_intact.trial_dir.unique())]

    # === Rewriting df
    df_old = df.copy()
    for index, row in df.iterrows():
        old_trial_dir = row.trial_dir
        new_trial_dir = convert_trial_dir(old_trial_dir, imaging=False)
        new_fly_dir = os.path.dirname(new_trial_dir)
        df["trial_dir"].iloc[df.index == index] = new_trial_dir
        df["fly_dir"].iloc[df.index == index] = new_fly_dir

    nan  = df["exclude"].iloc[0]
    df["comment"] = nan
    df.to_csv(os.path.join(HEADLESS_DIR, "predictions_summary_df.csv"), index=False)
    df = df_old

    for i_fly, (fly_id, fly_df) in enumerate(df.groupby("fly_id")):
        trial_dirs = list(fly_df.trial_dir.values)
        copy_one_fly(trial_dirs, base_dir=OTHER_DIR, overwrite=overwrite, imaging=False)


def compress_leg_cutting(overwrite=False, keep_videos=False):
    """
    Compress leg cutting data.

    Parameters:
        overwrite (bool): Whether to overwrite existing files. Defaults to False.
        keep_videos (bool): Whether to keep compressed video files. Defaults to False.
    """
    # check if the folder exists
    if not os.path.isdir(OTHER_ZIP_DIR):
        os.makedirs(OTHER_ZIP_DIR)
    old_wd = os.getcwd()
    os.chdir(OTHER_DIR)
    fly_names = os.listdir(".")
    fly_names = [fly_name for fly_name in fly_names if os.path.isdir(os.path.join(OTHER_DIR,fly_name))]

    for i_fly, fly_name in enumerate(fly_names):
        if keep_videos:
            source_folder = [fly_name]
            tar_file = os.path.join(OTHER_ZIP_DIR, fly_name+".tar.gz") 
            tar_file_origin = os.path.join(OTHER_DIR, fly_name+".tar.gz") 
        else:
            source_folder = []
            for path, subdirs, files in os.walk(os.path.join(OTHER_DIR, fly_name)):
                for name in files:
                    if not name.startswith("camera_") and not name.startswith("."):
                        source_folder.append(os.path.relpath(os.path.join(path, name),start=OTHER_DIR))
            tar_file = os.path.join(OTHER_ZIP_DIR, fly_name+"_novideo.tar.gz")
            tar_file_origin = os.path.join(OTHER_DIR, fly_name+"_novideo.tar.gz")
        if not (os.path.isfile(tar_file) or os.path.isfile(tar_file_origin)) or overwrite:
            compress(tar_file, source_folder)
    
    os.chdir(old_wd)        


def copy_one_fly(trial_dirs, imaging=True, overwrite=False, base_dir=None):
    """
    Copy data for one fly to an external directory

    Parameters:
        trial_dirs (list): List of trial directories to copy.
        imaging (bool): Whether there is imaging data to copy. Defaults to True.
        overwrite (bool): Whether to overwrite existing files. Defaults to False.
        base_dir (str): The base directory for copying. If None, use the appropriate data directory based on imaging. Defaults to None.
    """
    fly_dir = os.path.dirname(trial_dirs[0])
    date_genotype_dir, fly_name = os.path.split(fly_dir)
    _, date_genotype_name = os.path.split(date_genotype_dir)
    if base_dir is not None:
        new_fly_dir = os.path.join(base_dir, date_genotype_name + "_" + fly_name)
    elif imaging:
        new_fly_dir = os.path.join(IMAGING_DIR, date_genotype_name + "_" + fly_name)
    else:
        new_fly_dir = os.path.join(HEADLESS_DIR, date_genotype_name + "_" + fly_name)
    makedirs_safe(new_fly_dir)
    if imaging:
        makedirs_safe(os.path.join(new_fly_dir, "processed"))

        if not os.path.isfile(os.path.join(fly_dir, "processed", "roi_center_annotation.pdf")):
            try:
                make_roi_center_annotation_file(fly_dir)
            except FileNotFoundError:
                print(f"Warning: Could not make ROI center annotation file for fly {fly_dir}")
        if os.path.isfile(os.path.join(fly_dir, "processed", "roi_center_annotation.pdf")):
            copy_file(os.path.join(fly_dir, "processed", "roi_center_annotation.pdf"), fly_dir, new_fly_dir, overwrite=overwrite)
        else:
            print(f"Warning: Could not copy ROI center annotation file for fly {fly_dir}")

        if not os.path.isfile(os.path.join(fly_dir, "processed", "background_image.tif")):
            try:
                make_background_image(fly_dir)
            except FileNotFoundError:
                print(f"Warning: Could not make background file for fly {fly_dir}")
        if os.path.isfile(os.path.join(fly_dir, "processed", "background_image.tif")):
            copy_file(os.path.join(fly_dir, "processed", "background_image.tif"), fly_dir, new_fly_dir, overwrite=overwrite)
        else:
            print(f"Warning: Could not copy background file for fly {fly_dir}")
        if os.path.isfile(os.path.join(fly_dir, "processed", "ROI_centers.txt")):
            copy_file(os.path.join(fly_dir, "processed", "ROI_centers.txt"), fly_dir, new_fly_dir, overwrite=overwrite)
            copy_file(os.path.join(fly_dir, "processed", "ROI_mask.tif"), fly_dir, new_fly_dir, overwrite=overwrite)
        else:
            print(f"Warning: Could not copy ROI_centers and ROI_mask file for fly {fly_dir}")

    for trial_dir in trial_dirs:
        copy_one_trial(new_fly_dir, trial_dir, imaging=imaging, overwrite=overwrite)


def copy_one_trial(new_fly_dir, trial_dir, imaging=True, overwrite=False):
    """
    Copy data for one trial to an external directory.

    Parameters:
        new_fly_dir (str): The destination directory for the fly.
        trial_dir (str): The trial directory to copy.
        imaging (bool): Whether there is imaging data to copy. Defaults to True.
        overwrite (bool): Whether to overwrite existing files. Defaults to False.
    """
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), trial_dir)
    old_fly_dir, trial_name = os.path.split(trial_dir)
    new_trial_dir = os.path.join(new_fly_dir, trial_name)
    makedirs_safe(new_trial_dir)
    makedirs_safe(os.path.join(new_trial_dir, "2p"))
    makedirs_safe(os.path.join(new_trial_dir, "behData", "images"))
    makedirs_safe(os.path.join(new_trial_dir, "processed"))

    copy_file(os.path.join(trial_dir, "processed", "beh_df.pkl"), trial_dir, new_trial_dir, overwrite=overwrite)
    copy_file(os.path.join(trial_dir, "behData", "images", "camera_3.mp4"), trial_dir, new_trial_dir, overwrite=overwrite)
    copy_file(os.path.join(trial_dir, "behData", "images", "camera_5.mp4"), trial_dir, new_trial_dir, overwrite=overwrite)
    if "wheel" in trial_dir:
        copy_file(os.path.join(trial_dir, "behData", "images", "camera_1.mp4"), trial_dir, new_trial_dir, overwrite=overwrite)
    
    copy_file(utils2p.find_sync_file(trial_dir), trial_dir, new_trial_dir, overwrite=overwrite)
    copy_file(utils2p.find_sync_metadata_file(trial_dir), trial_dir, new_trial_dir, overwrite=overwrite)
    copy_file(utils2p.find_seven_camera_metadata_file(trial_dir), trial_dir, new_trial_dir, overwrite=overwrite)

    if imaging and os.path.isfile(os.path.join(trial_dir, "processed", "green_com_warped.tif")):
        copy_file(os.path.join(trial_dir, "processed", "twop_df.pkl"), trial_dir, new_trial_dir, overwrite=overwrite)
        copy_file(os.path.join(trial_dir, "processed", "green_com_warped.tif"), trial_dir, new_trial_dir, overwrite=overwrite)
        copy_file(utils2p.find_metadata_file(trial_dir), trial_dir, new_trial_dir, overwrite=overwrite)
    elif imaging:
        print(f"Warning: could not copy imaging data for trial {trial_dir}")
        

def copy_file(file_path, old_trial_dir, new_trial_dir, overwrite=False):
    """
    Copy a file from one directory to another.

    Parameters:
        file_path (str): The source file path.
        old_trial_dir (str): The source trial directory.
        new_trial_dir (str): The destination trial directory.
        overwrite (bool): Whether to overwrite existing files. Defaults to False.
    """
    new_file_path = file_path.replace(old_trial_dir, new_trial_dir)
    new_dir = os.path.dirname(new_file_path)
    makedirs_safe(new_dir)
    if not os.path.isfile(new_file_path) or overwrite:
        shutil.copy(file_path, new_file_path)


def compress_headless_predictions(overwrite=False, keep_videos=False):
    """
    Compress headless and prediction data.

    Parameters:
        overwrite (bool): Whether to overwrite existing files. Defaults to False.
        keep_videos (bool): Whether to keep compressed video files. Defaults to False.
    """
    old_wd = os.getcwd()
    os.chdir(HEADLESS_DIR)
    fly_names = os.listdir(".")
    fly_names = [fly_name for fly_name in fly_names if os.path.isdir(os.path.join(HEADLESS_DIR,fly_name))]

    for i_fly, fly_name in enumerate(fly_names):
        if "GCaMP6s_tdTom_CsChrimson" in fly_name:  # imaging in headless flies
            if os.path.isdir(os.path.join(HEADLESS_DIR, fly_name)):
                print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), i_fly, "out of", len(fly_names), fly_name)
                compress_one_fly_imaging(os.path.join(HEADLESS_DIR, fly_name), out_dir=HEADLESS_ZIP_DIR,
                                         overwrite=overwrite, keep_videos=keep_videos)
        else:
            if keep_videos:
                source_folder = [fly_name]
                tar_file = os.path.join(HEADLESS_ZIP_DIR, fly_name+".tar.gz")
            else:
                source_folder = []
                for path, subdirs, files in os.walk(os.path.join(HEADLESS_DIR, fly_name)):
                    for name in files:
                        if not name.startswith("camera_") and not name.startswith("."):
                            source_folder.append(os.path.relpath(os.path.join(path, name),start=HEADLESS_DIR))
                tar_file = os.path.join(HEADLESS_ZIP_DIR, fly_name+"_novideo.tar.gz")
            if not os.path.isfile(tar_file) or overwrite:
                compress(tar_file, source_folder)
    
    os.chdir(old_wd)


def compress_sleap(overwrite=False):
    """
    Compress SLEAP model data.

    Parameters:
        overwrite (bool): Whether to overwrite existing files. Defaults to False.
    """
    old_wd = os.getcwd()
    os.chdir(OTHER_DIR)
    source_folder = ["sleap_model"]
    tar_file = os.path.join(OTHER_DIR, "sleap_model.tar.gz")
    compress(tar_file, source_folder)
    os.chdir(old_wd)


def compress_stim_control(overwrite=False, keep_videos=False):
    """
    Compress stimulation control data.

    Parameters:
        overwrite (bool): Whether to overwrite existing files. Defaults to False.
        keep_videos (bool): Whether to keep compressed video files. Defaults to False.
    """
    old_wd = os.getcwd()
    os.chdir(OTHER_DIR)
    fly_names = [
        # "220816_MDN3xCsChrimson_Fly5",
        # "230125_BPNxCsChrimson_Fly1"
        "230914_MDN3xCsChrimson_Fly1",
        "230914_MDN3xCsChrimson_Fly2",
        "230914_BPNxCsChrimson_Fly3",
        "230914_BPNxCsChrimson_Fly4",
        "230915_MDN3xCsChrimson_Fly11",
        "230921_BPNxCsChrimson_Fly5",
        "230921_BPNxCsChrimson_Fly6",
        "231115_DfdxGtACR1_Fly1",
        "231115_DfdxGtACR1_Fly2",
        "231115_DfdxGtACR1_Fly3"
    ]
    for fly_name in fly_names:
        if keep_videos:
            source_folder = [fly_name]
            tar_file = os.path.join(OTHER_DIR, fly_name+".tar.gz")
        else:
            source_folder = []
            for path, subdirs, files in os.walk(os.path.join(OTHER_DIR, fly_name)):
                for name in files:
                    if not name.startswith("camera_") and not name.startswith("."):
                        source_folder.append(os.path.relpath(os.path.join(path, name),start=OTHER_DIR))
            tar_file = os.path.join(OTHER_DIR, fly_name+"_novideo.tar.gz")
        if not os.path.isfile(tar_file) or overwrite:
            compress(tar_file, source_folder)
    
    os.chdir(old_wd)


def compress_imaging(overwrite=False, keep_videos=False):
    """
    Compress imaging data.

    Parameters:
        overwrite (bool): Whether to overwrite existing files. Defaults to False.
        keep_videos (bool): Whether to keep compressed behavioural and fluorescence video files. Defaults to False.
    """
    base_dir = IMAGING_DIR
    out_dir = IMAGING_ZIP_DIR
    fly_names = os.listdir(base_dir)
    for i_fly, fly_name in enumerate(fly_names):
        if os.path.isdir(os.path.join(base_dir,fly_name)):
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), i_fly, "out of", len(fly_names), fly_name)
            compress_one_fly_imaging(os.path.join(base_dir, fly_name), out_dir=out_dir, overwrite=overwrite, keep_videos=keep_videos)


def compress_one_fly_imaging(fly_dir, out_dir, overwrite=False, keep_videos=False):
    """
    Compress data for one fly's imaging.

    Parameters:
        fly_dir (str): The directory containing the fly data.
        out_dir (str): The destination directory for compressed data.
        overwrite (bool): Whether to overwrite existing files. Defaults to False.
        keep_videos (bool): Whether to keep compressed behavioural and fluorescence video files. Defaults to False.
    """
    base_dir, fly_name = os.path.split(fly_dir)
    subfolders = os.listdir(fly_dir)
    for subfolder in subfolders:
        if not os.path.isdir(os.path.join(fly_dir, subfolder)):
            continue
        if subfolder == "processed":
            old_wd = os.getcwd()
            os.chdir(fly_dir)
            tar_file = os.path.join(out_dir, fly_name + "__processed.tar.gz")
            if not os.path.isfile(tar_file) or overwrite:  # TODO
                compress(tar_file, ["processed"])
            os.chdir(old_wd)
        else:
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), subfolder)
            compress_one_trial_imaging(trial_dir=os.path.join(fly_dir, subfolder), out_dir=out_dir, overwrite=overwrite, keep_videos=keep_videos)


def compress_one_trial_imaging(trial_dir, out_dir, overwrite=False, keep_videos=False):
    """
    Compress data for one trial's imaging.

    Parameters:
        trial_dir (str): The trial directory containing the data.
        out_dir (str): The destination directory for compressed data.
        overwrite (bool): Whether to overwrite existing files. Defaults to False.
        keep_videos (bool): Whether to keep compressed behavioural and fluorescence video files. Defaults to False.
    """
    old_wd = os.getcwd()
    os.chdir(trial_dir)
    fly_dir, trial_name = os.path.split(trial_dir)
    base_dir, fly_name = os.path.split(fly_dir)
    trial_name = fly_name + "__" + trial_name
    if keep_videos:
        # 2p subfolder
        twop_folder = ["2p"]
        twop_name = trial_name + "__2p.tar.gz"
        tar_file = os.path.join(out_dir, twop_name)
        if not os.path.isfile(tar_file) or overwrite:
            compress(tar_file, twop_folder)
        # behData subfolder
        os.chdir(os.path.join(trial_dir,"behData", "images"))
        beh_folder = ["camera_3.mp4"]
        beh_name = trial_name + "__behData__images__cam3.tar.gz"
        tar_file = os.path.join(out_dir, beh_name)
        if not os.path.isfile(tar_file) or overwrite:
            compress(tar_file, beh_folder)
        beh_folder = ["camera_5.mp4"]
        beh_name = trial_name + "__behData__images__cam5.tar.gz"
        tar_file = os.path.join(out_dir, beh_name)
        if not os.path.isfile(tar_file) or overwrite:
            compress(tar_file, beh_folder)
        if os.path.isfile(os.path.join(os.getcwd(), "camera_1.mp4")):
            beh_folder = ["camera_1.mp4", "capture_metadata.json"]
        else:
            beh_folder = ["capture_metadata.json"]
        beh_name = trial_name + "__behData__images__other.tar.gz"
        tar_file = os.path.join(out_dir, beh_name)
        if not os.path.isfile(tar_file) or overwrite:
            compress(tar_file, beh_folder)
        # processed subfolder
        os.chdir(os.path.join(trial_dir,"processed"))
        processed_folder = ["green_com_warped.tif"]
        processed_name = trial_name + "__processed__tif.tar.gz"
        tar_file = os.path.join(out_dir, processed_name)
        if not os.path.isfile(tar_file) or overwrite:
            compress(tar_file, processed_folder)
        processed_folder = ["beh_df.pkl", "twop_df.pkl"]
        processed_name = trial_name + "__processed__df.tar.gz"
        tar_file = os.path.join(out_dir, processed_name)
        if not os.path.isfile(tar_file) or overwrite:
            compress(tar_file, processed_folder)
    else:
        source_folder = []
        for path, subdirs, files in os.walk(trial_dir):
            for name in files:
                if not name.startswith("camera_") and not name.startswith(".") and not name.startswith("green_com_warped"):
                    source_folder.append(os.path.relpath(os.path.join(path, name),start=trial_dir))
        tar_file = os.path.join(out_dir, trial_name+"_novideo.tar.gz")
        if not os.path.isfile(tar_file) or overwrite:
            compress(tar_file, source_folder)
    os.chdir(old_wd)

def decompress_imaging(imaging_data_dir):
    """
    Decompress imaging data files and update the imaging summary DataFrame.
    Moves tar.gz files into a "compressed" subfolder.

    Parameters:
        imaging_data_dir (str): The directory path containing the imaging data files.

    Returns:
        None
    """
    old_wd = os.getcwd()
    os.chdir(imaging_data_dir)
    all_files = os.listdir(imaging_data_dir)
    compressed_files = [this_file for this_file in all_files if this_file.endswith("tar.gz")]
    trial_files = [this_file for this_file in all_files if this_file.endswith("novideo.tar.gz")]
    processed_files = [this_file for this_file in all_files if this_file.endswith("processed.tar.gz")]

    makedirs_safe(os.path.join(imaging_data_dir, "compressed"))

    for compressed_file in processed_files:
        fly_name, processed = compressed_file.split("__")
        new_fly_dir = os.path.join(imaging_data_dir, fly_name)
        makedirs_safe(new_fly_dir)
        decompress(tar_file=compressed_file, path=new_fly_dir)
        shutil.move(compressed_file, os.path.join("compressed", compressed_file))

    for compressed_file in trial_files:
        fly_name, trial_name = compressed_file.split("__")
        trial_name = trial_name.replace("_novideo.tar.gz", "")
        new_trial_dir = os.path.join(imaging_data_dir, fly_name, trial_name)
        makedirs_safe(new_trial_dir)
        decompress(tar_file=compressed_file, path=new_trial_dir)
        shutil.move(compressed_file, os.path.join("compressed", compressed_file))

    os.chdir(old_wd)

    update_summary_df(df_path=os.path.join(imaging_data_dir, "imaging_summary_df.csv"), base_dir=imaging_data_dir)

    
def decompress_headless_predictions(headless_predictions_data_dir):
    """
    Decompress headless prediction data files and update the respective summary DataFrames.
    Moves tar.gz files into a "compressed" subfolder.

    Parameters:
        headless_predictions_data_dir (str): The directory path containing headless prediction data files.

    Returns:
        None
    """
    old_wd = os.getcwd()
    os.chdir(headless_predictions_data_dir)
    all_files = os.listdir(headless_predictions_data_dir)
    compressed_files = [this_file for this_file in all_files if this_file.endswith("tar.gz")]
    makedirs_safe(os.path.join(headless_predictions_data_dir, "compressed"))
    compressed_imaging_files = []
    for compressed_file in compressed_files:
        if "processed" in compressed_file or "xz_t1" in compressed_file:
            compressed_imaging_files.append(compressed_file)
            continue
        decompress(tar_file=compressed_file, path=headless_predictions_data_dir)
        shutil.move(compressed_file, os.path.join("compressed", compressed_file))

    # handle the headless flies that also have imaging data
    trial_files = [this_file for this_file in compressed_imaging_files if this_file.endswith("novideo.tar.gz")]
    processed_files = [this_file for this_file in compressed_imaging_files if this_file.endswith("processed.tar.gz")]
    for compressed_file in processed_files:
        fly_name, processed = compressed_file.split("__")
        new_fly_dir = os.path.join(imaging_data_dir, fly_name)
        makedirs_safe(new_fly_dir)
        decompress(tar_file=compressed_file, path=new_fly_dir)
        shutil.move(compressed_file, os.path.join("compressed", compressed_file))

    for compressed_file in trial_files:
        fly_name, trial_name = compressed_file.split("__")
        trial_name = trial_name.replace("_novideo.tar.gz", "")
        new_trial_dir = os.path.join(imaging_data_dir, fly_name, trial_name)
        makedirs_safe(new_trial_dir)
        decompress(tar_file=compressed_file, path=new_trial_dir)
        shutil.move(compressed_file, os.path.join("compressed", compressed_file))

    os.chdir(old_wd)

    update_summary_df(df_path=os.path.join(headless_predictions_data_dir, "headless_summary_df.csv"), base_dir=headless_predictions_data_dir)
    update_summary_df(df_path=os.path.join(headless_predictions_data_dir, "predictions_summary_df.csv"), base_dir=headless_predictions_data_dir)

def decompress_other(other_data_dir):
    """
    Decompress supplementary data files.
    Moves tar.gz files into a "compressed" subfolder.

    Parameters:
        other_data_dir (str): The directory path containing supplementary data files.

    Returns:
        None
    """
    old_wd = os.getcwd()
    os.chdir(other_data_dir)
    all_files = os.listdir(other_data_dir)
    compressed_files = [this_file for this_file in all_files if this_file.endswith("tar.gz")]
    makedirs_safe(os.path.join(other_data_dir, "compressed"))
    for compressed_file in compressed_files:
        decompress(tar_file=compressed_file, path=other_data_dir)
        shutil.move(compressed_file, os.path.join("compressed", compressed_file))
    os.chdir(old_wd)

def update_summary_df(df_path, base_dir):
    """
    Update a summary DataFrame with new directory paths.

    Parameters:
        df_path (str): The path to the summary DataFrame CSV file.
        base_dir (str): The new base directory path of the data.

    Returns:
        None
    """
    _, df_name = os.path.split(df_path)
    df = summarydf.load_data_summary(path=df_path)
    def update_trial(trial_dir, base_dir):
        fly_dir, trial_name = os.path.split(trial_dir)
        old_base_dir, fly_name = os.path.split(fly_dir)
        new_fly_dir = os.path.join(base_dir, fly_name)
        return os.path.join(new_fly_dir, trial_name)

    for index, row in df.iterrows():
        old_trial_dir = row.trial_dir
        new_trial_dir = update_trial(old_trial_dir, base_dir=base_dir)
        new_fly_dir = os.path.dirname(new_trial_dir)
        df["trial_dir"].iloc[df.index == index] = new_trial_dir
        df["fly_dir"].iloc[df.index == index] = new_fly_dir

    df.to_csv(os.path.join(base_dir, df_name), index=False)

def compress(tar_file, members):
    """
    Compress files using tar and gzip.
    Copied from https://thepythoncode.com/article/compress-decompress-files-tarfile-python#google_vignette

    Parameters:
        tar_file (str): The path to the tar.gz file to create.
        members (list): List of file/folder names to include in the archive.
    """
    # open file for gzip compressed writing
    tar = tarfile.open(tar_file, mode="w:gz")
    # with progress bar
    # set the progress bar
    progress = tqdm(members)
    for member in progress:
        # set the progress description of the progress bar
        progress.set_description(f"Compressing {member}")
        # add file/folder/link to the tar file (compress)
        tar.add(member)
    # close the file
    tar.close()


def decompress(tar_file, path, members=None):
    """
    Decompress files from a tar.gz archive.
    Extracts `tar_file` and puts the `members` to `path`.
    If members is None, all members on `tar_file` will be extracted.
    Copied from: https://thepythoncode.com/article/compress-decompress-files-tarfile-python#google_vignette

    Parameters:
        tar_file (str): The path to the tar.gz file to decompress.
        path (str): The destination directory for decompressed files.
        members (list): List of specific members to extract. Defaults to None (extract all).
    """
    tar = tarfile.open(tar_file, mode="r:gz")
    if members is None:
        members = tar.getmembers()
    # with progress bar
    # set the progress bar
    progress = tqdm(members)
    for member in progress:
        # set the progress description of the progress bar
        progress.set_description(f"Extracting {member.name}")
        
        tar.extract(member, path=path)
    # or use this
    # tar.extractall(members=members, path=path)
    # close the file
    tar.close()

if __name__ == "__main__":
    ##### TO COMPRESS THE DATA UNCOMMENT THE FOLLOWING
    compress_sleap()

    copy_stim_control_trials()
    compress_stim_control(keep_videos=False)

    copy_all_predictions_trials()
    copy_all_headless_trials()
    compress_headless_predictions(keep_videos=False)  # TODO: check how compression of MDN > CsChrimson, GCaMP compression works
    
    copy_all_imaging_trials()
    compress_imaging(keep_videos=False)

    copy_leg_cutting_trials(overwrite=False)
    compress_leg_cutting(overwrite=False, keep_videos=False)

    ##### TO DECOMPRESS THE DATA UNCOMMENT THE FOLLOWING
    """
    decompress_imaging("PATH/TO/THE/DATAVERSE/DOWNLOADS/FROM/Optogenetics_Dfd_population_imaging")
    decompress_headless_predictions("PATH/TO/THE/DATAVERSE/DOWNLOADS/FROM/Optogenetics_headless_behaviour")
    decompress_other("PATH/TO/THE/DATAVERSE/DOWNLOADS/FROM/Supplementary_Data")
    """
     

