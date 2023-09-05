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
params = PreProcessParams()
import utils2p
import summarydf, params, loaddata


BASE_DIR = "/mnt/nas2/JB/_paper_data"
IMAGING_DIR = os.path.join(BASE_DIR, "imaging")
IMAGING_ZIP_DIR = os.path.join(BASE_DIR, "imaging_zip_novideo")
OPTOGENETICS_DIR = os.path.join(BASE_DIR, "optogenetics")
HEADLESS_DIR = os.path.join(BASE_DIR, "headless")
HEADLESS_ZIP_DIR = os.path.join(BASE_DIR, "headless_zip_novideo")
OTHER_DIR = os.path.join(BASE_DIR, "other")

def copy_stim_control_trials(overwrite=False):
    pass

def make_roi_center_annotation_file(fly_dir, quantile=0.99):
    pca_map_file = os.path.join(fly_dir, "processed", params.pca_maps)
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
    background_image = loaddata.get_background_image(fly_dir)
    save_stack(os.path.join(fly_dir, "processed", "background_image.tif"), background_image)


def copy_all_imaging_trials(overwrite=False):
    df = summarydf.load_data_summary()
    df = summarydf.filter_data_summary(df, no_co2=False)
    df = summarydf.get_selected_df(df, [{"GCaMP": "Dfd"}])
    ball_df = summarydf.get_selected_df(df, [{"walkon": "ball"}])
    wheel_df = summarydf.get_selected_df(df, [{"walkon": "wheel"}])
    
    fly_ids_DNp09 = [15,18,44,46,61]
    fly_ids_aDN2 = [25,32,60,63,72,74]
    fly_ids_MDN = [3,5,7,20,28,34,42,50,51,62,71]
    fly_ids_PR = [8,10,65]
    fly_ids_ball = fly_ids_DNp09 + fly_ids_aDN2 + fly_ids_MDN + fly_ids_PR
    fly_id_natbeh_wheel = 72

    selected_trials = np.zeros((len(df)), dtype=bool)
    for fly_id in fly_ids_ball:
        new_trials = np.logical_and(df.walkon.values == "ball", df.fly_id.values == fly_id)
        selected_trials = np.logical_or(selected_trials, new_trials)

    new_trials = np.logical_and(df.walkon.values == "wheel", df.fly_id.values == fly_id_natbeh_wheel)
    selected_trials = np.logical_or(selected_trials, new_trials)

    df = df[selected_trials]

    df.to_csv(os.path.join(IMAGING_DIR, "imaging_summary_df.csv"), index=False)

    for i_fly, (fly_id, fly_df) in enumerate(df.groupby("fly_id")):
        trial_dirs = fly_df.trial_dir.values
        copy_one_fly(trial_dirs, imaging=True, overwrite=overwrite)


def copy_all_headless_trials(overwrite=False):
    df = summarydf.get_headless_df()
    df.to_csv(os.path.join(HEADLESS_DIR, "headless_summary_df.csv"), index=False)

    for i_fly, (fly_id, fly_df) in enumerate(df.groupby("fly_id")):
        trial_dirs = list(fly_df.trial_dir.values)
        for trial_dir in trial_dirs:
            if "noball" in trial_dir and not "nohead" in trial_dir:
                trial_dirs.remove(trial_dir)
        copy_one_fly(trial_dirs, imaging=False, overwrite=overwrite)


def copy_all_predictions_trials(genotypes=["DNa01", "DNa02", "DNb02", "aDN1", "DNg14", "mute"], overwrite=False):
    df = summarydf.get_predictions_df()
    df = summarydf.get_selected_df(df, select_dicts=[{"CsChrimson": genotype} for genotype in genotypes])
    df.to_csv(os.path.join(HEADLESS_DIR, "predictions_summary_df.csv"), index=False)

    for i_fly, (fly_id, fly_df) in enumerate(df.groupby("fly_id")):
        trial_dirs = list(fly_df.trial_dir.values)
        for trial_dir in trial_dirs:
            if "noball" in trial_dir and not "nohead" in trial_dir:
                trial_dirs.remove(trial_dir)
        copy_one_fly(trial_dirs, imaging=False, overwrite=overwrite)


def copy_one_fly(trial_dirs, imaging=True, overwrite=False):
    fly_dir = os.path.dirname(trial_dirs[0])
    date_genotype_dir, fly_name = os.path.split(fly_dir)
    _, date_genotype_name = os.path.split(date_genotype_dir)

    if imaging:
        new_fly_dir = os.path.join(IMAGING_DIR, date_genotype_name + "_" + fly_name)
    else:
        new_fly_dir = os.path.join(HEADLESS_DIR, date_genotype_name + "_" + fly_name)
    makedirs_safe(new_fly_dir)
    if imaging:
        makedirs_safe(os.path.join(new_fly_dir, "processed"))

        if not os.path.isfile(os.path.join(fly_dir, "processed", "roi_center_annotation.pdf")):
            make_roi_center_annotation_file(fly_dir)
        copy_file(os.path.join(fly_dir, "processed", "roi_center_annotation.pdf"), fly_dir, new_fly_dir, overwrite=overwrite)

        if not os.path.isfile(os.path.join(fly_dir, "processed", "background_image.tif")):
            make_background_image(fly_dir)
        copy_file(os.path.join(fly_dir, "processed", "background_image.tif"), fly_dir, new_fly_dir, overwrite=overwrite)

        copy_file(os.path.join(fly_dir, "processed", "ROI_centers.txt"), fly_dir, new_fly_dir, overwrite=overwrite)
        copy_file(os.path.join(fly_dir, "processed", "ROI_mask.tif"), fly_dir, new_fly_dir, overwrite=overwrite)

    for trial_dir in trial_dirs:
        copy_one_trial(new_fly_dir, trial_dir, imaging=imaging, overwrite=overwrite)


def copy_one_trial(new_fly_dir, trial_dir, imaging=True, overwrite=False):
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

    if imaging:
        copy_file(os.path.join(trial_dir, "processed", "twop_df.pkl"), trial_dir, new_trial_dir, overwrite=overwrite)
        copy_file(os.path.join(trial_dir, "processed", "green_com_warped.tif"), trial_dir, new_trial_dir, overwrite=overwrite)
        copy_file(utils2p.find_metadata_file(trial_dir), trial_dir, new_trial_dir, overwrite=overwrite)
        

def copy_file(file_path, old_trial_dir, new_trial_dir, overwrite=False):
    new_file_path = file_path.replace(old_trial_dir, new_trial_dir)
    new_dir = os.path.dirname(new_file_path)
    makedirs_safe(new_dir)
    if not os.path.isfile(new_file_path) or overwrite:
        shutil.copy(file_path, new_file_path)


def compress_headless_predictions(overwrite=False, keep_videos=False):
    old_wd = os.getcwd()
    os.chdir(HEADLESS_DIR)
    fly_names = os.listdir(".")
    fly_names = [fly_name for fly_name in fly_names if os.path.isdir(os.path.join(HEADLESS_DIR,fly_name))]

    for fly_name in fly_names:
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
    old_wd = os.getcwd()
    os.chdir(OTHER_DIR)
    source_folder = ["sleap_model"]
    tar_file = os.path.join(OTHER_DIR, "sleap_model.tar.gz")
    compress(tar_file, source_folder)
    os.chdir(old_wd)

def compress_imaging(overwrite=False, keep_videos=False):
    base_dir = IMAGING_DIR
    out_dir = IMAGING_ZIP_DIR
    fly_names = os.listdir(base_dir)
    for i_fly, fly_name in enumerate(fly_names):
        if os.path.isdir(os.path.join(base_dir,fly_name)):
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), i_fly, "out of", len(fly_names), fly_name)
            compress_one_fly_imaging(os.path.join(base_dir, fly_name), out_dir=out_dir, overwrite=overwrite, keep_videos=keep_videos)


def compress_one_fly_imaging(fly_dir, out_dir, overwrite=False, keep_videos=False):
    base_dir, fly_name = os.path.split(fly_dir)
    subfolders = os.listdir(fly_dir)
    for subfolder in subfolders:
        if not os.path.isdir(os.path.join(fly_dir, subfolder)):
            continue
        if subfolder == "processed":
            old_wd = os.getcwd()
            os.chdir(fly_dir)
            tar_file = os.path.join(out_dir, fly_name + "__processed.tar.gz")
            if True:  # not os.path.isfile(tar_file) or overwrite:  # TODO
                compress(tar_file, ["processed"])
            os.chdir(old_wd)
        else:
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), subfolder)
            compress_one_trial_imaging(trial_dir=os.path.join(fly_dir, subfolder), out_dir=out_dir, overwrite=overwrite, keep_videos=keep_videos)


def compress_one_trial_imaging(trial_dir, out_dir, overwrite=False, keep_videos=False):
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


def compress(tar_file, members):
    """
    Adds files (`members`) to a tar_file and compress it
    Copied from https://thepythoncode.com/article/compress-decompress-files-tarfile-python#google_vignette
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
    Extracts `tar_file` and puts the `members` to `path`.
    If members is None, all members on `tar_file` will be extracted.
    Copied from: https://thepythoncode.com/article/compress-decompress-files-tarfile-python#google_vignette
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
    # compress_sleap()
    # copy_all_predictions_trials()
    # copy_all_headless_trials()
    compress_headless_predictions(keep_videos=False)
    
    # copy_all_imaging_trials()
    # compress_imaging(keep_videos=False)

