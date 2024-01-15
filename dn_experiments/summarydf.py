import os
import sys
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json

from datetime import date

today = date.today().strftime("%y%m%d")

from twoppp import load, rois
from utils2p import find_seven_camera_metadata_file

sys.path.append(os.path.dirname(__file__))
import params


def find_genotyp_dirs(
    nas_base_dir="/mnt/nas2/JB",
    min_date: int = None,
    max_date: int = None,
    contains=None,
    exclude=None,
):
    all_dirs = [
        this_dir.split(os.sep)[-2]
        for this_dir in glob(f"{nas_base_dir}/*/", recursive=True)
    ]
    all_dirs = [this_dir for this_dir in all_dirs if this_dir.startswith("2")]
    if min_date is not None:
        all_dirs = [this_dir for this_dir in all_dirs if int(this_dir[:6]) >= min_date]
    if max_date is not None:
        all_dirs = [this_dir for this_dir in all_dirs if int(this_dir[:6]) <= max_date]
    if contains is not None:
        all_dirs = [this_dir for this_dir in all_dirs if contains in this_dir]
    if exclude is not None:
        all_dirs = [this_dir for this_dir in all_dirs if exclude not in this_dir]
    all_genotype_dirs = [os.path.join(nas_base_dir, this_dir) for this_dir in all_dirs]
    return all_genotype_dirs


def frame_count(video_path, manual=False):
    """Counts the number of frames in a video file.
    Args:
        video_path (str): Path to video file.
        manual (bool): Whether to use the slow but accurate method or the fast but inaccurate method.
    Returns:
        frames (int): Number of frames in video.
    """

    def manual_count(handler):
        frames = 0
        while True:
            status, frame = handler.read()
            if not status:
                break
            frames += 1
        return frames

    cap = cv2.VideoCapture(video_path)
    # Slow, inefficient but 100% accurate method
    if manual:
        frames = manual_count(cap)
    # Fast, efficient but inaccurate method
    else:
        try:
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        except:
            frames = manual_count(cap)
    cap.release()
    return frames


def add_flies_to_data_summary(
    genotype_dir, path=None, headless=False, predictions=False, revisions=False
):
    if path is None and not headless and not predictions and not revisions:
        path = params.data_summary_csv_dir
    elif path is None and headless:
        path = params.data_headless_summary_csv_dir
    elif path is None and predictions:
        path = params.data_predictions_summary_csv_dir
    elif path is None and revisions:
        path = params.data_revisions_summary_dir
    df = pd.read_csv(path)

    if not isinstance(genotype_dir, list):
        genotype_dir = [genotype_dir]
    all_fly_dirs = []
    for this_dir in genotype_dir:
        all_fly_dirs += load.get_flies_from_datedir(this_dir)
    all_trial_dirs = load.get_trials_from_fly(all_fly_dirs)

    # TODO: check whether flies / trials are already in DF

    base_df = pd.DataFrame(columns=df.columns, index=[0])  # df[:1].copy()
    min_fly_id = df.fly_id.max() + 1
    if np.isnan(min_fly_id):
        min_fly_id = 0

    if headless or predictions or revisions:
        all_trial_dfs = get_trial_info_headless(base_df, all_trial_dirs, min_fly_id)
    else:
        all_trial_dfs = get_trial_info(base_df, all_trial_dirs, min_fly_id)

    new_trials_df = pd.concat(
        all_trial_dfs
    )  # TODO: check whether some flies are already in dataframe
    df = pd.concat([df, new_trials_df])
    df.to_csv(path, index=False)


def get_trial_info(base_df, all_trial_dirs, min_fly_id):
    all_trial_dfs = []
    for i_fly, fly_trial_dirs in enumerate(all_trial_dirs):
        fly_dir = os.path.dirname(fly_trial_dirs[0])
        for i_trial, trial_dir in enumerate(fly_trial_dirs):
            trial_df = base_df.copy()
            trial_df.fly_dir = fly_dir
            trial_df.trial_dir = trial_dir

            genotype_dir, fly_name = os.path.split(fly_dir)
            base_dir, date_name = os.path.split(genotype_dir)

            trial_df.fly_number = (
                int(fly_name[3:]) if len(fly_name) in [4, 5] else fly_name[3:]
            )
            trial_df.date = int(date_name[:6])
            genotype = date_name[7:]
            trial_df.genotype = genotype
            trial_df.trial_number = i_trial
            _, trial_name = os.path.split(trial_dir)
            trial_df.trial_name = trial_name

            trial_df.GCaMP = genotype.split("xGCaMP")[0]
            trial_df.tdTomato = "tdT" in genotype
            trial_df.CsChrimson = genotype.split("xCsChr")[0].split("_")[-1]
            trial_df.CsChrimson_expression_system = (
                "LeXA" if "BO" in genotype else "GAL4"
            )
            if "wheel" in trial_name:
                trial_df.walkon = "wheel"
            elif "ball" in trial_name:
                trial_df.walkon = "ball"
            elif "CO2" in trial_name or "co2" in trial_name:
                trial_df.walkon = "no"
            else:
                trial_df.walkon = "ball"
            trial_df.image_type = "xz" if "xz" in trial_name else trial_name[4:]
            if "CO2" in trial_name or "co2" in trial_name:
                trial_df.CO2 = True
            else:
                trial_df.CO2 = True
            if "plevels" in trial_name or "p10" in trial_name or "p20" in trial_name:
                trial_df.laser_stim = True
            else:
                trial_df.laser_stim = False

            trial_df.olfac_stim = True if "olfac" in trial_name else False
            if "p10" in trial_name:
                trial_df.laser_power = "10"
            elif "10_20" in trial_name:
                trial_df.laser_power = "10_20"
            elif "p20" in trial_name:
                trial_df.laser_power = "20"
            elif "plevels" in trial_name:
                trial_df.laser_power = "1_5_10_20"
            else:
                trial_df.laser_power = ""
            if "thorax" in trial_name:
                trial_df.stim_location = "thorax"
            elif "_t1_" in trial_name:
                trial_df.stim_location = "t1"
            elif "head" in trial_name:
                trial_df.stim_location = "head"
            elif trial_df.iloc[0].laser_stim:
                trial_df.stim_location = "cc"
            else:
                trial_df.stim_location = ""

            trial_df["fly_id"] = min_fly_id + i_fly

            if os.path.isfile(
                os.path.join(fly_dir, load.PROCESSED_FOLDER, "ROI_centers.txt")
            ):
                trial_df.roi_selection_done = True
                roi_centers = rois.read_roi_center_file(
                    os.path.join(fly_dir, load.PROCESSED_FOLDER, "ROI_centers.txt")
                )
                trial_df.n_neurons = len(roi_centers)
            else:
                trial_df.roi_selection_done = False
                trial_df.n_neurons = 0

            all_trial_dfs.append(trial_df)
    return all_trial_dfs


def get_trial_info_headless(base_df, all_trial_dirs, min_fly_id):
    all_trial_dfs = []
    for i_fly, fly_trial_dirs in enumerate(all_trial_dirs):
        fly_dir = os.path.dirname(fly_trial_dirs[0])
        for i_trial, trial_dir in enumerate(fly_trial_dirs):
            trial_df = base_df.copy()
            trial_df.fly_dir = fly_dir
            trial_df.trial_dir = trial_dir

            genotype_dir, fly_name = os.path.split(fly_dir)
            base_dir, date_name = os.path.split(genotype_dir)
            nas_dir, experimenter = os.path.split(base_dir)

            trial_df.experimenter = experimenter
            trial_df.fly_number = (
                int(fly_name[3:]) if len(fly_name) in [4, 5] else fly_name[3:]
            )
            trial_df.date = int(date_name[:6])
            genotype = date_name[7:]
            trial_df.genotype = genotype
            trial_df.trial_number = i_trial
            _, trial_name = os.path.split(trial_dir)
            trial_df.trial_name = trial_name

            trial_df.GCaMP = genotype.split("xGCaMP")[0]
            trial_df.tdTomato = "tdT" in genotype
            trial_df.CsChrimson = genotype.split("xCsChr")[0].split("_")[-1]
            trial_df.CsChrimson_expression_system = (
                "LeXA" if "BO" in genotype else "GAL4"
            )
            if "wheel" in trial_name:
                trial_df.walkon = "wheel"
            elif "noball" in trial_name:
                trial_df.walkon = "no"
            elif "ball" in trial_name:
                trial_df.walkon = "ball"
            elif "CO2" in trial_name or "co2" in trial_name:
                trial_df.walkon = "no"
            else:
                trial_df.walkon = "ball"
            # trial_df.image_type = "xz" if "xz" in trial_name else trial_name[4:]
            # if "CO2" in trial_name or "co2" in trial_name:
            #     trial_df.CO2 = True
            # else:
            #     trial_df.CO2 = True
            if "plevels" in trial_name or "p10" in trial_name or "p20" in trial_name:
                trial_df.laser_stim = True
            else:
                trial_df.laser_stim = False

            # trial_df.olfac_stim = True if "olfac" in trial_name else False
            if "p10" in trial_name:
                trial_df.laser_power = "10"
            elif "10_20" in trial_name:
                trial_df.laser_power = "10_20"
            elif "p20" in trial_name:
                trial_df.laser_power = "20"
            elif "plevels" in trial_name:
                trial_df.laser_power = "1_5_10_20"
            else:
                trial_df.laser_power = ""
            if "thorax" in trial_name:
                trial_df.stim_location = "thorax"
            elif "_t1_" in trial_name:
                trial_df.stim_location = "t1"
            elif "head" in trial_name:
                trial_df.stim_location = "head"
            elif trial_df.iloc[0].laser_stim:
                trial_df.stim_location = "cc"
            else:
                trial_df.stim_location = ""

            if ("nohead" in trial_name) or ("headless" in trial_name):
                trial_df["head"] = False
            else:
                trial_df["head"] = True

            # Leg amputation
            if "amp" in trial_name:
                if "HL" in trial_name:
                    trial_df["leg_amp"] = "HL"
                elif "ML" in trial_name:
                    trial_df["leg_amp"] = "ML"
                elif "FL" in trial_name:
                    trial_df["leg_amp"] = "FL"
                else:
                    trial_df["leg_amp"] = "no"

                if "FeTi" in trial_name:
                    trial_df["joint_amp"] = "FeTi"
                elif "TiTa" or "tarsus" in trial_name:
                    trial_df["joint_amp"] = "TiTa"
                else:
                    trial_df["joint_amp"] = "no"

            # camera frame counts to see if there has been an aqcuisisiton issue
            seven_camera_metadata_file = find_seven_camera_metadata_file(trial_dir)
            with open(seven_camera_metadata_file, "r") as f:
                capture_info = json.load(f)
            list_of_cameras = [
                cam_idx for cam_idx in capture_info["Frame Counts"].keys()
            ]
            for cam_num in list_of_cameras:
                video_path = os.path.join(
                    trial_dir, "behData", "images", f"camera_{cam_num}.mp4"
                )
                trial_df[f"frames_{cam_num}"] = frame_count(video_path)

            trial_df["n_trials"] = len(fly_trial_dirs)
            trial_df["fly_id"] = min_fly_id + i_fly

            # if os.path.isfile(os.path.join(fly_dir, load.PROCESSED_FOLDER, "ROI_centers.txt")):
            #     trial_df.roi_selection_done = True
            #     roi_centers = rois.read_roi_center_file(os.path.join(fly_dir, load.PROCESSED_FOLDER, "ROI_centers.txt"))
            #     trial_df.n_neurons = len(roi_centers)
            # else:
            #     trial_df.roi_selection_done = False
            #     trial_df.n_neurons = 0

            all_trial_dfs.append(trial_df)

    return all_trial_dfs


def load_data_summary(path=params.data_summary_csv_dir):
    """load a dataframe with data summary from a csv file

    Parameters
    ----------
    path : str, optional
        location of csv file with data summary, by default data_summary_csv_dir

    Returns
    -------
    pd.DataFrame
        data summary DataFrame
    """
    return pd.read_csv(path)


def filter_data_summary(
    df,
    imaging_type="xz",
    no_co2=True,
    q_thres_neural=params.q_thres_neural,
    q_thres_beh=params.q_thres_beh,
    q_thres_stim=params.q_thres_stim,
):
    """
    filter the summary data frame for trials that are not of interest or of bad quality

    Parameters
    ----------
    df : pd.DataFrame
    imaging_type : str, optional
        filters the imag_type field if not None, by default "xz"
    no_co2 : bool, optional
        return only trials without CO2, by default True
    q_thres_neural : int, optional
        apply a threshold on the neural recording quality, by default 3
    q_thres_beh : int, optional
        apply a threshold on the behavioural recording quality, by default 3
    q_thres_stim : int, optional
        apply a threshold on stimulus response quality, by default 3

    Returns
    -------
    pd.DataFrame
    """
    df_copy = df.copy()

    df_copy = df_copy[np.logical_not(df_copy.exclude == True)]
    if imaging_type is not None:
        df_copy = df_copy[df_copy.image_type == imaging_type]
    if no_co2:
        df_copy = df_copy[
            np.logical_and(
                df_copy.CO2 == False,
                [not "co2_puff" in trial_name for trial_name in df_copy.trial_name],
            )
        ]
    if (
        q_thres_neural is not None
        and q_thres_beh is not None
        and q_thres_stim is not None
    ):
        df_copy = df_copy[
            np.logical_and.reduce(
                (
                    df_copy.neural_quality <= q_thres_neural,
                    df_copy.behaviour_quality <= q_thres_beh,
                    df_copy.stim_response_quality <= q_thres_stim,
                )
            )
        ]

    return df_copy


def get_selected_df(df, select_dicts):
    """
    filter df for certain arguments, e.g. GCaMP == "Dfd" and CsChrimson == "DNp09"

    Parameters
    ----------
    df : pd.DataFrame
        summary dataframe of all trials
    select_dicts : list(dict)
        list of dictionaries. each dictionary contains argument + value pairs. e.g.:
        [{"GCaMP": "Dfd", "CsChrimson": "DNp09"}, {"GCaMP": "Dfd", "CsChrimson": "DNP9"}]
        inside dict is interpreted as logical and.
        results from multiple dicts will be appended

    Returns
    -------
    pd.DataFrame
        dataframe containing only selected trials
    """
    if not isinstance(select_dicts, list):
        select_dicts = [select_dicts]
    selected_dfs = []
    for select_dict in select_dicts:
        dict_keys = list(select_dict.keys())
        selected_dfs.append(
            df[
                np.logical_and.reduce(
                    [df[this_key] == select_dict[this_key] for this_key in dict_keys]
                )
            ]
        )
    if len(selected_dfs) > 1:
        return pd.concat(selected_dfs)
    else:
        return selected_dfs[0]


def get_olfac_df(df=None):
    if df is None:
        df = filter_data_summary(load_data_summary())
    return get_selected_df(df, [{"GCaMP": "Dfd", "olfac_stim": True}])


def get_ball_df(df=None):
    if df is None:
        df = filter_data_summary(load_data_summary())
    return get_selected_df(df, [{"GCaMP": "Dfd", "walkon": "ball"}])


def get_wheel_df(df=None):
    if df is None:
        df = filter_data_summary(load_data_summary())
    return get_selected_df(df, [{"GCaMP": "Dfd", "walkon": "wheel"}])


def get_aDN2_olfac_df(df=None):
    if df is None:
        df = filter_data_summary(load_data_summary())
    df_aDN = get_selected_df(df, [{"GCaMP": "Dfd", "CsChrimson": "aDN2"}])
    df_olfac = get_selected_df(df, [{"GCaMP": "Dfd", "olfac_stim": True}])
    fly_ids = list(
        set(np.unique(df_aDN.fly_id)).intersection(np.unique(df_olfac.fly_id))
    )

    # df = pd.concat((df_aDN.iloc[[this_id in fly_ids for this_id in df_aDN.fly_id.values]], df_olfac.iloc[[this_id in fly_ids for this_id in df_olfac.fly_id.values]]))
    return df_aDN.iloc[[this_id in fly_ids for this_id in df_aDN.fly_id.values]]


def get_stim_ball_df(df):
    return get_selected_df(df, [{"laser_stim": True, "walkon": "ball"}])


def get_stim_wheel_df(df):
    return get_selected_df(df, [{"laser_stim": True, "walkon": "wheel"}])


def get_new_flies_df(
    df=None,
    date=230401,
    no_co2=True,
    q_thres_neural=params.q_thres_neural,
    q_thres_beh=params.q_thres_beh,
    q_thres_stim=params.q_thres_stim,
):
    if df is None:
        df = filter_data_summary(
            df=load_data_summary(),
            no_co2=no_co2,
            q_thres_neural=q_thres_neural,
            q_thres_beh=q_thres_beh,
            q_thres_stim=q_thres_stim,
        )
    df = df[df.date >= date]
    return df


def get_genotype_dfs_of_interest(df=None):
    if df is None:
        df = filter_data_summary(load_data_summary())
    df_Dfd_MDN = get_selected_df(df, [{"GCaMP": "Dfd", "CsChrimson": "MDN3"}])
    df_Dfd_DNp09 = get_selected_df(
        df,
        [
            {"GCaMP": "Dfd", "CsChrimson": "DNp09"},
            {"GCaMP": "Dfd", "CsChrimson": "DNP9"},
        ],
    )
    df_Dfd_aDN = get_selected_df(df, [{"GCaMP": "Dfd", "CsChrimson": "aDN2"}])
    df_Dfd_PR = get_selected_df(df, [{"GCaMP": "Dfd", "CsChrimson": "PR"}])
    df_Scr = get_selected_df(df, [{"GCaMP": "Scr"}])  # , "CsChrimson": "MDN"
    df_BO = get_selected_df(df, [{"GCaMP": "BO"}])  # , "CsChrimson": "MDN"

    dfs = [df_Dfd_MDN, df_Dfd_DNp09, df_Dfd_aDN, df_Dfd_PR, df_Scr, df_BO]
    df_names = [
        "Dfd & MDN",
        "Dfd & DNp09",
        "Dfd & aDN2",
        "Dfd & ___",
        "Scr & MDN",
        "BO  & MDN",
    ]
    return dfs, df_names


def get_headless_df(
    path=params.data_headless_summary_csv_dir,
    q_check=params.q_check_headless,
    q_thres_legs_intact=params.q_thres_headless_legs_intact,
    q_thres_beh=params.q_thres_headless_beh,
    q_thres_stim=params.q_thres_headless_stim,
):
    df = load_data_summary(path=path)

    df = df[np.logical_not(df.exclude == True)]
    if q_check:
        df = df[
            np.logical_and.reduce(
                (
                    df.legs_intact_post <= q_thres_legs_intact,
                    df.behaviour_quality <= q_thres_beh,
                    df.stim_response_quality_pre <= q_thres_stim,
                )
            )
        ]
    return df


def get_predictions_df(
    path=params.data_predictions_summary_csv_dir,
    q_check=params.q_check_headless,
    q_thres_legs_intact=params.q_thres_headless_legs_intact,
    q_thres_beh=params.q_thres_headless_beh,
    q_thres_stim=params.q_thres_headless_stim,
):
    return get_headless_df(
        path=path,
        q_check=q_check,
        q_thres_legs_intact=q_thres_legs_intact,
        q_thres_beh=q_thres_beh,
        q_thres_stim=q_thres_stim,
    )


def plot_trial_number_summary(dfs, df_names, plot_base_dir=params.data_summary_dir):
    n_flies = np.array([len(np.unique(this_df.fly_dir)) for this_df in dfs])
    n_trials = np.array([len(this_df) for this_df in dfs])
    n_stim_trials = np.array([np.sum(df.laser_stim) for df in dfs])

    fig, axs = plt.subplots(1, 2, figsize=(9.0, 4), sharex=True)
    for i_n_fly, n_fly in enumerate(n_flies):
        axs[0].bar(i_n_fly, n_fly)
    axs[0].set_title("# flies per genotype")

    for i_n_fly, n_trial in enumerate(n_trials):
        axs[1].bar(i_n_fly, n_trial, alpha=0.5)
    axs[1].set_prop_cycle(None)
    for i_n_fly, n_stim_trial in enumerate(n_stim_trials):
        axs[1].bar(i_n_fly, n_stim_trial, alpha=1)
    axs[1].set_title("# of trials per genotype (of which stimulation trials)")

    axs[1].set_yticks(np.arange(5) * 5)

    _ = [ax.set_xticks(np.arange(len(dfs))) for ax in axs]
    _ = [ax.set_xticklabels(df_names, rotation=45, ha="right") for ax in axs]
    _ = [ax.spines["top"].set_visible(False) for ax in axs]
    _ = [ax.spines["right"].set_visible(False) for ax in axs]

    fig.tight_layout()
    fig.savefig(os.path.join(plot_base_dir, f"n_good_flies_{today}.png"), dpi=300)


if __name__ == "__main__":
    genotype_dirs = find_genotyp_dirs(
        nas_base_dir="/mnt/nas2/FH",
        min_date=240101,
        max_date=240110,
        contains="CsChrimson",
        exclude="BAD",
    )  # exclude = "headless"
    add_flies_to_data_summary(
        genotype_dir=genotype_dirs,
        path=None,
        headless=False,
        predictions=False,
        revisions=True,
    )


"""

    df = load_data_summary()
    df = filter_data_summary(df, imaging_type="xz", no_co2=False, q_thres_neural=params.q_thres_neural, q_thres_beh=params.q_thres_beh, q_thres_stim=params.q_thres_stim)
    df = get_selected_df(df, [{"walkon": "wheel"}, {"walkon": "ball"}])
    trial_dirs = df.trial_dir.values
    image_dirs = [os.path.join(trial_dir, "behData", "images") for trial_dir in reversed(trial_dirs)]

    with open('sleap_dirs.txt', 'w') as f:
        for line in image_dirs: 
            f.write(f"{line}\n")

    # sleap script in  /mnt/labserver/BRAUN_Jonas/Other/sleap/data

    df = get_headless_df()
    trial_dirs = df.trial_dir.values
    image_dirs = [os.path.join(trial_dir, "behData", "images").replace("nas2", "data2") for trial_dir in reversed(trial_dirs)]

    with open('sleap_dirs_headless.txt', 'w') as f:
        for line in image_dirs:
            f.write(f"{line}\n")
"""
