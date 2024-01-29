"""
module for behavioural processing with functions for behaviour classification and behaviour onset detection.
Author: jonas.braun@epfl.ch
"""

import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle

from scipy.ndimage import gaussian_filter1d, gaussian_filter, median_filter

from twoppp.behaviour import synchronisation, fictrac
import twoppp.plot as myplt

import params

"""
beh_mapping = {
    0: 'background',
    1: 'walk',
    2: 'rest',
    3: 'back',
    4: 'groom',
    5: 'leg rub',
    6: 'posterior'
    }
"""
beh_mapping = [
    "background",
    "walk",
    "rest",
    "back",
    "groom",
    "leg rub",
    "posterior",
]
collapse_groom = True
n_beh = len(beh_mapping)
# beh_cmaplist = [myplt.WHITE, myplt.DARKGREEN, myplt.DARKBLUE, myplt.DARKCYAN, myplt.DARKRED, myplt.DARKORANGE, myplt.DARKPINK]
# here: grooming and leg rubbing is collapsed
beh_cmaplist = [
    myplt.WHITE,
    myplt.DARKGREEN,
    myplt.DARKBLUE,
    myplt.DARKCYAN,
    myplt.DARKRED,
    myplt.DARKRED,
    myplt.DARKPINK,
]

beh_cmap = mpl.colors.LinearSegmentedColormap.from_list(
    name="behaviour", colors=beh_cmaplist, N=256
)  # , N=6, )
beh_cbounds = np.linspace(-0.5, n_beh - 0.5, n_beh + 1)
beh_cnorm = mpl.colors.BoundaryNorm(beh_cbounds, ncolors=256)


def get_beh_color(beh_name):
    """
    Returns the color associated with a behavior label.
    If not one of the following: [walk, back, groom, rest, olfac]
    returns black

    Args:
        beh_name (str): The behavior label.

    Returns:
        str: The color code for the behavior label.
    """
    if beh_name == "walk":
        beh_color = myplt.DARKGREEN
    elif beh_name == "back":
        beh_color = myplt.DARKCYAN
    elif beh_name == "groom" or beh_name == "olfac":
        beh_color = myplt.DARKRED
    elif beh_name == "rest":
        beh_color = myplt.DARKBLUE
    else:
        beh_color = myplt.BLACK
    return beh_color


def get_beh_info_in_twop_df(
    beh_df,
    twop_df,
    keys=["silent", "fast", "me_all"],
    reduce_fns=[
        synchronisation.reduce_max_bool,
        synchronisation.reduce_max_bool,
        synchronisation.reduce_mean,
    ],
):
    """
    Retrieves behavior information from beh_df and adds it to twop_df.

    Args:
        beh_df (DataFrame): DataFrame containing behavior information.
        twop_df (DataFrame): DataFrame containing 2P imaging data.
        keys (list): List of behavior keys to extract from beh_df.
        reduce_fns (list): List of reduction functions to apply to behavior data.

    Returns:
        DataFrame: Updated twop_df with behavior information.
    """
    for key in keys:
        if key not in twop_df.keys():
            twop_df.insert(
                loc=len(twop_df.columns),
                column=key,
                value=np.zeros((len(twop_df)), dtype=beh_df[key].dtype),
            )

    for trial_name in np.unique(beh_df.index.get_level_values("TrialName")):
        beh_df_index = beh_df.index.get_level_values("TrialName") == trial_name
        twop_df_index = (
            twop_df.index.get_level_values("TrialName") == trial_name
        )
        signals_2p = []
        for key, reduce_fn in zip(keys, reduce_fns):
            signals_2p.append(
                synchronisation.reduce_during_2p_frame(
                    beh_df.loc[beh_df_index, "twop_index"].values,
                    beh_df.loc[beh_df_index, key].values,
                    reduce_fn,
                )
            )
        index_values = np.unique(beh_df.loc[beh_df_index, "twop_index"].values)
        index_values = index_values[index_values >= 0]
        min_index = np.min(index_values)
        max_index = np.max(index_values)
        assert len(index_values) == max_index - min_index + 1
        new_twop_df_index = np.logical_and.reduce(
            (
                twop_df.index.get_level_values("Frame").values >= min_index,
                twop_df.index.get_level_values("Frame").values <= max_index,
                twop_df.index.get_level_values("TrialName") == trial_name,
            )
        )
        for i_key, (key, signal_2p) in enumerate(zip(keys, signals_2p)):
            twop_df.loc[new_twop_df_index, key] = signal_2p

    return twop_df


def get_natural_beh_starts_trial(
    beh_df,
    beh="back",
    trial_name="",
    remove_laser=True,
    return_beh_prob=True,
    t_win=[-5, 10],
    min_dur=params.backwalk_min_dur,
    min_dist=params.backwalk_min_dist,
):
    """
    Get the start and stop frames of natural behavior events within a beh_df.

    Args:
        beh_df (DataFrame): DataFrame containing behavior information for one or multiple trials.
        beh (str): Behavior label of interest. Should be either of: back, walk, rest, groom, frub
        trial_name (str): Name of the trial. Not used
        remove_laser (bool): Whether to ignore spontaneous behaviours during laser stimulation events.
        return_beh_prob (bool): Whether to return behavior probability.
        t_win (list): Time window around behavior event extraction. Used when behaviour probability should be returned.
        min_dur (int): Minimum duration of a spontaneous behavior event.
        min_dist (int): Minimum distance between spontaneous behavior events.

    Returns:
        tuple: Tuple containing start and stop frames of behavior events.
    """
    if "v_forw" in beh_df.keys():
        v_forw_beh_filt = beh_df.v_forw.values
    else:
        v_forw_beh_filt = beh_df.v.values
    t_beh = beh_df.t.values
    fs_beh = params.fs_beh

    if (
        "back" in beh_df.keys()
        and "walk" in beh_df.keys()
        and "rest" in beh_df.keys()
        and "groom" in beh_df.keys()
        and "frub" in beh_df.keys()
    ):
        back_beh = beh_df.back.values
        rest_beh = beh_df.rest.values
        walk_beh = beh_df.walk.values
        groom_beh = beh_df.groom.values + beh_df.frub.values
    else:
        (
            walk_beh,
            rest_beh,
            back_beh,
            groom_beh,
            frub_beh,
            post_beh,
        ) = discriminate_beh(beh_df)
        groom_beh = groom_beh + frub_beh
    if beh == "back":
        beh_of_interest = back_beh
    elif beh == "walk":
        beh_of_interest = walk_beh
    elif beh == "rest":
        beh_of_interest = rest_beh
    elif beh == "groom":
        beh_of_interest = groom_beh
    else:
        raise NotImplementedError(
            f"Warning: behaviour of interest must be 'walk', 'back', 'groom', or 'rest', but was {beh}."
        )

    beh_starts = np.where(np.diff(beh_of_interest.astype(int)) == 1)[0]
    beh_stops = np.where(np.diff(beh_of_interest.astype(int)) == -1)[0]

    if len(beh_starts) == len(beh_stops) + 1:
        beh_starts = beh_starts[:-1]
    elif len(beh_starts) + 1 == len(beh_stops):
        beh_stops = beh_stops[1:]
    if remove_laser and "laser_stim" in beh_df.keys():
        to_keep = np.zeros_like(beh_starts).astype(bool)
        for i_s, (start, stop) in enumerate(zip(beh_starts, beh_stops)):
            if (
                not beh_df.laser_stim.values[start]
                and not beh_df.laser_stim.values[stop]
            ):
                to_keep[i_s] = True
        beh_starts = beh_starts[to_keep]
        beh_stops = beh_stops[to_keep]

    beh_dur = beh_stops - beh_starts
    beh_cum = np.array(
        [
            np.sum(v_forw_beh_filt[beh_start:beh_stop] / fs_beh)
            for beh_start, beh_stop in zip(beh_starts, beh_stops)
        ]
    )
    beh_pre = np.array(
        [
            np.sum(beh_of_interest[beh_start - min_dur : beh_start - 1])
            for beh_start in beh_starts
        ]
    )
    if beh == "back":
        beh_event_filter_strict = np.logical_and.reduce(
            (beh_pre == 0, beh_dur >= min_dur, beh_cum <= min_dist)
        )
    elif beh == "walk":
        beh_event_filter_strict = np.logical_and.reduce(
            (beh_pre == 0, beh_dur >= min_dur, beh_cum >= min_dist)
        )
    elif beh == "rest" or beh == "groom":
        beh_event_filter_strict = np.logical_and(
            beh_pre == 0, beh_dur >= min_dur
        )
    beh_starts_filt_strict = beh_starts[beh_event_filter_strict]
    beh_stops_filt_strict = beh_stops[beh_event_filter_strict]

    twop_starts_filt_strict = beh_df.twop_index.values[beh_starts_filt_strict]
    twop_stops_filt_strict = beh_df.twop_index.values[beh_stops_filt_strict]

    max_twop_index = beh_df.twop_index.values.max()

    # remove the ones in the very beginning and end of the trial where the denoising as removed 2p data
    twop_stops_filt_strict = twop_stops_filt_strict[
        twop_starts_filt_strict >= -1 * params.fs_int * t_win[0] + 4
    ]
    beh_starts_filt_strict = beh_starts_filt_strict[
        twop_starts_filt_strict >= -1 * params.fs_int * t_win[0] + 4
    ]
    beh_stops_filt_strict = beh_stops_filt_strict[
        twop_starts_filt_strict >= -1 * params.fs_int * t_win[0] + 4
    ]
    twop_starts_filt_strict = twop_starts_filt_strict[
        twop_starts_filt_strict >= -1 * params.fs_int * t_win[0] + 4
    ]

    twop_starts_filt_strict = twop_starts_filt_strict[
        twop_stops_filt_strict <= max_twop_index - params.fs_int * t_win[1] - 4
    ]
    beh_starts_filt_strict = beh_starts_filt_strict[
        twop_stops_filt_strict <= max_twop_index - params.fs_int * t_win[1] - 4
    ]
    beh_stops_filt_strict = beh_stops_filt_strict[
        twop_stops_filt_strict <= max_twop_index - params.fs_int * t_win[1] - 4
    ]
    twop_stops_filt_strict = twop_stops_filt_strict[
        twop_stops_filt_strict <= max_twop_index - params.fs_int * t_win[1] - 4
    ]

    if return_beh_prob and len(twop_stops_filt_strict):
        i_win = (fs_beh * np.array(t_win)).astype(int)
        beh_df_aligned = beh_df[
            beh_starts_filt_strict[0]
            + i_win[0] : beh_starts_filt_strict[0]
            + i_win[1]
        ][["t", "twop_index"]]
        beh_df_aligned["t"] -= beh_df_aligned.iloc[-i_win[0]]["t"]
        beh_df_aligned["twop_index"] -= int(
            beh_df_aligned.iloc[-i_win[0]]["twop_index"]
        )
        beh_df_aligned["back"] = np.zeros((len(beh_df_aligned)))
        beh_df_aligned["walk"] = np.zeros((len(beh_df_aligned)))
        beh_df_aligned["rest"] = np.zeros((len(beh_df_aligned)))
        beh_df_aligned["v_forw"] = np.zeros((len(beh_df_aligned)))

        for start in beh_starts_filt_strict:
            beh_df_aligned["back"] += back_beh[
                start + i_win[0] : start + i_win[1]
            ]
            beh_df_aligned["rest"] += rest_beh[
                start + i_win[0] : start + i_win[1]
            ]
            beh_df_aligned["walk"] += walk_beh[
                start + i_win[0] : start + i_win[1]
            ]
            beh_df_aligned["v_forw"] += v_forw_beh_filt[
                start + i_win[0] : start + i_win[1]
            ]

    elif not len(twop_stops_filt_strict):
        beh_df_aligned = None
    if return_beh_prob:
        return (
            twop_starts_filt_strict,
            twop_stops_filt_strict,
            beh_starts_filt_strict,
            beh_stops_filt_strict,
            beh_df_aligned,
        )
    else:
        return (
            twop_starts_filt_strict,
            twop_stops_filt_strict,
            beh_starts_filt_strict,
            beh_stops_filt_strict,
        )


def get_beh_trigger_into_dfs(twop_df, beh_df, beh="back"):
    """
    Add behavior trigger information to both twop_df and beh_df DataFrames.

    Args:
        twop_df (DataFrame): DataFrame containing 2P imaging data.
        beh_df (DataFrame): DataFrame containing behavior information.
        beh (str): Behavior label of interest.

    Returns:
        tuple: Updated twop_df and beh_df with behavior trigger information.
    """
    if beh == "back":
        min_dur = params.backwalk_min_dur
        min_dist = params.backwalk_min_dist
    elif beh == "walk":
        min_dur = params.walk_min_dur
        min_dist = params.walk_min_dist
    elif beh == "rest":
        min_dur = params.rest_min_dur
        min_dist = params.rest_min_dist
    elif beh == "groom":
        min_dur = params.groom_min_dur
        min_dist = params.groom_min_dist
    else:
        raise NotImplementedError(
            f"Warning: behaviour of interest must be 'walk', 'back', or 'rest', but was {beh}."
        )

    if twop_df is not None:
        twop_df.insert(
            loc=len(twop_df.columns),
            column=f"{beh}_trig_start",
            value=np.zeros((len(twop_df)), dtype=bool),
        )
        twop_df.insert(
            loc=len(twop_df.columns),
            column=f"{beh}_trig_stop",
            value=np.zeros((len(twop_df)), dtype=bool),
        )
        twop_df.insert(
            loc=len(twop_df.columns),
            column=f"{beh}_trig",
            value=np.zeros((len(twop_df)), dtype=bool),
        )
    beh_df.insert(
        loc=len(beh_df.columns),
        column=f"{beh}_trig_start",
        value=np.zeros((len(beh_df)), dtype=bool),
    )
    beh_df.insert(
        loc=len(beh_df.columns),
        column=f"{beh}_trig_stop",
        value=np.zeros((len(beh_df)), dtype=bool),
    )
    beh_df.insert(
        loc=len(beh_df.columns),
        column=f"{beh}_trig",
        value=np.zeros((len(beh_df)), dtype=bool),
    )

    for trial_name in np.unique(beh_df.index.get_level_values("TrialName")):
        beh_df_index = beh_df.index.get_level_values("TrialName") == trial_name
        if twop_df is not None:
            twop_df_index = (
                twop_df.index.get_level_values("TrialName") == trial_name
            )

        starts, stops, beh_starts, beh_stops = get_natural_beh_starts_trial(
            beh_df.loc[beh_df_index],
            beh=beh,
            remove_laser=True,
            return_beh_prob=False,
            t_win=[-5, 10],
            min_dur=min_dur,
            min_dist=min_dist,
        )
        for start, stop, beh_start, beh_stop in zip(
            starts, stops, beh_starts, beh_stops
        ):
            new_beh_df_index_start = np.logical_and(
                beh_df_index,
                beh_df.index.get_level_values("Frame") == beh_start,
            )
            beh_df.loc[new_beh_df_index_start, f"{beh}_trig_start"] = True
            new_beh_df_index_stop = np.logical_and(
                beh_df_index,
                beh_df.index.get_level_values("Frame") == beh_stop,
            )
            beh_df.loc[new_beh_df_index_stop, f"{beh}_trig_stop"] = True
            new_beh_df_index_start = np.argwhere(new_beh_df_index_start)[0][0]
            new_beh_df_index_stop = np.argwhere(new_beh_df_index_stop)[0][0]
            beh_df.loc[
                new_beh_df_index_start:new_beh_df_index_stop, f"{beh}_trig"
            ] = True

            if twop_df is not None:
                new_twop_df_index_start = np.logical_and(
                    twop_df_index,
                    twop_df.index.get_level_values("Frame") == start,
                )
                twop_df.loc[
                    new_twop_df_index_start, f"{beh}_trig_start"
                ] = True
                new_twop_df_index_stop = np.logical_and(
                    twop_df_index,
                    twop_df.index.get_level_values("Frame") == stop,
                )
                twop_df.loc[new_twop_df_index_stop, f"{beh}_trig_stop"] = True
                new_twop_df_index_start = np.argwhere(new_twop_df_index_start)[
                    0
                ][0]
                new_twop_df_index_stop = np.argwhere(new_twop_df_index_stop)[
                    0
                ][0]
                twop_df.loc[
                    new_twop_df_index_start:new_twop_df_index_stop,
                    f"{beh}_trig",
                ] = True
    return twop_df, beh_df


def discriminate_beh(
    beh_df=None,
    v=None,
    v_forw=None,
    me_q=None,
    mef_q=None,
    mem_q=None,
    meb_q=None,
    method=params.beh_class_method,
):
    """
    Discriminates behavioral states such as 'walk', 'rest', 'back', 'groom', 'frub', and 'post' based on various features.

    Parameters:
    - beh_df (DataFrame): DataFrame containing behavioral data (optional).
    - v (numpy.ndarray): Velocity data (if beh_df is not provided).
    - v_forw (numpy.ndarray): Forward velocity data (if beh_df is not provided).
    - me_q (numpy.ndarray): Motion energy data (if beh_df is not provided).
    - mef_q (numpy.ndarray): Front motion energy data (if beh_df is not provided).
    - mem_q (numpy.ndarray): Mid motion energy data (if beh_df is not provided).
    - meb_q (numpy.ndarray): Back motion energy data (if beh_df is not provided).
    - method (str): Method for behavior discrimination, either "motionenergy" or "sleap" (default is params.beh_class_method).

    Returns:
    - walk (numpy.ndarray): Boolean array indicating 'walk' behavior.
    - rest (numpy.ndarray): Boolean array indicating 'rest' behavior.
    - back (numpy.ndarray): Boolean array indicating 'back' behavior.
    - groom (numpy.ndarray): Boolean array indicating 'groom' behavior.
    - frub (numpy.ndarray): Boolean array indicating 'frub' behavior.
    - post (numpy.ndarray): Boolean array indicating 'post' behavior.
    """
    wheel = False
    if beh_df is not None:
        v = beh_df.v.values
        if (
            "v_forw" not in beh_df.keys()
            and "delta_rot_lab_forward" in beh_df.keys()
        ):
            v_forw = fictrac.filter_fictrac(
                beh_df["delta_rot_lab_forward"], 5, 10
            )
        elif "v_forw" not in beh_df.keys():  # wheel trials
            wheel = True
            v_forw = v.copy()
            v = np.abs(v_forw)

        else:
            v_forw = beh_df.v_forw.values
    if method == "motionenergy":
        me_q = beh_df.me_all_q.values
        mef_q = beh_df.me_front_q.values
        mem_q = beh_df.me_mid_q.values
        meb_q = beh_df.me_back_q.values
        rest = get_rest_me(v, v_forw, me_q, mef_q, mem_q, meb_q)
        walk = get_walk_me(v, v_forw, me_q, mef_q, mem_q, meb_q)
        back = get_back_me(v, v_forw, me_q, mef_q, mem_q, meb_q)
        groom = np.logical_and(
            get_groom_me(v, v_forw, me_q, mef_q, mem_q, meb_q),
            np.logical_not(rest),
        )
        frub = np.logical_and(
            get_frub_me(v, v_forw, me_q, mef_q, mem_q, meb_q),
            np.logical_not(rest),
        )
        post = np.logical_and(
            get_post_me(v, v_forw, me_q, mef_q, mem_q, meb_q),
            np.logical_not(rest),
        )
    elif method == "sleap":
        frleg_height = beh_df.frleg_height.values
        mef_tita = beh_df.mef_tita.values
        mem_tita = beh_df.mem_tita.values
        meh_tita = beh_df.meh_tita.values
        rest = get_rest_sleap(
            v, v_forw, frleg_height, mef_tita, mem_tita, meh_tita
        )
        walk = get_walk_sleap(
            v, v_forw, frleg_height, mef_tita, mem_tita, meh_tita
        )
        back = get_back_sleap(
            v, v_forw, frleg_height, mef_tita, mem_tita, meh_tita
        )
        groom = get_groom_sleap(
            v, v_forw, frleg_height, mef_tita, mem_tita, meh_tita
        )
        frub = get_frub_sleap(
            v, v_forw, frleg_height, mef_tita, mem_tita, meh_tita
        )
        post = get_post_sleap(
            v, v_forw, frleg_height, mef_tita, mem_tita, meh_tita
        )
    else:
        raise NotImplementedError
    return walk, rest, back, groom, frub, post


def get_beh_me(v, v_forw, me_q, mef_q, mem_q, meb_q, thr, sign):
    """
    Determines behavioral states based on motion energy data and thresholds.

    Parameters:
    - v (numpy.ndarray): Velocity data.
    - v_forw (numpy.ndarray): Forward velocity data.
    - me_q (numpy.ndarray): Motion energy data.
    - mef_q (numpy.ndarray): Front motion energy data.
    - mem_q (numpy.ndarray): Mid motion energy data.
    - meb_q (numpy.ndarray): Back motion energy data.
    - thr (list): Threshold values for each feature.
    - sign (list): Comparison functions for each feature (e.g., np.greater, np.less_equal).

    Returns:
    - beh (numpy.ndarray): Boolean array indicating behavioral states.
    """
    beh = np.logical_and.reduce(
        (
            sign[0](v, thr[0]),
            sign[1](v_forw, thr[1]),
            sign[2](me_q, thr[2]),
            sign[3](mef_q, thr[3]),
            sign[4](mem_q, thr[4]),
            sign[5](meb_q, thr[5]),
        )
    )
    return beh


def get_rest_me(
    v,
    v_forw,
    me_q,
    mef_q,
    mem_q,
    meb_q,
    thr=[1, 1, 0.33, 0.33, -1 * np.inf, -1 * np.inf],
    sign=[
        np.less_equal,
        np.less_equal,
        np.less_equal,
        np.less_equal,
        np.greater,
        np.greater,
    ],
):
    return get_beh_me(v, v_forw, me_q, mef_q, mem_q, meb_q, thr, sign)


def get_walk_me(
    v,
    v_forw,
    me_q,
    mef_q,
    mem_q,
    meb_q,
    thr=[1, 1, -1 * np.inf, -1 * np.inf, -1 * np.inf, -1 * np.inf],
    sign=[
        np.greater,
        np.greater,
        np.greater,
        np.greater,
        np.greater,
        np.greater,
    ],
):
    return get_beh_me(v, v_forw, me_q, mef_q, mem_q, meb_q, thr, sign)


def get_back_me(
    v,
    v_forw,
    me_q,
    mef_q,
    mem_q,
    meb_q,
    thr=[1, -1, -1 * np.inf, -1 * np.inf, -1 * np.inf, -1 * np.inf],
    sign=[
        np.greater,
        np.less_equal,
        np.greater,
        np.greater,
        np.greater,
        np.greater,
    ],
):
    return get_beh_me(v, v_forw, me_q, mef_q, mem_q, meb_q, thr, sign)


def get_groom_me(
    v,
    v_forw,
    me_q,
    mef_q,
    mem_q,
    meb_q,
    thr=[1, 1, np.inf, 0.33, 0.33, np.inf],
    sign=[
        np.less_equal,
        np.less_equal,
        np.less_equal,
        np.greater,
        np.less_equal,
        np.less_equal,
    ],
):
    return get_beh_me(v, v_forw, me_q, mef_q, mem_q, meb_q, thr, sign)


def get_frub_me(
    v,
    v_forw,
    me_q,
    mef_q,
    mem_q,
    meb_q,
    thr=[1, 1, np.inf, 0.33, 0.33, np.inf],
    sign=[
        np.less_equal,
        np.less_equal,
        np.less_equal,
        np.less_equal,
        np.greater,
        np.less_equal,
    ],
):
    return get_beh_me(v, v_forw, me_q, mef_q, mem_q, meb_q, thr, sign)


def get_post_me(
    v,
    v_forw,
    me_q,
    mef_q,
    mem_q,
    meb_q,
    thr=[1, 1, np.inf, 0.33, 0.33, 0.33],
    sign=[
        np.less_equal,
        np.less_equal,
        np.less_equal,
        np.less_equal,
        np.less_equal,
        np.greater,
    ],
):
    return get_beh_me(v, v_forw, me_q, mef_q, mem_q, meb_q, thr, sign)


def get_beh_sleap(
    v, v_forw, frleg_height, mef_tita, mem_tita, meh_tita, thr, sign
):
    """
    Determines behavioral states based on Sleap features and thresholds.

    Parameters:
    - v (numpy.ndarray): Velocity data.
    - v_forw (numpy.ndarray): Forward velocity data.
    - frleg_height (numpy.ndarray): Front leg height feature from Sleap.
    - mef_tita (numpy.ndarray): Front leg motion energy feature from Sleap.
    - mem_tita (numpy.ndarray): Mid leg motion energy feature from Sleap.
    - meh_tita (numpy.ndarray): Hind leg motion energy feature from Sleap.
    - thr (list): Threshold values for each feature.
    - sign (list): Comparison functions for each feature (e.g., np.greater, np.less_equal).

    Returns:
    - beh (numpy.ndarray): Boolean array indicating behavioral states.
    """
    beh = np.logical_and.reduce(
        (
            sign[0](v, thr[0]),
            sign[1](v_forw, thr[1]),
            sign[2](frleg_height, thr[2]),
            sign[3](mef_tita, thr[3]),
            sign[4](mem_tita, thr[4]),
            sign[5](meh_tita, thr[5]),
        )
    )
    return beh


def get_rest_sleap(
    v,
    v_forw,
    frleg_height,
    mef_tita,
    mem_tita,
    meh_tita,
    thr=[0.75, 0.75, 0, 0.75, 0.75, 0.75],
    sign=[
        np.less_equal,
        np.less_equal,
        np.greater,
        np.less_equal,
        np.less_equal,
        np.less_equal,
    ],
):
    return get_beh_sleap(
        v, v_forw, frleg_height, mef_tita, mem_tita, meh_tita, thr, sign
    )


def get_walk_sleap(
    v,
    v_forw,
    frleg_height,
    mef_tita,
    mem_tita,
    meh_tita,
    thr=[1, 1, -1 * np.inf, -1 * np.inf, -1 * np.inf, -1 * np.inf],
    sign=[
        np.greater,
        np.greater,
        np.greater,
        np.greater,
        np.greater,
        np.greater,
    ],
):
    return get_beh_sleap(
        v, v_forw, frleg_height, mef_tita, mem_tita, meh_tita, thr, sign
    )


def get_back_sleap(
    v,
    v_forw,
    frleg_height,
    mef_tita,
    mem_tita,
    meh_tita,
    thr=[1, -1, -1 * np.inf, -1 * np.inf, -1 * np.inf, -1 * np.inf],
    sign=[
        np.greater,
        np.less_equal,
        np.greater,
        np.greater,
        np.greater,
        np.greater,
    ],
):
    return get_beh_sleap(
        v, v_forw, frleg_height, mef_tita, mem_tita, meh_tita, thr, sign
    )


def get_groom_sleap(
    v,
    v_forw,
    frleg_height,
    mef_tita,
    mem_tita,
    meh_tita,
    thr=[0.75, 0.75, 0, np.inf, 0.75, 0.75],
    sign=[
        np.less_equal,
        np.less_equal,
        np.less_equal,
        np.less_equal,
        np.less_equal,
        np.less_equal,
    ],
):
    return get_beh_sleap(
        v, v_forw, frleg_height, mef_tita, mem_tita, meh_tita, thr, sign
    )


def get_frub_sleap(
    v,
    v_forw,
    frleg_height,
    mef_tita,
    mem_tita,
    meh_tita,
    thr=[0.75, 0.75, 0, 0.75, 0.75, 0.75],
    sign=[
        np.less_equal,
        np.less_equal,
        np.greater,
        np.greater,
        np.less_equal,
        np.less_equal,
    ],
):
    return get_beh_sleap(
        v, v_forw, frleg_height, mef_tita, mem_tita, meh_tita, thr, sign
    )


def get_post_sleap(
    v,
    v_forw,
    frleg_height,
    mef_tita,
    mem_tita,
    meh_tita,
    thr=[0.75, 0.75, 0, 0.75, np.inf, 0.75],
    sign=[
        np.less_equal,
        np.less_equal,
        np.greater,
        np.less_equal,
        np.less_equal,
        np.greater,
    ],
):
    return get_beh_sleap(
        v, v_forw, frleg_height, mef_tita, mem_tita, meh_tita, thr, sign
    )


def reduce_behaviour_class(values, thres=0.67, default=0):
    """
    Reduces behavioral class values to the most frequent class within a time bin.

    Parameters:
    - values (numpy.ndarray): Array of behavioral class values.
    - thres (float): Threshold for class reduction (default is 0.67).
    - default: Default class value (default is 0).

    Returns:
    - reduced_class (int): Reduced behavioral class value.
    """
    return synchronisation.reduce_most_freq(
        values=values, thres=thres, default=default
    )


def add_beh_class_to_dfs(twop_df, beh_df, allow_exceptions=False):
    """
    Adds behavioral class information to twop_df and beh_df DataFrames.

    Parameters:
    - twop_df (DataFrame): 2-photon imaging data DataFrame.
    - beh_df (DataFrame): Behavioral data DataFrame.
    - allow_exceptions (bool): Whether to allow exceptions in behaviour classification.
      If exception, replace all classifications by 0.

    Returns:
    - twop_df (DataFrame): Updated 2-photon imaging data DataFrame.
    - beh_df (DataFrame): Updated behavioral data DataFrame.
    """
    if twop_df is not None:
        if not "walk" in twop_df.keys():
            twop_df.insert(
                loc=len(twop_df.columns),
                column="walk",
                value=np.zeros((len(twop_df)), dtype=bool),
            )
        if not "rest" in twop_df.keys():
            twop_df.insert(
                loc=len(twop_df.columns),
                column="rest",
                value=np.zeros((len(twop_df)), dtype=bool),
            )
        if not "back" in twop_df.keys():
            twop_df.insert(
                loc=len(twop_df.columns),
                column="back",
                value=np.zeros((len(twop_df)), dtype=bool),
            )
        twop_df.insert(
            loc=len(twop_df.columns),
            column="groom",
            value=np.zeros((len(twop_df)), dtype=bool),
        )
        twop_df.insert(
            loc=len(twop_df.columns),
            column="frub",
            value=np.zeros((len(twop_df)), dtype=bool),
        )
        twop_df.insert(
            loc=len(twop_df.columns),
            column="post",
            value=np.zeros((len(twop_df)), dtype=bool),
        )
        twop_df.insert(
            loc=len(twop_df.columns),
            column="beh_class",
            value=np.zeros((len(twop_df)), dtype=int),
        )
    if not "walk" in beh_df.keys():
        beh_df.insert(
            loc=len(beh_df.columns),
            column="walk",
            value=np.zeros((len(beh_df)), dtype=bool),
        )
    if not "rest" in beh_df.keys():
        beh_df.insert(
            loc=len(beh_df.columns),
            column="rest",
            value=np.zeros((len(beh_df)), dtype=bool),
        )
    if not "back" in beh_df.keys():
        beh_df.insert(
            loc=len(beh_df.columns),
            column="back",
            value=np.zeros((len(beh_df)), dtype=bool),
        )
    beh_df.insert(
        loc=len(beh_df.columns),
        column="groom",
        value=np.zeros((len(beh_df)), dtype=bool),
    )
    beh_df.insert(
        loc=len(beh_df.columns),
        column="frub",
        value=np.zeros((len(beh_df)), dtype=bool),
    )
    beh_df.insert(
        loc=len(beh_df.columns),
        column="post",
        value=np.zeros((len(beh_df)), dtype=bool),
    )
    beh_df.insert(
        loc=len(beh_df.columns),
        column="beh_class",
        value=np.zeros((len(beh_df)), dtype=int),
    )

    for trial_name in np.unique(beh_df.index.get_level_values("TrialName")):
        beh_df_index = beh_df.index.get_level_values("TrialName") == trial_name
        if twop_df is not None:

            twop_df_index = twop_df.index.get_level_values("TrialName") == trial_name
        if allow_exceptions:
            try:
                walk, rest, back, groom, frub, post = discriminate_beh(beh_df=beh_df.loc[beh_df_index])
            except AttributeError:
                print("Warning: could not run behaviour classification")
                walk = np.zeros(len(beh_df.loc[beh_df_index]))
                rest = np.zeros(len(beh_df.loc[beh_df_index]))
                back = np.zeros(len(beh_df.loc[beh_df_index]))
                groom = np.zeros(len(beh_df.loc[beh_df_index]))
                frub = np.zeros(len(beh_df.loc[beh_df_index]))
                post = np.zeros(len(beh_df.loc[beh_df_index]))
        else:
            walk, rest, back, groom, frub, post = discriminate_beh(beh_df=beh_df.loc[beh_df_index])

        beh_class = walk + 2*rest + 3*back + 4*groom + 5*frub + 6*post

        beh_df.loc[beh_df_index,"walk"] = walk
        beh_df.loc[beh_df_index,"rest"] = rest
        beh_df.loc[beh_df_index,"back"] = back
        beh_df.loc[beh_df_index,"groom"] = groom
        beh_df.loc[beh_df_index,"frub"] = frub
        beh_df.loc[beh_df_index,"post"] = post
        beh_df.loc[beh_df_index,"beh_class"] = beh_class

        if twop_df is not None:
            beh_class_2p = synchronisation.reduce_during_2p_frame(
                twop_index=beh_df.loc[beh_df_index, "twop_index"].values,
                values=beh_class,
                function=reduce_behaviour_class,
            )
            index_values = np.unique(
                beh_df.loc[beh_df_index, "twop_index"].values
            )
            index_values = index_values[index_values >= 0]
            min_index = np.min(index_values)
            max_index = np.max(index_values)
            assert len(index_values) == max_index - min_index + 1
            new_twop_df_index = np.logical_and.reduce(
                (
                    twop_df.index.get_level_values("Frame").values
                    >= min_index,
                    twop_df.index.get_level_values("Frame").values
                    <= max_index,
                    twop_df.index.get_level_values("TrialName") == trial_name,
                )
            )
            twop_df.loc[new_twop_df_index, "walk"] = beh_class_2p == 1
            twop_df.loc[new_twop_df_index, "rest"] = beh_class_2p == 2
            twop_df.loc[new_twop_df_index, "back"] = beh_class_2p == 3
            twop_df.loc[new_twop_df_index, "groom"] = beh_class_2p == 4
            twop_df.loc[new_twop_df_index, "frub"] = beh_class_2p == 5
            twop_df.loc[new_twop_df_index, "post"] = beh_class_2p == 6
            twop_df.loc[new_twop_df_index, "beh_class"] = beh_class_2p

    return twop_df, beh_df


def discriminate_walk_rest_pre_stim(
    v, v_forw, me_q, thr_rest=[1, 1, 0.33], thr_walk=[0, 1, 0]
):
    """
    Discriminates 'walk' and 'rest' states before stimulation based on velocity and motion energy features.

    Parameters:
    - v (numpy.ndarray): Velocity data.
    - v_forw (numpy.ndarray): Forward velocity data.
    - me_q (numpy.ndarray): Motion energy data.
    - thr_rest (list): Threshold values for 'rest' state discrimination (default is [1, 1, 0.33]).
    - thr_walk (list): Threshold values for 'walk' state discrimination (default is [0, 1, 0]).

    Returns:
    - walk (numpy.ndarray): Boolean array indicating 'walk' state.
    - rest (numpy.ndarray): Boolean array indicating 'rest' state.
    """
    rest = np.logical_and.reduce(
        (v <= thr_rest[0], v_forw <= thr_rest[1], me_q <= thr_rest[2])
    )
    walk = np.logical_and.reduce(
        (v > thr_walk[0], v_forw > thr_walk[1], me_q > thr_walk[2])
    )
    return walk, rest


def get_pre_stim_beh(
    beh_df,
    trigger="laser_start",
    stim_p=[10, 20],
    n_pre_stim=100,
    trials=None,
    return_starts=False,
):  # TODO: look at behavioural classification instead
    """
    Obtains 'walk' and 'rest' states before stimulation based on behavioral data.

    Parameters:
    - beh_df (DataFrame): Behavioral data DataFrame.
    - trigger (str): Trigger signal for stimulation (default is "laser_start").
    - stim_p (list): List of stimulation powers (default is [10,20]).
    - n_pre_stim (int): Number of time points before stimulation (default is 100).
    - trials (list): List of trial names to consider (optional).
    - return_starts (bool): Whether to return trigger indices (default is False).

    Returns:
    - walk_pre (numpy.ndarray): Boolean array indicating 'walk' state before stimulation.
    - rest_pre (numpy.ndarray): Boolean array indicating 'rest' state before stimulation.
    - beh_starts (list): List of trigger indices (if return_starts is True).
    """
    fs_int = int(1 / np.mean(np.diff(beh_df.t.values[:1000])))
    if trigger in beh_df.keys():
        beh_starts = np.where(beh_df[trigger])[0]
    else:
        beh_starts = []
    # TODO: check these filters
    if trials is not None:
        beh_starts = [
            start
            for start in beh_starts
            if any(
                [
                    beh_df.iloc[start].name[3].startswith(trial[:3])
                    for trial in trials
                ]
            )
        ]
    if "laser" in trigger and stim_p is not None:
        try:
            beh_starts = [
                start
                for start in beh_starts
                if any(
                    [
                        int(beh_df.iloc[start + fs_int].laser_power_uW) == p
                        for p in stim_p
                    ]
                )
            ]
        except:
            try:
                print(
                    f"will look at laser_power_mW instead of laser_power_uW. unique values are: {np.unique(twop_df.laser_power_mW)}"
                )
                beh_starts = [
                    start
                    for start in beh_starts
                    if any(
                        [
                            int(beh_df.iloc[start + fs_int].laser_power_mW)
                            == p
                            for p in stim_p
                        ]
                    )
                ]
            except:
                print(
                    f"WARNING: laser_power_mW and laser_power_uW were not found. Will use laser_power."
                )
                stim_power_uW = stimulation.get_stim_p_uW(
                    beh_df.laser_power.values
                )
                print(f"Converted to uW: {np.unique(stim_power_uW)}")
                beh_starts = [
                    start
                    for start in beh_starts
                    if any(
                        [
                            int(stim_power_uW[start + fs_int]) == p
                            for p in stim_p
                        ]
                    )
                ]

    # make sure that the ones that might be too early or too late are ignored.
    beh_starts = [
        start
        for start in beh_starts
        if all(
            [
                start - params.response_t_params_beh[0] > 0,
                start + np.sum(params.response_t_params_beh[1:]) < len(beh_df),
            ]
        )
    ]
    # if "olfac" in trigger and olfac_cond is not None:
    rest_pre = np.zeros_like(beh_starts, dtype=bool)
    walk_pre = np.zeros_like(beh_starts, dtype=bool)

    v = beh_df.v.values
    if (
        "v_forw" not in beh_df.keys()
        and "delta_rot_lab_forward" in beh_df.keys()
    ):
        v_forw = fictrac.filter_fictrac(beh_df["delta_rot_lab_forward"], 5, 10)
    elif "v_forw" not in beh_df.keys():  # wheel trials
        wheel = True
        v_forw = v.copy()
        v = np.abs(v_forw)
    else:
        v_forw = beh_df.v_forw.values

    for i_s, beh_start in enumerate(beh_starts):
        walk_pre[i_s], rest_pre[i_s] = discriminate_walk_rest_pre_stim(
            v=np.mean(v[beh_start - n_pre_stim : beh_start]),
            v_forw=np.mean(v_forw[beh_start - n_pre_stim : beh_start]),
            me_q=np.mean(
                beh_df.me_all_q.iloc[beh_start - n_pre_stim : beh_start]
            ),
        )
    if return_starts:
        return walk_pre, rest_pre, beh_starts
    else:
        return walk_pre, rest_pre
