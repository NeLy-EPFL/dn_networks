"""
Module to analyse stimulations. e.g., getting stimulation starts, aligning variables around stimulation, ...
Author: jonas.braun@epfl.ch
"""
import os
import sys
import numpy as np
import pandas as pd

from twoppp import utils
from twoppp.behaviour import stimulation as twoppp_stimulation

sys.path.append(os.path.dirname(__file__))
import params

def fix_stim_power_signal(df):
    """
    Fixes a backwards incompatibility issue where the 'laser_power_uW' signal was missing in older versions of `twoppp`.

    Parameters
    ----------
    df : DataFrame
        The DataFrame containing the data.

    Returns
    -------
    df : DataFrame
        The DataFrame with the 'laser_power_uW' signal added or modified.
    """
    if "laser_power_uW" not in df.keys():
        if "laser_power_mW" in df.keys():
            print(f"WARNING: laser_power_uW was not found. Will use laser_power_mW. Unique values are: {np.unique(df.laser_power_mW.values)}")
            df["laser_power_uW"] = df.laser_power_mW
        elif "laser_power" in df.keys():
            print(f"WARNING: laser_power_mW and laser_power_uW were not found. Will use laser_power.")
            df["laser_power_uW"] = twoppp_stimulation.get_stim_p_uW(df.laser_power.values)
            print(f"Unique values are now: {np.unique(df.laser_power_uW.values)}")
        else:
            print(f"WARNING: laser_power, laser_power_mW, laser_power_uW were not found.")
    return df

def get_stim_starts(df, stim_signals, stim_levels, stim_level_signals, return_stops=False):
    """
    Retrieves the stimulation start times based on specified stimulation signals and levels.

    Parameters
    ----------
    df : DataFrame
        The DataFrame containing the data.
    stim_signals : str or list of str
        Stimulation signal(s) to detect.
    stim_levels : int, list of int, or None
        Stimulation level(s) to consider. Use None if not applicable.
    stim_level_signals : str, list of str, or None
        Stimulation level signal(s) to consider. Use None if not applicable.
    return_stops : bool, optional
        Whether to also return the stimulation stop times.

    Returns
    -------
    stim_starts : array
        Array containing the stimulation start times.
    stim_stops : array, optional
        Array containing the stimulation stop times if return_stops=True.
    """
    if not isinstance(stim_signals, list):
        stim_signals = [stim_signals]
    if not isinstance(stim_level_signals, list):
        stim_level_signals = [stim_level_signals]
    if not isinstance(stim_signals, list) or len(stim_levels) != len(stim_signals):
        stim_levels = [stim_levels]
    stim = np.zeros((len(df)), dtype=int)
    for stim_signal, stim_level, stim_level_signal in zip(stim_signals, stim_levels, stim_level_signals):
        if stim_signal in df.keys():
            this_stim = np.nan_to_num(df[stim_signal].values)
            if stim_level is not None:
                correct_stim_levels = np.logical_or.reduce([np.nan_to_num(df[stim_level_signal].values) == this_stim_level for this_stim_level in stim_level])
                this_stim = np.logical_and(this_stim, correct_stim_levels)
            stim += this_stim.astype(int)
            # TODO: check how this handles the case where some trials have 'olfac_stim' and others not
        else:
            print(f"could not find {stim_signal} in {' '.join([str(df.index.get_level_values(i)[0]) for i in range(4)])}")
    stim = stim > 0
    stim_starts = np.where(np.diff(stim.astype(int)) == 1)[0]
    stim_stops = np.where(np.diff(stim.astype(int)) == -1)[0]

    if return_stops:
        return stim_starts, stim_stops
    else:
        return stim_starts

def get_olfac_stim_starts(df, return_stops=False):
    """
    Retrieves the start times of olfactory stimulation.

    Parameters
    ----------
    df : DataFrame
        The DataFrame containing the data.
    return_stops : bool, optional
        Whether to also return the olfactory stimulation stop times.

    Returns
    -------
    olfac_stim_starts : array
        Array containing the olfactory stimulation start times.
    olfac_stim_stops : array, optional
        Array containing the olfactory stimulation stop times if return_stops=True.
    """
    return get_stim_starts(df, stim_signals=["olfac_stim"], stim_levels=[None], stim_level_signals=None, return_stops=return_stops)

def get_laser_stim_starts(df, stim_levels, return_stops=False):
    """
    Retrieves the start times of laser stimulation with specified power levels.

    Parameters
    ----------
    df : DataFrame
        The DataFrame containing the data.
    stim_levels : int or list of int
        Laser stimulation power levels to consider.
    return_stops : bool, optional
        Whether to also return the laser stimulation stop times.

    Returns
    -------
    laser_stim_starts : array
        Array containing the laser stimulation start times.
    laser_stim_stops : array, optional
        Array containing the laser stimulation stop times if return_stops=True.
    """
    return get_stim_starts(df, stim_signals=["laser_stim"], stim_levels=[stim_levels], stim_level_signals=["laser_power_uW"], return_stops=return_stops)

def get_laser_stim_starts_10uW(df, return_stops=False):
    """
    Retrieves the start times of laser stimulation with 10uW power.

    Parameters
    ----------
    df : DataFrame
        The DataFrame containing the data.
    return_stops : bool, optional
        Whether to also return the laser stimulation stop times.

    Returns
    -------
    laser_stim_starts : array
        Array containing the laser stimulation start times.
    laser_stim_stops : array, optional
        Array containing the laser stimulation stop times if return_stops=True.
    """
    return get_stim_starts(df, stim_signals=["laser_stim"], stim_levels=[[10]], stim_level_signals=["laser_power_uW"], return_stops=return_stops)

def get_laser_stim_starts_20uW(df, return_stops=False):
    """
    Retrieves the start times of laser stimulation with 20uW power.

    Parameters
    ----------
    df : DataFrame
        The DataFrame containing the data.
    return_stops : bool, optional
        Whether to also return the laser stimulation stop times.

    Returns
    -------
    laser_stim_starts : array
        Array containing the laser stimulation start times.
    laser_stim_stops : array, optional
        Array containing the laser stimulation stop times if return_stops=True.
    """
    return get_stim_starts(df, stim_signals=["laser_stim"], stim_levels=[[20]], stim_level_signals=["laser_power_uW"], return_stops=return_stops)

def get_laser_stim_starts_10_20uW(df, return_stops=False):
    """
    Retrieves the start times of laser stimulation with 10 or 20uW power.

    Parameters
    ----------
    df : DataFrame
        The DataFrame containing the data.
    return_stops : bool, optional
        Whether to also return the laser stimulation stop times.

    Returns
    -------
    laser_stim_starts : array
        Array containing the laser stimulation start times.
    laser_stim_stops : array, optional
        Array containing the laser stimulation stop times if return_stops=True.
    """
    return get_stim_starts(df, stim_signals=["laser_stim"], stim_levels=[[10, 20]], stim_level_signals=["laser_power_uW"], return_stops=return_stops)

def get_laser_stim_starts_all(df, return_stops=False):
    """
    Retrieves the start times of laser stimulation irrespective of power level.

    Parameters
    ----------
    df : DataFrame
        The DataFrame containing the data.
    return_stops : bool, optional
        Whether to also return the laser stimulation stop times.

    Returns
    -------
    laser_stim_starts : array
        Array containing the laser stimulation start times.
    laser_stim_stops : array, optional
        Array containing the laser stimulation stop times if return_stops=True.
    """
    return get_stim_starts(df, stim_signals=["laser_stim"], stim_levels=[None], stim_level_signals=[None], return_stops=return_stops)

def get_all_stim_starts(df, return_stops=False):
    """
    Retrieves all start times of laser stimulation and olfactory stimulation.

    Parameters
    ----------
    df : DataFrame
        The DataFrame containing the data.
    return_stops : bool, optional
        Whether to also return the laser stimulation stop times.

    Returns
    -------
    laser_stim_starts : array
        Array containing the laser stimulation start times.
    laser_stim_stops : array, optional
        Array containing the laser stimulation stop times if return_stops=True.
    """
    return get_stim_starts(df, stim_signals=["laser_stim", "olfac_stim"], stim_levels=[None, None], stim_level_signals=[None, None], return_stops=return_stops)

def get_backwalk_starts(df, return_stops=False):
    """
    Retrieves the start times of spontaneous backward walking.

    Parameters
    ----------
    df : DataFrame
        The DataFrame containing the data.
    return_stops : bool, optional
        Whether to also return the backward walking stop times.

    Returns
    -------
    backwalk_starts : array
        Array containing the backward walking start times.
    backwalk_stops : array, optional
        Array containing the backward walking  stop times if return_stops=True.
    """
    return get_stim_starts(df, stim_signals=["back_trig",], stim_levels=[None], stim_level_signals=[None], return_stops=return_stops)

def get_walk_starts(df, return_stops=False):
    """
    Retrieves the start times of spontaneous walking.

    Parameters
    ----------
    df : DataFrame
        The DataFrame containing the data.
    return_stops : bool, optional
        Whether to also return the walking stop times.

    Returns
    -------
    walk_starts : array
        Array containing the walking start times.
    walk_stops : array, optional
        Array containing the walking stop times if return_stops=True.
    """
    return get_stim_starts(df, stim_signals=["walk_trig",], stim_levels=[None], stim_level_signals=[None], return_stops=return_stops)

def get_rest_starts(df, return_stops=False):
    """
    Retrieves the start times of spontaneus resting.

    Parameters
    ----------
    df : DataFrame
        The DataFrame containing the data.
    return_stops : bool, optional
        Whether to also return the resting stop times.

    Returns
    -------
    rest_starts : array
        Array containing the resting start times.
    rest_stops : array, optional
        Array containing the resting stop times if return_stops=True.
    """
    return get_stim_starts(df, stim_signals=["rest_trig",], stim_levels=[None], stim_level_signals=[None], return_stops=return_stops)

def get_groom_starts(df, return_stops=False):
    """
    Retrieves the start times of spontaneous grooming.

    Parameters
    ----------
    df : DataFrame
        The DataFrame containing the data.
    return_stops : bool, optional
        Whether to also return the grooming stop times.

    Returns
    -------
    groom_starts : array
        Array containing the grooming start times.
    groom_stops : array, optional
        Array containing the grooming stop times if return_stops=True.
    """
    return get_stim_starts(df, stim_signals=["groom_trig",], stim_levels=[None], stim_level_signals=[None], return_stops=return_stops)


def get_neural_responses(twop_df, trigger, neural_regex=params.default_response_regex, trials=None, stim_p=[10,20], return_var=None,
                  response_t_params=params.response_t_params_2p):
    """
    Retrieves neural responses to stimulation from twop_df.

    Parameters
    ----------
    twop_df : DataFrame
        The DataFrame containing the neural data.
    trigger : str
        The trigger for stimulation detection.
    neural_regex : str, optional
        Regular expression for neural signal column selection.
    trials : list of str or None, optional
        List of trial identifiers to filter data by trial names.
    stim_p : list of int, optional
        Laser stimulation power levels to consider.
    return_var : str or None, optional
        The variable to return along with the neural responses, for example a synchronised behavioural variable.
    response_t_params : array, optional
        Parameters specifying the response time window.

    Returns
    -------
    stim_responses : array
        Array containing neural responses to stimulation. Format: time x neurons x trials.
    to_return : array or None, optional
        Array containing additional, potentially behavioural, data specified by `return_var` if provided.
    """
    if trigger == "laser_start":
        starts = get_laser_stim_starts(twop_df, stim_levels=stim_p)
    elif trigger == "olfac_start":
        starts = get_olfac_stim_starts(twop_df)
    elif trigger == "back_trig_start":
        starts, stops = get_backwalk_starts(twop_df, return_stops=True)  # TODO: process stops
    elif trigger == "walk_trig_start":
        starts, stops = get_walk_starts(twop_df, return_stops=True)  # TODO: process stops
    elif trigger == "rest_trig_start":
        starts, stops = get_rest_starts(twop_df, return_stops=True)  # TODO: process stops
    elif trigger == "groom_trig_start":
        starts, stops = get_groom_starts(twop_df, return_stops=True)  # TODO: process stops
    else:
        raise NotImplementedError(f"'trigger' must be either 'laser_start' 'olfac_start', 'back_trig_start', 'walk_trig_start', 'rest_trig_start', 'groom_trig_start', but was {trigger}")

    # reject starts that are too close to the beginning or end of a trial
    # stops = [stop for start, stop in zip(starts, stops)
    #               if all([start - response_t_params[0] > 0,
    #                       start+np.sum(response_t_params[1:]) < len(twop_df),
    #                       ])]
    starts = [start for start in starts
                  if all([start - response_t_params[0] > 0,
                          start+np.sum(response_t_params[1:]) < len(twop_df),
                          ])]
    

    # check whether stimulation happens in selected trials
    if trials is not None:
        starts = [start for start in starts
                  if any([twop_df.iloc[start].name[3].startswith(trial[:3])
                          for trial in trials])]
    
    signals = twop_df.filter(regex=neural_regex).values
    if np.sum(np.isnan(signals)):
        print(f"PROBLEM: some signals are nan: {np.sum(np.isnan(signals))}")

    stim_responses = np.zeros((int(np.sum(response_t_params)), signals.shape[1], len(starts)))
    to_return = np.zeros((int(np.sum(response_t_params)), len(starts))) if return_var is not None else None
    i_0 = int(-response_t_params[0])
    i_1 = int(response_t_params[1]+response_t_params[2])
    for i_start, stim_start in enumerate(starts):
        stim_responses[:,:,i_start] = signals[stim_start+i_0:stim_start+i_1, :]
        if return_var is not None and return_var in twop_df.keys():
            to_return[:,i_start] = twop_df[return_var][stim_start+i_0:stim_start+i_1]
        elif return_var is not None and i_start == 0:
            print(f"Warning: could not find return_var in twop_df: {return_var}")
    return stim_responses, to_return

def get_beh_responses(beh_df, trigger, trials, beh_var="v_forw", stim_p=[10,20], response_t_params=params.response_t_params_beh):
    """
    Retrieves behavioral responses to stimulation from beh_df.

    Parameters
    ----------
    beh_df : DataFrame
        The DataFrame containing the behavioral data.
    trigger : str
        The trigger for stimulation detection.
    trials : list of str
        List of trial identifiers to filter data by trial names.
    beh_var : str, optional
        The behavioral variable to retrieve.
    stim_p : list of int, optional
        Laser stimulation power levels to consider.
    response_t_params : array, optional
        Parameters specifying the response time window.

    Returns
    -------
    beh_out : array
        Array containing behavioral responses to stimulation.
    """
    if trigger == "laser_start":
        starts = get_laser_stim_starts(beh_df, stim_levels=stim_p)
    elif trigger == "olfac_start":
        starts = get_olfac_stim_starts(beh_df)
    elif trigger == "back_trig_start":
        starts, stops = get_backwalk_starts(beh_df, return_stops=True)  # TODO: process stops
    elif trigger == "walk_trig_start":
        starts, stops = get_walk_starts(beh_df, return_stops=True)  # TODO: process stops
    elif trigger == "rest_trig_start":
        starts, stops = get_rest_starts(beh_df, return_stops=True)  # TODO: process stops
    elif trigger == "groom_trig_start":
        starts, stops = get_groom_starts(beh_df, return_stops=True)  # TODO: process stops
    else:
        raise NotImplementedError(f"'trigger' must be either 'laser_start' 'olfac_start', 'back_trig_start', 'walk_trig_start', 'rest_trig_start', 'groom_trig_start', but was {trigger}")

    # reject starts that are too close to the beginning or end of a trial
    # stops = [stop for start, stop in zip(starts, stops)
    #               if all([start - response_t_params[0] > 0,
    #                       start+np.sum(response_t_params[1:]) < len(beh_df),
    #                       ])]
    starts = [start for start in starts
                  if all([start - response_t_params[0] > 0,
                          start+np.sum(response_t_params[1:]) < len(beh_df),
                          ])]

    # check whether stimulation happens in selected trials
    if trials is not None:
        starts = [start for start in starts
                  if any([beh_df.iloc[start].name[3].startswith(trial[:3])
                          for trial in trials])]

    beh_out = np.zeros((int(np.sum(response_t_params)), len(starts)))
    i_0 = int(-response_t_params[0])
    i_1 = int(response_t_params[1]+response_t_params[2])
    for i_start, stim_start in enumerate(starts):
        beh_out[:,i_start] = beh_df[beh_var][stim_start+i_0:stim_start+i_1]
    return beh_out

def get_beh_class_responses(beh_df, trigger, trials=None, stim_p=[10,20], response_t_params=params.response_t_params_beh):
    """
    Retrieves behavioral class responses to stimulation.

    Parameters
    ----------
    beh_df : DataFrame
        The DataFrame containing the behavioral data.
    trigger : str
        The trigger for stimulation detection.
    trials : list of str or None, optional
        List of trial identifiers to filter data by trial names.
    stim_p : list of int, optional
        Laser stimulation power levels to consider.
    response_t_params : array, optional
        Parameters specifying the response time window.

    Returns
    -------
    beh_class_responses : array
        Array containing behavioral class responses to stimulation.
    """
    return get_beh_responses(beh_df, beh_var="beh_class", trigger=trigger, trials=trials, stim_p=stim_p,
    response_t_params=response_t_params)

def summarise_responses(stim_responses, n_avg=params.response_n_avg, only_confident=params.response_n_confident,
                        n_baseline=params.response_n_baseline, n_latest_max=params.response_n_latest_max,
                        response_t_params=params.response_t_params_2p, return_conf_int=False):
    """
    Summarizes neural responses to stimulation and compute stimulation response values.

    Parameters
    ----------
    stim_responses : array
        Array containing neural responses over time to stimulation.
    n_avg : int, optional
        Number of samples to average around the maximum response.
    only_confident : int, optional
        Minimum number of confident responses required.
    n_baseline : int, optional
        Number of samples to consider as the baseline.
    n_latest_max : int, optional
        Maximum number of samples to consider to find the maximum response after stimulus onset.
    response_t_params : array, optional
        Parameters specifying the response time window.
    return_conf_int : bool, optional
        Whether to return confidence intervals.

    Returns
    -------
    response_values : array
        Array containing summarized response values.
    response_conf_ints : array, optional
        Array containing confidence intervals if return_conf_int=True.
    """
    n_avg = n_avg // 2
    if np.sum(np.isnan(stim_responses)):
        print(f"PROBLEM: some stim_responses are nan: {np.sum(np.isnan(stim_responses))}")
    stim_response = np.mean(stim_responses, axis=-1)
    i_b = int(response_t_params[0] - n_baseline)
    i_0 = int(response_t_params[0])
    rel_stim_response = stim_response - np.mean(stim_response[i_b:i_0], axis=0)

    stim_conf = utils.conf_int(stim_responses, axis=-1)
    # rel_resp_conf = rel_stim_response.copy()
    # rel_resp_conf[np.abs(rel_stim_response) < stim_conf] = np.nan

    i_m = int(i_0 + n_latest_max)
    # find the index of the maximum absolute stimulation response for each neuron
    i_stim_response_max = np.argmax(np.abs(rel_stim_response[i_0:i_m]), axis=0) + i_0  # TODO: allow different summaries

    response_values = np.zeros((len(i_stim_response_max)))
    response_conf_ints = np.zeros((len(i_stim_response_max)))
    for i, this_i_max in enumerate(i_stim_response_max):
        # compute mean response during n_avg samples around the maximum response identified before
        response_value = np.mean(rel_stim_response[this_i_max-n_avg:this_i_max+n_avg, i])
        response_conf_int = utils.conf_int(np.mean(stim_responses[this_i_max-n_avg:this_i_max+n_avg, i,:], axis=0), axis=-1)
        # check whether the absolute response within the period defined above exceeds the confidence interval.
        n_confident = np.sum(np.abs(rel_stim_response[this_i_max-n_avg:this_i_max+n_avg, i]) > \
                             stim_conf[this_i_max-n_avg:this_i_max+n_avg, i])
        if n_confident >= only_confident:
            response_values[i] = response_value
        response_conf_ints[i] = response_conf_int
    if return_conf_int:
        return response_values, response_conf_ints
    return response_values
