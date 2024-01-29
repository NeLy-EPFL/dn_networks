
from params import n_s_beh_1s
import numpy as np

beh_mapping = {
    0: 'background',
    1: 'walk',
    2: 'rest',
    3: 'back',
    4: 'groom',
    5: 'leg rub',
    6: 'posterior'
    }

def remove_trials_beh_pre(
    fly_data,
    pre_beh_to_filter_for: str,
    threshold: float = 0.25,
    ):
    """
    Removes trials from all_fly_data for which the behaviour probability is
    below the threshold for the pre-behaviour.

    Parameters
    ----------
    all_fly_data : dict
        Dictionary containing all the data for all flies.
    pre_beh_to_filter_for : str
        Behaviour to filter for.
    threshold : float, optional
        Threshold for the pre-behaviour number of instances.

    Returns
    -------
    all_fly_data : dict
        Dictionary containing all the data for all flies, but with the
        filtered trials removed.
    """
    if (not isinstance(fly_data["beh_class_responses_pre"], np.ndarray) or (
        np.isnan(fly_data["beh_class_responses_pre"].all()))):
        return fly_data

    # number in beh_mapping corresponding to pre_beh_to_filter_for
    pre_beh_key = [k for k, v in beh_mapping.items()
                    if v == pre_beh_to_filter_for][0]
    trials_match_pre_beh = []
    for trial in fly_data['beh_class_responses_pre'].T:
        relevant_data = trial[-1*n_s_beh_1s:] # last second before stimulus
        # if more than threshold precent of the last second is the pre_beh
        if sum(relevant_data == pre_beh_key)/n_s_beh_1s > threshold:
            trials_match_pre_beh.append(True)
        else:
            trials_match_pre_beh.append(False)

    # remove trials that don't match pre_beh
    for key in [
        'beh_class_responses_pre',
        'beh_class_responses_post',
        'beh_responses_post',
        'beh_responses_pre',
        ]:
        if fly_data[key] is not None:
            data = fly_data[key].T
            filtered_data = data[trials_match_pre_beh].T.copy()
            fly_data[key] = filtered_data
    return fly_data
