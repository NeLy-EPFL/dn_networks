"""
module to export plotted points for Nature's sourcedata requirements.
"""

import os
import sys
import numpy as np
import pandas as pd

def genotype_from_suptitle(figure_params):
    string = figure_params["suptitle"]
    if "MDN" in string:
        return "MDN"
    elif "DNp09" in string:
        return "DNp09"
    elif "aDN2" in string:
        return "aDN2"
    elif "PR" in string or "control" in string or "__" in string:
        return "PR"
    elif "DNb02" in string:
        return "DNb02"
    elif "DNg14" in string:
        return "DNg14"
    else:
        return "DNXXX"


def export_sourcedata(data, file_name):
    if isinstance(data, np.ndarray):
        np.savetxt(file_name, data, delimiter=",")
    elif isinstance(data, pd.DataFrame):
        data.to_csv(file_name)
    else:
        raise(NotImplementedError)

def behaviour_to_sourcedata(t, var, var_ci, varname, figure_params, figure_number: str, base_dir, var_post=None, var_post_ci=None):
    if var_post is None or var_post_ci is None:
        df = pd.DataFrame(np.array([t, var, var_ci]).T, columns=["t", varname, varname + "_CI"])
    else:
        df = pd.DataFrame(np.array([t, var, var_ci, var_post, var_post_ci]).T,
                          columns=["t", varname, varname + "_CI", varname + "_post", varname + "_post_CI"])

    genotype = genotype_from_suptitle(figure_params)
    file_name = f"sourcedata_{figure_number}_{varname}_{genotype}.csv"

    export_sourcedata(df, os.path.join(base_dir, file_name))

def behclass_to_sourcedata(t, behclass, behclass_name, figure_params, figure_number: str, base_dir, behclass_post=None):
    behclass_name = [behclass_name] if not isinstance(behclass_name, list) else behclass_name
    if len(behclass.shape) == 1:
        behclass = np.expand_dims(behclass, axis=-1)
    if behclass_post is None:
        df = pd.DataFrame(np.concatenate([np.expand_dims(t, axis=-1), behclass], axis=1), columns=["t"] + behclass_name)
    else:
        if len(behclass_post.shape) == 1:
            behclass_post = np.expand_dims(behclass_post, axis=-1)
        behclass_post_name = [b + "_post" for b in behclass_name]
        df = pd.DataFrame(np.concatenate([np.expand_dims(t, axis=-1), behclass, behclass_post], axis=1), columns=["t"] + behclass_name + behclass_post_name)

    genotype = genotype_from_suptitle(figure_params)
    file_name = f"sourcedata_{figure_number}_behclass_{genotype}.csv"

    export_sourcedata(df, os.path.join(base_dir, file_name))

def rois_to_sourcedata(roi_centers, response_values, figure_params, figure_number: str, base_dir):
    df = pd.DataFrame(np.concatenate([roi_centers, np.expand_dims(response_values, axis=-1)], axis=1), columns=["x", "y", "response"])

    genotype = genotype_from_suptitle(figure_params)
    file_name = f"sourcedata_{figure_number}_roi_response_{genotype}.csv"

    export_sourcedata(df, os.path.join(base_dir, file_name))

def density_to_sourcedata(density, figure_params, figure_number: str, base_dir):
    genotype = genotype_from_suptitle(figure_params)
    file_name = f"sourcedata_{figure_number}_density_{genotype}.csv"

    export_sourcedata(density, os.path.join(base_dir, file_name))

def neuronstime_to_sourcedata(neuronstime, figure_params, figure_number: str, base_dir):
    genotype = genotype_from_suptitle(figure_params)
    file_name = f"sourcedata_{figure_number}_neuronstime_{genotype}.csv"

    export_sourcedata(neuronstime, os.path.join(base_dir, file_name))