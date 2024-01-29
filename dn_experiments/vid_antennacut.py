"""
This module generates videos comparing intact and antennaless flies behavioural respones
Author: jonas.braun@epfl.ch
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import pickle
from tqdm import tqdm

import params, summarydf, loaddata, stimulation, behaviour, plotpanels

from twoppp.plot import videos
from twoppp import load

def make_antennacut_summary_stim_videos():
    """
    Create antennacut summary stimulus videos.

    """
    out_dir = os.path.join(params.video_base_dir, "revision")
    n_stim = 3
    n_frames = params.n_s_beh_15s * n_stim

    df = summarydf.get_headless_df()

    genotype_generators = {
        "aDN2": [],
    }
    genotype_fly_ids = {
        "aDN2": [154,156,158,162],
    }
    N_fly = {
        "aDN2": 1,
    }
    for i_fly, (fly_id, fly_df) in enumerate(df.groupby("fly_id")):
        genotype = np.unique(fly_df.CsChrimson)[0]
        if genotype not in list(genotype_fly_ids.keys()):
            continue
        if fly_id not in genotype_fly_ids[genotype]:
            continue

        for index, trial_df in fly_df.iterrows():

            trial_dir = trial_df.trial_dir
            beh_df = pd.read_pickle(os.path.join(trial_dir, load.PROCESSED_FOLDER, "beh_df.pkl"))
            video_dir = os.path.join(trial_dir, "behData", "images", "camera_5.mp4")

            stim_starts = np.array(stimulation.get_laser_stim_starts_all(beh_df)[:n_stim])
            video_start = stim_starts[0] - params.n_s_beh_5s
            stim_starts = stim_starts - video_start
            generator = videos.generator_video(video_dir, start=video_start, stop=video_start+n_frames, size=(240, -1))
            generator = videos.stimulus_dot_generator(generator, start_stim=list(stim_starts), stop_stim=list(stim_starts + params.n_s_beh_5s))

            if trial_df["head"]:
                generator_head = videos.utils_video.generators.add_text(generator, text=f"Fly {N_fly[genotype]}", pos=(10,70))  # f"ID {fly_id}"
                if N_fly[genotype] == 1:
                    generator_head = videos.utils_video.generators.add_text(generator_head, text=f"intact fly", pos=(10,230))
                N_fly[genotype] += 1
            elif N_fly[genotype] <= 2:
                generator_nohead = videos.utils_video.generators.add_text(generator, text=f"antennae removed", pos=(10,230))  # f"ID {fly_id}"
            else:
                generator_nohead = generator

        generator = videos.utils_video.generators.stack([generator_head, generator_nohead], axis=0)
        genotype_generators[genotype].append(generator)

    for genotype, generators in genotype_generators.items():
        generator = videos.utils_video.generators.stack(generators, axis=1)
        videos.make_video(os.path.join(out_dir, f"antennacut_summary_{genotype}_cam5.mp4"), generator, params.fs_beh, n_frames=n_frames)


if __name__ == "__main__":
    make_antennacut_summary_stim_videos()
