import os
import sys

import numpy as np
import pandas as pd

import utils_video.generators
from twoppp.plot import videos
from twoppp import load

import params
import summarydf
import stimulation

def make_headless_beh_video(fly_df, out_dir, video_params):
    video_name = f"{video_params['video_base_name']}_{fly_df.CsChrimson.iloc[0]}_{fly_df.date.iloc[0]}_Fly{fly_df.fly_number.iloc[0]}_id{fly_df.fly_id.iloc[0]}.mp4"
    print(video_name)
    n_frames = params.n_s_beh_15s
    stim_range = [params.n_s_beh_5s, params.n_s_beh_10s]

    generators = []

    for i_t, (index, row) in enumerate(fly_df.iterrows()):
        # headless = np.logical_not(row["head"])
        # onball = row.walkon == "ball"
        power = row.laser_power
        text = f"{i_t+1}: p {power} uW"
        trial_dir = row.trial_dir
        video_dir = os.path.join(trial_dir, "behData", "images", video_params["camera"])
        beh_df = os.path.join(trial_dir, load.PROCESSED_FOLDER, "beh_df.pkl")
        beh_df = pd.read_pickle(beh_df)

        stim_starts = stimulation.get_laser_stim_starts_all(beh_df)
        stim_starts = np.array(stim_starts) - params.n_s_beh_5s
        
        row_generators = videos.make_behaviour_grid_video(
            video_dirs=[video_dir],
            start_frames=[stim_starts],
            N_frames=n_frames,
            stim_range=stim_range,
            out_dir=None,
            video_name=None,
            frame_rate=params.fs_beh,
            asgenerator=True,
            size=(video_params["video_height"],-1),
        )
        row_generator = utils_video.generators.stack(row_generators, axis=1)
        row_generator = utils_video.generators.add_text(row_generator, text=text, pos=(10,150))
        generators.append(row_generator)


    generator = utils_video.generators.stack(generators, axis=0)
    videos.make_video(os.path.join(out_dir, video_name), generator, fps=params.fs_beh)


def make_all_headless_beh_videos(exp_df, out_dir, video_params):
    for i_vid, (fly_id, fly_df) in enumerate(exp_df.groupby("fly_id")):
        
        if fly_id <= 132:
            print("skipping fly_id", fly_id)
            continue
        make_headless_beh_video(fly_df, out_dir, video_params)


if __name__ == "__main__":
    out_dir = os.path.join(params.data_summary_dir, "videos", "headless")
    video_params = {
        "video_base_name": "headless",
        "video_height": 80,
        "camera": "camera_5.mp4",
    }

    df = summarydf.get_headless_df()
    make_all_headless_beh_videos(df, out_dir, video_params)
