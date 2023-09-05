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



def make_headless_example_stim_videos(walkon="ball"):
    out_dir = os.path.join(params.video_base_dir, "presentation")
    n_frames = params.n_s_beh_10s + params.n_s_beh_2s
    fly_dicts = [
        {"id": 110, "trial_head": 3, "trial_no_head": 0},  # DNp09
        {"id": 112, "trial_head": 1, "trial_no_head": 0},  # MDN3
        {"id": 117, "trial_head": 0, "trial_no_head": 1}  # MDN3
    ]
    fly_ids = [fly_dict["id"] for fly_dict in fly_dicts]

    df = summarydf.get_headless_df()

    for i_fly, (fly_id, fly_df) in enumerate(df.groupby("fly_id")):
        if fly_id not in fly_ids:
            continue
        else:
            fly_dict = fly_dicts[fly_ids.index(fly_id)]
        for index, trial_df in fly_df.iterrows():
            if not trial_df.walkon == walkon:
                continue  

            if trial_df["head"]:
                video_name = f"{trial_df['CsChrimson']}_stim_example_head.mp4"
                i_stim_start = fly_dict["trial_head"]
            else:
                video_name = f"{trial_df['CsChrimson']}_stim_example_nohead.mp4"
                i_stim_start = fly_dict["trial_no_head"]

            trial_dir = trial_df.trial_dir
            beh_df = pd.read_pickle(os.path.join(trial_dir, load.PROCESSED_FOLDER, "beh_df.pkl"))
            video_dir = os.path.join(trial_dir, "behData", "images", "camera_5.mp4")

            stim_starts = stimulation.get_laser_stim_starts_all(beh_df)
            stim_start = stim_starts[i_stim_start] - params.n_s_beh_2s
            generator = videos.generator_video(video_dir, start=stim_start, stop=stim_start+n_frames, size=(480, -1))
            generator = videos.stimulus_dot_generator(generator, params.n_s_beh_2s, params.n_s_beh_2s+params.n_s_beh_5s)
            videos.make_video(os.path.join(out_dir, video_name), generator, params.fs_beh, n_frames=n_frames)


def make_headless_summary_stim_videos(walkon="ball"):
    out_dir = os.path.join(params.video_base_dir, "presentation")
    n_stim = 3
    n_frames = params.n_s_beh_15s * n_stim

    df = summarydf.get_headless_df()

    genotype_generators = {
        "MDN3": [],
        "DNp09": [],
        "aDN2": [],
        "PR": []
    }
    genotype_fly_ids = {
        "MDN3": [112,137,139],
        "DNp09": [110,111,119],
        "aDN2": [115,116,117],
        "PR": [131,140,143]
    }
    N_fly = {
        "MDN3": 1,
        "DNp09": 1,
        "aDN2": 1,
        "PR": 1
    }
    for i_fly, (fly_id, fly_df) in enumerate(df.groupby("fly_id")):
        genotype = np.unique(fly_df.CsChrimson)[0]
        if genotype == "MDN":
            genotype = "MDN3"
        elif genotype == "DNP9":
            genotype = "DNp09"

        if fly_id not in genotype_fly_ids[genotype]:
            continue

        for index, trial_df in fly_df.iterrows():
            if not trial_df.walkon == "ball" and trial_df["head"]:
                continue

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
                N_fly[genotype] += 1
            elif trial_df["walkon"] == "ball":
                generator_nohead = generator
            else:
                generator_nohead_noball = generator

        generator = videos.utils_video.generators.stack([generator_head, generator_nohead, generator_nohead_noball], axis=0)
        genotype_generators[genotype].append(generator)

    for genotype, generators in genotype_generators.items():
        generator = videos.utils_video.generators.stack(generators, axis=1)
        videos.make_video(os.path.join(out_dir, f"headless_summary_{genotype}.mp4"), generator, params.fs_beh, n_frames=n_frames)



def make_prediction_stim_videos():
    out_dir = os.path.join(params.video_base_dir, "presentation")
    n_stim = 3
    n_frames = params.n_s_beh_25s * n_stim

    df = summarydf.get_predictions_df()

    genotype_generators = {
        "DNa01": [],
        "DNa02": [],
        "DNb02": [],
        "aDN1": [],
        "DNg14": [],
        "mute": [],
        # "web": [],
        # "DNp24": [],
        # "DNg30": [],
    }
    N_fly = {
        "DNa01": 1,
        "DNa02": 1,
        "DNb02": 1,
        "aDN1": 1,
        "DNg14": 1,
        "mute": 1,
        # "web": 1,
        # "DNp24": 1,
        # "DNg30": 1,
    }

    genotype_fly_ids = {
        # "DNa01": [206,209,210],
        # "DNa02": [297,299,306],
        # "DNb02": [281,282,283],
        # "aDN1": [[217,233,233],[231,260,260],[232,263,263]],
        # "DNg14": [276,277,310],
        "mute": [313,[289,272,272],[314,271,314]],
    }
    
    shape = (480, 960, 3)
    black_image = np.zeros(shape, dtype=np.uint8)

    for genotype, this_genotype_fly_ids in genotype_fly_ids.items():
        genotpye_df = df[df.CsChrimson == genotype]
        n_fly = 0
        prev_fly_id = -1
        prev_fly_id_2 = -1
        for replicate in this_genotype_fly_ids:
            replicate_generator_list = [videos.utils_video.generators.static_image(black_image),
                                        videos.utils_video.generators.static_image(black_image),
                                        videos.utils_video.generators.static_image(black_image)]
            if isinstance(replicate, int):
                replicate = [replicate, replicate, replicate]

            for i_trial, fly_id in enumerate(replicate):
                fly_df = genotpye_df[genotpye_df.fly_id == fly_id]
                head = np.logical_or.reduce((fly_df["head"].values == "True", fly_df["head"].values == "TRUE", fly_df["head"].values == "1", fly_df["head"].values == True))

                if i_trial == 0:
                    trial_df = fly_df[np.logical_and(head, fly_df.walkon == "ball")]
                elif i_trial == 1:
                    trial_df = fly_df[np.logical_and(np.logical_not(head), fly_df.walkon == "ball")]
                elif i_trial == 2:
                    trial_df = fly_df[np.logical_and(np.logical_not(head), np.logical_not(fly_df.walkon == "ball"))]
                else:
                    raise NotImplementedError

                trial_df = trial_df.iloc[-1]  # make sure only one trial is used. Use the last one available by default

                trial_dir = trial_df.trial_dir
                beh_df = pd.read_pickle(os.path.join(trial_dir, load.PROCESSED_FOLDER, "beh_df.pkl"))
                video_dir = os.path.join(trial_dir, "behData", "images", "camera_5.mp4")

                stim_starts = np.array(stimulation.get_laser_stim_starts_all(beh_df))  # [:n_stim]
                video_start = stim_starts[0] - params.n_s_beh_5s
                stim_starts = stim_starts - video_start
                stim_starts = stim_starts[stim_starts < n_frames - params.n_s_beh_5s]
                generator = videos.generator_video(video_dir, start=video_start, stop=video_start+n_frames, size=(240, -1))
                generator = videos.stimulus_dot_generator(generator, start_stim=list(stim_starts), stop_stim=list(stim_starts + params.n_s_beh_5s))

                prev_fly_id_2 = prev_fly_id
                if fly_id != prev_fly_id and fly_id == prev_fly_id_2:
                    n_fly -= 1
                    prev_fly_id = fly_id
                elif fly_id != prev_fly_id:
                    n_fly += 1
                    prev_fly_id = fly_id
                replicate_generator_list[i_trial] = videos.utils_video.generators.add_text(generator, text=f"Fly {n_fly}", pos=(10,70))  # ID {fly_id}
            
            generator = videos.utils_video.generators.stack(replicate_generator_list, axis=0)
            genotype_generators[genotype].append(generator)

        generator = videos.utils_video.generators.stack(genotype_generators[genotype], axis=1)
        videos.make_video(os.path.join(out_dir, f"prediction_summary_{genotype}.mp4"), generator, params.fs_beh, n_frames=n_frames)
    """
    for i_fly, (fly_id, fly_df) in enumerate(df.groupby("fly_id")):
        genotype = np.unique(fly_df.CsChrimson)[0]

        if genotype not in genotype_generators.keys():
            continue

        generator_head = videos.utils_video.generators.static_image(black_image)
        generator_nohead = videos.utils_video.generators.static_image(black_image)
        generator_nohead_noball = videos.utils_video.generators.static_image(black_image)

        for index, trial_df in fly_df.iterrows():
            head = trial_df["head"] == "True" or trial_df["head"] == "TRUE" or trial_df["head"] == "1" or trial_df["head"] == True
            if not trial_df.walkon == "ball" and head:
                continue

            trial_dir = trial_df.trial_dir
            beh_df = pd.read_pickle(os.path.join(trial_dir, load.PROCESSED_FOLDER, "beh_df.pkl"))
            video_dir = os.path.join(trial_dir, "behData", "images", "camera_5.mp4")

            stim_starts = np.array(stimulation.get_laser_stim_starts_all(beh_df)[:n_stim])
            video_start = stim_starts[0] - params.n_s_beh_5s
            stim_starts = stim_starts - video_start
            generator = videos.generator_video(video_dir, start=video_start, stop=video_start+n_frames, size=(240, -1))
            generator = videos.stimulus_dot_generator(generator, start_stim=list(stim_starts), stop_stim=list(stim_starts + params.n_s_beh_5s))

            if head:
                generator_head = videos.utils_video.generators.add_text(generator, text=f"Fly ID {fly_id}", pos=(10,70))  # f"ID {fly_id}"  # {N_fly[genotype]} 
                # N_fly[genotype] += 1
            elif trial_df["walkon"] == "ball":
                generator_nohead = videos.utils_video.generators.add_text(generator, text=f"Fly ID {fly_id}", pos=(10,70))
            else:
                generator_nohead_noball = videos.utils_video.generators.add_text(generator, text=f"Fly ID {fly_id}", pos=(10,70))

        generator = videos.utils_video.generators.stack([generator_head, generator_nohead, generator_nohead_noball], axis=0)
        genotype_generators[genotype].append(generator)

    for genotype, generators in genotype_generators.items():
        generator = videos.utils_video.generators.stack(generators, axis=1)
        videos.make_video(os.path.join(out_dir, f"prediction_summary_{genotype}.mp4"), generator, params.fs_beh, n_frames=n_frames)
    """ 

if __name__ == "__main__":
    # make_headless_example_stim_videos()
    make_headless_summary_stim_videos()
    # make_prediction_stim_videos()