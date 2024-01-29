"""
Script to regenerate all videos related to experimental data analysis.
This script also generates the plots shown in Figure 2a.
Running this script requires raw data that is ~1 TB large and can be requested from the Authors of the manuscript.
Author: jonas.braun@epfl.ch
"""

from vid_functional import make_functional_example_stim_video, make_DNp09_natbeh_video, make_aDN2_natbeh_video, make_MDN_natbeh_video
from vid_headless import make_headless_summary_stim_videos, make_prediction_stim_videos
from vid_vnccut import make_vnc_cut_video
if __name__ == "__main__":
    # Videos 1-4: Stimulation response videos
    # This function also generates the plots shown in Figure 2b.
    make_functional_example_stim_video(pre_stim="walk", overwrite=False,
        subtract_pre=True, show_centers=False, show_mask=False, show_beh=True,
        make_fig=True, make_vid=False, select_group_of_flies="presentation", trigger="laser_start")

    # Video 5: Comparing optogenetic DNp09 stimulation with natural walking
    make_DNp09_natbeh_video()
    # Video 6: Comparing optogenetic aDN stimulation with natural grooming
    make_aDN2_natbeh_video()
    # Video 7: Comparing optogenetic MDN stimulation with natural backward walking on the wheel
    make_MDN_natbeh_video()

    # Video 8: DNp09-driven and control trial-averaged GNG-DN population activity after resecting the T1 neuromere of the VNC.
    make_vnc_cut_video()

    # Videos 9-12: comparing intact and headless stimulation response behaviours for DNp09, aDN2, MDN, wild type flies.
    make_headless_summary_stim_videos()
    # Vides 13-21: comparing intact and headless stimulation response behaviours for new DN lines
    make_prediction_stim_videos()
   