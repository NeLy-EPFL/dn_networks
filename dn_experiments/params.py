"""
Parameters used for analysis of 2p and behavioural data
"""
import os
import matplotlib.pyplot as plt

from datetime import date

today = date.today().strftime("%y%m%d")


# handling summary data of all recording
data_summary_dir_JB = "/mnt/nas2/JB/_data_summary"
data_summary_dir_FH = "/mnt/nas2/FH/_data_summary"
data_summary_dir = data_summary_dir_FH

# data_summary_csv_dir = os.path.join(data_summary_dir, "fly_selection_manual_230224.csv")
data_summary_csv_dir = os.path.join(
    data_summary_dir, "fly_selection_revisions_231222.csv"
)
data_headless_summary_csv_dir = os.path.join(
    data_summary_dir, "fly_selection_nohead_230406.csv"
)
plot_base_dir = os.path.join(data_summary_dir_FH, "plots")
plotdata_base_dir = os.path.join(data_summary_dir_FH, "plotdata")
# data_predictions_summary_csv_dir = os.path.join(data_summary_dir, "fly_selection_predictions_230712.csv")
# plot_base_dir = os.path.join(data_summary_dir, "plots")
# plotdata_base_dir = os.path.join(data_summary_dir, "plotdata")
predictionsdata_base_dir = os.path.join(data_summary_dir, "predictionsdata")
predictionsplot_base_dir = os.path.join(data_summary_dir, "predictionsplots")
data_predictions_summary_csv_dir = os.path.join(
    data_summary_dir_FH, "fly_selection_revisions_231222.csv"
)

data_revisions_summary_dir = os.path.join(
    data_summary_dir_FH, "fly_selection_revisions_231222.csv"
)

video_base_dir = os.path.join(data_summary_dir, "videos")
twop_df_save_name = "twop_df_comp.pkl"
beh_df_save_name = "beh_df_comp.pkl"

q_thres_neural = 3
q_thres_beh = 3
q_thres_stim = 3
q_check_headless = False
q_thres_headless_legs_intact = 3
q_thres_headless_beh = 3
q_thres_headless_stim = 3

# general parameters
fs_int = int(16)
n_s_2p_int_0_5s = int(fs_int * 0.5)
n_s_2p_int_1s = fs_int * 1
n_s_2p_int_2s = fs_int * 2
n_s_2p_int_2_5s = int(fs_int * 2.5)
n_s_2p_int_5s = fs_int * 5
n_s_2p_int_10s = fs_int * 10
n_s_2p_int_15s = fs_int * 15

fs_2p = 16.23
n_s_2p_0_5s = int(fs_2p * 0.5)
n_s_2p_1s = int(fs_2p * 1)
n_s_2p_2s = int(fs_2p * 2)
n_s_2p_2_5s = int(fs_2p * 2.5)
n_s_2p_5s = int(fs_2p * 5)
n_s_2p_10s = int(fs_2p * 10)
n_s_2p_15s = int(fs_2p * 15)

fs_beh = 100
n_s_beh_0_5s = int(fs_beh * 0.5)
n_s_beh_1s = int(fs_beh * 1)
n_s_beh_2s = int(fs_beh * 2)
n_s_beh_2_5s = int(fs_beh * 2.5)
n_s_beh_5s = int(fs_beh * 5)
n_s_beh_10s = int(fs_beh * 10)
n_s_beh_15s = int(fs_beh * 15)
n_s_beh_25s = int(fs_beh * 25)


## neuronal processing
# filtering
neurons_med = 3  # median filterin applied to ROI signals
neurons_med_xy = 3  # spatial median filter applied to raw data for visialisation
neurons_sigma = 3  # Gaussian filter applied to ROI signals
neurons_sigma_xy = 2  # spatial Gaussian filter applied to raw data for visialisation
neurons_filt_regex = "neuron_filt"  # how to call the filtered neurons in dataframe
# baseline computation
baseline_exclude_stim_range = [
    0,
    n_s_2p_5s,
]  # how long before and after a stimulation period to exclude data for baseline computation
baseline_exclude_stim_low_high = [
    True,
    False,
]  # whether to apply the stimulus period exclusion to lower and/or upper normalisation point. Only used for normalisation that are calculated based on 2 points.
baseline_neurons_regex = "neuron_filt"  # which neural signal to load for dff compuation. could be "neuron_filt" (see above) or "neuron_denoised" from deepinterpolation.
all_normalisation_types = ["dff", "std", "rest_qmax", "trough_qmax"]
# "silent", "fastsilent", , "me_trough_qmax"  # (exclude motion energy based baselines)
baseline_qmax = 0.95  # quantile used for upper limit normalisation
rest_base_min_rest_s = 1  # how long a resting bout has to be in order to be considered for baseline computation
rest_base_frac_rest = 0.75  # what the fraction of resting classification during this bout has to be at least
rest_base_max_search_t_s = [
    -1,
    2,
]  # which time window around resting onset to look for the minimum
trough_baseline_n_peaks = (
    10  # how many peaks (inverted troughs) to consider while computing trough baseline
)
trough_baseline_min_height = (
    0.5  # minimum heights of standardised mean neural signal to consider a peak
)
trough_baseline_min_distance = n_s_2p_2s  # minimum distance between peaks
trough_baseline_n_samples_around = (
    n_s_2p_1s  # how many samples around the peak to check for each individual neuron
)
me_trough_baseline_min_height = (
    0.75  # minimum heights of standardised motion energy to consider a peak
)
me_trough_baseline_sigma_me = (
    5  # smoothing coefficient before motion energy peak detection
)
# response computation and plotting
default_response_regex = "dff_rest_qmax"  # which signals to look at as default
response_t_params_2p = [
    n_s_2p_5s,
    n_s_2p_5s,
    n_s_2p_5s,
]  # defining number of pre, during, and post stimulus samples
response_t_params_2p_label = [
    5,
    5,
    5,
]  # same as above but in seconds to be used as a label
response_n_baseline = n_s_2p_1s  # number of samples before stimulation that are used as baseline for relative stim responses
response_n_latest_max = n_s_2p_2_5s  # number of samples after stim start up to which the maximum response can occur  # TODO: check the effect of 2.5s vs 5s
response_n_avg = n_s_2p_1s  # number of samples to average across after the maximum response location has been found
response_n_confident = n_s_2p_0_5s  # number of samples with in the range defined above that have to be above the confidence interval.

## behavioural data processing
# motion energy processing
me_cam = "camera_5.mp4"
me_cam_mean = "camera_5_mean.jpg"
q_me = 0.95  # quantile to which to normalise motion energy to
thres_silent_me_front = 0.2  # front motion energy quantile has to be smaller than this to count into silent state
thres_silent_me_all = 0.2  # total motion energy quantile has to be smaller than this to count into silent state
thres_silent_v = (
    0.5  # absolut velocity has to be smaller than this to count into silent state
)
thres_fast_me_all = 0.8  # # total motion energy quantile has to be larger than this to count into fast state
# sleap processing
sleap_med_filt = 9
sleap_sigma_gauss = 5
joint_motion_energy_moving_average = n_s_beh_0_5s
#  response computation and plotting
pre_stim_n_samples_beh = n_s_beh_1s
response_t_params_beh = [n_s_beh_5s, n_s_beh_5s, n_s_beh_5s]
# behaviour classification
beh_class_method = "sleap"  # "motionenergy"
# behaviour onset detection
backwalk_min_dur = n_s_beh_1s
backwalk_min_dist = -1  # mm
walk_min_dur = n_s_beh_1s
walk_min_dist = 1  # mm
rest_min_dur = n_s_beh_1s
rest_min_dist = None
groom_min_dur = n_s_beh_1s
groom_min_dist = None

# plotting parameters for showing maps of data
cmap_groom = plt.cm.get_cmap(
    "PiYG_r"
)  # mpl.colors.ListedColormap(plt.cm.get_cmap('PiYG_r')(np.concatenate((np.linspace(0,0.375, 128), np.linspace(0.625,1,128)))))
cmap_back = plt.cm.get_cmap(
    "RdBu_r"
)  # mpl.colors.ListedColormap(plt.cm.get_cmap('RdBu_r')(np.concatenate((np.linspace(0,0.375, 128), np.linspace(0.625,1,128)))))
cmap_walk = plt.cm.get_cmap(
    "PuOr_r"
)  # mpl.colors.ListedColormap(plt.cm.get_cmap('PuOr_r')(np.concatenate((np.linspace(0,0.375, 128), np.linspace(0.625,1,128)))))
cmap_ci = cmap_back

background_key = "green_stds_raw"  # which image to show as background
background_crop = [80, 80, 0, 0]  # how to show the std image in background
background_med = 1  # parameter for median filter on background image
background_sigma = 3  # how to smooth the background image

map_min_dot_size = 10
map_max_dot_size = 250
map_min_dot_alpha = 0.5
map_max_dot_alpha = 1
map_cmap_dft = cmap_back
map_crop_x = 40
map_q_max = 0.96
