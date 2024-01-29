"""
Script to regenerate all figures related to experimental data analysis.
Author: jonas.braun@epfl.ch
"""
import os
from pathlib import Path

import params, copy_data
import fig_functional, fig_headless, fig_predictions, supfig_natbeh, supfig_bpn_mdn, supfig_stim_powers, 
import supfig_dfd_inhibit, fig_antennacut, fig_headless_imaging, fig_vnccut


if __name__ == "__main__":
    # TO DECOMPRESS THE DATA DOWNLOADED FROM THE DATAVERSE UNCOMMENT THE FOLLOWING
    # MAKE SURE THE PATHS TO THE DATA IN THE params.py FILE ARE SET TO POINT TO YOUR DATA
    """
    copy_data.decompress_imaging("PATH/TO/THE/DATAVERSE/DOWNLOADS/FROM/Optogenetics_Dfd_population_imaging")
    copy_data.decompress_headless_predictions("PATH/TO/THE/DATAVERSE/DOWNLOADS/FROM/Optogenetics_headless_behaviour")
    copy_data.decompress_other("PATH/TO/THE/DATAVERSE/DOWNLOADS/FROM/Supplementary_Data")
    """

    figures_path = os.path.join(Path(__file__).absolute().parent, "figures")
    os.makedirs(figures_path, exist_ok=True)
    tmpdata_path = os.path.join(Path(__file__).absolute().parent, "tmpdata")
    os.makedirs(tmpdata_path, exist_ok=True)
    
    # Figure 2: Activation of command-like DNs recruits larger, distinct DN populations.
    fig_functional.summarise_all_stim_resp(pre_stim="walk", overwrite=False, mode="presentation", natbeh=False, figures_path=figures_path, tmpdata_path=tmpdata_path)
    fig_functional.summarise_all_stim_resp(pre_stim="walk", overwrite=False, mode="presentationsummary", natbeh=False, figures_path=figures_path, tmpdata_path=tmpdata_path)
    fig_functional.plot_stat_comparison_activation(pre_stim="walk", figures_path=figures_path, tmpdata_path=tmpdata_path)
    
    # Supp. File 1: Individual fly neural and behavioral responses to opto-genetic stimulation
    fig_functional.summarise_all_stim_resp(pre_stim="walk", overwrite=False, mode="pdf", natbeh=False, figures_path=figures_path, tmpdata_path=tmpdata_path)
    fig_functional.summarise_all_stim_resp(pre_stim="rest", overwrite=False, mode="pdf", natbeh=False, figures_path=figures_path, tmpdata_path=tmpdata_path)
    
    # Supp. Figure 1d: DN driver lines
    supfig_stim_powers.plot_stim_p_effect(figures_path=figures_path)
    # Supp. Figure 1e-f: Optogenetic stimulation strategy.
    bpn_mdn_control_fly_names = ["230914_MDN3xCsChrimson_Fly1", "230914_MDN3xCsChrimson_Fly2",
                                 "230914_BPNxCsChrimson_Fly3", "230914_BPNxCsChrimson_Fly4",
                                 "230915_MDN3xCsChrimson_Fly11", "230921_BPNxCsChrimson_Fly5", "230921_BPNxCsChrimson_Fly6"]]
    bpn_mdn_control_fly_dirs = [os.path.join(params.other_data_dir, fly_name), for fly_name in bpn_mdn_control_fly_names]
    supfig_bpn_mdn.make_stim_loc_plots(figures_path=figures_path, fly_dirs=bpn_mdn_control_fly_dirs)
    # Supp. Figure 1g: Dfd inhibition drives grooming
    dfd_control_fly_dirs = [os.path.join(params.other_data_dir, "231115_DfdxGtACR1_Fly{i_fly}") for i_fly in range(1,4)]
    supfig_dfd_inhibit.make_dfd_inhibit_stim_loc_plots(fly_dirs=dfd_control_fly_dirs, figures_path=figures_path):
    
    # Supp. Figure 2: Comparison of GNG-DN population neural activity during optogenetic stimulation versus corresponding natural behaviors.
    fig_functional.summarise_all_stim_resp(pre_stim=None, overwrite=False, mode="presentation", natbeh=True, figures_path=figures_path, tmpdata_path=tmpdata_path)
    fig_functional.summarise_all_stim_resp(pre_stim=None, overwrite=False, mode="pdf", natbeh=False, figures_path=figures_path, tmpdata_path=tmpdata_path, compute_only=True)
    fig_functional.summarise_all_stim_resp(pre_stim="not_walk", overwrite=False, mode="pdf", natbeh=False, figures_path=figures_path, tmpdata_path=tmpdata_path, compute_only=True)
    supfig_natbeh.analyse_natbehaviour_responses_all_genotypes(figures_path=figures_path, tmpdata_path=tmpdata_path)
    supfig_natbeh.analyse_natbehaviour_responses_DNp09(figures_path=figures_path, tmpdata_path=tmpdata_path)

    # Supp. Figure 3: GNG DN activation upon comDN-activation after cut through the VNC.
    fig_vnccut.summarise_vnccut_resp(mode="pdf", figures_path=figures_path)

    # Figure 4: Recruited DN networks are required for forward walking and grooming, but not for backward walking
    fig_headless.summarise_all_headless(tmpdata_path=tmpdata_path, figures_path=figures_path, overwrite=False)
    fig_headless.headless_stat_test(tmpdata_path=tmpdata_path)

    # Figure 5f,g : Network connectivity accurately predicts the necessity for other DNs and flexi-bility of DN-driven behaviors.
    # Supp. Figures 6,7: Testing connectome-based predictions of DN-driven behavioral flexibility and its dependence upon downstream DNs.
    fig_predictions.make_all_predictions_figures(tmpdata_path=tmpdata_path, figures_path=figures_path, overwrite=False)
    fig_predictions.predictions_stats_tests(tmpdata_path=tmpdata_path)


    # Reviewer Figures
    # Reviewer Figure 1: Antennal grooming upon aDN2 stimulation with or without antennae.
    fig_antennacut.summarise_all_antennacut(tmpdata_path=tmpdata_path, figures_path=figures_path)
    fig_antennacut.antennacut_stat_test(tmpdata_path=tmpdata_path)

    # Reviewer Figure 2: Simultaneous optogenetic stimulation and neural recordings from DNp09 and MDN in intact and headless animals.
    fig_headless_imaging.make_headless_imaging_figure(figures_path=figures_path)