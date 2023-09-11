import os
from pathlib import Path

import params
import fig_functional, fig_headless, fig_predictions, supfig_natbeh, supfig_bpn_mdn, supfig_stim_powers


if __name__ == "__main__":
    local_data_dir = "path/to/data/DN_Networks"
    imaging_data_dir = os.path.join(local_data_dir, "Optogenetics_Dfd_population_imaging")
    headless_predictions_data_dir = os.path.join(local_data_dir, "Optogenetics_headless_behavior")
    other_data_dir = os.path.join(local_data_dir, "Supplementary_data")

    figures_path = os.path.join(Path(__file__).absolute().parent, "figures")
    os.makedirs(figures_path, exist_ok=True)
    tmpdata_path = os.path.join(Path(__file__).absolute().parent, "tmpdata")
    os.makedirs(tmpdata_path, exist_ok=True)

    # tmpdata_path = params.plotdata_base_dir  # TODO: remove

    # decrompress data if necessary

    # TODO: summary df handling
    
    # Figure 2: Activation of command-like DNs recruits larger, distinct DN populations.
    fig_functional.summarise_all_stim_resp(pre_stim="walk", overwrite=False, mode="presentation", natbeh=False, figures_path=figures_path, tmpdata_path=tmpdata_path)
    fig_functional.summarise_all_stim_resp(pre_stim="walk", overwrite=False, mode="presentationsummary", natbeh=False, figures_path=figures_path, tmpdata_path=tmpdata_path)
    fig_functional.plot_stat_comparison_activation(pre_stim="walk", figures_path=figures_path, tmpdata_path=tmpdata_path)
    
    # Supp. File 1: Individual fly neural and behavioral responses to opto-genetic stimulation
    fig_functional.summarise_all_stim_resp(pre_stim="walk", overwrite=False, mode="pdf", natbeh=False, figures_path=figures_path, tmpdata_path=tmpdata_path)
    fig_functional.summarise_all_stim_resp(pre_stim="rest", overwrite=False, mode="pdf", natbeh=False, figures_path=figures_path, tmpdata_path=tmpdata_path)
    
    # Supp. Figure 1d-f: DN driver lines and optogenetic stimulation strategy.
    supfig_stim_powers.plot_stim_p_effect(figures_path=figures_path)
    bpn_mdn_control_fly_dirs = None  # [os.path.join(other_data_dir, "220816_MDN3xCsChrimson_Fly1"), os.path.join(other_data_dir, "230125_BPNxCsChrimson_Fly1")]  # TODO: change back
    supfig_bpn_mdn.make_stim_loc_plots(figures_path=figures_path, fly_dirs=bpn_mdn_control_fly_dirs)

    # Supp. Figure 2: Comparison of GNG-DN population neural activity during optogenetic stimulation versus corresponding natural behaviors.
    fig_functional.summarise_all_stim_resp(pre_stim=None, overwrite=False, mode="presentation", natbeh=True, figures_path=figures_path, tmpdata_path=tmpdata_path)
    supfig_natbeh.analyse_natbehaviour_responses_all_genotypes(figures_path=figures_path, tmpdata_path=tmpdata_path)
    supfig_natbeh.analyse_natbehaviour_responses_DNp09(figures_path=figures_path, tmpdata_path=tmpdata_path)

    # Figure 4: Recruited DN networks are required for forward walking and grooming, but not for backward walking
    fig_headless.summarise_all_headless(tmpdata_path=tmpdata_path, figures_path=figures_path, overwrite=False)
    fig_headless.headless_stat_test(tmpdata_path=tmpdata_path)

    # Figure 5f,g : Network connectivity accurately predicts the necessity for other DNs and flexi-bility of DN-driven behaviors.
    # Supp. Figure 3: Testing connectome-based predictions of DN-driven behavioral flexibility and its dependence upon downstream DNs.
    # tmpdata_path = params.predictionsdata_base_dir  # TODO: remove

    fig_predictions.make_all_predictions_figures(tmpdata_path=tmpdata_path, figures_path=figures_path, overwrite=False)
    fig_predictions.predictions_stats_tests(tmpdata_path=tmpdata_path)

