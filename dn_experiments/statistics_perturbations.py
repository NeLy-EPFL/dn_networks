import os

import numpy as np
import pickle
from scipy.stats import mannwhitneyu

import params


def test_stats_pre_post(all_flies, i_beh, GAL4, beh_name,measured="", i_0=500, i_1=750,file=None):
    v_pre = []
    p_pre = []
    v_post = []
    p_post = []
    for fly in all_flies:
        if not fly["beh_responses_pre"] is None:
            v_pre.append(np.mean(fly["beh_responses_pre"][i_0:i_1], axis=0))
        if not fly["beh_responses_post"] is None:
            v_post.append(np.mean(fly["beh_responses_post"][i_0:i_1], axis=0))
        if not fly["beh_class_responses_pre"] is None:
            p_pre.append(np.mean(fly["beh_class_responses_pre"][i_0:i_1] == i_beh, axis=0))
        if not fly["beh_class_responses_post"] is None:
            p_post.append(np.mean(fly["beh_class_responses_post"][i_0:i_1] == i_beh, axis=0))
    v_pre = np.concatenate(v_pre).flatten()
    v_post = np.concatenate(v_post).flatten()
    p_pre = np.concatenate(p_pre).flatten()
    p_post = np.concatenate(p_post).flatten()
    if file is None:
        print(f"{GAL4} ", measured ," : ", mannwhitneyu(v_pre, v_post))
        print(f"{GAL4} {beh_name} beh class:", mannwhitneyu(p_pre, p_post))
    else:
        with open(file, "a") as f:
            f.write(f"{GAL4}  {measured}  :  {mannwhitneyu(v_pre, v_post)} \n")
            f.write(f"{GAL4} {beh_name} beh class: {mannwhitneyu(p_pre, p_post)}\n")
            f.write("\n")
    return

def test_stats_beh_control(all_flies, all_flies_control, GAL4, beh_name, i_0=500, i_1=750,file=None):
    beh_pre = []
    beh_pre_control = []
    beh_post = []
    beh_post_control = []
    for fly, fly_control in zip(all_flies, all_flies_control):
        if not fly["beh_responses_post"] is None:
            beh_post.append(np.mean(fly["beh_responses_post"][i_0:i_1], axis=0))
            beh_post_control.append(np.mean(fly_control["beh_responses_post"][i_0:i_1], axis=0))
        if not fly["beh_responses_pre"] is None:
            beh_pre.append(np.mean(fly["beh_responses_pre"][i_0:i_1], axis=0))
            beh_pre_control.append(np.mean(fly_control["beh_responses_pre"][i_0:i_1], axis=0))
    beh_post = np.concatenate(beh_post).flatten()
    beh_post_control = np.concatenate(beh_post_control).flatten()
    beh_pre = np.concatenate(beh_pre).flatten()
    beh_pre_control = np.concatenate(beh_pre_control).flatten()

    if file is None:
        print(f"{GAL4} {beh_name} before perturbation: {mannwhitneyu(beh_pre, beh_pre_control)}")
        print(f"{GAL4} {beh_name} after perturbation: {mannwhitneyu(beh_post, beh_post_control)}")
    else:
        with open(file, "a") as f:
            f.write(f"{GAL4} {beh_name} before perturbation: {mannwhitneyu(beh_pre, beh_pre_control)}\n")
            f.write(f"{GAL4} {beh_name} after perturbation: {mannwhitneyu(beh_post, beh_post_control)}\n")
            f.write("\n")
    return

def revisions_stat_test(overwrite=True):
    stats_file_summary = os.path.join(params.predictionsdata_base_dir, "stats.txt")
    if overwrite:
        with open(stats_file_summary, "w") as f:
            f.write("")

    revision_files = {
        "DNp42": os.path.join(params.predictionsdata_base_dir, "predictions_DNp42_v_forw.pkl"),
        "DNb01": os.path.join(params.predictionsdata_base_dir, "predictions_DNb01_v_forw.pkl"),
        "DNg16": os.path.join(params.predictionsdata_base_dir, "predictions_DNg16_v_forw.pkl"),
        "DNg11": os.path.join(params.predictionsdata_base_dir, "predictions_DNg11_frtita_y.pkl"),
        "oviDN": os.path.join(params.predictionsdata_base_dir, "predictions_oviDN_ovum_y.pkl"),
        "CS_v_forw": os.path.join(params.predictionsdata_base_dir, "predictions_CantonS_v_forw.pkl"),
        "CS_ovum": os.path.join(params.predictionsdata_base_dir, "predictions_CantonS_ovum_y.pkl"),
        "CS_frtita": os.path.join(params.predictionsdata_base_dir, "predictions_CantonS_frtita_y.pkl"),
    }

    with open(revision_files["DNp42"], "rb") as f:
        DNp42 = pickle.load(f)
    with open(revision_files["DNb01"], "rb") as f:
        DNb01 = pickle.load(f)
    with open(revision_files["DNg16"], "rb") as f:
        DNg16 = pickle.load(f)
    with open(revision_files["DNg11"], "rb") as f:
        DNg11 = pickle.load(f)
    with open(revision_files["oviDN"], "rb") as f:
        oviDN = pickle.load(f)
    with open(revision_files["CS_v_forw"], "rb") as f:
        CS_v_forw = pickle.load(f)
    with open(revision_files["CS_ovum"], "rb") as f:
        CS_ovum = pickle.load(f)
    with open(revision_files["CS_frtita"], "rb") as f:
        CS_frtita = pickle.load(f)

    # Tests from intact to headless
    # Add a sentence to the summary file
    with open(stats_file_summary, "a") as f:
        f.write("Tests from intact to headless\n")
    test_stats_pre_post(DNp42,i_beh=3,GAL4="DNp42",beh_name="back",measured="v_forw",file=stats_file_summary)
    test_stats_pre_post(DNb01,i_beh=1,GAL4="DNb01",beh_name="walk",measured="v_forw",file=stats_file_summary)
    test_stats_pre_post(DNg16,i_beh=1,GAL4="DNg16",beh_name="walk",measured="v_forw",file=stats_file_summary)
    test_stats_pre_post(DNg11,i_beh=4,GAL4="DNg11",beh_name="groom",measured="y_frtita",file=stats_file_summary)
    test_stats_pre_post(oviDN,i_beh=4,GAL4="oviDN",beh_name="groom",measured="y_ovum",file=stats_file_summary)

    # Tests between experiment and control
    with open(stats_file_summary, "a") as f:
        f.write("\n")
        f.write("Tests from modified genotype to control on identical settings\n")
    test_stats_beh_control(DNp42, CS_v_forw, GAL4="DNp42", beh_name="v_forw", i_0=500, i_1=750,file=stats_file_summary)
    test_stats_beh_control(DNb01, CS_v_forw, GAL4="DNb01", beh_name="v_forw", i_0=500, i_1=750,file=stats_file_summary)
    test_stats_beh_control(DNg16, CS_v_forw, GAL4="DNg16", beh_name="v_forw", i_0=500, i_1=750,file=stats_file_summary)
    test_stats_beh_control(DNg11, CS_frtita, GAL4="DNg11", beh_name="y_frtita", i_0=500, i_1=750,file=stats_file_summary)
    test_stats_beh_control(oviDN, CS_ovum, GAL4="oviDN", beh_name="y_ovum", i_0=500, i_1=750,file=stats_file_summary)




def headless_stat_test():
    headless_files = {
        "MDN": os.path.join(params.plotdata_base_dir, "headless_MDN3.pkl"),
        "DNp09": os.path.join(params.plotdata_base_dir, "headless_DNp09.pkl"),
        "aDN2": os.path.join(params.plotdata_base_dir, "headless_aDN2.pkl"),
        "PR": os.path.join(params.plotdata_base_dir, "headless_PR.pkl"),
    }
    with open(headless_files["MDN"], "rb") as f:
        MDN = pickle.load(f)
    with open(headless_files["DNp09"], "rb") as f:
        DNp09 = pickle.load(f)
    with open(headless_files["aDN2"], "rb") as f:
        aDN2 = pickle.load(f)
    with open(headless_files["PR"], "rb") as f:
        PR = pickle.load(f)
    
    test_stats_pre_post(MDN, i_beh=3, GAL4="MDN", beh_name="back")
    test_stats_pre_post(DNp09, i_beh=1, GAL4="DNp09", beh_name="walk")
    test_stats_pre_post(aDN2, i_beh=4, GAL4="aDN2", beh_name="groom")
    test_stats_pre_post(PR, i_beh=2, GAL4="PR", beh_name="rest")

    detailled_files = {
        "DNp09_anus": os.path.join(params.plotdata_base_dir, "headless_DNp09_anus.pkl"),
        "PR_anus": os.path.join(params.plotdata_base_dir, "headless_PR_anus.pkl"),
        "aDN2_height": os.path.join(params.plotdata_base_dir, "headless_aDN2_frleg_height.pkl"),
        "PR_height": os.path.join(params.plotdata_base_dir, "headless_PR_frleg_height.pkl"),
        "aDN2_angle": os.path.join(params.plotdata_base_dir, "headless_aDN2_tibia_angle.pkl"),
        "PR_angle": os.path.join(params.plotdata_base_dir, "headless_PR_tibia_angle.pkl"),
        "aDN2_dist_tita": os.path.join(params.plotdata_base_dir, "headless_aDN2_frtita_dist.pkl"),
        "PR_dist_tita": os.path.join(params.plotdata_base_dir, "headless_PR_frtita_dist.pkl"),
        "aDN2_dist_feti": os.path.join(params.plotdata_base_dir, "headless_aDN2_frfeti_dist.pkl"),
        "PR_dist_feti": os.path.join(params.plotdata_base_dir, "headless_PR_frfeti_dist.pkl"),
    }
    with open(detailled_files["DNp09_anus"], "rb") as f:
        DNp09_anus = pickle.load(f)
    with open(detailled_files["PR_anus"], "rb") as f:
        PR_anus = pickle.load(f)
    with open(detailled_files["aDN2_height"], "rb") as f:
        aDN2_height = pickle.load(f)
    with open(detailled_files["PR_height"], "rb") as f:
        PR_height = pickle.load(f)
    with open(detailled_files["aDN2_angle"], "rb") as f:
        aDN2_angle = pickle.load(f)
    with open(detailled_files["PR_angle"], "rb") as f:
        PR_angle = pickle.load(f)
    with open(detailled_files["aDN2_dist_tita"], "rb") as f:
        aDN2_dist_tita = pickle.load(f)
    with open(detailled_files["PR_dist_tita"], "rb") as f:
        PR_dist_tita = pickle.load(f)
    with open(detailled_files["aDN2_dist_feti"], "rb") as f:
        aDN2_dist_feti = pickle.load(f)
    with open(detailled_files["PR_dist_feti"], "rb") as f:
        PR_dist_feti = pickle.load(f)

    

    test_stats_beh_control(DNp09_anus, PR_anus, GAL4="DNp09", beh_name="anus", i_0=500, i_1=750)

    # test_stats_beh_control(aDN2_height, PR_height, GAL4="aDN2", beh_name="height", i_0=750, i_1=1000)
    # test_stats_beh_control(aDN2_angle, PR_angle, GAL4="aDN2", beh_name="angle", i_0=750, i_1=1000)
    test_stats_beh_control(aDN2_dist_tita, PR_dist_tita, GAL4="aDN2", beh_name="dist tita", i_0=500, i_1=750)
    test_stats_beh_control(aDN2_dist_feti, PR_dist_feti, GAL4="aDN2", beh_name="dist feti", i_0=500, i_1=750)


if __name__ == "__main__":
    #headless_stat_test()
    revisions_stat_test()