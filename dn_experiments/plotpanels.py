import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from twoppp import plot as myplt
from twoppp import utils

import params, behaviour

linewidth = 2
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['axes.labelpad'] = 5

def make_nice_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('outward', 2*linewidth))
    ax.spines['bottom'].set_position(('outward',2*linewidth))
    ax.tick_params(width=linewidth)
    ax.tick_params(length=2.5*linewidth)
    ax.tick_params(labelsize=16)
    ax.spines["left"].set_linewidth(linewidth)
    ax.spines["bottom"].set_linewidth(linewidth)
    ax.spines["top"].set_linewidth(0)
    ax.spines["right"].set_linewidth(0)

def plot_ax_behavioural_response(beh_responses, response_name=None, response_ylabel=None, ax=None, beh_responses_2=None, beh_response_2_color=myplt.BLACK, x="2p", ylim=None):

    ax = plt.gca() if ax is None else ax
    if x == "2p":
        response_t_params = params.response_t_params_2p
        fs = params.fs_int
    elif x == "beh":
        response_t_params = params.response_t_params_beh
        fs = params.fs_beh
    x = np.arange(np.sum(response_t_params))/fs-params.response_t_params_2p_label[0]
    
    myplt.shade_categorical(catvar=np.concatenate((np.zeros((response_t_params[0])),
                                                           np.ones((response_t_params[1])),
                                                           np.zeros((response_t_params[2])))),
                                    x=x,colors=[myplt.WHITE, myplt.BLACK], ax=ax)

    if beh_responses is not None and beh_responses != [] and not np.all(np.isnan(beh_responses)):
        myplt.plot_mu_sem(mu=np.mean(beh_responses, axis=-1), err=utils.conf_int(beh_responses, axis=-1),
                            x=x, ax=ax, color=myplt.BLACK, linewidth=2*linewidth)
    if beh_responses_2 is not None:
        myplt.plot_mu_sem(mu=np.mean(beh_responses_2, axis=-1), err=utils.conf_int(beh_responses, axis=-1),
                            x=x, ax=ax, color=beh_response_2_color, linewidth=2*linewidth)
            
    ax.set_xticks([0,params.response_t_params_2p_label[1]])
    ax.set_xticklabels(["", ""])
    # ax.set_xlabel("t (s)")    
    ax.set_ylabel(response_ylabel)
    ax.set_title(response_name)  # +f": {beh_responses.shape[-1]} trials")
    if ylim is not None:
        ax.set_ylim(ylim)
    make_nice_spines(ax)
    ax.spines['bottom'].set_position('zero')


def plot_ax_allneurons_response(stim_responses, response_ylabel=None, ax=None):
    ax = plt.gca() if ax is None else ax
    x = np.arange(np.sum(params.response_t_params_2p))/params.fs_int-params.response_t_params_2p_label[0]
    stim_response = np.mean(stim_responses, axis=-1)


    myplt.shade_categorical(catvar=np.concatenate((np.zeros((params.response_t_params_2p[0])),
                                                           np.ones((params.response_t_params_2p[1])),
                                                           np.zeros((params.response_t_params_2p[2])))),
                                    x=x,colors=[myplt.WHITE, myplt.BLACK], ax=ax)
    if stim_response.size:
        ax.plot(x, stim_response, alpha=0.3)
 
    ax.set_xticks([0,params.response_t_params_2p_label[1]])
    # ax.set_xlabel("t (s)")
    ax.set_ylabel(response_ylabel)
    make_nice_spines(ax)
    ax.spines['bottom'].set_position('zero')


def plot_ax_allneurons_confidence(stim_responses, clim=None, sort_ind=None, cmap=params.cmap_ci, ax=None, ylabel="Annotated neurons", title="|mean| > CI"):
    ax = plt.gca() if ax is None else ax
    stim_response = np.mean(stim_responses, axis=-1)

    i_0 = int(params.response_t_params_2p[0])
    i_b = int(i_0 - params.response_n_baseline)
    rel_stim_response = stim_response-np.mean(stim_response[i_b:i_0], axis=0)
    stim_conf = utils.conf_int(stim_responses, axis=-1)
    rel_resp_conf = rel_stim_response.copy()
    rel_resp_conf[np.abs(rel_stim_response) < stim_conf] = np.nan

    clim = [np.min(rel_resp_conf), np.max(rel_resp_conf)] if clim is None else clim
    sort_ind = np.arange(rel_resp_conf.shape[1]) if sort_ind is None else sort_ind

    ax.imshow(rel_resp_conf.T[sort_ind], aspect=7, clim=[-clim,clim], cmap=cmap, interpolation='none')
    
    ax.set_xticks([params.response_t_params_2p[0],params.response_t_params_2p[0]+params.response_t_params_2p[1]])
    ax.set_xticklabels([0,params.response_t_params_2p_label[1]])
    ax.set_yticks([])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    make_nice_spines(ax)
    ax.spines['left'].set_visible(False)
    

def plot_ax_cbar(fig, ax, clim, cmap=params.cmap_ci, clabel=None):
    cbar1 = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.NoNorm(), cmap=cmap), cax=ax,
                         ticks=np.linspace(0,255, 3), label=clabel)
    cbar1.set_ticklabels([f"{-clim:.1f}", 0, f"{clim:.1f}"])
    cbar1.outline.set_visible(False)
    # cbar1.spines["right"].set_linewidth(2)
    cbar1.outline.set_linewidth(2)
    cbar1.ax.tick_params(width=2.5)
    cbar1.ax.tick_params(length=5)
    cbar1.ax.tick_params(labelsize=16)

# plot_responses_summary_on_ax
def plot_ax_response_summary(background_image, roi_centers, ax, 
                   response_values=None, response_values_left=None, response_values_right=None,
                   min_dot_size=params.map_min_dot_size, max_dot_size=params.map_max_dot_size,
                   min_dot_alpha=params.map_min_dot_alpha, max_dot_alpha=params.map_max_dot_alpha,
                   norm_name="", response_name="", response_name_left="", response_name_right="", fly_name="",
                   cmap=params.map_cmap_dft, crop_x=params.map_crop_x, q_max=params.map_q_max, clim=None, title=None):
    ax.imshow(background_image, cmap=plt.cm.binary, clim=[np.quantile(background_image, 0.01), np.quantile(background_image, 0.99)])
    
    if clim is not None:
        response_max = clim
    elif response_values is not None:
        if not len(response_values):
            print(response_values)
            print("COULD NOT PLOT")
            return ax
        response_max = np.quantile(np.abs(response_values), q=q_max)
    elif response_values_left is not None and response_values_right is not None:
        response_max = np.quantile(np.concatenate((np.abs(response_values_left), np.abs(response_values_right)), axis=0), q=q_max)
    else:
        raise NotImplementedError()
    cnorm = mpl.colors.CenteredNorm(vcenter=0, halfrange=response_max)
    
    no_response = (response_values is None or not response_values.size) and \
                  (response_values_left is None or not response_values_left.size or \
                   response_values_right is None or not response_values_right.size)
    if not len(roi_centers) or no_response:
        if len(response_name):
            ax.set_title(f"{norm_name.upper()} response to {response_name}\n{fly_name}")
        else:
            ax.set_title(f"{fly_name}")
    else:
        x = np.array(roi_centers)[:,1]
        y = np.array(roi_centers)[:,0]
        
        if response_values is not None:
            s = np.clip(np.abs(response_values) / response_max * (max_dot_size-min_dot_size) + min_dot_size,
                        a_min=min_dot_size, a_max=max_dot_size)
            a = np.clip(np.abs(response_values) / response_max * (max_dot_alpha-min_dot_alpha) + min_dot_alpha,
                        a_min=min_dot_alpha, a_max=max_dot_alpha)
            p = ax.scatter(x=x, y=y, s=s, c=response_values, edgecolors="none", marker=mpl.markers.MarkerStyle("o", fillstyle="full"),
                cmap=cmap, norm=cnorm)
            ax.set_title(f"{norm_name.upper()} response to {response_name}\n{fly_name}" if title is None else title)
        elif response_values_left is not None and response_values_right is not None:
            s1 = np.clip(np.abs(response_values_left) / response_max * (max_dot_size-min_dot_size) + min_dot_size,
                        a_min=min_dot_size, a_max=max_dot_size)
            s2 = np.clip(np.abs(response_values_right) / response_max * (max_dot_size-min_dot_size) + min_dot_size,
                        a_min=min_dot_size, a_max=max_dot_size)

            a1 = np.clip(np.abs(response_values_left) / response_max * (max_dot_alpha-min_dot_alpha) + min_dot_alpha,
                        a_min=min_dot_alpha, a_max=max_dot_alpha)
            a2 = np.clip(np.abs(response_values_right) / response_max * (max_dot_alpha-min_dot_alpha) + min_dot_alpha,
                        a_min=min_dot_alpha, a_max=max_dot_alpha)

            if not isinstance(cmap, list):
                cmap = [cmap, cmap]
            p1 = ax.scatter(x=x, y=y, s=s1, c=response_values_left, edgecolors="none", marker=mpl.markers.MarkerStyle("o", fillstyle="left"),
                    cmap=cmap[0], norm=cnorm)
            p2 = ax.scatter(x=x, y=y, s=s2, c=response_values_right, edgecolors="none", marker=mpl.markers.MarkerStyle("o", fillstyle="right"),
                    cmap=cmap[1], norm=cnorm)
            ax.set_title(f"{norm_name.upper()} response to {response_name_left} (left) and {response_name_right} (right)\n{fly_name}" if title is None else title)

    # ax.set_ylim([250,50])
    ax.set_xlim([crop_x,background_image.shape[1]-crop_x])
    ax.axis("off")
    return ax

def plot_ax_multi_fly_response_density(roi_centers, response_values, background_image, ax, n_flies=None, clim=[-1,1],
                                       cmap=params.map_cmap_dft, crop_x=params.map_crop_x):
    ax = plt.gca() if ax is None else ax
    ax.imshow(background_image, cmap=plt.cm.binary, clim=[np.quantile(background_image, 0.01), np.quantile(background_image, 0.99)])

    response_img = np.zeros_like(background_image, dtype=float)

    for roi_center, response_value in zip(roi_centers, response_values):
        response_img[roi_center[0], roi_center[1]] += response_value
    
    response_img_filt = gaussian_filter(response_img, sigma=(25,25))  # TODO: set external params
    factor_filt_25 = 0.00025467751790182654  # computed by appling Gaussian filter to Diraq pulse and checking value at center location

    response_img_filt /= factor_filt_25
    if n_flies is not None:
        response_img_filt /= n_flies

    ax.imshow(response_img_filt, clim=[-clim,clim], cmap=cmap)
    # ax.set_xlim([crop_x,background_image.shape[1]-crop_x])
    
    # ax.axis("off")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("density")

def plot_ax_behclass(labels, ax=None, cmap=behaviour.beh_cmap, cnorm=behaviour.beh_cnorm, dy=0.1, xlabel="Time (s)", ylabel=""):
    ax = plt.gca() if ax is None else ax
    x = np.arange(np.sum(params.response_t_params_beh))/params.fs_beh-params.response_t_params_2p_label[0]

    labels = labels.T
    # def plot_behaviour_classes(labels, x=None, classes=None, ):
    if len(labels.shape) == 1:
        labels = np.expand_dims(labels, axis=0)
    N_rep, N_s = labels.shape
    x = np.arange(N_s) if x is None else x
    dx = np.mean(np.diff(x))
    
    # if cnorm is None:
    #     bounds = np.linspace(-0.5, len(classes)-0.5, len(classes)+1)
    #     cnorm =  mpl.colors.BoundaryNorm(bounds, ncolors=256)
    cmap = plt.cm.get_cmap(cmap) if (cmap is None or isinstance(cmap, str)) else cmap
    
    for i_rep in range(N_rep):
        i_start = 0
        for i_s in range(N_s):
            if i_s < N_s-1:
                if labels[i_rep, i_s] == labels[i_rep, i_s+1]:
                    continue
            rect = plt.Rectangle((x[i_start], i_rep), width=(i_s-i_start+1)*dx, height=1-dy, color=cmap(cnorm(labels[i_rep, i_s])),
                                snap=False)
            ax.add_patch(rect)
            i_start = i_s + 1
    ax.set_xlim((x[0], x[-1]))
    ax.set_ylim((N_rep, 0))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("behaviour classes")

    make_nice_spines(ax)

def plot_ax_behprob(labels, ax=None, cmap=behaviour.beh_cmaplist.copy(), beh_mapping=behaviour.beh_mapping.copy(),
                    collapse_groom=behaviour.collapse_groom, xlabel="Time (s)", ylabel="Behaviour\nprobability", labels_2=None, beh_2=None):
    ax = plt.gca() if ax is None else ax
    x = np.arange(np.sum(params.response_t_params_beh))/params.fs_beh-params.response_t_params_2p_label[0]
    if labels is None or np.all(np.isnan(labels)):
        return ax
    n_beh = len(cmap)
    beh_prob = np.zeros((labels.shape[0], n_beh))
    for i in range(n_beh):
        beh_prob[:,i] = np.mean(labels == i, axis=1)
    if collapse_groom:  # TODO: make this more general to fit any behavioural mapping.
        beh_prob[:,4] = beh_prob[:,4] + beh_prob[:,5]
        beh_prob[:,5] = beh_prob[:,6]
        beh_prob = beh_prob[:,:6]
        cmap[5] = cmap[6]
        cmap = cmap[:6]
        beh_mapping[5] = beh_mapping[6]
        beh_mapping = beh_mapping[:6]
        # n_beh -= 1

    cmap[0] = myplt.DARKGRAY

    myplt.shade_categorical(catvar=np.concatenate((np.zeros((params.response_t_params_beh[0])),
                                                           np.ones((params.response_t_params_beh[1])),
                                                           np.zeros((params.response_t_params_beh[2])))),
                                    x=x,colors=[myplt.WHITE, myplt.BLACK], ax=ax)

    if labels_2 is None or beh_2 is None:
         _ = [ax.plot(x, beh_prob[:,i], color=cmap[i], linewidth=2*linewidth) for i in range(len(cmap))]
    else:
        beh_prob_2 = np.zeros((labels_2.shape[0], n_beh))
        for i in range(n_beh):
            beh_prob_2[:,i] = np.mean(labels_2 == i, axis=1)
        if collapse_groom:  # TODO: make this more general to fit any behavioural mapping.
            beh_prob_2[:,4] = beh_prob_2[:,4] + beh_prob_2[:,5]
            beh_prob_2[:,5] = beh_prob_2[:,6]
            beh_prob_2 = beh_prob_2[:,:6]

        if beh_2 == "olfac":
            beh_2 = "groom"
        ax.plot(x, beh_prob[:,beh_mapping.index(beh_2)], color=myplt.BLACK, linewidth=2*linewidth)
        ax.plot(x, beh_prob_2[:,beh_mapping.index(beh_2)], color=cmap[beh_mapping.index(beh_2)], linewidth=2*linewidth)
        

    ax.set_xticks([0,params.response_t_params_2p_label[1]])
    ax.set_ylim([-0.01,1.01])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.set_title("behaviour classes")

    make_nice_spines(ax)
