"""
2023.08.30
author: femke.hurtak@epfl.ch
Script to refer to the neurons used in the experiments.
"""

from loaddata import get_rootids_from_name
import plot_params


# Reference DNs in our work
def make_dict_special_neurons(name, database_name, color=plot_params.DARKBLUE):
    """
    Define a dictionary of special neurons for the validation experiments.
    """
    dict_ = {
        neuron_id: {
            "root_id": neuron_id,
            "color": color,
            "name": name,
            "database_name": database_name,
        }
        for neuron_id in get_rootids_from_name(database_name)
    }
    return dict_


aDN2 = make_dict_special_neurons("DNg_aDN2", "DNg_aDN2", color=plot_params.DARKRED)
DNp09 = make_dict_special_neurons(
    "DNp09", "DNp09", color=plot_params.DARKGREEN
)
MDN = make_dict_special_neurons("DNp_MDN", "DNp_MDN", color=plot_params.DARKCYAN)
REF_DNS = {**aDN2, **DNp09, **MDN}


# DNs used in the validation experiments


aDN1 = make_dict_special_neurons("aDN1", "DNg_aDN1", color="#1034A6")
DNa02 = make_dict_special_neurons("DNa02", "DNa02", color="#412F88")
DNa01 = make_dict_special_neurons("DNa01", "DNa01", color="#722B6A")
DNb02 = make_dict_special_neurons("DNb02", "DNb02", color="#A2264B")
mute = make_dict_special_neurons("mute", "DNg_mute", color="#D3212D")
DNg14 = make_dict_special_neurons("DNg14", "DNg14", color="#F62D2D")

VALIDATION_DNS = {**aDN1, **DNa02, **DNa01, **DNb02, **mute, **DNg14}
KNOWN_DNS = {**REF_DNS, **VALIDATION_DNS}

#FLOWER_PLOTS_TO_MAKE = [
#    v["database_name"] for _, v in KNOWN_DNS.items()
#]
FLOWER_PLOTS_TO_MAKE = ['DNb01', 'DNg_BDN2', 'DNa03', 'DNa06', 'DNg_oDN1','DNg16', 'DNp42','DNg_aDN2']
FLOWER_PLOTS_TO_MAKE.extend([
    v["database_name"] for _, v in KNOWN_DNS.items()
])
#FLOWER_PLOTS_TO_MAKE = ['DNg_BDN2','DNg_oDN1']