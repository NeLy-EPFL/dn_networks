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
DNp42 = make_dict_special_neurons("DNp42", "DNp42", color="#00D0FF")
aDN1 = make_dict_special_neurons("aDN1", "DNg_aDN1", color="#00B5DB")
DNa02 = make_dict_special_neurons("DNa02", "DNa02", color="#00A4B0")
DNa01 = make_dict_special_neurons("DNa01", "DNa01", color="#006270")
DNb02 = make_dict_special_neurons("DNb02", "DNb02", color="#000000")
oviDNa = make_dict_special_neurons("oviDNa", "DNp_oviDNa", color="#700000")
oviDNb = make_dict_special_neurons("oviDNb", "DNp_oviDNb", color="#700000")
oviDN = {**oviDNa, **oviDNb}
DNg11 = make_dict_special_neurons("DNg11", "DNg11", color="#A50000")
mute = make_dict_special_neurons("mute", "DNg_mute", color="#E3212D")
DNg14 = make_dict_special_neurons("DNg14", "DNg14", color="#F62D2D")

VALIDATION_DNS = {**DNp42,**aDN1, **DNa02, **DNa01, **DNb02, **oviDN,**DNg11,**mute, **DNg14}
KNOWN_DNS = {**REF_DNS, **VALIDATION_DNS}

#FLOWER_PLOTS_TO_MAKE = [
#    v["database_name"] for _, v in KNOWN_DNS.items()
#]
FLOWER_PLOTS_TO_MAKE = ['DNb01', 'DNg_BDN2', 'DNa03', 'DNa06', 'DNg_oDN1','DNg16', 'DNp42','DNg_aDN2','DNg11','DNp_oviDNa', 'DNp_oviDNb']
FLOWER_PLOTS_TO_MAKE.extend([
    v["database_name"] for _, v in KNOWN_DNS.items()
])


# DNs having similar behaviour phenotypes
grooming_color = plot_params.DARKRED
walking_color = plot_params.DARKGREEN
turning_color = plot_params.LIGHTGREEN
abdomen_color = plot_params.DARKPURPLE
backwards_color = plot_params.DARKCYAN
grooming_aDN1 = make_dict_special_neurons("aDN1", "DNg_aDN1", color= grooming_color)
groomig_aDN2 = make_dict_special_neurons("aDN2", "DNg_aDN2", color= grooming_color)
turning_DNa01 = make_dict_special_neurons("DNa01", "DNa01", color= turning_color)
turning_DNa02 = make_dict_special_neurons("DNa02", "DNa02", color= turning_color)
truning_DNb02 = make_dict_special_neurons("DNb02", "DNb02", color= turning_color)
walking_BDN2 = make_dict_special_neurons("BDN2", "DNg_BDN2", color= walking_color)
walking_oDN1 = make_dict_special_neurons("oDN1", "DNg_oDN1", color= walking_color)
walking_DNp09 = make_dict_special_neurons("DNp09", "DNp09", color= walking_color)
abdomen_DNg14 = make_dict_special_neurons("DNg14", "DNg14", color= abdomen_color)
abdomen_ovDNb = make_dict_special_neurons("oviDNb", "DNp_oviDNb", color= abdomen_color)
abdomen_ovDNa = make_dict_special_neurons("oviDNa", "DNp_oviDNa", color= abdomen_color)
backwards_MDN = make_dict_special_neurons("MDN", "DNp_MDN", color= backwards_color)
DNg11 = make_dict_special_neurons("DNg11", "DNg11", color= plot_params.DARKYELLOW)

BEHAVIOUR_DNS = {
    **walking_BDN2, 
    **walking_oDN1, 
    **walking_DNp09,
    **turning_DNa01,
    **turning_DNa02,
    **truning_DNb02,
    **abdomen_DNg14,
    **abdomen_ovDNb, 
    **abdomen_ovDNa,
    **DNg11,
    **grooming_aDN1,
    **groomig_aDN2,
    **backwards_MDN,
    }

# DNS strongly connected
DNb01 = make_dict_special_neurons("DNb01", "DNb01", color="#1034A6")
DNg16 = make_dict_special_neurons("DNg16", "DNg16", color="#722B6A")
DNp42 = make_dict_special_neurons("DNp42", "DNp42", color="#D3212D")

STRONGLY_CONNECTED_DNS = {
    **DNp42,
    **DNb01,
    **DNg16,
    }