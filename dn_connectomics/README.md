Code used for the connectomics analysis.

# Installation
This part of the code used different libraries than the experimental part. 
Therefore, it is recommended to install the code in a separate virtual environment. 

Installing another virtual environment:
```
conda deactivate
conda env create -f connectomics_environment.yml
conda activate connectomics
```

# Free parameters
- All the parameters that are used in the analysis are stored in the `params.py` file. This file is used by all the other scripts to load the parameters. This includes the paths to the data and the parameters for the analysis.
- All the parameters used for specific neurons used in this work are defined in the `neuron_params.py` file.
- All the parameters used for plotting are defined in the `plot_params.py` file.

# Data

## Raw data
The raw data is taken from the FlyWire project (https://www.flywire.ai/). The data is not included in this repository, but can be downloaded from the FlyWire website (https://codex.flywire.ai/api/download).

## Preprocessed data
The data is preprocessed using the `data_prep.py` script. This script takes the raw data and converts it to a format that is easier to work with. This dat is stored in the `data` folder as .pkl files. 

## Processed data
To generate all the connectomics plots displayed in the paper, run the script `make_all_plots.py`. This script will generate all the plots and store them in the `figures` folder, along with the data used to generate the plots.