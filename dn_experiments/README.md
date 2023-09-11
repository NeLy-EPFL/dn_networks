# Code used analyse experimental data and to reproduce the figures.

**Installation:**
Create and activate a new conda environment
- ```conda create -n twoppp37 python=3.7```
- ```conda activate twoppp37```
Clone and install th dn_interactions repository:
- ```git clone https://github.com/NeLy-EPFL/dn_interations```
- ```cd dn_interations```
- ```pip install -e .```
- fix numpy installation: ```pip install numpy --upgrade```

**Making all figures:**
Run the [make_all_plots.py](make_all_plots.py) script in this folder to create all plots from experimental data.
