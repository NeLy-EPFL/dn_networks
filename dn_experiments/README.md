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

**Downloading all data:**
Please download all files from the respsective Dataset on [Harvard Dataverse](https://dataverse.harvard.edu/dataverse/dn_networks) as follows:

Make a base directory called ```DN_Networks```. The code to decompress and analyse the data assumes a structure as follows:
- ```data_summary_dir = "path/to/data/DN_Networks"```
- imaging data directory --> all data from [the Optogenetics_Dfd_population_imaging dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/INYAYV)
    - ```data_summary_dir/Optogenetics_Dfd_population_imaging```
- headless and predictions directory --> all data from [the Optogenetics_headless_behavior dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6IL0X3)
    - ```data_summary_dir/Optogenetics_headless_behavior```
- supplementary directory --> all data from [the Supplementary_data dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/TZK8FA)
    - ```data_summary_dir/Supplementary_data```

**Update folder names in params.py:**
Open the [params.py](params.py) file and update the directory as follows:
- ```data_summary_dir = "path/to/data/DN_Networks"```
- Verify that the relative folder names for imaging_data_dir, headless_predictions_data_dir, other_data_dir are correct. (They should be in case you followed above instructions.)

**Decompress the data**
Run the [decompress_data.py](decompress_data.py) script in this folder to assemble the data into its original folder structure.

**Making all figures:**
Run the [make_all_plots.py](make_all_plots.py) script in this folder to create all plots from experimental data.
