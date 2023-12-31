"""
Script used to decompress all data downloaded from Harvard Dataverse.
Please read the README.md and follow the instructions where to download the data.
Author: jonas.braun@epfl.ch
"""
import os

from copy_data import decompress_imaging, decompress_headless_predictions, decompress_other
import params
if __name__ == "__main__":
    print("Decompressing other data")
    decompress_other(params.other_data_dir)
    print("Decompressing headless and predictions data")
    decompress_headless_predictions(params.headless_predictions_data_dir)
    print("Decompressing imaging data")
    decompress_imaging(params.imaging_data_dir)