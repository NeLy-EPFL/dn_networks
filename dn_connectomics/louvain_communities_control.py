"""
2023.08.30
author: femke.hurtak@epfl.ch
Comparison of the clustering results with a shuffled control.
"""

import os
import pandas as pd
from scipy import stats

import plot_params


def get_cluster_stats(working_folder):
    # Open clustering_stats.csv as pd.DataFrame
    df_data = pd.read_csv(
        os.path.join(working_folder, "data", "clustering_stats.csv"),
        index_col=0,
    )
    df_control = pd.read_csv(
        os.path.join(
            working_folder, "shuffled_control", "clustering_stats.csv"
        ),
        index_col=0,
    )

    stats_df = (
        pd.DataFrame()
    )  # rows: data, control; columns: features [mean, std, ttest]
    for column in df_data.columns:
        stats_df[column + "_mean"] = [
            df_data[column].mean(),
            df_control[column].mean(),
        ]
        stats_df[column + "_std"] = [
            df_data[column].std(),
            df_control[column].std(),
        ]
        stats_df[column + "_less_ttest"] = stats.ttest_ind(
            df_data[column],
            df_control[column],
            equal_var=False,
            alternative="less",
        )
        stats_df[column + "_greater_ttest"] = stats.ttest_ind(
            df_data[column],
            df_control[column],
            equal_var=False,
            alternative="greater",
        )
    stats_df.index = ["data", "control"]
    stats_df = stats_df.T
    return stats_df


def compare_clustering_with_random():
    working_folder = plot_params.CLUSTERING_ARGS["folder"]
    stats_df = get_cluster_stats(working_folder)
    stats_df.to_csv(
        os.path.join(working_folder, "clustering_stats_comparison.csv")
    )


if __name__ == "__main__":
    compare_clustering_with_random()
