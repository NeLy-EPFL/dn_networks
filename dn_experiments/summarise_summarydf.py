import pandas as pd

import params


def summarise_flies_from_data_summary(path=None, headless=False, predictions=False):
    if path is None and not headless and not predictions:
        path = params.data_summary_csv_dir
    elif path is None and headless:
        path = params.data_headless_summary_csv_dir
    elif path is None and predictions:
        path = params.data_predictions_summary_csv_dir
    df = pd.read_csv(path)

    summary_df = pd.DataFrame()
    summary_df['genotype'] = df['genotype'].unique()
    summary_df['n_flies'] = summary_df['genotype'].apply(
        lambda x: len(df[df['genotype'] == x]['fly_id'].unique()))
    summary_df['n_trials'] = summary_df['genotype'].apply(
        lambda x: len(df[df['genotype'] == x]['trial_name']))
    
    summary_df['n_good_trials'] = summary_df['genotype'].apply(
        lambda x: len(df[(df['genotype'] == x) & (df['exclude'] != True)]))
    
    summary_df['head_ball'] = summary_df['genotype'].apply(
            lambda x: len(df[(df['genotype'] == x) & (df['exclude'] != True) & (df['head'] == 1) & (df['walkon'] == 'ball')]))
    
    summary_df['nohead_ball'] = summary_df['genotype'].apply(
            lambda x: len(df[(df['genotype'] == x) & (df['exclude'] != True) & (df['head'] == 0) & (df['walkon'] == 'ball')]))
    
    summary_df['head_noball'] = summary_df['genotype'].apply(
            lambda x: len(df[(df['genotype'] == x) & (df['exclude'] != True) & (df['head'] == 1) & (df['walkon'] == 'no')]))
    
    summary_df['nohead_noball'] = summary_df['genotype'].apply(
            lambda x: len(df[(df['genotype'] == x) & (df['exclude'] != True) & (df['head'] == 0) & (df['walkon'] == 'no')]))
    
    # for each acronym in the 'experimenter' column, count the number of trials
    # that experimenter performed
    for experimenter in df['experimenter'].unique():
        summary_df[f'experimenter_{experimenter}'] = summary_df['genotype'].apply(
            lambda x: len(df[(df['genotype'] == x) & (df['experimenter'] == experimenter)]))
    
    # for each acronym in the 'experimenter' column, count the number of trials
    # that experimenter performed and for which 'exclude' is no TRUE
    for experimenter in df['experimenter'].unique():
        summary_df[f'experimenter_{experimenter}_noexclude'] = summary_df['genotype'].apply(
            lambda x: len(df[(df['genotype'] == x) & (df['experimenter'] == experimenter) & (df['exclude'] != True)]))
    
    # for each acronym in the 'experimenter', separate exp. conditions
    for experimenter in df['experimenter'].unique():
        summary_df[f'experimenter_{experimenter}_noexclude_nohead_walkonball'] = summary_df['genotype'].apply(
            lambda x: len(df[(df['genotype'] == x) & (df['experimenter'] == experimenter) & (df['exclude'] != True) & (df['head'] == 0) & (df['walkon'] == 'ball')]))
        
    for experimenter in df['experimenter'].unique():
        summary_df[f'experimenter_{experimenter}_noexclude_nohead_noball'] = summary_df['genotype'].apply(
            lambda x: len(df[(df['genotype'] == x) & (df['experimenter'] == experimenter) & (df['exclude'] != True) & (df['head'] == 0) & (df['walkon'] == 'no')]))
        
    for experimenter in df['experimenter'].unique():
        summary_df[f'experimenter_{experimenter}_noexclude_head_walkonball'] = summary_df['genotype'].apply(
            lambda x: len(df[(df['genotype'] == x) & (df['experimenter'] == experimenter) & (df['exclude'] != True) & (df['head'] == 1) & (df['walkon'] == 'ball')]))
        
    for experimenter in df['experimenter'].unique():
        summary_df[f'experimenter_{experimenter}_noexclude_head_noball'] = summary_df['genotype'].apply(
            lambda x: len(df[(df['genotype'] == x) & (df['experimenter'] == experimenter) & (df['exclude'] != True) & (df['head'] == 1) & (df['walkon'] == 'no')]))
    
    # save summary_df to csv by appending '_summary' to the original filename
    summary_df.to_csv(path[:-4] + '_summary.csv')

if __name__ == '__main__':
    summarise_flies_from_data_summary(predictions=True)





