import os
import pandas as pd
import math
import ast
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

# Disable scientific notation for pandas display
pd.set_option('display.float_format', lambda x: '%.10f' % x)




def main_routine():

    # Directory path to search through
    directory_path = r'H:\biomass\model_REG16_test_train'

    # directory_path = r'H:\biomass'

    # List to store the DataFrames
    df_list = []

    # Walk through the directory and subdirectories
    for root, dirs, files in os.walk(directory_path):
        #print(root,dirs, directory_path)
        for file in files:
            if file.endswith("metrics.csv"):
                # Full path to the CSV file
                file_path = os.path.join(root, file)
                print(file_path)
                # Read the CSV file and append to the list
                df = pd.read_csv(file_path)
                print(list(df))

                # Convert r2_x and other columns from scientific notation to float if needed
                df['r2'] = df['r2'].astype(float)
                df['rmse'] = df['rmse'].astype(float)
                # Repeat for any other columns that might be in scientific notation

                df_list.append(df)

    # Concatenate all DataFrames in the list
    combined_df = pd.concat(df_list, ignore_index=True)
    # combined_df['file'] = combined_df.apply(lambda row: f"{row['var']}_{row['fac']}_{row['mdl']}_sel_{int(row['sel_num'])}_variable_score.csv", axis=1)

    combined_df['file'] = combined_df.apply(
        lambda
            row: f"{row['var']}_{row['fac']}_{row['mdl']}_sel_{int(row['sel_num']) if not math.isnan(row['sel_num']) else ''}_variable_score.csv",
        axis=1
    )
    # import ace_tools as tools; tools.display_dataframe_to_user(name="Combined Metrics DataFrame", dataframe=combined_df)

    # Create 'season' feature from the first two characters of 'var'
    combined_df['season'] = combined_df['var'].str[:2]

    # Create 'group' feature by splitting 'var' on "_" and keeping the last part
    combined_df['group'] = combined_df['var'].str.split('_').str[-1]

    # Display the updated DataFrame
    print("DataFrame with new features 'season' and 'group':")
    print(combined_df[['var', 'season', 'group']])

    # Count the occurrences of each feature group
    group_counts = combined_df['group'].value_counts()

    # Create a new column with the total times each group exists
    combined_df['group_count'] = combined_df['group'].map(group_counts)

    # Count the occurrences of each feature group
    group_counts = combined_df['season'].value_counts()

    # Create a new column with the total times each group exists
    combined_df['season_count'] = combined_df['season'].map(group_counts)


    combined_df.to_csv(os.path.join(directory_path, "total_metrics.csv"), index=False)


    # ----------------------------------- Predicted_Data -------------------
    # List to store the DataFrames
    df_list = []

    # Walk through the directory and subdirectories
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith("retest_predicted_data.csv"):
                # Full path to the CSV file
                file_path = os.path.join(root, file)
                print(file_path)
                # Read the CSV file and append to the list
                df = pd.read_csv(file_path)

                df_list.append(df)

    # Concatenate all DataFrames in the list
    combined_df = pd.concat(df_list, ignore_index=True)
    # combined_df['file'] = combined_df.apply(lambda row: f"{row['var']}_{row['fac']}_{row['mdl']}_sel_{int(row['sel_num'])}_variable_score.csv", axis=1)

    combined_df['file'] = combined_df.apply(
        lambda
            row: f"{row['var']}_{row['fac']}_{row['mdl']}_sel_{int(row['sel_num']) if not math.isnan(row['sel_num']) else ''}_variable_score.csv",
        axis=1
    )
    combined_df.to_csv(os.path.join(directory_path, "anova_predicted_metrics.csv"), index=False)


if __name__ == '__main__':
    main_routine()