import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pingouin as pg
import scipy.stats as stats
import numpy as np

def get_classification_type(row):
    # Participant said "Depressed" if Confidence_Level > 5, otherwise "Non-Depressed"
    if row['Confidence_Level'] > 5:
        # Participant said depressed: if the clip is actually depressed, label it TP; otherwise, FP.
        if row['Clip_Type'] in ['TP', 'FN']:
            return 'TP'
        else:
            return 'FP'
    else:
        # Participant said non-depressed: if the clip is actually non-depressed, label it TN; otherwise, FN.
        if row['Clip_Type'] in ['TN', 'FP']:
            return 'TN'
        else:
            return 'FN'

def format_series(label, series):
    # Start with a comment indicating the label (e.g., //Salient or //Selected)
    s = f"[//{label}\n"
    for item in series:
        s += f'    {{axis:"{item["axis"]}", value:{item["value"]}}},\n'
    s += "],"
    return s

def split_features(s):
    if pd.isna(s):
        return []
    return [x.strip() for x in s.split(';') if x.strip()]

def calculate_radar_data(df):
    # Process each of the three feature columns separately
    # feature_columns = ['Influential_Features-Eyebrows', 
    #                 'Influential_Features-Eyes', 
    #                 'Influential_Features-Mouth']
    feature_columns = ['Influential_Features-Voice']

    for col in feature_columns:
        df[col + '_list'] = df[col].apply(split_features)

    # Combine the lists from the three columns into one column per row
    # df['All_Features'] = df[[col + '_list' for col in feature_columns]].apply(
    #     lambda row: row[0] + row[1] + row[2], axis=1
    # )
    df['All_Features'] = df[[col + '_list' for col in feature_columns]].apply(
    lambda row: row[0], axis=1
    )

    # Explode the All_Features column for the entire DataFrame
    exploded = df.explode('All_Features')
    
    # Count the occurrences of each feature across all clips
    feature_counts = exploded['All_Features'].value_counts().reset_index()
    feature_counts.columns = ['axis', 'value']
    
    # Normalize the counts (e.g., scale the highest count to 1)
    max_value = feature_counts['value'].max()
    feature_counts['value'] = feature_counts['value'] / max_value
    
    # Return the aggregated radar data as a list of dictionaries
    return feature_counts.to_dict(orient='records')

    # #  Group by Clip_ID (assuming there's a Clip_ID column)
    # grouped = df.groupby('Clip_ID')

    # # Create a dictionary to hold the radar data for each clip
    # radar_data_by_clip = {}

    # for clip_id, group in grouped:
    #     # Explode the All_Features column for the current clip
    #     exploded = group.explode('All_Features')
    #     feature_counts = exploded['All_Features'].value_counts().reset_index()
    #     feature_counts.columns = ['axis', 'value']

    #     #     # Print the non-normalized counts for this clip
    #     # print(f"Non-normalized counts for {clip_id}:")
    #     # print(feature_counts)
        
    #     # Optional: Normalize the counts (e.g., max becomes 1)
    #     max_value = feature_counts['value'].max()
    #     feature_counts['value'] = feature_counts['value'] / max_value
        
    #     # Save as a list of dicts for this clip
    #     radar_data_by_clip[clip_id] = feature_counts.to_dict(orient='records')
    # return radar_data_by_clip

folder = os.path.join(os.path.expanduser("~"), "Downloads")

data = pd.read_excel('real_experiment_results.xlsx', sheet_name='table')
data2 = pd.read_excel('real_experiment_results.xlsx', sheet_name='salient-segments')

# radar_data_by_clip_selected = calculate_radar_data(data)
# radar_data_by_clip_salient = calculate_radar_data(data2)
# Compute the Classification_Type column first
data['Classification_Type'] = data.apply(get_classification_type, axis=1)
data2['Classification_Type'] = data['Classification_Type']  # assuming same order

# Filter to include only clips that are correctly classified as depressed (TP).
# Here, "TP" means the clip is depressed (Clip_Type in ['TP','FN']) and the participant said depressed (Confidence_Level > 5).
# correct_depressed_data = data[(data['Classification_Type'] == 'TP') & (data['Clip_Type'].isin(['TP','FN']))]
# correct_depressed_salient = data2[(data2['Classification_Type'] == 'TP') & (data2['Clip_Type'].isin(['TP','FN']))]
correct_depressed_data = data[(data['Classification_Type'] == 'TN') & (data['Clip_Type'] == 'FP')]
correct_depressed_salient = data2[(data2['Classification_Type'] == 'TN') & (data2['Clip_Type'] == 'FP')]


# Compute the radar data dictionaries from the filtered data.
selected_clip = calculate_radar_data(correct_depressed_data)
salient_clip = calculate_radar_data(correct_depressed_salient)
# For the comparison, we want to compare the data for the same clip.
# Let's choose "Clip 2". We need to make sure both series have the same set of axes.
# selected_clip = radar_data_by_clip_selected['Clip 3']
# salient_clip = radar_data_by_clip_salient['Clip 3']

# Create a master list (union) of axes from both series
all_axes = set(item['axis'] for item in selected_clip) | set(item['axis'] for item in salient_clip)
all_axes = sorted(all_axes)  # sort for consistency

# Function to fill in missing axes with a value of 0
def fill_missing(series, master_axes):
    d = {item['axis']: item['value'] for item in series}
    return [{"axis": ax, "value": d.get(ax, 0)} for ax in master_axes]

# Fill missing features for both series
selected_filled = fill_missing(selected_clip, all_axes)
salient_filled = fill_missing(salient_clip, all_axes)
# Example output for one clip:
# print(radar_data_by_clip['Clip 1'])
# print(radar_data_by_clip['Clip2'])
# print(radar_data_by_clip_selected['Clip 2'])

# Now, format the output for the radar chart
formatted_output = "[\n" + format_series("Salient", salient_filled) + "\n" + format_series("Selected", selected_filled) + "\n]"
print(formatted_output)


