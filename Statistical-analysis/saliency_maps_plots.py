import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from scipy.stats import pearsonr
from scipy.stats import pointbiserialr
# def parse_interval(interval_str):
#     """Parse something like '2-3 sec' into numeric floats (2.0, 3.0)."""
#     interval_str = interval_str.replace(' sec', '').strip()
#     start_str, end_str = interval_str.split('-')
#     return float(start_str), float(end_str)

# # -------------------- LOAD PROCESSED SALIENCY DATA -------------------- #
# saliency_values = np.load('saliency_values_308.npy', allow_pickle=True)
# salient_time = np.load('salient_time_308.npy', allow_pickle=True)
# smoothed_saliencies = np.load('smoothed_saliencies_308.npy', allow_pickle=True)

# # Create time axis from 0..8.5 seconds
# time_axis = np.linspace(0, 8.5, num=len(saliency_values))

# -------------------- LOAD & FILTER EXCEL DATA -------------------- #
# df = pd.read_excel('real_experiment_results.xlsx')
# df_clip = df[df['Clip_ID'] == 'Clip 1']  # adjust as needed

# # # Get unique clip IDs from the Excel file
# # clip_ids = df['Clip_ID'].unique()

# # If timestamps are slightly off but should be considered the same,
# # you might do: round_ts = df_clip['Selected_Timestamp'].round(1)
# # and then convert to list. For now, we assume exact timestamps.
# selected_timestamps = df_clip['Selected_Timestamp'].astype(float).tolist()
# print(selected_timestamps)
# selected_timestamps[0] = selected_timestamps[1]
# -------------------- STACKED RUG LOGIC -------------------- #
# # Count how many participants share each *exact* timestamp
# time_groups = defaultdict(int)
# for ts in selected_timestamps:
#     time_groups[ts] += 1

# # We'll remove the old "density" approach entirely,
# # focusing on just the saliency + stacked rug.

# plt.figure(figsize=(10, 5))
# ax = plt.gca()

# # Plot raw and smoothed saliency
# # ax.plot(time_axis, saliency_values, 'o-', alpha=0.3, label='Raw Saliency')
# ax.plot(time_axis, smoothed_saliencies, 'r-', label='Smoothed Saliency')
# ax.axvline(x=salient_time, color='maroon', linestyle='--', label='Salient Segment Start')
# ax.set_xlabel('Time (seconds)')
# ax.set_ylabel('Saliency Score')
# ax.legend(loc='upper right')

# # We'll place stacked markers at the bottom of the plot.
# y_min, y_max = ax.get_ylim()

# # Decide how tall each "stack level" should be
# tick_height = 0.02 * (y_max - y_min)  # space between stacks
# marker_size = 5  # vertical marker size,

# # Make sure we have enough vertical space for the highest stack
# max_stack = max(time_groups.values()) if time_groups else 0
# extra_space = max_stack * tick_height
# if y_min + extra_space > y_max:
#     ax.set_ylim(y_min, y_min + extra_space + 0.1*(y_max - y_min))

# # For each unique time, stack the markers from bottom up
# for t in sorted(time_groups.keys()):
#     count = time_groups[t]
#     # We stack them from y_min upward
#     for level in range(count):
#         y_stack = y_min + level * tick_height
#         # Plot a small vertical marker
#         # marker='|' draws a vertical line; you could do marker='o' for dots
#         ax.plot(t, y_stack, marker='o', color='navy', markersize=marker_size)

# plt.title('Saliency vs. Stacked Rug of Participant Timestamps (Clip 1)')
# plt.tight_layout()
# plt.show()


# # --------------------  -------------------- -------------------------------------------------------------------------------------------#
# # Lists to store our aggregate metrics across clips
# clip_ids = ['Clip 1']  # adjust as needed
# correlations = []   # for Option 2
# # Loop over each clip
# for clip in clip_ids:
#     # 1. Load the saliency data for this clip.
#     #    (Assume files are named like 'saliency_values_<clip>.npy'. Adjust as needed.)
#     try:
#         saliency_values = np.load(f'saliency_values_308.npy', allow_pickle=True)
#     except Exception as e:
#         print(f"Could not load saliency data for {clip}: {e}")
#         continue

#     # Create a time axis (assuming saliency covers 0 to 8.5 seconds)
#     time_axis = np.linspace(0, 8.5, num=len(saliency_values))
    
#     # 2. Filter the Excel data for this clip
#     df_clip = df[df['Clip_ID'] == clip]
#     intervals = []
#     for _, row in df_clip.iterrows():
#         start, end = parse_interval(row['Selected_Timestamp'])
#         intervals.append((start, end))
    
#     # 3. Compute the human vote density along the time axis
#     # Use a fixed number of bins, e.g., equal to the length of saliency_values
#     time_bins = np.linspace(0, 8.5, len(saliency_values) + 1)
#     density_counts = np.zeros(len(time_bins) - 1)
#     for (start, end) in intervals:
#         for i in range(len(time_bins) - 1):
#             bin_center = 0.5 * (time_bins[i] + time_bins[i+1])
#             if start <= bin_center <= end:
#                 density_counts[i] += 1
#     # Note: density_counts will be an array of the same length as saliency_values
    
#     # -------------------- Option 2: Correlation Analysis -------------------- #
#     # Compute Pearson correlation between saliency and density counts
#     if len(saliency_values) == len(density_counts):
#         corr, _ = pearsonr(saliency_values, density_counts)
#         correlations.append(corr)
#     else:
#         # If lengths differ, skip or interpolate (for now, we assume they match)
#         continue

# print("Correlations:", correlations)
# print("Number of correlation values:", len(correlations)) 
#     # Option 2: Correlation Histogram (or Boxplot)
# plt.figure(figsize=(8, 5))
# plt.hist(correlations, bins=10, color='skyblue', edgecolor='black')
# plt.xlabel("Pearson Correlation")
# plt.ylabel("Number of Clips")
# plt.title("Distribution of Saliency vs. Selection Density Correlations")
# plt.show()
# --------------------  -------------------- -----------------------------------------------------------------------------------------------#  

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from scipy.stats import pearsonr

# # -------------------- LOAD PROCESSED SALIENCY DATA -------------------- #
# saliency_values = np.load('saliency_values_308.npy', allow_pickle=True)
# # (Adjust filename if needed: e.g., "saliency_values_Clip 1.npy")

# # Create a time axis from 0..8.5 seconds
# time_axis = np.linspace(0, 8.5, num=len(saliency_values))

# # -------------------- LOAD & FILTER EXCEL DATA -------------------- #
# df = pd.read_excel('experiment_results.xlsx')
# df_clip = df[df['Clip_ID'] == 'Clip 1']  # or whichever clip you want

# # Convert 'Selected_Timestamp' to a float list
# timestamps = df_clip['Selected_Timestamp'].astype(float).tolist()

# # -------------------- COMPUTE HUMAN "VOTE" DENSITY FROM TIMESTAMPS -------------------- #
# # We'll create bins matching the saliency array length.
# time_bins = np.linspace(0, 8.5, len(saliency_values) + 1)
# density_counts = np.zeros(len(saliency_values))  # one count per bin

# # For each participant's single timestamp, increment exactly one bin
# for ts in timestamps:
#     for i in range(len(time_bins) - 1):
#         if time_bins[i] <= ts < time_bins[i+1]:
#             density_counts[i] += 1
#             break  # move to next participant once the bin is found

# # -------------------- CORRELATION ANALYSIS -------------------- #
# if len(saliency_values) == len(density_counts):
#     corr, _ = pearsonr(saliency_values, density_counts)
#     print(f"Pearson correlation between saliency & single-timestamp density: {corr:.3f}")
# else:
#     print("Lengths differ; cannot compute correlation.")

# # -------------------- OPTIONAL: PLOT -------------------- #
# plt.figure(figsize=(10,5))
# plt.plot(time_axis, saliency_values, 'r-', label='Saliency')
# plt.xlabel("Time (seconds)")
# plt.ylabel("Saliency Score")

# # Plot the density on a secondary axis (just to visualize)
# ax1 = plt.gca()
# ax2 = ax1.twinx()
# ax2.plot(time_axis, density_counts, color='blue', label='Timestamp Density')
# ax2.set_ylabel('Vote Count')

# # Combine legends
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

# plt.title(f"Clip 1: Saliency vs. Single-Timestamp Density (corr={corr:.3f})")
# plt.tight_layout()
# plt.show()


# --------------------  -------------------- ---------------------------#
# Pearson Correlation and Percentage of Votes in Peak Bins
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.interpolate import make_interp_spline
from scipy.signal import find_peaks

def compute_timestamp_density(timestamps, num_bins, t_min=0, t_max=8.5):
    """
    Given a list of timestamps and the desired number of bins,
    returns an array (of length num_bins) with vote counts per bin.
    """
    time_bins = np.linspace(t_min, t_max, num_bins + 1)
    density = np.zeros(num_bins)
    for ts in timestamps:
        # Find the bin index for ts
        for i in range(num_bins):
            if time_bins[i] <= ts < time_bins[i+1]:
                density[i] += 1
                break
    return density

# -------------------- LOAD NP FILES  -------------------- #
# Those come from the file clips_saliency_maps.py
saliency_values_all = np.load('saliency_values_arr.npy', allow_pickle=True)
saliency_times_all = np.load('salient_time_arr.npy', allow_pickle=True)
saliency_values_all_TP = np.load('saliency_values_arr_TN.npy', allow_pickle=True)
saliency_times_all_TP = np.load('saliency_times_arr_TN.npy', allow_pickle=True)
print(saliency_values_all.shape)  # (8, 50)
# smoothed_saliencies_all = np.load('/tudelft.net/staff-umbrella/EleniSalient/Saliency_graphs_per_clip/smoothed_saliencies.npy', allow_pickle=True)
# salient_time_all = np.load('/tudelft.net/staff-umbrella/EleniSalient/Saliency_graphs_per_clip/salient_time.npy', allow_pickle=True)

# Subset the first 8 clips (assuming axis 0 indexes clips) / because for the fake tests we only have fake 8 participants.
#
# saliency_values_all = saliency_values_all[:8, :]  # shape (8, 50)
# saliency_times_all = saliency_times_all[:8]
# (We won't use smoothed_saliencies or salient_time for the correlation analysis here.)

# Create a time axis for each clip (0 to 8.5 s over 50 points)
num_points = saliency_values_all.shape[1]
time_axis = np.linspace(0, 8.5, num_points)

patient_ids= [308, 321, 337, 338, 344, 365, 367, 440, 459, 483,303, 323, 349, 401, 409117, 411, 427, 445, 477, 490, 324, 379, 472, 478, 409982, 352, 353, 405, 433, 448]

# -------------------- LOAD EXCEL DATA -------------------- #
df = pd.read_excel('real_experiment_results.xlsx')

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

# Create a new column with descriptive labels
df['Classification_Type'] = df.apply(get_classification_type, axis=1)

# Filter the DataFrame to include only rows where participants correctly identified depression (TP)
df_tp = df[df['Classification_Type'] == 'TN']

# Get unique clip IDs from the filtered DataFrame
unique_clip_ids = df_tp['Clip_ID'].unique()  # e.g., array(['Clip 308', 'Clip 321', ...])

# Convert them to numeric values (assuming the format is "Clip {number}")
tp_patient_ids = [int(clip.split()[1]) for clip in unique_clip_ids]

print("Patient IDs (TP only):", tp_patient_ids)

# We assume that the Excel "Clip_ID" values are like "Clip 1", "Clip 2", ..., "Clip 8".
# For each clip, we will compute the vote density from the "Selected_Timestamp" column.
correlations = []  # to store correlation for each clip
clip_list = []     # to store clip IDs for reporting
pvalues = []
# Pearson Correlatin per Clip
# 1. Lets start with clips that are TP from the participant and TP salient moments map.

# Iterate over each clip using the real patient_ids list
for clip_id in tp_patient_ids:
    # Create the string as it appears in Excel (e.g., "Clip 303")
    clip_id_str = f"Clip {clip_id}"
    print(f"Processing {clip_id_str}...")
    
    # Get saliency vector for this clip from saliency_values_all.
    # We assume that the order in saliency_values_all corresponds to the order in patient_ids.
    idx = patient_ids.index(clip_id)
    saliency_vec = saliency_values_all_TP[idx, :]  # vector of length num_points
    saliency_time_vec = saliency_times_all[idx]  # single salient time
    
    # Filter Excel rows for this clip using the clip string
    df_clip = df_tp[df_tp['Clip_ID'] == clip_id_str]
    
    # Convert "Selected_Timestamp" to floats
    timestamps = df_clip['Selected_Timestamp'].astype(float).tolist()
    if len(timestamps) == 0:
        print(f"  No participant timestamps for {clip_id_str}. Skipping.")
        continue
    
    # Compute density using the single timestamps.
    # We use the same number of bins as the saliency vector length.
    density_vec = compute_timestamp_density(timestamps, num_bins=num_points, t_min=0, t_max=8.5)
    
    # Compute Pearson correlation between saliency and density, if lengths match
    if len(saliency_vec) == len(density_vec):
        # Compute Pearson correlation between saliency and density, if lengths match
        # Convert saliency_vec to a float array explicitly.
        saliency_vec = np.array(saliency_vec, dtype=float)
        # First, create a mask to filter out NaN values in saliency_vec
        mask = ~np.isnan(saliency_vec)
        if np.sum(mask) < 2:
            print(f"Not enough valid data for {clip_id_str}. Skipping.")
            continue

        valid_saliency_vec = saliency_vec[mask]
        valid_density_vec = density_vec[mask]
        corr, p_val = pearsonr(valid_saliency_vec, valid_density_vec)
        correlations.append(corr)
        pvalues.append(p_val)
        clip_list.append(clip_id_str)
        print(f"  {clip_id_str} correlation = {corr:.3f}")
    else:
        print(f"  Length mismatch for {clip_id_str}: {len(saliency_vec)} vs {len(density_vec)}. Skipping.")

# Optionally, print overall results
print("Processed clips:", clip_list)
print("Correlations:", correlations)
print("P-Values:", pvalues)

# correlations = []  # to store point-biserial correlation for each clip
# clip_list = []     # to store clip IDs for reporting

# # Iterate over each clip using the real patient_ids list
# for clip_id in tp_patient_ids:
#     # Create the string as it appears in Excel (e.g., "Clip 303")
#     clip_id_str = f"Clip {clip_id}"
#     print(f"Processing {clip_id_str}...")
    
#     # Get saliency vector for this clip from saliency_values_all.
#     # We assume that the order in saliency_values_all corresponds to the order in patient_ids.
#     idx = patient_ids.index(clip_id)
#     saliency_vec = saliency_values_all_TP[idx, :]  # vector of length num_points
#     saliency_time_vec = saliency_times_all[idx]     # single salient time (if needed)
    
#     # Filter Excel rows for this clip using the clip string
#     df_clip = df_tp[df_tp['Clip_ID'] == clip_id_str]
    
#     # Convert "Selected_Timestamp" to floats
#     timestamps = df_clip['Selected_Timestamp'].astype(float).tolist()
#     if len(timestamps) == 0:
#         print(f"  No participant timestamps for {clip_id_str}. Skipping.")
#         continue
    
#     # Compute density using the single timestamps.
#     # We use the same number of bins as the saliency vector length.
#     density_vec = compute_timestamp_density(timestamps, num_bins=num_points, t_min=0, t_max=8.5)
    
#     # Convert density vector to a binary vector:
#     # 1 if at least one timestamp falls in that bin, 0 otherwise.
#     binary_vec = np.array([1 if count > 0 else 0 for count in density_vec])
    
#     # Ensure saliency_vec is a numpy array of floats.
#     saliency_vec = np.array(saliency_vec, dtype=float)
    
#     # Create a mask for non-NaN saliency values.
#     mask = ~np.isnan(saliency_vec)
#     if np.sum(mask) < 2:
#         print(f"Not enough valid data for {clip_id_str}. Skipping.")
#         continue

#     valid_saliency_vec = saliency_vec[mask]
#     valid_binary_vec = binary_vec[mask]
    
#     # Compute the point-biserial correlation.
#     corr, _ = pointbiserialr(valid_binary_vec, valid_saliency_vec)
#     correlations.append(corr)
#     clip_list.append(clip_id_str)
#     print(f"  {clip_id_str} point-biserial correlation = {corr:.3f}")

# # Optionally, print overall results
# print("Processed clips:", clip_list)
# print("Point-biserial correlations:", correlations)

# Suppose you want to plot the saliency map for a specific clip, e.g., "Clip 303"
target_clip_id = 490
clip_index = patient_ids.index(target_clip_id)  # Find its index in the patient_ids list
clip_title = f"Clip {target_clip_id}"  # Title for the plot

saliency_vec = saliency_values_all_TP[clip_index]  # No comma, returns the saliency vector (shape, e.g., (50,))
saliency_vec2 = saliency_values_all[clip_index]
saliency_time_vec = saliency_times_all[clip_index]  # Single salient time
# Create a time axis from 0 to 8.5 seconds
time_axis = np.linspace(0, 8.5, num=len(saliency_vec))

# Plot the saliency map
plt.figure(figsize=(10, 5))
saliency_vec = saliency_vec.astype(float)
mask = ~np.isnan(saliency_vec)
plt.plot(time_axis[mask], saliency_vec[mask], marker='o', color='darkslategrey', linestyle='-')
plt.axvline(x=saliency_time_vec, color='maroon', linestyle='--', label="Salient Segment Start")
plt.xlabel("Time (seconds)")
plt.ylabel("Saliency Score")
plt.xticks(np.arange(0, 9, 0.5))  # Set X-axis ticks every 0.5s
plt.title(f"Saliency Map for Clip {target_clip_id}")
plt.legend()
plt.grid(True)
plt.show()


# Smoothed line
# Filter out NaNs from saliency_vec and corresponding time_axis entries
mask = ~np.isnan(saliency_vec)
if np.sum(mask) < 2:
    raise ValueError("Not enough valid points to create a spline.")
valid_time = time_axis[mask]
valid_saliency = saliency_vec[mask]

# 1) Create a new set of x-values at higher resolution
x_new = np.linspace(time_axis[0], time_axis[-1], 200)  # 200 points

# 2) Create a cubic spline of df_clip(time_axis, saliency_vec)
# spline = make_interp_spline(time_axis, saliency_vec, k=3)
spline = make_interp_spline(valid_time, valid_saliency, k=3)
saliency_smooth = spline(x_new)

df_clip = df[df['Clip_ID'] == 'Clip 490']
print(df_clip)
selected_timestamps = df_clip['Selected_Timestamp'].astype(float).tolist()
print(selected_timestamps)
# 3) Plot the smoothed curve
plt.figure(figsize=(10, 5))
plt.plot(time_axis, saliency_vec, 'o', color='darkslategrey', alpha=0.4, label='Original Points')
plt.plot(x_new, saliency_smooth, color='goldenrod', linestyle='-', label='Smoothed Saliency')
plt.xlabel("Time (seconds)")
plt.ylabel("Saliency Score")
plt.title("Smoothed Saliency Map (Cubic Spline)")
plt.grid(True)
plt.legend()
# Add the stacked rug of participant timestamps.
ax = plt.gca()
y_min, y_max = ax.get_ylim()
tick_height = 0.02 * (y_max - y_min)  # vertical spacing for each stacked marker
marker_size = 5

# Count how many participants selected each exact timestamp.
time_groups = defaultdict(int)
for ts in selected_timestamps:
    time_groups[ts] += 1

# Plot the stacked markers along the bottom of the plot.
for t in sorted(time_groups.keys()):
    count = time_groups[t]
    for level in range(count):
        y_stack = y_min + level * tick_height
        ax.plot(t, y_stack, marker='o', color='red', markersize=marker_size)

plt.show()



# -------------------- PLOT THE CORRELATION DISTRIBUTION -------------------- #

# We have clip list and correlations from above
if len(correlations) == 0:
    print("No valid correlations computed.")
else:
    plt.figure(figsize=(8, 5))
    plt.bar(clip_list, correlations, color='skyblue', edgecolor='black')
    plt.xlabel("Clips")
    plt.ylabel("Pearson Correlation")
    plt.title("Correlation per Clip (Saliency vs. Timestamp Density)")
    plt.axhline(y=0, color='gray', linestyle='--')  # horizontal line at 0
    
    # Rotate x-axis labels 45 degrees and align them to the right
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

    # Print a summary table
    summary_df = pd.DataFrame({"Clip_ID": clip_list, "Correlation": correlations})
    print("Correlation Summary:")
    print(summary_df)
    avg_corr = np.mean(correlations)
    print(f"Average correlation across {len(correlations)} clips: {avg_corr:.3f}")

# -------------------- PEAK VS. NONPEAK VOTE ANALYSIS -------------------- #
# For each clip, determine the fraction of votes that fall into "peak" bins.
# "Peak" bins are defined as time points where the saliency value is above the 80th percentile.

numeric_clip_list = [int(x.split()[1]) for x in clip_list]
print(numeric_clip_list)

peak_vote_fractions = []  # fraction of votes in peak bins for each clip
clip_ids_used = []        # the corresponding clip IDs

for i, clip_id in enumerate(numeric_clip_list):
    clip_id_str = f"Clip {clip_id}"
    print(f"Processing {clip_id_str}...")

    # 1) Retrieve the saliency vector for this clip (e.g., shape=(50,))
    s = saliency_values_all_TP[i, :]
    
    # 2) Compute the 80th percentile threshold
    threshold = np.percentile(s, 70)

    # 3) Filter the Excel data for this clip
    df_clip = df_tp[df_tp['Clip_ID'] == clip_id_str]
    timestamps = df_clip['Selected_Timestamp'].astype(float).tolist()
    total_votes = len(timestamps)

    # 4) Create a time axis for interpolation, matching saliency vector length
    # time_axis = np.linspace(0, 8.5, num=len(s))
    time_axis = np.linspace(0, 8.5, num=len(s), dtype=float)

    # Convert s to float
    s = np.asarray(s, dtype=float)

    # 5) Count how many votes fall in a "peak" bin (saliency > threshold)
    votes_in_peak = 0
    for ts in timestamps:
        vote_sal = np.interp(ts, time_axis, s)
        if vote_sal > threshold:
            votes_in_peak += 1

    # 6) Compute fraction; if no votes, set NaN
    fraction = votes_in_peak / total_votes if total_votes > 0 else np.nan
    peak_vote_fractions.append(fraction)
    clip_ids_used.append(clip_id_str)

# Create a summary table
summary_df = pd.DataFrame({
    "Clip_ID": clip_ids_used,
    "% Votes in Peak Bins": [
        f"{100 * f:.1f}%" if not np.isnan(f) else "N/A"
        for f in peak_vote_fractions
    ]
})
print(summary_df)
# --- Now aggregate duplicate clip entries ---
# Convert the "% Votes in Peak Bins" back to numeric values.
summary_df['Votes_Numeric'] = summary_df['% Votes in Peak Bins'].str.rstrip('%').astype(float)

# Group by Clip_ID and compute the mean percentage.
agg_df = summary_df.groupby('Clip_ID', as_index=False)['Votes_Numeric'].mean()

# Convert the numeric values back to a percentage string.
agg_df['% Votes in Peak Bins'] = agg_df['Votes_Numeric'].apply(lambda x: f"{x:.1f}%")
agg_df = agg_df.drop(columns='Votes_Numeric')

print("Aggregated Summary:")
print(agg_df)
# Compute and print the average fraction across all processed clips
avg_fraction = np.nanmean(peak_vote_fractions)
print(f"Average fraction of votes in peak bins: {100 * avg_fraction:.1f}%")


print(1/0)



# peak_vote_fractions = []  # to store fraction per clip
# clip_ids_used = []        # to store corresponding clip IDs

# for i in range(8):
#     clip_id = f"Clip {i+1}"
#     # Get the saliency vector for this clip (length = 50)
#     s = saliency_values_all[i, :]
#     # Compute the 80th percentile threshold for this clip
#     threshold = np.percentile(s, 80)
    
#     # Filter the Excel data for this clip
#     df_clip = df[df['Clip_ID'] == clip_id]
#     # Extract the single vote timestamps (Selected_Timestamp) as floats
#     timestamps = df_clip['Selected_Timestamp'].astype(float).tolist()
#     total_votes = len(timestamps)
    
#     # Count how many votes fall in a "peak" bin.
#     # For each vote, interpolate the saliency value at that time.
#     votes_in_peak = 0
#     for ts in timestamps:
#         vote_sal = np.interp(ts, time_axis, s)
#         if vote_sal > threshold:
#             votes_in_peak += 1
            
#     # Compute fraction; if no votes, assign NaN.
#     fraction = votes_in_peak / total_votes if total_votes > 0 else np.nan
#     peak_vote_fractions.append(fraction)
#     clip_ids_used.append(clip_id)

# # Create a summary table
# summary_df = pd.DataFrame({
#     "Clip_ID": clip_ids_used,
#     "% Votes in Peak Bins": [f"{100 * f:.1f}%" if not np.isnan(f) else "N/A" for f in peak_vote_fractions]
# })
# print(summary_df)

# # Compute and print the average fraction across clips
# avg_fraction = np.nanmean(peak_vote_fractions)
# print(f"Average fraction of votes in peak bins: {100 * avg_fraction:.1f}%")

# # ----------------------Overall pearson Correlation across all clips----------------------#
# # Global Pearson correlation computation

# global_saliency = []
# global_density = []

# # Use the full length of patient_ids (or the number of clips in saliency_values_all)
# for i in range(len(patient_ids)):
#     # Construct the clip ID string (e.g., "Clip 308")
#     clip_id = f"Clip {patient_ids[i]}"
#     print(f"Processing {clip_id}...")
    
#     # Filter the Excel rows for this clip
#     df_clip = df[df['Clip_ID'] == clip_id]
#     timestamps = df_clip['Selected_Timestamp'].astype(float).tolist()
#     if len(timestamps) == 0:
#         print(f"No participant timestamps for {clip_id}. Skipping.")
#         continue

#     # Compute density vector for the current clip (ensure num_points is defined)
#     density_vec = compute_timestamp_density(timestamps, num_bins=num_points, t_min=0, t_max=8.5)
    
#     # Get saliency vector for the current clip from your saved saliency array.
#     # Here we assume that saliency_values_all is ordered corresponding to patient_ids.
#     saliency_vec = saliency_values_all[i, :]  # e.g., shape (num_points,)
    
#     # Append values to global lists
#     global_saliency.extend(saliency_vec.tolist())
#     global_density.extend(density_vec.tolist())

# # Compute the global Pearson correlation coefficient
# global_corr, global_p = pearsonr(global_saliency, global_density)
# print(f"Global Pearson correlation = {global_corr:.3f} (p-value = {global_p:.3f})")

# # Convert global_saliency and global_density to numpy arrays (if not already)
# global_saliency = np.array(global_saliency)
# global_density = np.array(global_density)

# # Compute linear regression (best-fit line)
# slope, intercept = np.polyfit(global_saliency, global_density, 1)
# best_fit_line = slope * global_saliency + intercept

# # Create scatter plot
# plt.figure(figsize=(8, 6))
# plt.scatter(global_saliency, global_density, color='blue', label='Data Points')
# plt.plot(global_saliency, best_fit_line, color='red', 
#          label=f'Best-fit Line (r={global_corr:.3f}, p={global_p:.3f})')
# plt.xlabel('Saliency Values')
# plt.ylabel('Timestamp Density')
# plt.title('Global Scatter Plot: Saliency vs. Timestamp Density')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Print a summary table
# summary_df = pd.DataFrame({"Clip_ID": [f"Clip {pid}" for pid in patient_ids[:len(correlations)]],
#                            "Correlation": correlations})
# print("Correlation Summary:")
# print(summary_df)
# avg_corr = np.mean(correlations)
# print(f"Average correlation across {len(correlations)} clips: {avg_corr:.3f}")


# ------------------------------------------------------------------------------------------------------------------
# global_saliency = []
# global_density = []

# # Loop through all clips to accumulate saliency values and density values
# for i in range(8):
#     clip_id = f"Clip {i+1}"
#     df_clip = df[df['Clip_ID'] == clip_id]
#     timestamps = df_clip['Selected_Timestamp'].astype(float).tolist()
#     if len(timestamps) == 0:
#         print(f"No participant timestamps for {clip_id}. Skipping.")
#         continue
#     # Compute density vector for the current clip
#     density_vec = compute_timestamp_density(timestamps, num_bins=num_points, t_min=0, t_max=8.5)
#     # Get saliency vector for the current clip
#     saliency_vec = saliency_values_all[i, :]
    
#     # Append values to global lists
#     global_saliency.extend(saliency_vec.tolist())
#     global_density.extend(density_vec.tolist())

# # Compute the global Pearson correlation coefficient
# global_corr, global_p = pearsonr(global_saliency, global_density)
# print(f"Global Pearson correlation = {global_corr:.3f} (p-value = {global_p:.3f})")

# # Convert global_saliency and global_density to numpy arrays if not already
# global_saliency = np.array(global_saliency)
# global_density = np.array(global_density)

# # Compute linear regression (best-fit line)
# slope, intercept = np.polyfit(global_saliency, global_density, 1)
# best_fit_line = slope * global_saliency + intercept

# # Create scatter plot
# plt.figure(figsize=(8, 6))
# plt.scatter(global_saliency, global_density, color='blue', label='Data Points')
# plt.plot(global_saliency, best_fit_line, color='red', 
#          label=f'Best-fit Line (r={global_corr:.3f}, p={global_p:.3f})')
# plt.xlabel('Saliency Values')
# plt.ylabel('Timestamp Density')
# plt.title('Global Scatter Plot: Saliency vs. Timestamp Density')
# plt.legend()
# plt.grid(True)
# plt.show()