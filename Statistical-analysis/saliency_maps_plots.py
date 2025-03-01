import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from scipy.stats import pearsonr

def parse_interval(interval_str):
    """Parse something like '2-3 sec' into numeric floats (2.0, 3.0)."""
    interval_str = interval_str.replace(' sec', '').strip()
    start_str, end_str = interval_str.split('-')
    return float(start_str), float(end_str)

# -------------------- LOAD PROCESSED SALIENCY DATA -------------------- #
saliency_values = np.load('saliency_values_308.npy', allow_pickle=True)
salient_time = np.load('salient_time_308.npy', allow_pickle=True)
smoothed_saliencies = np.load('smoothed_saliencies_308.npy', allow_pickle=True)

# Create time axis from 0..8.5 seconds
time_axis = np.linspace(0, 8.5, num=len(saliency_values))

# -------------------- LOAD & FILTER EXCEL DATA -------------------- #
df = pd.read_excel('experiment_results.xlsx')
df_clip = df[df['Clip_ID'] == 'Clip 1']  # adjust as needed

# # Get unique clip IDs from the Excel file
# clip_ids = df['Clip_ID'].unique()

# If timestamps are slightly off but should be considered the same,
# you might do: round_ts = df_clip['Selected_Timestamp'].round(1)
# and then convert to list. For now, we assume exact timestamps.
selected_timestamps = df_clip['Selected_Timestamp'].astype(float).tolist()
print(selected_timestamps)
selected_timestamps[0] = selected_timestamps[1]
# -------------------- STACKED RUG LOGIC -------------------- #
# Count how many participants share each *exact* timestamp
time_groups = defaultdict(int)
for ts in selected_timestamps:
    time_groups[ts] += 1

# We'll remove the old "density" approach entirely,
# focusing on just the saliency + stacked rug.

plt.figure(figsize=(10, 5))
ax = plt.gca()

# Plot raw and smoothed saliency
# ax.plot(time_axis, saliency_values, 'o-', alpha=0.3, label='Raw Saliency')
ax.plot(time_axis, smoothed_saliencies, 'r-', label='Smoothed Saliency')
ax.axvline(x=salient_time, color='maroon', linestyle='--', label='Salient Segment Start')
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Saliency Score')
ax.legend(loc='upper right')

# We'll place stacked markers at the bottom of the plot.
y_min, y_max = ax.get_ylim()

# Decide how tall each "stack level" should be
tick_height = 0.02 * (y_max - y_min)  # space between stacks
marker_size = 5  # vertical marker size

# Make sure we have enough vertical space for the highest stack
max_stack = max(time_groups.values()) if time_groups else 0
extra_space = max_stack * tick_height
if y_min + extra_space > y_max:
    ax.set_ylim(y_min, y_min + extra_space + 0.1*(y_max - y_min))

# For each unique time, stack the markers from bottom up
for t in sorted(time_groups.keys()):
    count = time_groups[t]
    # We stack them from y_min upward
    for level in range(count):
        y_stack = y_min + level * tick_height
        # Plot a small vertical marker
        # marker='|' draws a vertical line; you could do marker='o' for dots
        ax.plot(t, y_stack, marker='o', color='navy', markersize=marker_size)

plt.title('Saliency vs. Stacked Rug of Participant Timestamps (Clip 1)')
plt.tight_layout()
plt.show()


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
# # Pearson Correlation and Percentage of Votes in Peak Bins
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.stats import pearsonr

# def compute_timestamp_density(timestamps, num_bins, t_min=0, t_max=8.5):
#     """
#     Given a list of timestamps and the desired number of bins,
#     returns an array (of length num_bins) with vote counts per bin.
#     """
#     time_bins = np.linspace(t_min, t_max, num_bins + 1)
#     density = np.zeros(num_bins)
#     for ts in timestamps:
#         # Find the bin index for ts
#         for i in range(num_bins):
#             if time_bins[i] <= ts < time_bins[i+1]:
#                 density[i] += 1
#                 break
#     return density

# # -------------------- LOAD NP FILES (First 8 clips) -------------------- #
# # Files have shape (12, 50); we'll use only the first 8 clips.
# saliency_values_all = np.load('saliency_values_arr.npy', allow_pickle=True)
# print(saliency_values_all.shape)  # (8, 50)
# # smoothed_saliencies_all = np.load('/tudelft.net/staff-umbrella/EleniSalient/Saliency_graphs_per_clip/smoothed_saliencies.npy', allow_pickle=True)
# # salient_time_all = np.load('/tudelft.net/staff-umbrella/EleniSalient/Saliency_graphs_per_clip/salient_time.npy', allow_pickle=True)

# # Subset the first 8 clips (assuming axis 0 indexes clips)
# saliency_values_all = saliency_values_all[:8, :]  # shape (8, 50)
# # (We won't use smoothed_saliencies or salient_time for the correlation analysis here.)

# # Create a time axis for each clip (0 to 8.5 s over 50 points)
# num_points = saliency_values_all.shape[1]
# time_axis = np.linspace(0, 8.5, num_points)

# # -------------------- LOAD EXCEL DATA -------------------- #
# df = pd.read_excel('experiment_results.xlsx')

# # We assume that the Excel "Clip_ID" values are like "Clip 1", "Clip 2", ..., "Clip 8".
# # For each clip, we will compute the vote density from the "Selected_Timestamp" column.
# correlations = []  # to store correlation for each clip
# clip_list = []     # to store clip IDs for reporting

# for i in range(8):
#     clip_id = f"Clip {i+1}"
#     print(f"Processing {clip_id}...")
    
#     # Get saliency vector for this clip from the npy file
#     saliency_vec = saliency_values_all[i, :]  # vector of length 50
    
#     # Filter Excel rows for this clip
#     df_clip = df[df['Clip_ID'] == clip_id]
#     # Convert "Selected_Timestamp" to floats
#     timestamps = df_clip['Selected_Timestamp'].astype(float).tolist()
#     if len(timestamps) == 0:
#         print(f"  No participant timestamps for {clip_id}. Skipping.")
#         continue
    
#     # Compute density using the single timestamps.
#     # We use the same number of bins as saliency vector length.
#     density_vec = compute_timestamp_density(timestamps, num_bins=num_points, t_min=0, t_max=8.5)
    
#     # Compute Pearson correlation between saliency and density
#     if len(saliency_vec) == len(density_vec):
#         corr, _ = pearsonr(saliency_vec, density_vec)
#         correlations.append(corr)
#         clip_list.append(clip_id)
#         print(f"  {clip_id} correlation = {corr:.3f}")
#     else:
#         print(f"  Length mismatch for {clip_id}: {len(saliency_vec)} vs {len(density_vec)}. Skipping.")

# # -------------------- PLOT THE CORRELATION DISTRIBUTION -------------------- #
# # Example data
# clip_list = ["Clip 1", "Clip 2", "Clip 3", "Clip 4", 
#              "Clip 5", "Clip 6", "Clip 7", "Clip 8"]
# if len(correlations) == 0:
#     print("No valid correlations computed.")
# else:
#     plt.figure(figsize=(8, 5))
#     plt.bar(clip_list, correlations, color='skyblue', edgecolor='black')
#     plt.xlabel("Clips")
#     plt.ylabel("Pearson Correlation")
#     plt.title("Correlation per Clip (Saliency vs. Timestamp Density)")
#     plt.axhline(y=0, color='gray', linestyle='--')  # horizontal line at 0
#     plt.tight_layout()
#     plt.show()

#     # Print a summary table
#     summary_df = pd.DataFrame({"Clip_ID": clip_list, "Correlation": correlations})
#     print("Correlation Summary:")
#     print(summary_df)
#     avg_corr = np.mean(correlations)
#     print(f"Average correlation across {len(correlations)} clips: {avg_corr:.3f}")



# # -------------------- PEAK VS. NONPEAK VOTE ANALYSIS -------------------- #
# # For each clip, determine the fraction of votes that fall into "peak" bins.
# # "Peak" bins are defined as time points where the saliency value is above the 80th percentile.
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