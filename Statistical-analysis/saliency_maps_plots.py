import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from scipy.stats import pearsonr
from scipy.stats import pointbiserialr
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

# Create a new set of x-values at higher resolution
x_new = np.linspace(time_axis[0], time_axis[-1], 200)  # 200 points

# Create a cubic spline of df_clip(time_axis, saliency_vec)
spline = make_interp_spline(valid_time, valid_saliency, k=3)
saliency_smooth = spline(x_new)

df_clip = df[df['Clip_ID'] == 'Clip 490']
print(df_clip)
selected_timestamps = df_clip['Selected_Timestamp'].astype(float).tolist()
print(selected_timestamps)
# Plot the smoothed curve
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
