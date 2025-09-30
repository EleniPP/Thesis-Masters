import os

def get_segment_index(timestamp, segment_duration, stride, start_timestamp):

    if timestamp < start_timestamp:
        return None, "Timestamp is before the first segment starts."

    # Calculate the segment index (starting from 1)
    segment_number = round((timestamp - start_timestamp) / stride) + 1


    # Calculate the start and end timestamps of the identified segment
    start_time_seconds = start_timestamp + (segment_number - 1) * stride
    end_time_seconds = start_time_seconds + segment_duration

    return segment_number, (start_time_seconds, end_time_seconds)

def parse_patient_time_file(file_path):
    patient_times = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 3):  # Process in blocks of 3 lines
            patient_number = int(lines[i].split(':')[1].strip())
            visual_start_time = float(lines[i + 1].split(':')[1].strip())
            visual_end_time = float(lines[i + 2].split(':')[1].strip())
            patient_times[patient_number] = visual_start_time
    # print(patient_times)
    return patient_times

# Those ones are for the TPs
downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads", "experiment-clips")
patients_folder = os.path.join(os.path.expanduser("~"), "Downloads")
patient_times = parse_patient_time_file(patients_folder + "/timestamps_per_patient.txt")	

patients = []
timestamps = []
# List all files in the directory
if os.path.exists(downloads_folder):
    files = os.listdir(downloads_folder)
    print("Files in 'Downloads/TP_4each':")
    for file in files:
        patient = int(file.split("_")[0])
        patients.append(patient)
        timestamp = file.split("_")[1]
        timestamps.append(timestamp)
else:
    print("The directory does not exist.")

patient_ids = []
segments = []
for patient in patients:
    if patient in patient_times:
        start_timestamp = patient_times[patient]
        timestamp = timestamps[patients.index(patient)]
        patient_ids.append(patient)
        print('For patient:', patient)
        segment = get_segment_index(float(timestamp), 3.5, 0.1, start_timestamp)
        print('Segment:', segment[0])
        segments.append(segment[0])
