def get_timestamp(segment_number, segment_duration):
    # Calculate the timestamp in seconds
    timestamp_seconds = (segment_number - 1) * segment_duration

    # Convert the timestamp to minutes and seconds
    minutes = int(timestamp_seconds // 60)
    seconds = timestamp_seconds % 60

    return f"{minutes} minutes and {seconds:.1f} seconds"

# Example usage
segment_duration = 3.5  # Each segment is 3.5 seconds
segment_number = 183    # You can change this to any segment number

timestamp = get_timestamp(segment_number, segment_duration)
print(f"Timestamp of segment {segment_number}: {timestamp}")
