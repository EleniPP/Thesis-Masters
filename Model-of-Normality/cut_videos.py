import wave
import pandas as pd

def cut_txt_file(txt_file_path, output_txt_path, start_time, end_time):
    """
    Cut the .txt file by filtering rows within the timestamp range.

    Args:
        txt_file_path (str): Path to the input .txt file.
        output_txt_path (str): Path to save the filtered .txt file.
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.
    """
    # Load the .txt file into a DataFrame
    df = pd.read_csv(txt_file_path)
    print(df.columns)
    # Filter rows based on the timestamp column
    filtered_df = df[(df[" timestamp"] >= start_time) & (df[" timestamp"] <= end_time)]
    
    # Save the filtered DataFrame to a new .txt file
    filtered_df.to_csv(output_txt_path, index=False)
    print(f"Saved filtered text data to {output_txt_path}")


def cut_wav_file_with_wave(input_wav_path, output_wav_path, start_time, end_time):
    """
    Cut the .wav file based on the given start and end timestamps using the built-in wave module.

    Args:
        input_wav_path (str): Path to the input .wav file.
        output_wav_path (str): Path to save the output .wav file.
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.
    """
    print('hello')
    with wave.open(input_wav_path, 'rb') as wav:
        frame_rate = wav.getframerate()
        channels = wav.getnchannels()
        sampwidth = wav.getsampwidth()

        # Calculate the start and end frames
        start_frame = int(start_time * frame_rate)
        end_frame = int(end_time * frame_rate)

        # Set the position to the start frame and read frames
        wav.setpos(start_frame)
        frames = wav.readframes(end_frame - start_frame)

    # Save the cut audio
    with wave.open(output_wav_path, 'wb') as output_wav:
        output_wav.setnchannels(channels)
        output_wav.setsampwidth(sampwidth)
        output_wav.setframerate(frame_rate)
        output_wav.writeframes(frames)
    
    print(f"Saved cut audio to {output_wav_path}")


if __name__ == "__main__":
    base_path = "/tudelft.net/staff-umbrella/EleniSalient/"
    patient = "_P/"
    audio_extension = "_AUDIO.wav"
    features_extension = "_CLNF_features.txt"
    clips = "Clips/"
    clip_extension = ".wav"
    features_clip_extension = ".txt"

    # numbers = list(range(300, 491))
    numbers = [423]

    for number in numbers:
        start_time = 574.2
        end_time = 577.7
        input_audio = f"{base_path}{number}{patient}{number}{audio_extension}"
        output_clip = f"{base_path}{clips}{number}_{start_time}_{end_time}{clip_extension}"
        input_features = f"{base_path}{number}{patient}{number}{features_extension}"
        output_features_clip = f"{base_path}{clips}{number}_{start_time}_{end_time}{features_clip_extension}"

        cut_txt_file(input_features, output_features_clip, start_time, end_time)
        cut_wav_file_with_wave(input_audio, output_clip, start_time, end_time)