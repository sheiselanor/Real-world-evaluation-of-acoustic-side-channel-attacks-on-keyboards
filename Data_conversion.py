from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip

# Define the time intervals for each session
sessions = [
    (), #session duration 1
    (), #session duration 2
    (), #session duration 3
    () #session duration 4
]

# Original video path
video_path = r'E:\University\FYP\FYP_A\MSUTypingDatabase\Phase1\005\S1.mp4'

# Output audio path
output_audio_path = r'E:\University\FYP\FYP_A\dataset\PREPROCESSING_DATA\audio\phase1_005_s1'

# Output video path
output_video_path = r'E:\University\FYP\FYP_A\dataset\PREPROCESSING_DATA\video\phase1_005_s1'

# Create VideoFileClip object
video_clip = VideoFileClip(video_path)

# Extract audio for each session
for i, session in enumerate(sessions):
    start_time, end_time = session
    session_clip = video_clip.subclip(start_time, end_time)
    
    # Create output audio file path
    output_audio_file = f'{output_audio_path}\\session_{i+1}.mp3'
    
    # Write audio file
    session_clip.audio.write_audiofile(output_audio_file, codec='mp3', bitrate='192k')
    
    # Create output video file path
    output_video_file = f'{output_video_path}\\session_{i+1}.mp4'
    
    # Write video file
    session_clip.write_videofile(output_video_file, codec='libx264', audio_codec='aac', fps=30)

# Close the video clip
video_clip.close()
