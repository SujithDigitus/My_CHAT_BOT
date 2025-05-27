import cv2
import os
import shutil
import numpy as np
import whisper
import glob
import subprocess
from PIL import Image as PilImage
from fpdf import FPDF
import ffmpeg
from moviepy import VideoFileClip

def extracting_transcript(video_path,audio_path,output_path):
    video_clip = VideoFileClip(video_path)
    video_clip.audio.write_audiofile(audio_path)
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)

    # Extract transcript segments with timestamps
    segments = result['segments']

    # Combine segments into 1-minute intervals
    combined_transcript = []
    current_start_time = 0
    current_end_time = 60
    current_text = []

    for segment in segments:
        start = segment['start']
        end = segment['end']
        text = segment['text']

        if start < current_end_time:
            current_text.append(text)
        else:
            combined_transcript.append({
                'start_time': current_start_time,
                'end_time': current_end_time,
                'text': ' '.join(current_text)
            })
            current_start_time = current_end_time
            current_end_time += 60
            current_text = [text]

    # Add the last segment
    if current_text:
        combined_transcript.append({
            'start_time': current_start_time,
            'end_time': current_end_time,
            'text': ' '.join(current_text)
        })

    # Write the combined transcript to a text file
    with open(output_path, 'w',encoding="utf-8") as file:
        for segment in combined_transcript:
            file.write(f"{segment['start_time']:.2f} - {segment['end_time']:.2f}: {segment['text']}\n")

    print("Transcript has been saved to", output_path)



# Function to extract frames at intervals and skip similar frames
def extract_frames(video_file,output_folder,interval_sec):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * interval_sec)

    frame_count = 0
    last_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval == 0:
            timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
            frame_filename = os.path.join(output_folder, f"frame_{timestamp:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Extracted frame at {timestamp:.2f} sec: {frame_filename}")

        frame_count += 1

    cap.release()

# Function to parse the transcript
def parse_transcript(transcript_file):
    with open(transcript_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    transcript_segments = []
    for line in lines:
        timestamp, text = line.split(':', 1)
        start_time, end_time = timestamp.split('-')
        segment = {
            'start_time': float(start_time.replace(':', '')),
            'end_time': float(end_time.replace(':', '')),
            'text': text.strip()
        }
        transcript_segments.append(segment)

    return transcript_segments

# Function to get frame timestamp from filename
def get_frame_timestamp(frame_file):
    frame_timestamp = int(os.path.basename(frame_file).split('_')[1].split('.')[0])
    return frame_timestamp

# Function to map transcript segments to frames
def map_transcript_to_frames(transcript_segments, frame_folder):
    frame_files = sorted(glob.glob(os.path.join(frame_folder, 'frame_*.jpg')))
    mapped_data = []

    for segment in transcript_segments:
        segment_frames = []
        for frame_file in frame_files:
            frame_timestamp = get_frame_timestamp(frame_file)
            if segment['start_time'] <= frame_timestamp <= segment['end_time']:
                segment_frames.append(frame_file)

        mapped_data.append({'segment': segment, 'frames': segment_frames})

    return mapped_data

# PDF Class
class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_font("Arial", size=12)  # Using the default Arial font

    def add_transcript_segment(self, text, frames):
        # Start a new page for each transcript segment
        self.add_page()
        self.multi_cell(0, 6, text)  # Write transcript text
        self.ln(2)  # Reduced spacing

        if frames:  # Ensure there is at least one frame to display
            frame = frames[0]  # Only take the first frame
            self.add_frame_image(frame)

    def add_frame_image(self, frame_file):
        img = PilImage.open(frame_file)
        img_width, img_height = img.size
        max_width = 140  # Further increased image width
        max_height = 120  # Further increased image height
        if img_width > max_width:
            img_height = int((max_width / img_width) * img_height)
            img_width = max_width

        if self.get_y() + img_height > 270:  # Ensure that the image fits on the page
            self.add_page()

        self.image(frame_file, x=15, w=img_width, h=img_height)
        self.ln(2)  # Reduced spacing after image

# Function to save mapped data into PDF
def save_mapped_data_to_pdf(mapped_data, pdf_filename):
    pdf = PDF()
    pdf.add_font("arialuni", "", "C:/Users\DELL/Desktop/Chatbot/My_chat_bot/Video_to_audio/Frames/arialuni.ttf", uni=True)
    pdf.set_font("arialuni", "", 12)
    for data in mapped_data:
        pdf.add_transcript_segment(data['segment']['text'], data['frames'])
        pdf.ln(2)  # Reduced spacing between segments

    pdf.output(pdf_filename, 'F')
    print(f"PDF saved as {pdf_filename}")

# Paths
pdf_filename = r"C:\Users\DELL\Desktop\Chatbot\My_chat_bot\work_folder\Inlay.pdf"

# Execution
def Video_to_pdf(filename):
    print(filename)
    video_path=fr"C:\Users\DELL\Desktop\Chatbot\My_chat_bot\Video_to_audio\{filename}"
    audio_path = r"C:\Users\DELL\Desktop\Chatbot\My_chat_bot\Video_to_audio\extracted_audio\extracted_audio.wav"
    output_path= r"C:\Users\DELL\Desktop\Chatbot\My_chat_bot\Video_to_audio\transcript.txt"
    try:
        os.mkdir(r"C:\Users\DELL\Desktop\Chatbot\My_chat_bot\Video_to_audio\Frames\Inlay")
    except FileExistsError:
        print("folder exist")
    pdf_filename = fr"C:\Users\DELL\Desktop\Chatbot\My_chat_bot\work_folder\{os.path.splitext(filename)[0]}.pdf"
    frame_folder = r"C:\Users\DELL\Desktop\Chatbot\My_chat_bot\Video_to_audio\Frames\Inlay"
    extracting_transcript(video_path,audio_path,output_path)
    extract_frames(video_path,frame_folder,interval_sec=60)
    transcript_segments = parse_transcript(output_path)
    mapped_data = map_transcript_to_frames(transcript_segments, frame_folder)
    save_mapped_data_to_pdf(mapped_data, pdf_filename)
    os.remove(audio_path)
    shutil.rmtree(frame_folder)
    os.remove(output_path)
    return pdf_filename
