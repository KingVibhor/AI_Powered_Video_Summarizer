import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp as ydl
from PIL import Image
import cv2
import subprocess
import whisper
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message="You are using torch.load with weights_only=False.*", category=FutureWarning)
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".use_column_width.", category=DeprecationWarning)

# Load environment variables from .env file
load_dotenv()

# Configure Google Generative AI with the API key
genai.configure(api_key="YOUR_API_KEY")  # Replace with your actual API key

# Define the prompt for summarization
prompt = """You are a YouTube video summarizer. You will take the transcript text in any language but provide summary STRICTLY in English language and summarize the entire video in a detailed, comprehensive manner. Provide the notes in a descriptive, point-wise format covering all key points: """

# Function to extract transcript from YouTube video
from youtube_transcript_api import NoTranscriptFound, TranscriptsDisabled

def extract_transcript_details(video_source, whisper_model_size="small", is_youtube=True):
    try:
        if is_youtube:
            video_id = video_source.split("v=")[-1]

            try:
                transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
                transcript = " ".join([item["text"] for item in transcript_data])
                return transcript
            except NoTranscriptFound:
                st.warning("No English transcript found. Trying other languages...")
                try:
                    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                    for transcript in transcript_list:
                        if transcript.language_code in ['hi', 'mr']:
                            transcript_data = transcript.fetch()
                            transcript_text = " ".join([item["text"] for item in transcript_data])
                            st.success(f"Transcript fetched in language: {transcript.language_code}")
                            return transcript_text
                    st.warning("No preferred language transcript found. Proceeding with audio transcription.")
                except TranscriptsDisabled:
                    st.error("Transcripts are disabled for this video. Proceeding with audio transcription.")

            st.info("Downloading audio for Whisper transcription...")
            audio_path = "temp_audio.mp3"
            try:
                download_youtube_audio(video_source, output_path=audio_path)
                st.info("Transcribing audio using Whisper...")
                segments, transcript_text = transcribe_and_translate_audio(audio_path, model_size=whisper_model_size)
                os.remove(audio_path)
                return transcript_text
            except Exception as e:
                st.error(f"Error downloading or transcribing audio: {str(e)}")
                return None
        else:
            st.info("Transcribing offline video/audio using Whisper...")
            transcript_text = transcribe_and_translate_audio(audio_path=video_source, model_size=whisper_model_size)
            return transcript_text
    except Exception as e:
        st.error(f"Error fetching transcript: {str(e)}")
        return None

def extract_chapters(youtube_video_url):
    try:
        with ydl.YoutubeDL({"quiet": True}) as ydl_instance:
            video_info = ydl_instance.extract_info(youtube_video_url, download=False)
            return video_info.get("chapters", None), video_info["id"], video_info["duration"]
    except Exception as e:
        st.error(f"Error extracting chapters: {str(e)}")
        return None, None, None

def extract_keyframe(video_path, time_in_seconds, title):
    try:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, time_in_seconds * 1000)
        ret, frame = cap.read()
        cap.release()
        if ret:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            return {"title": title, "keyframe": img}
        else:
            return None
    except Exception as e:
        st.error(f"Error extracting keyframe at {time_in_seconds} seconds: {str(e)}")
        return None

def extract_keyframes_by_duration(video_path, duration, interval_count):
    try:
        total_duration = int(duration)
        interval = total_duration // interval_count
        keyframes = []

        for i in range(interval_count):
            frame_time = (i + 1) * interval
            keyframe = extract_keyframe(video_path, frame_time, f"Keyframe {i+1}")
            if keyframe:
                keyframes.append(keyframe)
        return keyframes
    except Exception as e:
        st.error(f"Error extracting keyframes by duration: {str(e)}")
        return []

def generate_gemini_content(translated_text, prompt):
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt + translated_text)
        return response.text
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return None

def pair_notes_with_keyframes(summary, keyframes, last_keyframe):
    notes_sections = summary.split("\n\n")
    paired_content = []

    for i, section in enumerate(notes_sections):
        paired_content.append({"note": section})
        if i < len(keyframes):
            paired_content.append({"keyframe": keyframes[i]})

    for extra_keyframe in keyframes[len(notes_sections):]:
        paired_content.append({"keyframe": extra_keyframe})
    for extra_note in notes_sections[len(keyframes):]:
        paired_content.append({"note": extra_note})

    if last_keyframe:
        paired_content.append({"keyframe": last_keyframe})

    return paired_content

def download_youtube_audio(link, output_path="audio.mp3"):
    command = ["yt-dlp", "-x", "--audio-format", "mp3", "-o", output_path, link]
    subprocess.run(command, check=True)

def transcribe_and_translate_audio(audio_path, model_size="small"):
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)
    return result["text"]

# Streamlit app layout
st.title("Chapter-Based Video Notes with Keyframes")
option = st.selectbox("Choose the type of video:", ("YouTube Link", "Offline Video"))

if option == "YouTube Link":
    youtube_link = st.text_input("Enter YouTube Video Link:")

    if youtube_link:
        video_id = youtube_link.split("v=")[-1]
        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

        if st.button("Get Detailed Notes with Keyframes"):
            chapters, video_id, duration = extract_chapters(youtube_link)
            transcript_text = extract_transcript_details(youtube_link)

            if transcript_text:
                summary = generate_gemini_content(transcript_text, prompt)

                ydl_opts = {"format": "best", "outtmpl": f"D:\\Downloads\\{video_id}.mp4", "noplaylist": True}
                with ydl.YoutubeDL(ydl_opts) as ydl_instance:
                    video_info = ydl_instance.extract_info(youtube_link, download=True)
                    video_path = f"D:\\Downloads\\{video_info['id']}.mp4"

                keyframes = extract_keyframes_by_duration(video_path, duration, 5)
                last_keyframe = extract_keyframe(video_path, duration - 10, "Last Keyframe")
                paired_content = pair_notes_with_keyframes(summary, keyframes, last_keyframe)

                st.markdown("## Notes with Keyframes:")
                for content in paired_content:
                    if "note" in content:
                        st.markdown(content["note"])
                    if "keyframe" in content:
                        st.image(content["keyframe"]["keyframe"], caption=content["keyframe"]["title"], use_column_width=True)

elif option == "Offline Video":
    video_file = st.file_uploader("Upload your video file", type=["mp4", "avi", "mov"])

    if video_file:
        video_path = f"temp_video.{video_file.name.split('.')[-1]}"
        with open(video_path, "wb") as f:
            f.write(video_file.read())

        duration = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT)) // 30
        transcript_text = transcribe_and_translate_audio(video_path)
        summary = generate_gemini_content(transcript_text, prompt)

        keyframes = extract_keyframes_by_duration(video_path, duration, 5)
        last_keyframe = extract_keyframe(video_path, duration - 10, "Last Keyframe")
        paired_content = pair_notes_with_keyframes(summary, keyframes, last_keyframe)

        st.markdown("## Notes with Keyframes:")
        for content in paired_content:
            if "note" in content:
                st.markdown(content["note"])
            if "keyframe" in content:
                st.image(content["keyframe"]["keyframe"], caption=content["keyframe"]["title"], use_column_width=True)
