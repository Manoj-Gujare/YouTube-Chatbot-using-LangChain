# utils.py
import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def extract_video_id(url):
    """Extract YouTube video ID from URL using regex"""
    patterns = [
        r"youtube\.com/watch\?v=([^&]+)",
        r"youtu\.be/([^?]+)",
        r"youtube\.com/embed/([^?]+)",
        r"youtube\.com/v/([^?]+)",
        r"youtube\.com/shorts/([^?]+)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    # If no match, try to extract from path
    if "youtube.com" in url:
        path = url.split("youtube.com")[1]
        if path.startswith("/watch?v="):
            return path.split("/watch?v=")[1].split("&")[0]
    return None

def get_transcript(video_id, language="en"):
    """Fetch YouTube transcript with specified language"""
    try:
        # Try to get transcript in the specified language
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
        return " ".join(chunk["text"] for chunk in transcript_list)
    except (TranscriptsDisabled, NoTranscriptFound):
        try:
            # If specified language not available, try English
            if language != "en":
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
                return " ".join(chunk["text"] for chunk in transcript_list)
            return None
        except (TranscriptsDisabled, NoTranscriptFound):
            return None

def process_transcript(transcript):
    """Process transcript into vector store"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    return FAISS.from_documents(chunks, embeddings)