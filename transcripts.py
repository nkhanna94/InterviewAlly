import os
from dotenv import load_dotenv
import videodb
import ollama
from pathlib import Path
import json
import time
import streamlit as st
from typing import Optional, Dict, Any, List

# Load environment variables
load_dotenv()

class TranscriptGenerator:
    """Handle VideoDB operations for transcript generation"""
    
    def __init__(self):
        self.api_key = os.getenv("VIDEODB_API_KEY")
        if not self.api_key:
            raise ValueError("VIDEODB_API_KEY not found in environment variables")
        self.conn = videodb.connect(api_key=self.api_key)
    
    def upload_and_process_video(self, video_file, progress_callback=None) -> Optional[Dict[str, Any]]:
        """
        Upload video file and generate transcript
        
        Args:
            video_file: Streamlit uploaded file object
            progress_callback: Optional callback for progress updates
            
        Returns:
            dict: Transcript data or None if failed
        """
        try:
            if progress_callback:
                progress_callback("Uploading video to VideoDB...", 20)
            
            # Save uploaded file temporarily
            temp_path = f"temp_{video_file.name}"
            with open(temp_path, "wb") as f:
                f.write(video_file.getbuffer())
            
            if progress_callback:
                progress_callback("Processing video upload...", 40)
                
            # Upload to VideoDB
            video = self.conn.upload(temp_path)
            
            if progress_callback:
                progress_callback("Indexing spoken words...", 60)
            
            # Index spoken words (this may take time)
            video.index_spoken_words()
            
            if progress_callback:
                progress_callback("Generating transcript...", 80)
            
            # Get transcript with timestamps
            transcript = video.get_transcript()
            
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            if progress_callback:
                progress_callback("Transcript generated successfully!", 100)
                
            return transcript
            
        except Exception as e:
            # Clean up temporary file on error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e
    
    def save_transcript_to_file(self, transcript: List[Dict], filename: str = "interview_transcript.json"):
        """Save transcript to JSON file"""
        try:
            with open(filename, "w") as f:
                json.dump(transcript, f, indent=2)
            return True
        except Exception as e:
            st.error(f"Failed to save transcript: {str(e)}")
            return False
    
    def validate_video_file(self, video_file) -> bool:
        """Validate uploaded video file"""
        if video_file is None:
            return False
            
        # Check file size (limit to 100MB)
        if video_file.size > 100 * 1024 * 1024:
            st.error("File too large. Please upload a video smaller than 100MB.")
            return False
            
        # Check file type
        allowed_types = ['mp4', 'avi', 'mov', 'mkv', 'webm']
        file_extension = video_file.name.split('.')[-1].lower()
        
        if file_extension not in allowed_types:
            st.error(f"Unsupported file type. Please upload one of: {', '.join(allowed_types)}")
            return False
            
        return True

def create_transcript_generator():
    """Factory function to create TranscriptGenerator with error handling"""
    try:
        return TranscriptGenerator()
    except ValueError as e:
        st.error(f"Configuration Error: {str(e)}")
        st.info("Please ensure VIDEODB_API_KEY is set in your .env file")
        return None
    except Exception as e:
        st.error(f"Failed to initialize VideoDB connection: {str(e)}")
        return None
