import os
import shutil
import uuid
import json
import sqlite3
from typing import Dict, Optional, Any
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from brain import InterviewBrain
from rag import TranscriptProcessor
from transcripts import create_transcript_generator

app = FastAPI(title="InterviewAlly Backend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Database Configuration ---
DB_NAME = "jobs.db"

def init_db():
    """Initialize the SQLite database and create the table if it doesn't exist."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # We store complex structures (transcript, analysis) as JSON TEXT
    c.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            status TEXT,
            filename TEXT,
            transcript TEXT,
            analysis TEXT,
            error TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Initialize DB on startup
init_db()

# --- Database Helper Functions ---

def save_new_job(job_id: str, filename: str, status: str = "queued"):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        "INSERT INTO jobs (job_id, filename, status) VALUES (?, ?, ?)",
        (job_id, filename, status)
    )
    conn.commit()
    conn.close()

def update_job(job_id: str, status: str, transcript=None, analysis=None, error=None):
    """
    Dynamically updates provided fields for a job.
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    fields = ["status = ?"]
    values = [status]
    
    if transcript is not None:
        fields.append("transcript = ?")
        values.append(json.dumps(transcript)) # Serialize list/dict to JSON string
    
    if analysis is not None:
        fields.append("analysis = ?")
        values.append(json.dumps(analysis))   # Serialize dict to JSON string
        
    if error is not None:
        fields.append("error = ?")
        values.append(error)
        
    values.append(job_id)
    
    query = f"UPDATE jobs SET {', '.join(fields)} WHERE job_id = ?"
    c.execute(query, tuple(values))
    conn.commit()
    conn.close()

def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves a job and deserializes JSON fields back to Python objects.
    """
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row # Allows accessing columns by name
    c = conn.cursor()
    c.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
    row = c.fetchone()
    conn.close()
    
    if row:
        return {
            "job_id": row["job_id"],
            "status": row["status"],
            "filename": row["filename"],
            "transcript": json.loads(row["transcript"]) if row["transcript"] else None,
            "analysis": json.loads(row["analysis"]) if row["analysis"] else None,
            "error": row["error"]
        }
    return None

# --- Models ---
class ChatRequest(BaseModel):
    job_id: str
    message: str

class RewriteRequest(BaseModel):
    job_id: str          
    gap_text: str        
    profile_context: str 

# --- Background Task ---

def process_video_task(job_id: str, file_path: str):
    """
    Background task: Transcript -> Indexing -> DB Update.
    """
    try:
        print(f"[{job_id}] Starting background processing...")
        update_job(job_id, "processing")
        
        # 1. Generate Transcript (VideoDB / Whisper)
        generator = create_transcript_generator()
        transcript_data = generator.process_video_file(file_path)
        
        # 2. Index for RAG (LangChain + Chroma)
        print(f"[{job_id}] Starting RAG indexing...")
        rag_processor = TranscriptProcessor()
        rag_processor.index_transcript(job_id, transcript_data)
        
        # 3. Save Success to DB
        update_job(job_id, "completed", transcript=transcript_data)
        print(f"[{job_id}] Processing & Indexing complete.")
        
    except Exception as e:
        print(f"[{job_id}] Failed: {str(e)}")
        update_job(job_id, "failed", error=str(e))
    
    finally:
        # Cleanup temp file
        if os.path.exists(file_path):
            os.remove(file_path)

# --- Endpoints ---

@app.post("/upload-video")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    temp_filename = f"temp_{job_id}_{file.filename}"
    
    # Save file to disk temporarily
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 1. Create DB Entry
    save_new_job(job_id, file.filename)
    
    # 2. Trigger Background Task
    background_tasks.add_task(process_video_task, job_id, temp_filename)
    
    return {"job_id": job_id, "message": "Video uploaded and processing started."}

@app.get("/get-status/{job_id}")
async def get_status_endpoint(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return job

@app.post("/analyze/{job_id}")
async def analyze_interview(job_id: str):
    # 1. Fetch Job
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found")
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job processing not finished")
    
    # Return cached analysis if exists
    if job.get("analysis"):
        return job["analysis"]

    # 2. Prepare Data
    raw_transcript = job["transcript"]
    # Reconstruct dialogue with speakers for better context
    full_text = "\n".join([f"{chunk.get('speaker', 'Speaker')}: {chunk['text']}" for chunk in raw_transcript])

    # 3. Run Analysis
    brain = InterviewBrain()
    analysis_result = brain.generate_analysis(full_text)
    
    # 4. Save Result to DB
    update_job(job_id, "completed", analysis=analysis_result)
    
    return analysis_result

@app.post("/rewrite")
async def rewrite_answer_endpoint(request: RewriteRequest):
    job = get_job(request.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    raw_transcript = job["transcript"]
    full_text = "\n".join([f"{chunk.get('speaker', 'Speaker')}: {chunk['text']}" for chunk in raw_transcript])
    
    brain = InterviewBrain()
    better_answer = brain.rewrite_answer(request.gap_text, full_text, request.profile_context)
    
    return {"rewritten_answer": better_answer}

@app.post("/chat")
async def chat(request: ChatRequest):
    job = get_job(request.job_id)
    if not job:
         raise HTTPException(status_code=404, detail="Job ID not found")
         
    # 1. Setup RAG
    rag_processor = TranscriptProcessor()
    retriever = rag_processor.get_retriever()
    retriever.search_kwargs = {"k": 4, "filter": {"job_id": request.job_id}}
    
    # 2. Prepare Context (Analysis Summary)
    analysis_context = ""
    if job.get("analysis"):
        a = job["analysis"]
        analysis_context = (
            f"Candidate Summary: {a.get('summary', 'N/A')}\n"
            f"Technical Score: {a.get('technical_score', '0')}/10\n"
            f"Critical Gaps: {', '.join(a.get('critical_gaps', []))}"
        )
    
    # 3. Get Answer
    brain = InterviewBrain()
    answer = brain.get_chat_response(request.message, retriever, analysis_context)
    
    return {"response": answer}