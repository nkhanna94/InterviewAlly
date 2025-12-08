# 🚀 InterviewAlly

**Your Personal AI Interview Coach.**
InterviewAlly is a local, privacy-first application that ingests interview recordings (video/audio), analyzes them using **RAG (Retrieval-Augmented Generation)**, and provides brutal, actionable feedback to help candidates get hired.

Unlike generic chatbots, InterviewAlly uses **Speaker Diarization** to distinguish between the interviewer and candidate, and employs a **specialized RAG pipeline** to rewrite weak answers using the STAR method based on the candidate's actual experience.

-----

## ⚡ Key Features

  * **🎧 Multimodal Ingestion:** Upload `.mp4`, `.mov`, `.mp3`, or `.wav` files. The system uses **Faster-Whisper** for transcription and **Pyannote.audio** for Speaker Diarization (who said what).
  * **🧠 RAG-Powered Analysis:**
      * Indexes transcripts into **ChromaDB** using **Nomic Embeddings** (`nomic-embed-text-v1.5`).
      * Preserves temporal metadata (timestamps) for precise context retrieval.
  * **📊 Automated Scoring:** Uses **Llama 3.2** (via Ollama) to grade Technical Depth, Communication, and Cultural Fit on a 1-10 scale.
  * **✨ Magic Rewriter:** Identifies weak answers and rewrites them into "Gold Standard" responses using the **STAR Method** (Situation, Task, Action, Result), grounded strictly in the transcript data (no hallucinations).
  * **💬 Chat with your Interview:** A Q\&A interface to ask specific questions like *"Did I sound nervous?"* or *"How can I improve my explanation of SQL joins?"*

-----

## 🛠️ Tech Stack

  * **LLM & Orchestration:** [LangChain](https://www.langchain.com/), [Ollama](https://ollama.com/) (Llama 3.2), Pydantic (Structured Output).
  * **Speech Processing:** [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper), [Pyannote.audio](https://github.com/pyannote/pyannote-audio).
  * **Vector Database:** [ChromaDB](https://www.trychroma.com/) (Local persistence).
  * **Backend:** FastAPI, SQLite (Job management), BackgroundTasks.
  * **Frontend:** Streamlit.

-----

## 🏗️ Architecture

1.  **Ingestion:** Video is uploaded -\> Audio extracted -\> Transcribed (Whisper) -\> Speakers Identified (Pyannote).
2.  **Indexing:** Transcript is split into **semantic chunks** (grouped by speaker turns & time) -\> Embedded -\> Stored in ChromaDB.
3.  **Analysis:** Background task runs a "Coach" chain to generate a JSON report saved to SQLite.
4.  **Interaction:** User views scores/feedback in Streamlit and triggers RAG-based rewrites or chat.

-----

## 🚀 Setup & Installation

### Prerequisites

  * Python 3.10+
  * [Ollama](https://ollama.com/) installed and running.
  * **FFmpeg** installed (required for audio processing).

### 1\. Clone the Repository

```bash
git clone https://github.com/nkhanna94/InterviewAlly.git
cd InterviewAlly
```

### 2\. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3\. Setup Models

**Pull the LLM:**

```bash
ollama pull llama3.2:latest
```

**Environment Variables:**
Create a `.env` file in the root directory:

```env
# Required for Pyannote Speaker Diarization
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
```

### 4\. Run the Application

You need to run the Backend and Frontend in separate terminals.

**Terminal 1: Backend (FastAPI)**

```bash
uvicorn backend.main:app --reload
```

**Terminal 2: Frontend (Streamlit)**

```bash
streamlit run frontend/app.py
```

-----

## 📂 Project Structure

```
InterviewAlly/
├── backend/
|   ├── chroma_db/           # Local Vector Store
│   ├── main.py          # FastAPI endpoints & background tasks
│   ├── brain.py         # LLM logic, Chains, & Structured Output
│   ├── rag.py           # ChromaDB setup & Custom Chunking logic
│   ├── transcripts.py   # Whisper + Pyannote pipeline
│   └── jobs.db          # SQLite database for job status
├── frontend/
│   └── app.py           # Streamlit Dashboard
└── requirements.txt
```

-----

## 🔮 Future Improvements

  * **LangGraph Integration:** Move from linear Chains to a stateful Agent for iterative critique/refinement of answers.
  * **SQL Agent:** Allow users to query their history (e.g., *"Show me all interviews where I failed System Design"*).
  * **Cloud Deployment:** Dockerize the application for AWS/GCP deployment.

-----