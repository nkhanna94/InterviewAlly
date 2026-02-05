# 🚀 InterviewAlly

**Your Personal AI Interview Coach.**

InterviewAlly is a **privacy-first, local RAG application** that helps candidates ace technical interviews. It ingests video/audio recordings, creates a structured transcript, and uses **Local LLMs (Phi-3.5 / Llama 3.2)** to provide brutal, actionable feedback.

Unlike generic tools that treat interviews as a blob of text, InterviewAlly uses **Speaker Diarization** and **Semantic Q&A Chunking** to understand the *structure* of the conversation, ensuring feedback is grounded in specific candidate answers.

---

## ⚡ Technical Differentiators

### 1. 🧠 Structure-Aware RAG (The "Smart Chunking" Engine)

Most RAG apps fail on interviews because they split text by word count (e.g., every 500 words), cutting answers in half.

* **Our Solution:** InterviewAlly uses a custom **Semantic Chunker** that respects conversational boundaries.
* It merges fragmented speech into coherent **"Turns."**
* It explicitly pairs **Interviewer Questions** with **Candidate Answers** into a single retrievable unit.
* **Result:** When the AI analyzes a skill, it retrieves the *entire context* of that specific answer, eliminating hallucinations.

### 2. 🏷️ Intelligent Metadata Filtering

* Every chunk is auto-tagged by topic (e.g., `Technical`, `Behavioral`, `Introduction`).
* This allows specialized queries like *"How was my performance on **Technical** questions?"* to ignore unrelated small talk.

### 3. ✨ The "Magic Rewriter" (Fact-Checked STAR Method)

* Identifies weak answers and rewrites them using the **STAR Method** (Situation, Task, Action, Result).
* Includes a **Constraint-Based Prompting** layer to prevent "yapping" (excessive conversational filler) and enforces technical accuracy (e.g., correcting fake terms).

### 4. 🔒 100% Local Privacy

* Interviews contain sensitive career data. No data leaves the user's machine.
* **Inference:** Ollama (Phi-3.5/Llama 3.2).
* **Vector Store:** ChromaDB (Persisted locally).
* **Transcription:** Faster-Whisper (On-device).

---

## 🛠️ Tech Stack

| Component | Technology | Why? |
| --- | --- | --- |
| **LLM Inference** | [Ollama](https://ollama.com/) | Runs Phi-3.5/Llama 3.2 locally with zero latency penalty. |
| **Orchestration** | [LangChain](https://www.langchain.com/) | Manages retrieval chains and structured output parsing. |
| **Speech-to-Text** | [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) | 4x faster than standard Whisper using CTranslate2. |
| **Diarization** | [Pyannote 3.1](https://github.com/pyannote/pyannote-audio) | SOTA speaker separation to distinguish "Interviewer" vs "Candidate." |
| **Vector DB** | [ChromaDB](https://www.trychroma.com/) | Lightweight, file-based vector storage (no Docker required). |
| **Backend** | FastAPI + SQLite | Async endpoint handling to prevent blocking during inference. |
| **Frontend** | Streamlit | Rapid UI prototyping. |

---

## 🏗️ Architecture Pipeline

1. **Ingestion:** Video Upload -> Audio Extraction (FFmpeg).
2. **Processing:**
* **Transcribe:** Whisper generates text segments with timestamps.
* **Diarize:** Pyannote identifies *who* is speaking when.
* **Merge:** A custom algorithm aligns Whisper segments with Speaker timestamps.


3. **Indexing (The "Secret Sauce"):**
* Transcript is parsed into **Q&A Pairs**.
* Metadata is extracted (`topic`, `duration`, `question_type`).
* Embedded via `nomic-embed-text-v1.5` and stored in ChromaDB.


4. **Inference:**
* **Analysis:** LLM generates a JSON scorecard (0-10 rubric).
* **Rewriter:** RAG retrieves the specific "weak" chunk and generates a STAR-based improvement.



---

## 🚀 Setup & Installation

### Prerequisites

* Python 3.10+
* [Ollama](https://ollama.com/) installed.
* **FFmpeg** installed (Required for audio processing).

### 1. Clone & Install

```bash
git clone https://github.com/nkhanna94/InterviewAlly.git
cd InterviewAlly
pip install -r requirements.txt

```

### 2. Model Setup

We recommend **Phi-3.5** for its balance of speed and reasoning capability on consumer hardware.

```bash
ollama pull phi3.5:latest

```

*(Note: You can swap this for `llama3.2` in `backend/brain.py` if preferred.)*

### 3. Environment Config

Create a `.env` file in the root directory. **Crucial:** You must accept user conditions for `pyannote/speaker-diarization-3.1` on HuggingFace to get a token.

```env
HUGGINGFACEHUB_API_TOKEN=hf_your_token_here

```

### 4. Run the App

Open two separate terminals:

**Terminal 1: Backend**

```bash
uvicorn backend.main:app --reload

```

**Terminal 2: Frontend**

```bash
streamlit run frontend/app.py

```

---

## ⚠️ Troubleshooting

* **PyTorch/Pyannote Error:** If you see `WeightsUnpickler error`, ensure you are using the patched `transcripts.py` which allows safe globals for Pyannote.
* **"Model Loading" hang:** If the Rewriter spins forever, check `backend/brain.py` and ensure `num_predict` is set (e.g., 500 tokens) to prevent infinite generation.

---

## 🔮 Future Roadmap

* **Hybrid Search:** Implement BM25 + Vector Search to better capture specific technical keywords (e.g., "Postgres", "AWS").
* **Video Analysis:** Use multimodal models (Llava) to analyze body language and eye contact from video frames.
* **Resume Integration:** RAG over the candidate's resume to check if their spoken answers match their claimed experience.