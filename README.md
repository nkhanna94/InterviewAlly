# InterviewAlly

**InterviewAlly** is an AI-powered interview analysis tool that helps candidates and HR professionals improve the interview process.  
Simply upload your interview videos, and InterviewAlly will generate transcripts, run AI analysis, and give targeted feedback — powered by **VideoDB**, **Ollama**, and **RAG**.

---

## 🚀 Features

- **🎥 Video Upload & Processing**  
  Upload interview recordings directly to the platform.

- **📝 Automatic Transcription**  
  Generates high-quality transcripts using **VideoDB**.

- **🤖 AI-Powered Feedback**  
  Get detailed analysis on communication skills, confidence, and clarity.

- **🧠 RAG (Retrieval-Augmented Generation)**  
  Uses contextual retrieval to provide feedback with evidence from the transcript.

- **👥 Two Analysis Modes**
  - **HR Mode:** Evaluate candidates, identify strengths & weaknesses.
  - **Candidate Mode:** Get actionable tips to improve future interviews.

---

## 🛠️ Tech Stack

- **Backend:** Python 
- **AI Model:** [Ollama LLama 3](https://ollama.ai/) for LLM-based analysis  
- **Transcription Engine:** [VideoDB](https://videodb.io/)  
- **RAG Pipeline:** Custom embeddings & retrieval system for transcript-based Q&A  
- **Frontend:** (React / Next.js / Streamlit — specify your choice)  

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/nkhanna94/InterviewAlly.git
   cd interviewally
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   Create a `.env` file with:

   ```env
   VIDEODB_API_KEY=your_videodb_api_key
   ```

4. **Run the app**

   ```bash
   streamlit run app.py
   ```

---

## 📄 Usage

1. Upload an interview video.
2. Select **HR Mode** or **Candidate Mode**.
3. Receive:

   * AI-generated transcript
   * Key highlights from the interview
   * Strengths & areas for improvement

---

## 📅 Roadmap

* [ ] Multi-language support
* [ ] Live interview feedback
* [ ] Integration with ATS systems

---

## 🤝 Contributing

Contributions are welcome!
Fork the repo and submit a pull request with your improvements.

---
