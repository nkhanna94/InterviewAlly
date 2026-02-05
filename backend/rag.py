from typing import List, Dict, Optional
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import re

load_dotenv()

# Configuration
PERSIST_DIRECTORY = "./chroma_db"

EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1.5",
    model_kwargs={'trust_remote_code': True}
)

class TranscriptProcessor:
    
    def __init__(self):
        self.embeddings = EMBEDDING_MODEL
        # Heuristics for identifying speakers
        self.interviewer_keywords = ["interviewer", "recruiter", "manager", "speaker_0"]
        self.candidate_keywords = ["candidate", "speaker_1", "speaker_2"]
    
    def identify_role(self, speaker: str) -> str:
        """Classify speaker as 'interviewer' or 'candidate'"""
        speaker_lower = speaker.lower()
        
        if any(kw in speaker_lower for kw in self.interviewer_keywords):
            return "interviewer"
        elif any(kw in speaker_lower for kw in self.candidate_keywords):
            return "candidate"
        else:
            # Fallback: First speaker is usually interviewer
            return "unknown"
    
    def detect_question(self, text: str) -> bool:
        """Heuristic to detect if text is a question"""
        text = text.strip()
        
        # Explicit question marks
        if '?' in text:
            return True
        
        # Question starters
        question_starters = [
            "what", "why", "how", "when", "where", "who", "which",
            "can you", "could you", "would you", "have you",
            "tell me about", "describe", "explain", "walk me through"
        ]
        
        text_lower = text.lower()
        return any(text_lower.startswith(starter) for starter in question_starters)
    
    def create_qa_chunks(self, raw_transcript: List[Dict], job_id: str) -> List[Document]:
        """
        Strategy: Group by Q&A pairs with context windows
        
        Each chunk contains:
        - The question (interviewer's turn)
        - The answer (candidate's turn)
        - Optional: Previous question for context
        """
        
        documents = []
        
        # Step 1: Group consecutive segments by speaker
        speaker_turns = []
        current_turn = None
        
        for segment in raw_transcript:
            speaker = segment.get("speaker", "Unknown")
            text = segment.get("text", "").strip()
            
            if not text:
                continue
            
            # If same speaker, append to current turn
            if current_turn and current_turn["speaker"] == speaker:
                current_turn["text"] += " " + text
                current_turn["end"] = segment.get("end")
            else:
                # New speaker turn
                if current_turn:
                    speaker_turns.append(current_turn)
                
                current_turn = {
                    "speaker": speaker,
                    "role": self.identify_role(speaker),
                    "text": text,
                    "start": segment.get("start"),
                    "end": segment.get("end")
                }
        
        # Don't forget the last turn
        if current_turn:
            speaker_turns.append(current_turn)
        
        # Step 2: Pair questions with answers
        i = 0
        previous_question = None
        
        while i < len(speaker_turns):
            turn = speaker_turns[i]
            
            # Case 1: This is a question from interviewer
            if turn["role"] == "interviewer" and self.detect_question(turn["text"]):
                question = turn
                
                # Look ahead for candidate's answer
                if i + 1 < len(speaker_turns) and speaker_turns[i + 1]["role"] == "candidate":
                    answer = speaker_turns[i + 1]
                    
                    # Create Q&A chunk
                    doc = self._create_qa_document(
                        question=question,
                        answer=answer,
                        previous_question=previous_question,
                        job_id=job_id
                    )
                    documents.append(doc)
                    
                    previous_question = question
                    i += 2  # Skip both question and answer
                else:
                    # Question with no answer (edge case)
                    i += 1
            
            # Case 2: Candidate's long monologue (no clear question)
            elif turn["role"] == "candidate" and len(turn["text"].split()) > 100:
                # Split long answers into sub-chunks
                sub_chunks = self._split_long_answer(turn, max_words=150)
                
                for sub_chunk in sub_chunks:
                    doc = Document(
                        page_content=sub_chunk["text"],
                        metadata={
                            "job_id": job_id,
                            "type": "candidate_monologue",
                            "speaker": turn["speaker"],
                            "role": "candidate",
                            "start_time": sub_chunk["start"],
                            "end_time": sub_chunk["end"],
                            "has_context": False
                        }
                    )
                    documents.append(doc)
                
                i += 1
            
            else:
                # Other cases (small talk, clarifications, etc.)
                i += 1
        
        print(f"[{job_id}] Created {len(documents)} Q&A chunks from {len(speaker_turns)} speaker turns.")
        return documents
    
    def _create_qa_document(
        self, 
        question: Dict, 
        answer: Dict, 
        previous_question: Optional[Dict],
        job_id: str
    ) -> Document:
        """
        Create a rich document with question + answer + context
        """
        
        # Build the chunk text
        text_parts = []
        
        # Optional: Add previous question for context
        if previous_question:
            text_parts.append(f"[Previous Question: {previous_question['text']}]")
        
        # Main question
        text_parts.append(f"Question: {question['text']}")
        
        # Candidate's answer
        text_parts.append(f"Answer: {answer['text']}")
        
        combined_text = "\n\n".join(text_parts)
        
        # Classify question type (for better retrieval)
        question_type = self._classify_question(question["text"])
        
        return Document(
            page_content=combined_text,
            metadata={
                "job_id": job_id,
                "type": "qa_pair",
                "question": question["text"],
                "answer": answer["text"],
                "question_type": question_type,
                "interviewer": question["speaker"],
                "candidate": answer["speaker"],
                "start_time": question["start"],
                "end_time": answer["end"],
                "duration": answer["end"] - question["start"],
                "answer_length_words": len(answer["text"].split()),
                "has_context": previous_question is not None
            }
        )
    
    def _classify_question(self, question_text: str) -> str:
        """
        Tag questions by category for better filtering
        """
        text_lower = question_text.lower()
        
        if any(kw in text_lower for kw in ["experience", "worked", "previous role", "projects"]):
            return "experience"
        elif any(kw in text_lower for kw in ["algorithm", "code", "implement", "solve", "technical"]):
            return "technical"
        elif any(kw in text_lower for kw in ["team", "conflict", "challenge", "difficult"]):
            return "behavioral"
        elif any(kw in text_lower for kw in ["design", "architecture", "scale", "system"]):
            return "system_design"
        else:
            return "general"
    
    def _split_long_answer(self, turn: Dict, max_words: int = 150) -> List[Dict]:
        """
        Split a long candidate answer into sentence-based sub-chunks
        """
        text = turn["text"]
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            if current_word_count + sentence_words > max_words and current_chunk:
                # Seal current chunk
                chunks.append({
                    "text": " ".join(current_chunk),
                    "start": turn["start"],  # Approximate
                    "end": turn["end"]
                })
                current_chunk = [sentence]
                current_word_count = sentence_words
            else:
                current_chunk.append(sentence)
                current_word_count += sentence_words
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                "text": " ".join(current_chunk),
                "start": turn["start"],
                "end": turn["end"]
            })
        
        return chunks
    
    def index_transcript(self, job_id: str, raw_transcript: List[Dict]):
        """
        Main entry point - replaces your old method
        """
        docs = self.create_qa_chunks(raw_transcript, job_id)
        
        if not docs:
            print(f"[{job_id}] Warning: No documents created.")
            return
        
        # Index into ChromaDB
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            collection_name="interview_transcripts",
            persist_directory=PERSIST_DIRECTORY
        )
        
        print(f"[{job_id}] ✅ Indexed {len(docs)} Q&A chunks.")

    def get_retriever(self):
        return Chroma(
            collection_name="interview_transcripts",
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=self.embeddings
        ).as_retriever()