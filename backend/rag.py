from typing import List, Dict
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# Configuration
PERSIST_DIRECTORY = "./chroma_db"


EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1.5",
    model_kwargs={'trust_remote_code': True}
)

class TranscriptProcessor:
    """
    Handles the conversion of raw transcripts into vector-searchable documents.
    """
    
    def __init__(self):
        self.embeddings = EMBEDDING_MODEL

    def create_semantic_chunks(self, raw_transcript: List[Dict], job_id: str, chunk_size_words: int = 50) -> List[Document]:
        """
        Custom Chunker: Groups small transcript segments into meaningful paragraphs 
        while preserving temporal metadata (Start/End times).
        """
        documents = []
        current_chunk_text = []
        current_start_time = None
        current_speakers = set()
        
        # Iterate through raw segments (which are often just single words or short phrases)
        for segment in raw_transcript:
            text = segment.get("text", "").strip()
            start = segment.get("start")
            end = segment.get("end")
            speaker = segment.get("speaker", "Unknown")
            
            if not text:
                continue

            # Capture start time of the FIRST segment in the chunk
            if current_start_time is None:
                current_start_time = start
            
            current_chunk_text.append(text)
            current_speakers.add(speaker)
            
            # If chunk is big enough, seal it and start a new one
            if len(current_chunk_text) >= chunk_size_words:
                combined_text = " ".join(current_chunk_text)
                
                # Create a LangChain Document
                doc = Document(
                    page_content=combined_text,
                    metadata={
                        "job_id": job_id,
                        "start_time": current_start_time,
                        "end_time": end,  # End time of the last segment
                        "speakers": ", ".join(current_speakers)
                    }
                )
                documents.append(doc)
                
                # Reset accumulators
                current_chunk_text = []
                current_start_time = None
                current_speakers = set()
        
        # Handle any remaining text in the buffer
        if current_chunk_text:
            combined_text = " ".join(current_chunk_text)
            doc = Document(
                page_content=combined_text,
                metadata={
                    "job_id": job_id,
                    "start_time": current_start_time,
                    "end_time": raw_transcript[-1].get("end"),
                    "speakers": ", ".join(current_speakers)
                }
            )
            documents.append(doc)
            
        print(f"[{job_id}] Created {len(documents)} semantic chunks.")
        return documents

    def index_transcript(self, job_id: str, raw_transcript: List[Dict]):
        """
        Main pipeline: Chunk -> Embed -> Store in VectorDB
        """
        # 1. Chunking
        docs = self.create_semantic_chunks(raw_transcript, job_id)
        
        if not docs:
            print(f"[{job_id}] Warning: No documents created from transcript.")
            return

        # 2. Vector Store (ChromaDB)
        # This will embed the text and store it locally in ./chroma_db
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            collection_name="interview_transcripts",
            persist_directory=PERSIST_DIRECTORY
        )
        print(f"[{job_id}] Successfully indexed {len(docs)} chunks into ChromaDB.")

    def get_retriever(self):
        """Returns a retriever object for querying the database"""
        return Chroma(
            collection_name="interview_transcripts",
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=self.embeddings
        ).as_retriever()
    
    