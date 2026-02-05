from typing import List
import re
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
import traceback

LLM_MODEL = ChatOllama(
    model="phi3.5:latest", 
    temperature=0.1,     # Lower temperature for precision
    num_ctx=8192,
    keep_alive="5m",
    num_predict=500,    
    repeat_penalty=1.1   # <--- PREVENT LOOPS
)

# --- 1. Structure (Pydantic) ---
class InterviewAnalysis(BaseModel):
    summary: str = Field(description="Honest, direct summary of performance.")
    technical_score: int = Field(description="Score 1-10 based on the STRICT rubric.")
    communication_score: int = Field(description="Score 1-10 based on clarity and conciseness.")
    cultural_fit_score: int = Field(description="Score 1-10 based on attitude and honesty.")
    key_strengths: List[str] = Field(description="List of 3 distinct strengths found in the transcript.")
    critical_gaps: List[str] = Field(description="List of 3 distinct missing skills, wrong answers, or red flags.")
    timestamps_of_interest: List[str] = Field(description="List of timestamps (e.g., '00:12:30') where significant moments occurred.")

# --- 2. The Analysis Engine ---
class InterviewBrain:
    def __init__(self):
        self.llm = LLM_MODEL

    def _extract_json(self, text: str) -> str:
        """
        Robust cleaner: Finds the substring between the first { and last }.
        This ignores any "Here is your JSON:" prefixes that llm sometimes adds.
        """
        try:
            # 1. Strip whitespace
            text = text.strip()
            
            # 2. Find JSON pattern
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json_match.group(0)
                
            return text
        except:
            return text
        
    def generate_analysis(self, full_transcript_text: str) -> dict:
        parser = JsonOutputParser(pydantic_object=InterviewAnalysis)
        
        # Check for empty transcript BEFORE sending to LLM
        if not full_transcript_text or len(full_transcript_text) < 50:
            print("❌ Error: Transcript is empty or too short!")
            return {
                "summary": "Error: The video transcription failed. Please check backend logs.",
                "technical_score": 0,
                "communication_score": 0,
                "cultural_fit_score": 0,
                "key_strengths": ["Error in Transcription"],
                "critical_gaps": ["Please check server logs", "Pyannote might have failed"],
                "timestamps_of_interest": []
            }

        # UPDATED PROMPT: Uses generic placeholders to prevent "Copying"
        prompt = PromptTemplate(
            template="""
            You are a **Skeptical, High-Standards Hiring Manager**. 
            Analyze the transcript below using this **STRICT RUBRIC**:
            - **1-3 (Unqualified):** Confused, wrong answers.
            - **4-6 (Junior):** Knows terms but lacks depth.
            - **7-8 (Competent):** Good answers, clear reasoning.
            - **9-10 (Expert):** Deep insight, perfect answers.
            
            **Transcript:**
            {transcript}
            
            **INSTRUCTIONS:**
            1. Identify RED FLAGS (guessing, lying, vague answers).
            2. Output ONLY a raw JSON object. Do NOT wrap it in markdown codes.
            3. Do NOT output the schema. Output the DATA.

            **REQUIRED JSON FORMAT:**
            {{
                "summary": "Brief summary of candidate performance...",
                "technical_score": 0,
                "communication_score": 0,
                "cultural_fit_score": 0,
                "key_strengths": ["Strength 1", "Strength 2", "Strength 3"],
                "critical_gaps": ["Gap 1", "Gap 2", "Gap 3"],
                "timestamps_of_interest": ["00:00:00"]
            }}
            """,
            input_variables=["transcript"],
        )

        chain = prompt | self.llm 
        
        try:
            print("🧠 Sending transcript to phi3.5 ...")
            # Get raw string response first
            raw_response = chain.invoke({"transcript": full_transcript_text[:25000]})
            
            # Handle LangChain response types
            if hasattr(raw_response, 'content'):
                text_output = raw_response.content
            else:
                text_output = str(raw_response)
                
            # Clean and Parse
            cleaned_json_str = self._extract_json(text_output)
            print("✅ Parsing Analysis...")
            return parser.parse(cleaned_json_str)
            
        except Exception as e:
            print(f"❌ Analysis Failed: {e}")
            print("Raw Output was:", text_output if 'text_output' in locals() else "N/A")
            traceback.print_exc()
            return None

    def get_chat_response(self, query: str, retriever, analysis_context: str = ""):
        """
        RAG Chatbot with Hybrid Knowledge.
        """
        system_prompt = (
            "You are an expert **AI Interview Coach**. "
            "Your goal is to help the candidate get hired, but you must be honest about their current gaps.\n\n"
            
            "**CONTEXT FROM INTERVIEW:**\n"
            "{context}\n\n"
            
            "**PREVIOUS ANALYSIS SUMMARY:**\n"
            f"{analysis_context}\n\n"
            
            "**INSTRUCTIONS:**\n"
            "1. **Check Evidence:** Did the user mention this topic in the interview? If yes, critique their specific answer.\n"
            "2. **If Evidence is Missing:** Explicitly state: 'You didn't cover this in the interview, but here is the general advice...'\n"
            "3. **Be Concise:** No long bullet points. Keep it conversational and short (under 100 words)."
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        response = rag_chain.invoke({"input": query})
        return response["answer"]

    def rewrite_answer(self, gap_description: str, transcript: str, profile_context: str) -> str:
        prompt = PromptTemplate(
            template="""
            TASK: Replace a weak interview answer with a short "Gold Standard" technical response (under 100 words).
            
            INPUTS:
            - Candidate Profile: {profile}
            - Critique: "{gap}"
            - Transcript Segment: {transcript}
            
            INSTRUCTIONS:
            1. FIND the weak answer in the transcript matching the critique.
            2. REWRITE it using the STAR method (Situation, Task, Action, Result).
            3. CORRECT any technical errors (e.g., if they cite fake algorithms, use real ones like QuickSort/HashMap).
            4. STOP immediately after the rewrite.
            
            STRICT OUTPUT FORMAT:
            **Original Weak Answer:**
            "[Exact quote from transcript]"
            
            **✨ Gold Standard Rewrite:**
            "[The polished, technical answer using 'In my experience...' style]"
            """,
            input_variables=["gap", "transcript", "profile"],
        )
        
        # Bind stop tokens to ensure it doesn't run forever
        chain = prompt | self.llm 
        
        # Reduced context to 10k chars for speed
        response = chain.invoke({
            "gap": gap_description, 
            "transcript": transcript[:25000], 
            "profile": profile_context
        })
        
        if hasattr(response, 'content'):
            return response.content
        return str(response)