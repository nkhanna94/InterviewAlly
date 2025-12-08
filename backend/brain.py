from typing import List
import re
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
import traceback

LLM_MODEL = ChatOllama(
    model="llama3.2:latest", 
    temperature=0.2,
    keep_alive="5m"
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
        This ignores any "Here is your JSON:" prefixes that Llama sometimes adds.
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
        
        # ONE-SHOT PROMPT: Explicit example prevents hallucinated schemas
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

            **REQUIRED JSON FORMAT (Example):**
            {{
                "summary": "Candidate struggled with basic SQL but showed good attitude...",
                "technical_score": 4,
                "communication_score": 7,
                "cultural_fit_score": 6,
                "key_strengths": ["Honesty", "Python Basics", "Enthusiasm"],
                "critical_gaps": ["Did not know Joins", "Guessed on system design", "Vague answers"],
                "timestamps_of_interest": ["00:05:30", "00:12:15"]
            }}
            """,
            input_variables=["transcript"],
        )

        chain = prompt | self.llm 
        
        try:
            print("🧠 Sending transcript to Llama 3.1...")
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
            "1. **Diagnosis:** Refer to the transcript context for evidence.\n"
            "2. **Prescription:** Use your general knowledge to give study plans/advice.\n"
            "3. **Style:** Be encouraging but specific."
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
        """
        Intelligent Rewriter: 
        1. Finds the specific weak answer in the transcript.
        2. Rewrites it using the STAR method (grounded in reality).
        """
        
        # We use a specialized prompt that forces the AI to "Quote" the bad answer first
        prompt = PromptTemplate(
            template="""
            You are an expert Interview Coach. 
            
            **CANDIDATE PROFILE:**
            {profile}
            
            **CRITICISM TO FIX:**
            "{gap}"
            
            **TRANSCRIPT:**
            {transcript}
            
            **TASK:**
            1. **LOCATE:** Find the specific answer(s) in the transcript that triggered this criticism.
            2. **REWRITE:** Rewrite that specific answer to be strong, professional, and detailed.
            
            **RULES FOR REWRITING:**
            - **DO NOT HALLUCINATE:** Do not invent jobs or projects. Use the real details found in the transcript (e.g., if they said "Flutter login page", polish that specific task).
            - **USE "STAR" METHOD:** (Situation, Task, Action, Result).
            - **VOCABULARY:** Turn "I did a project" into "I spearheaded a prototype..." or "I implemented..."
            - **TONE:** If the candidate is a student, keep it ambitious but realistic.
            
            **OUTPUT FORMAT:**
            **Original Weak Answer:** "[Quote the specific part of the transcript]"
            
            **✨ Better Version:**
            "[The polished, professional rewrite]"
            """,
            input_variables=["gap", "transcript", "profile"],
        )
        
        chain = prompt | self.llm
        
        response = chain.invoke({
            "gap": gap_description, 
            "transcript": transcript[:15000], 
            "profile": profile_context
        })
        
        if hasattr(response, 'content'):
            return response.content
        return str(response)