#InterviewAlly - AI Interview Coach

import streamlit as st
import json
import ollama
from transcripts import create_transcript_generator


# =============================================================================
# CONFIGURATION & STYLING
# =============================================================================

def configure_page():
    """Configure Streamlit page settings and custom CSS styling"""
    st.set_page_config(
        page_title="InterviewAlly - AI Interview Coach", 
        page_icon="🚀",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    def load_css():
        with open('.streamlit/style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    load_css()


# =============================================================================
# DATA PROCESSING UTILITIES
# =============================================================================

def preprocess_transcript(transcript, min_words=1):
    """
    Clean and filter transcript data
    
    Args:
        transcript (list): Raw transcript data
        min_words (int): Minimum word count threshold
        
    Returns:
        list: Cleaned transcript chunks
    """
    cleaned = []
    for chunk in transcript:
        text = chunk.get('text', '').strip()
        if isinstance(text, str) and len(text.split()) >= min_words and text not in ['-', '...', '']:
            cleaned.append(chunk)
    return cleaned

def combine_transcript_text(transcript):
    """
    Combine all transcript chunks into single text
    
    Args:
        transcript (list): Processed transcript chunks
        
    Returns:
        str: Combined text content
    """
    return ' '.join([chunk['text'].strip() for chunk in transcript])

def calculate_transcript_stats(text):
    """
    Calculate basic statistics for transcript
    
    Args:
        text (str): Combined transcript text
        
    Returns:
        dict: Statistics including word count and estimated duration
    """
    word_count = len(text.split())
    estimated_duration = max(1, word_count // 150)  # Rough speaking pace
    
    return {
        'word_count': word_count,
        'estimated_duration': estimated_duration
    }

# =============================================================================
# AI FEEDBACK GENERATION
# =============================================================================

def get_overall_feedback(full_text):
    """
    Generate comprehensive feedback for candidates
    
    Args:
        full_text (str): Complete interview transcript
        
    Returns:
        str: Structured feedback response
    """
    prompt = f"""
    You are an experienced interview coach giving direct, actionable feedback. Be concise but specific. Give gender-neutral feedback.

    Analyze this interview transcript and provide feedback in exactly this format:

    **Overall Performance:** One honest sentence about how they did.

    **Critical Gaps:** The #1 thing that hurt their chances most - be direct about what's missing.

    **Technical Assessment:** Rate their technical depth as Strong/Moderate/Weak and give ONE specific example from their answers.

    **Communication:** One sentence - were they clear or confusing? Give a quick example.

    **Next Interview:** Exactly 3 concrete actions they should take before their next interview.

    Keep each section to 1-2 sentences max. Focus on what actually matters for getting hired.

    Transcript:
    {full_text}
    """

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.7}
    )
    return response['message']['content']

def get_hr_feedback(full_text):
    """
    Generate HR assessment for candidate evaluation
    
    Args:
        full_text (str): Complete interview transcript
        
    Returns:
        str: HR evaluation response
    """
    prompt = f"""
    You are HR. Evaluate this candidate's interview. Be strict. Use gender-neutral language.

    Overall: Should we hire them? Yes/No and why in one line.

    Technical: Strong/Adequate/Weak - what's missing?

    Communication: Clear or confusing? One sentence.

    Problem-Solving: Good or bad analytical thinking? 

    Cultural Fit: Team player or not?

    Decision: Hire/No Hire/Another Round - give one sentence on why.

    Transcript:
    {full_text}
    """

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.5}
    )
    return response['message']['content']

def get_chatbot_response(user_question, interview_feedback, user_role):
    """
    Generate role-aware chatbot responses
    
    Args:
        user_question (str): User's question
        interview_feedback (str): Previously generated feedback
        user_role (str): Either 'candidate' or 'hr'
        
    Returns:
        str: Contextual response based on role
    """
    try:
        if user_role == "candidate":
            prompt = f"""
            You are a helpful interview coach. Answer the candidate's question directly and concisely.

            CONTEXT: The candidate received this interview feedback: {interview_feedback}

            User's question: {user_question}

            INSTRUCTIONS:
            - If they ask about their interview performance - reference the context above
            - If they ask general questions - just answer directly, do not reference the context 
            - Keep responses focused and practical

            Answer based on what they actually asked.
            """
        else:  # HR role
            prompt = f"""
            You are an HR consultant helping with candidate evaluation. Answer the HR professional's question directly.

            CONTEXT: You provided this assessment of a candidate: {interview_feedback}

            HR question: {user_question}

            INSTRUCTIONS:
            - If they ask about the candidate's performance - reference your assessment
            - If they ask about hiring processes, interview techniques, or evaluation criteria - provide professional guidance
            - Keep responses focused and actionable for HR decisions

            Answer from an HR perspective.
            """

        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3}
        )
        return response['message']['content']
        
    except Exception as e:
        return "Can't process your question right now. Please try again."

# =============================================================================
# UI COMPONENT FUNCTIONS
# =============================================================================

def render_header():
    """Render the main application header"""
    st.markdown("""
    <div class="main-header">
        <h1>🚀 InterviewAlly</h1>
        <p style="font-size: 1.2rem; margin: 0;">AI-Powered Interview Coach</p>
    </div>
    """, unsafe_allow_html=True)


def render_file_upload():
    """
    Render video upload section with role-aware messaging
    
    Returns:
        tuple: (transcript_data, upload_success)
    """
    user_role = st.session_state.get("user_role", "candidate")
    
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    
    # Role-aware messaging
    if user_role == "candidate":
        st.markdown("### 🎥 Upload Your Interview Video")
        st.markdown("*Upload your practice interview or real interview recording for personalized feedback*")
        help_text = "Upload your interview video to get detailed performance feedback and improvement recommendations"
    else:  # HR role
        st.markdown("### 🎥 Upload Candidate Interview Video") 
        st.markdown("*Upload the candidate's interview for comprehensive evaluation and assessment*")
        help_text = "Upload candidate interview videos to generate structured evaluation reports"
    
    uploaded_file = st.file_uploader(
        "",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        help=help_text
    )
    
    transcript_data = None
    upload_success = False
    
    if uploaded_file:
        # Initialize transcript generator
        generator = create_transcript_generator()
        if not generator:
            st.markdown('</div>', unsafe_allow_html=True)
            return None, False
        
        # Validate file
        if not generator.validate_video_file(uploaded_file):
            st.markdown('</div>', unsafe_allow_html=True)
            return None, False
        
        # Show file info
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"📁 **File**: {uploaded_file.name}")
        with col2:
            st.info(f"📏 **Size**: {uploaded_file.size / 1024 / 1024:.1f} MB")
        
        # Role-aware button text
        button_text = "🚀 **Generate Transcript**" if user_role == "candidate" else "🚀 **Process Interview**"
        
        if st.button(button_text, type="primary", use_container_width=True):
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(message, percentage):
                progress_bar.progress(percentage)
                status_text.text(message)
            
            try:
                with st.spinner("Processing your video..."):
                    transcript_data = generator.upload_and_process_video(
                        uploaded_file, 
                        progress_callback=update_progress
                    )
                
                if transcript_data:
                    upload_success = True
                    progress_bar.progress(100)
                    status_text.text("✅ Transcript generated successfully!")
                    
                    # Save transcript to session state
                    st.session_state.transcript_data = transcript_data
                    st.session_state.video_filename = uploaded_file.name
                    
                    # Optional: Save to file
                    generator.save_transcript_to_file(
                        transcript_data, 
                        f"transcript_{uploaded_file.name.split('.')[0]}.json"
                    )
                else:
                    st.error("Failed to generate transcript. Please try again.")
                    
            except Exception as e:
                st.error(f"Error processing video: {str(e)}")
                progress_bar.progress(0)
                status_text.text("❌ Processing failed")
    
    st.markdown('</div>', unsafe_allow_html=True)
    return transcript_data, upload_success


def render_transcript_stats(transcript, stats):
    """
    Render transcript statistics
    
    Args:
        transcript (list): Original transcript data
        stats (dict): Calculated statistics
    """
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="📄 Transcript Entries", 
            value=len(transcript),
            help="Total number of transcript segments"
        )
    
    with col2:
        st.metric(
            label="📝 Word Count", 
            value=f"{stats['word_count']:,}",
            help="Total words in processed transcript"
        )
    
    with col3:
        st.metric(
            label="⏱️ Est. Duration", 
            value=f"~{stats['estimated_duration']}m",
            help="Estimated interview duration"
        )

def render_role_selection():
    """Render role selection interface as the first step"""
    st.markdown("### 👥 Choose Your Role")

    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(
            "👤 **I'm a Candidate**\n\n*Get feedback to improve my performance*", 
            type="secondary", 
            use_container_width=True,
            help="Upload your interview video to receive detailed feedback and coaching"
        ):
            st.session_state.user_role = "candidate"
            st.session_state.role_selected = True
            st.rerun()

    with col2:
        if st.button(
            "💼 **I'm an HR/Interviewer**\n\n*Evaluate candidate performance*", 
            type="primary", 
            use_container_width=True,
            help="Upload candidate interviews to get structured assessment reports"
        ):
            st.session_state.user_role = "hr"
            st.session_state.role_selected = True
            st.rerun()
    
    # Return whether role has been selected
    return st.session_state.get("role_selected", False)


def render_analysis_button(full_text):
    """
    Render analysis button and handle feedback generation
    
    Args:
        full_text (str): Complete transcript text
    """
    if st.button("🎯 **Analyze Interview**", type="primary", use_container_width=True):
        with st.spinner("🤖 Analyzing interview performance..."):
            if st.session_state.user_role == "candidate":
                summary = get_overall_feedback(full_text)
            else:  # HR role
                summary = get_hr_feedback(full_text)
            st.session_state.feedback_summary = summary

def render_feedback_display():
    """Render feedback results and download option"""
    if "feedback_summary" not in st.session_state or not st.session_state.feedback_summary:
        return
        
    st.markdown("---")
    
    # Role-specific headers and filenames
    if st.session_state.get("user_role") == "hr":
        st.markdown("## 📋 Candidate Assessment Report")
        download_label = "📥 Download Assessment"
        download_filename = "hr_assessment.txt"
    else:
        st.markdown("## 📊 Your Interview Feedback")
        download_label = "📥 Download Analysis"
        download_filename = "candidate_feedback.txt"
    
    # Display feedback
    st.markdown('<div class="feedback-container">', unsafe_allow_html=True)
    st.markdown(st.session_state.feedback_summary)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Download section
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.download_button(
            download_label,
            data=st.session_state.feedback_summary,
            file_name=download_filename,
            mime="text/plain",
            type="secondary",
            use_container_width=True
        )

def render_chatbot_interface():
    """Render the interactive chatbot interface"""
    if "feedback_summary" not in st.session_state or not st.session_state.feedback_summary:
        return
        
    st.markdown("---")
    
    # Role-specific chatbot headers
    if st.session_state.get("user_role") == "hr":
        st.markdown("## 💬 HR Consultation")
        chat_placeholder = "Ask about candidate evaluation, hiring recommendations, or assessment criteria..."
    else:
        st.markdown("## 💬 Ask Your Interview Coach")
        chat_placeholder = "Ask about your feedback, request study resources, or get clarification..."

    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Display existing chat messages
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle new chat input
    if prompt := st.chat_input(chat_placeholder):
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                response = get_chatbot_response(
                    prompt, 
                    st.session_state.feedback_summary, 
                    st.session_state.get("user_role", "candidate")
                )
                st.markdown(response)

        # Add assistant response to history
        st.session_state.chat_messages.append({"role": "assistant", "content": response})

    # Clear chat history option
    if st.session_state.chat_messages:
        if st.button("🗑️ Clear Chat History", type="secondary"):
            st.session_state.chat_messages = []
            st.rerun()

def render_footer():
    """Render application footer"""
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; font-size: 13px; margin-top: 3rem; color: #666;">
            © 2025 niharika
        </div>
        """, unsafe_allow_html=True)     
    

# =============================================================================
# MAIN APPLICATION LOGIC
# =============================================================================


def main():
    """Main application entry point"""
    # Initialize page configuration
    configure_page()
    
    # Render main header
    render_header()
    
    # Step 1: Role Selection (if not already selected)
    if not st.session_state.get("role_selected", False):
        role_selected = render_role_selection()
        if not role_selected:
            render_footer()  # Show footer and stop here
            return
    
    # Show current role status
    role = st.session_state.get("user_role", "candidate")
    
    # Outer layout: center everything
    spacer_l, content_col, spacer_r = st.columns([1, 6, 1])
    
    with content_col:
        st.markdown("""
        <div style="display: flex; justify-content: center; align-items: center; gap: 1rem; margin-top: 12px;">
        """, unsafe_allow_html=True)
    
        col1, col2 = st.columns([6, 1])  # Inner layout: text and button
    
        with col1:
            if role == "candidate":
                st.markdown(
                    '<div style="font-weight: bold; font-size: 22px; text-align: center;">🎯 Candidate Mode</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div style="font-weight: bold; font-size: 22px; text-align: center;">📊 HR Mode</div>',
                    unsafe_allow_html=True
                )
    
        with col2:
            if st.button("🔄", type="secondary", help="Switch between Candidate and HR modes"):
                st.session_state.role_selected = False
                st.session_state.pop("user_role", None)
                st.rerun()
    
        st.markdown("</div>", unsafe_allow_html=True)


    st.markdown("---")

    # Step 2: Video upload and transcript generation
    transcript_data, upload_success = render_file_upload()
    
    # Check if we have transcript data (from current upload or session)
    if transcript_data or "transcript_data" in st.session_state:
        
        # Use current transcript or session transcript
        current_transcript = transcript_data or st.session_state.get("transcript_data")
        
        # Clean and validate transcript
        cleaned_transcript = preprocess_transcript(current_transcript)
        
        if not cleaned_transcript:
            st.error("❌ **Transcript Issue Found**\n\nNo usable entries detected. Please try uploading a different video.")
            render_footer()
            return
        
        # Process transcript and calculate stats
        full_text = combine_transcript_text(cleaned_transcript)
        stats = calculate_transcript_stats(full_text)
        
        # Display transcript statistics
        render_transcript_stats(current_transcript, stats)
        
        if upload_success or "transcript_data" in st.session_state:
            st.success("✅ Transcript successfully processed and ready for analysis!")
        
        # Analysis button (role is already set)
        render_analysis_button(full_text)
    
    # Display feedback (independent of upload state)
    render_feedback_display()
    
    # Chatbot interface (independent of upload state)
    render_chatbot_interface()
    
    # Application footer
    render_footer()


# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()