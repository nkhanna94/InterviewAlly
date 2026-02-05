import streamlit as st
import requests
import time
import pandas as pd

# Configuration
API_URL = "http://localhost:8000"

st.set_page_config(page_title="InterviewAlly - AI Coach", page_icon="🚀", layout="wide")

# --- Custom CSS for "Coach" feel ---
st.markdown("""
    <style>
    .big-score { font-size: 3rem; font-weight: bold; color: #4CAF50; }
    .weak-score { font-size: 3rem; font-weight: bold; color: #FF5252; }
    .stButton>button { width: 100%; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

def render_header():
    st.title("🚀 InterviewAlly")
    st.caption("Your Personal AI Interview Coach. Upload a mock interview to get brutal feedback & perfect answers.")

def upload_section():
    with st.sidebar:
        st.header("1. Upload Session")
        uploaded_file = st.file_uploader("Upload video/audio", type=["mp4", "mov", "avi", "mp3", "wav"])
        
        if uploaded_file is not None:
            if st.button("Analyze My Performance", type="primary"):
                with st.spinner("Uploading & processing..."):
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    try:
                        response = requests.post(f"{API_URL}/upload-video", files=files)
                        if response.status_code == 200:
                            st.session_state["job_id"] = response.json()["job_id"]
                            st.session_state["status"] = "processing"
                            st.success("Started!")
                    except Exception as e:
                        st.error(f"Error: {e}")

        # Status Polling
        if "job_id" in st.session_state and st.session_state.get("status") != "completed":
            with st.spinner("🎧 Listening & Analyzing..."):
                while True:
                    try:
                        res = requests.get(f"{API_URL}/get-status/{st.session_state['job_id']}")
                        status = res.json().get("status")
                        if status == "completed":
                            st.session_state["status"] = "completed"
                            st.rerun()
                        elif status == "failed":
                            st.error("Processing failed.")
                            break
                        time.sleep(2)
                    except:
                        break

def render_dashboard():
    if st.session_state.get("status") != "completed":
        st.info("👈 Upload your interview recording to start coaching.")
        return

    job_id = st.session_state["job_id"]

    # Fetch Analysis if not present
    if "analysis" not in st.session_state:
        with st.spinner("🧠 Generating Coach Report..."):
            res = requests.post(f"{API_URL}/analyze/{job_id}")
            if res.status_code == 200:
                st.session_state["analysis"] = res.json()
                st.rerun()

    data = st.session_state["analysis"]

    # --- TOP ROW: SCORES ---
    st.divider()
    st.subheader("📊 Performance Scorecard")
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.metric("Technical Depth", f"{data.get('technical_score')}/10")
    with c2:
        st.metric("Communication", f"{data.get('communication_score')}/10")
    with c3:
        st.metric("Culture Fit", f"{data.get('cultural_fit_score')}/10")
    with c4:
        # Simple Radar Chart logic could go here
        st.markdown(f"**Verdict:**")
        if data.get('technical_score') > 7:
            st.markdown(":green[**Ready for Onsite**]")
        else:
            st.markdown(":red[**Needs Prep**]")

    # --- MIDDLE ROW: FEEDBACK ---
    st.divider()
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("✅ What You Did Well")
        for item in data.get('key_strengths', []):
            st.success(f"• {item}")

    with col_right:
        st.subheader("⚠️ Critical Gaps (Fix These!)")
        for item in data.get('critical_gaps', []):
            st.error(f"• {item}")

    # ... inside render_dashboard() ...

    # --- FEATURE: MAGIC REWRITER ---
    st.divider()
    st.subheader("✨ Magic Rewriter")
    st.markdown("Turn a weak answer into a gold-standard response (grounded in **your** actual experience).")

    # 1. Initialize History in Session State
    if "rewrite_history" not in st.session_state:
        st.session_state.rewrite_history = []

    # 2. Input Selection
    col_input, col_btn = st.columns([3, 1])
    
    with col_input:
        rewrite_mode = st.radio("What do you want to fix?", ["Pick a Critical Gap", "Type my own answer"], horizontal=True)
        
        target_text = ""
        if rewrite_mode == "Pick a Critical Gap":
            target_text = st.selectbox("Select a gap:", options=data.get('critical_gaps', []) + ["General Introduction"])
        else:
            target_text = st.text_area("Paste your weak answer or concern here:", height=100, placeholder="e.g., I don't have much leadership experience...")

    # 3. Construct Candidate Profile for Context
    # This tells the AI: "She is an ECE student, NOT a Manager."
    candidate_profile_context = (
        f"Role: {data.get('role', 'Fresh Graduate/Student')}\n" 
        f"Summary: {data.get('summary', '')}\n"
        f"Key Strengths: {', '.join(data.get('key_strengths', []))}"
    )

    # 4. Generate Button
    with col_btn:
        st.write("") # Spacer
        st.write("") # Spacer
        # ... inside render_dashboard ...
        if st.button("✨ Rewrite It", type="primary"):
            if not target_text:
                st.warning("Please select or type an answer to rewrite.")
            else:
                with st.spinner("🔍 Analyzing transcript & drafting fix..."):
                    payload = {
                        "job_id": job_id,           # <--- Updated Payload
                        "gap_text": target_text, 
                        "profile_context": candidate_profile_context 
                    }
                    try:
                        res = requests.post(f"{API_URL}/rewrite", json=payload)
                        
                        if res.status_code == 200:
                            new_entry = {
                                "original": target_text,
                                "rewritten": res.json()["rewritten_answer"],
                                "timestamp": time.strftime("%H:%M")
                            }
                            st.session_state.rewrite_history.insert(0, new_entry)
                            st.rerun()
                        else:
                            st.error(f"Error: {res.text}")
                    except Exception as e:
                        st.error(f"Connection Error: {e}")

                        
    # 5. Display History (So previous answers don't vanish)
    if st.session_state.rewrite_history:
        st.markdown("### 📝 Rewritten Versions")
        for item in st.session_state.rewrite_history:
            with st.expander(f"Rewritten: {item['original'][:50]}...", expanded=True):
                st.markdown(f"**💡 Better Version:**")
                st.info(item['rewritten'])
                st.caption(f"Generated at {item['timestamp']}")
            
    # --- CHATBOT ---
    st.divider()
    st.subheader("💬 Ask Your Coach")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("E.g., 'How do I explain my gap year?'"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                res = requests.post(f"{API_URL}/chat", json={"job_id": job_id, "message": prompt})
                if res.status_code == 200:
                    ans = res.json()["response"]
                    st.write(ans)
                    st.session_state.messages.append({"role": "assistant", "content": ans})

def main():
    render_header()
    upload_section()
    render_dashboard()

if __name__ == "__main__":
    main()