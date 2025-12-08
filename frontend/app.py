# import streamlit as st
# import requests
# import time

# # Configuration
# API_URL = "http://localhost:8000"

# st.set_page_config(page_title="InterviewAlly 2.0", page_icon="🧠")

# def render_header():
#     st.title("🧠 InterviewAlly 2.0")
#     st.markdown("AI-Powered Interview Analysis with **RAG & Structured Reasoning**")

# def upload_video():
#     st.sidebar.header("1. Upload Interview")
#     uploaded_file = st.sidebar.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])
    
#     if uploaded_file is not None:
#         if st.sidebar.button("Process Video"):
#             with st.spinner("Uploading to Backend..."):
#                 files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
#                 try:
#                     response = requests.post(f"{API_URL}/upload-video", files=files)
#                     if response.status_code == 200:
#                         job_id = response.json()["job_id"]
#                         st.session_state["job_id"] = job_id
#                         st.session_state["status"] = "processing"
#                         st.sidebar.success(f"Upload Complete! Job ID: {job_id}")
#                     else:
#                         st.sidebar.error("Upload failed.")
#                 except Exception as e:
#                     st.sidebar.error(f"Connection Error: {e}")

# def poll_status():
#     if "job_id" in st.session_state and st.session_state.get("status") != "completed":
#         job_id = st.session_state["job_id"]
#         status_placeholder = st.empty()
        
#         while True:
#             try:
#                 response = requests.get(f"{API_URL}/get-status/{job_id}")
#                 data = response.json()
#                 status = data.get("status")
                
#                 if status == "completed":
#                     st.session_state["status"] = "completed"
#                     status_placeholder.success("✅ Processing & Indexing Complete!")
#                     st.rerun()
#                     break
#                 elif status == "failed":
#                     status_placeholder.error(f"❌ Processing Failed: {data.get('error')}")
#                     break
#                 else:
#                     status_placeholder.info(f"⚙️ Backend is processing... Status: {status}")
#                     time.sleep(2)
#             except:
#                 break

# def render_analysis():
#     # Only show if processing is done
#     if st.session_state.get("status") == "completed":
#         st.divider()
#         st.header("2. AI Analysis")
        
#         job_id = st.session_state["job_id"]
        
#         # Button to trigger analysis
#         if st.button("Generate Detailed Report"):
#             with st.spinner("🧠 The Brain is analyzing..."):
#                 try:
#                     # Call Backend
#                     response = requests.post(f"{API_URL}/analyze/{job_id}")
                    
#                     if response.status_code == 200:
#                         result = response.json()
#                         # CRITICAL FIX: Check if result is valid before storing
#                         if result:
#                             st.session_state["analysis"] = result
#                             st.rerun()
#                         else:
#                             st.error("⚠️ Analysis returned empty. Check backend terminal for errors.")
#                     else:
#                         st.error(f"Analysis failed. Status Code: {response.status_code}")
#                 except Exception as e:
#                     st.error(f"Connection Error: {e}")
        
#         # Display Results (Only if data exists)
#         if "analysis" in st.session_state and st.session_state["analysis"]:
#             data = st.session_state["analysis"]
            
#             # Use .get() to prevent crashes if a field is missing
#             col1, col2, col3 = st.columns(3)
#             col1.metric("Technical", f"{data.get('technical_score', 0)}/10")
#             col2.metric("Communication", f"{data.get('communication_score', 0)}/10")
#             col3.metric("Cultural Fit", f"{data.get('cultural_fit_score', 0)}/10")
            
#             st.info(f"**Summary:** {data.get('summary', 'N/A')}")
            
#             c1, c2 = st.columns(2)
#             with c1:
#                 st.subheader("✅ Key Strengths")
#                 for item in data.get('key_strengths', []):
#                     st.success(f"- {item}")
            
#             with c2:
#                 st.subheader("🚩 Critical Gaps")
#                 for item in data.get('critical_gaps', []):
#                     st.error(f"- {item}")

#             with st.expander("⏱️ Moments of Interest"):
#                 for ts in data.get('timestamps_of_interest', []):
#                     st.write(f"• **{ts}**")

# def render_chat():
#     if st.session_state.get("status") == "completed":
#         st.divider()
#         st.header("3. Chat with the Candidate (RAG)")
        
#         # Chat history
#         if "messages" not in st.session_state:
#             st.session_state.messages = []

#         for message in st.session_state.messages:
#             with st.chat_message(message["role"]):
#                 st.markdown(message["content"])

#         if prompt := st.chat_input("Ask about specific skills, answers, or red flags..."):
#             st.session_state.messages.append({"role": "user", "content": prompt})
#             with st.chat_message("user"):
#                 st.markdown(prompt)

#             with st.chat_message("assistant"):
#                 with st.spinner("Searching transcript..."):
#                     payload = {"job_id": st.session_state["job_id"], "message": prompt}
#                     response = requests.post(f"{API_URL}/chat", json=payload)
                    
#                     if response.status_code == 200:
#                         answer = response.json()["response"]
#                         st.markdown(answer)
#                         st.session_state.messages.append({"role": "assistant", "content": answer})
#                     else:
#                         st.error("Error getting response")

# def main():
#     render_header()
#     upload_video()
#     poll_status()
#     render_analysis()
#     render_chat()

# if __name__ == "__main__":
#     main()

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