import streamlit as st
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import requests

# Hugging Face Inference API Endpoint (No need to load model locally)
API_URL = "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct"
headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# Streamlit App UI
st.title("📚 AI Revision Assistant (Up to School Level)")
st.write("Hi! I'm your AI Study Buddy. I can help with any subject, topic, or syllabus up to school level. Let's start!")

if "history" not in st.session_state:
    st.session_state.history = []

subject = st.text_input("What subject are you studying?")
topic = st.text_input("What specific topic within this subject?")
syllabus = st.text_input("Which syllabus or board? (CBSE, ICSE, State Board, Cambridge, etc.)")

option = st.radio(
    "What would you like help with?",
    ["Summary (Bullet Points)", "Quiz (MCQ / True-False)", "Interactive Test (Fill Blanks, Match, Arrange Steps)",
     "Flashcards", "Code Completion / Output Prediction (if coding-related)"]
)

instruction_template = f"""
You are a friendly AI Revision Agent helping students up to school level.
Subject: {subject}
Topic: {topic}
Syllabus: {syllabus}

Option selected: {option}

Provide clear markdown-formatted outputs (headings, bullet points, numbered lists where appropriate).
Make the content detailed enough for meaningful revision.
If the output is long, ask: 'Would you like me to continue this in smaller parts? Reply "Continue".'

At the end of the output, add:
💡 A study tip (Pomodoro, Active Recall, etc.)
👏 A motivational reward ('You've earned ⭐️⭐️⭐️ for today!')

Always offer follow-up: 'Would you like more questions, bullet points, or another option?'
"""

if st.button("Generate Revision Material"):
    with st.spinner("Generating your revision material..."):
        output = query({"inputs": instruction_template})
        if "error" in output:
            st.error(output["error"])
        else:
            st.markdown(output[0]["generated_text"])
            st.session_state.history.append({
                "Subject": subject,
                "Topic": topic,
                "Syllabus": syllabus,
                "Option": option,
                "Output": output[0]["generated_text"]
            })

if st.checkbox("Show My Revision History"):
    for idx, item in enumerate(st.session_state.history, start=1):
        st.markdown(f"**{idx}. Subject:** {item['Subject']} | **Topic:** {item['Topic']} | **Option:** {item['Option']}")
        st.markdown(item["Output"])
        st.markdown("---")
