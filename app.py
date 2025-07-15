import streamlit as st
import requests

st.set_page_config(page_title="AI Exam Revision Agent", page_icon="📚")

st.title("📚 AI Revision Agent for School-Level Exams")
st.write("Helping you revise smarter with summaries, quizzes, and interactive tests!")

# ---- 1️⃣ Gather User Input ----
subject = st.text_input("Enter your subject (e.g., Science, Math, History):")
topic = st.text_input("Enter your specific topic:")
syllabus = st.text_area("Briefly describe the syllabus (chapters, scope):")

# ---- 2️⃣ Offer Task Menu ----
options = [
    "Summary Notes",
    "Quiz (MCQ/True-False)",
    "Fill in the Blanks",
    "Match the Columns",
    "Complete the Code"
]
choice = st.selectbox("What would you like to do?", options)

# ---- 3️⃣ Hugging Face API Setup ----
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-small"
headers = {"Authorization": f"Bearer {st.secrets['HUGGINGFACE_API_KEY']}"}

def query_huggingface(payload):
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()[0]['generated_text']
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

# ---- 4️⃣ Build Prompt ----
if st.button("Generate"):
    if not (subject and topic and syllabus):
        st.warning("Please fill in all the details first.")
    else:
        prompt = (
            f"You are a helpful AI teacher assisting a school-level student in exam preparation.\n\n"
            f"Subject: {subject}\n"
            f"Topic: {topic}\n"
            f"Syllabus: {syllabus}\n"
            f"Task: {choice}\n\n"
            "Provide clear, structured, easy-to-understand output suitable for school students. "
            "Make it engaging, educational, and concise."
        )

        with st.spinner("Generating..."):
            output = query_huggingface({"inputs": prompt})

        if output:
            st.success("Here’s your result:")
            st.markdown(output)

# ---- 5️⃣ Follow-up & Motivation ----
st.write("---")
st.write("🎉 Need more help? Just ask again!")
st.caption("🚀 Hack Tip: Study small, test often. You’ve got this! 💪 Stay confident.")
