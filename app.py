import streamlit as st
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Hugging Face Token Login
login(st.secrets["HF_TOKEN"])

# Load Phi-3 Model
model_id = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

revision_agent = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    pad_token_id=tokenizer.eos_token_id,
    device_map="auto",
    max_length=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

# Streamlit UI
st.title("📚 AI Revision Assistant (Up to School Level)")
st.write("Hi! I'm your AI Study Buddy. How can I help you today with your school studies?")

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

Provide clear markdown-formatted outputs (headings, bullet points, etc.). 
End with a study tip and motivational message.
"""

if st.button("Generate Revision Material"):
    with st.spinner("Generating content..."):
        response = revision_agent(
            instruction_template,
            max_new_tokens=300,
            pad_token_id=tokenizer.eos_token_id
        )
        output = response[0]['generated_text'].split("AI:")[-1].strip()

    st.markdown(output)
    st.session_state.history.append({
        "Subject": subject,
        "Topic": topic,
        "Syllabus": syllabus,
        "Option": option,
        "Output": output
    })

if st.checkbox("Show My Revision History"):
    for idx, item in enumerate(st.session_state.history, start=1):
        st.markdown(f"**{idx}. Subject:** {item['Subject']} | **Topic:** {item['Topic']} | **Option:** {item['Option']}")
        st.markdown(item["Output"])
        st.markdown("---")
