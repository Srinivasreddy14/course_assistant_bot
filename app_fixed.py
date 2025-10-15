import os
import base64
import json
from datetime import datetime
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv

# Optional features (voice input / TTS / translation)
try:
    import speech_recognition as sr  # type: ignore
except Exception:  # pragma: no cover
    sr = None  # type: ignore

try:
    from gtts import gTTS  # type: ignore
except Exception:  # pragma: no cover
    gTTS = None  # type: ignore

try:
    from deep_translator import GoogleTranslator  # type: ignore
except Exception:  # pragma: no cover
    GoogleTranslator = None  # type: ignore

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ===== Hero Section =====
st.markdown("""
<div style="display: flex; align-items: center; justify-content: center; gap: 0.5rem; margin: 0; padding: 0;">
  <span style="font-size: 3.2rem; padding: 0;">🎓</span>
  <h1 style="font-size: 3.2rem; font-weight: 800; margin: 0; padding: 10; line-height: 1;
             background: linear-gradient(135deg, #6366f1, #ec4899, #06b6d4);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
    NareshIT Course Assistant
  </h1>
</div>
""", unsafe_allow_html=True)

# ===== Registration Button in Main Area =====
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    registration_url = "https://docs.google.com/forms/d/e/1FAIpQLSdGnEC1sAMZ-ILOIArtd6-7C1yT-clPYx2g9gpaE8lgEq-WvQ/viewform?usp=header"
    st.markdown(
        f"""
        <div style="width:100%">
          <a href="{registration_url}" target="_blank" rel="noopener noreferrer"
             style="display:inline-block; width:100%; text-align:center; text-decoration:none; 
                    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
                    color:#fff; border:none; border-radius:12px; padding:12px 16px; font-weight:600;
                    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.3);">
            📋 Register for Courses Now
          </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ===== Styles =====
# (omitted here for brevity in this file - original styles can be pasted back if desired)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    /* ... full CSS omitted in this snippet - keep your original CSS here ... */
    </style>
    """, unsafe_allow_html=True)


# ===== Env / Setup =====
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
contact_number = os.getenv("CONTACT_PHONE", "+91-90000-00000").strip()


# ===== Helpers =====
LANGUAGE_MAP = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Marathi": "mr",
    "Punjabi": "pa",
    "Urdu": "ur",
}


@st.cache_resource(show_spinner=False)
def get_embeddings():
    # FastEmbed is a lightweight, CPU-friendly embedding backend that avoids heavy torch deps
    return FastEmbedEmbeddings()


@st.cache_data(show_spinner=False)
def load_and_split_from_url(url: str) -> List[Any]:
    loader = WebBaseLoader(url)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    return splitter.split_documents(docs)


@st.cache_resource(show_spinner=False)
def build_vectordb_for_url(url: str, persist_dir: str = "") -> FAISS:
    texts = load_and_split_from_url(url)
    embeddings = get_embeddings()
    # Use the correct keyword 'embeddings' accepted by LangChain's FAISS helper
    return FAISS.from_documents(texts, embeddings=embeddings)


def tts_to_audio_tag(text: str, lang_code: str) -> str:
    if not gTTS:
        return ""
    try:
        tts = gTTS(text, lang=lang_code)
        tmp_path = "response.mp3"
        tts.save(tmp_path)
        with open(tmp_path, "rb") as f:
            b64_audio = base64.b64encode(f.read()).decode()
        return f"""
        <audio controls>
            <source src="data:audio/mp3;base64,{b64_audio}" type="audio/mp3" />
        </audio>
        """
    except Exception:
        return ""


def maybe_translate(text: str, target_lang_code: str) -> str:
    if not GoogleTranslator or target_lang_code == "en":
        return text
    try:
        return GoogleTranslator(source="en", target=target_lang_code).translate(text)
    except Exception:
        return text


# ===== Sidebar =====
with st.sidebar:
    st.markdown('<div class="sidebar-card">🌐 Language</div>', unsafe_allow_html=True)
    target_language_name = st.selectbox("Response Language", list(LANGUAGE_MAP.keys()), index=0)
    target_lang_code = LANGUAGE_MAP[target_language_name]
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-card">🎛️ I/O Options</div>', unsafe_allow_html=True)
    enable_voice = st.checkbox("Microphone input", value=False, disabled=(sr is None))
    enable_tts = st.checkbox("Text-to-speech", value=False, disabled=(gTTS is None))
    st.markdown('</div>', unsafe_allow_html=True)

    # Export & Share
    st.markdown('<div class="sidebar-card">🔗 Export & Share</div>', unsafe_allow_html=True)
    has_msgs = bool(st.session_state.get("messages"))
    try:
        history_json_str = json.dumps(st.session_state.get("messages", []), ensure_ascii=False, indent=2)
    except Exception:
        history_json_str = "[]"

    lines = ["# Conversation Transcript\n"]
    for i, msg in enumerate(st.session_state.get("messages", []), start=1):
        who = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "").replace("\r", "")
        lines.append(f"## {i}. {who}\n\n{content}\n")
    transcript_md = "\n".join(lines)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.download_button(
            label="⬇️ JSON",
            data=history_json_str.encode("utf-8"),
            file_name=f"chat_history_{timestamp}.json",
            mime="application/json",
            use_container_width=True,
            disabled=not has_msgs,
        )
    with col_e2:
        st.download_button(
            label="⬇️ Markdown",
            data=transcript_md.encode("utf-8"),
            file_name=f"chat_history_{timestamp}.md",
            mime="text/markdown",
            use_container_width=True,
            disabled=not has_msgs,
        )
    if not has_msgs:
        st.markdown('<div class="muted">No messages yet. Start a chat to enable export.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Registration Form Button
    st.markdown('<div class="sidebar-card">📝 Course Registration</div>', unsafe_allow_html=True)
    registration_url = "https://docs.google.com/forms/d/e/1FAIpQLSdGnEC1sAMZ-ILOIArtd6-7C1yT-clPYx2g9gpaE8lgEq-WvQ/viewform?usp=header"
    st.markdown(
        f"""
        <div style="width:100%">
          <a href="{registration_url}" target="_blank" rel="noopener noreferrer"
             style="display:inline-block; width:100%; text-align:center; text-decoration:none; 
                    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
                    color:#fff; border:none; border-radius:12px; padding:12px 16px; font-weight:600;
                    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.3);">
            📋 Register for Courses
          </a>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="muted">Click to register for any of our courses</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ===== Session State =====
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # list of dicts {role: "user"|"assistant", content: str}

if "retriever_ready" not in st.session_state:
    st.session_state["retriever_ready"] = False

# ===== Actions =====
action_clear, action_adv = st.columns([1,1])

with action_clear:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state["messages"].clear()
        st.toast("💬 Chat history cleared")
    st.markdown('</div>', unsafe_allow_html=True)

with action_adv:
    course_options: Dict[str, str] = {
        "Select Course":"",
        "Full Stack Python Online Training": "https://nareshit.com/courses/full-stack-python-online-training",
        "Full Stack Data Science & AI": "https://nareshit.com/courses/full-stack-data-science-ai-online-training",
        "Full Stack Software Testing" : "https://nareshit.com/courses/full-stack-software-testing-online-training",
        "UI Full Stack Web Development With React":"https://nareshit.com/courses/ui-full-stack-web-development-with-react-online-training",
        "Full Stack Dot Net Core":"https://nareshit.com/courses/full-stack-dot-net-core-online-training",
        "Full Stack Python":"https://nareshit.com/courses/full-stack-python-online-training",
        "Full Stack Java":"https://nareshit.com/courses/full-stack-java-online-training",
        "Spring Boot MicroServices":"https://nareshit.com/courses/spring-boot-microservices-online-training",
        "Django":"https://nareshit.com/courses/django-online-training",
        "Tableau":"https://nareshit.com/courses/tableau-online-training",
        "Power BI":"https://nareshit.com/courses/power-bi-online-training",
        "MySQL":"https://nareshit.com/courses/mysql-online-training"
    }

    def _on_course_change():
        name = st.session_state.get("selected_course_name")
        url = course_options.get(name, "")
        st.session_state["active_url"] = url
        if not url:
            st.session_state["retriever_ready"] = False
            return
        with st.spinner("🤖 Processing course content with AI..."):
            try:
                persist_dir = os.path.join("./vectordb", "web")
                vectordb = build_vectordb_for_url(url, persist_dir=persist_dir)
                st.session_state["vectordb"] = vectordb
                st.session_state["retriever_ready"] = True
                st.toast("✅ Course content loaded and indexed successfully!")
            except Exception as err:
                st.session_state["retriever_ready"] = False
                st.error(f"❌ Failed to process course content: {err}")

    selected_course_name = st.selectbox("", list(course_options.keys()), key="selected_course_name", on_change=_on_course_change)
    selected_course_url = course_options[selected_course_name]
    active_url = st.session_state.get("active_url", selected_course_url)
    if not st.session_state.get("retriever_ready"):
        _on_course_change()
    st.markdown('</div>', unsafe_allow_html=True)


# ===== Chat =====
chat_tab, history_tab = st.tabs(["💬 Chat", "📜 History"])

with chat_tab:
    st.markdown('<div class="chat-container"> 💬 AI Course Assistant</div>', unsafe_allow_html=True)
    st.markdown("Ask intelligent questions about your course content")

    # Show message history
    for msg in st.session_state["messages"]:
        css_cls = "user-msg" if msg["role"] == "user" else "bot-msg"
        st.markdown(f"<div class='stChatMessage {css_cls}'>{msg['content']}</div>", unsafe_allow_html=True)

    # Process any pending query first (so answer appears above input)
    pending_query = st.session_state.get("pending_query")
    if pending_query:
        # Display user query immediately
        st.markdown(f"<div class='stChatMessage user-msg'>{pending_query}</div>", unsafe_allow_html=True)

        if not st.session_state.get("retriever_ready"):
            st.info("⏳ Loading course content automatically. Please wait a moment and try again.")
        elif not groq_api_key:
            st.stop()
        else:
            # Prepare RAG chain
            embeddings = get_embeddings()
            vectordb = st.session_state.get("vectordb")
            retriever = vectordb.as_retriever(search_kwargs={"k": 5}) if vectordb is not None else None

            prompt_template = (
                "You are a helpful course assistant. Answer using only the provided context.\n"
                "If unsure, say 'I am not certain; please check the course page.'\n\n"
                "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:" \
                "\n\nInstructions: Write a friendly, one-sentence popup message reminding users to save their work. "
                "Auto-generate suggested search text related to the user's question to reduce their time. "
                "Include brief helpful hints based on the loaded data while searching."
            )
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

            llm = ChatGroq(api_key=groq_api_key, model="llama-3.3-70b-versatile", temperature=0.4, max_tokens=512)

            qa = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff",
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt},
            )

            # Add user message to history
            st.session_state["messages"].append({"role": "user", "content": pending_query})

            with st.spinner("Thinking..."):
                try:
                    # Use the standard call interface for LangChain chains
                    result = qa({"query": pending_query})

                    # result shape can vary across LangChain versions; be robust
                    if isinstance(result, dict):
                        answer = result.get("result") or result.get("output_text") or result.get("answer") or ""
                        sources = result.get("source_documents") or result.get("source_documents", [])
                    else:
                        # If chain returns a string, treat it as the answer
                        answer = str(result)
                        sources = []
                except Exception as run_err:
                    answer = f"There was an error answering the question: {run_err}"
                    sources = []

            # Translate if needed
            final_answer = maybe_translate(answer, target_lang_code)
            st.session_state["messages"].append({"role": "assistant", "content": final_answer})

            # Render AI response
            st.markdown(f"<div class='stChatMessage bot-msg'>{final_answer}</div>", unsafe_allow_html=True)

            # TTS
            if enable_tts and final_answer:
                audio_tag = tts_to_audio_tag(final_answer, target_lang_code)
                if audio_tag:
                    st.markdown(audio_tag, unsafe_allow_html=True)

        # Clear pending query
        st.session_state["pending_query"] = None

    if not groq_api_key:
        st.warning("⚠️ GROQ_API_KEY is missing. Add it to your environment to use the AI assistant.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Input row (moved to bottom)
    with st.form(key="chat_form", clear_on_submit=True):
        user_query = st.text_input("💭 Ask about the course content", placeholder="e.g., What are the prerequisites for this course?")
        col1, col2 = st.columns([1, 4])
        with col1:
            submitted = st.form_submit_button("🚀 Send", type="primary", use_container_width=True)
        with col2:
            if enable_voice and sr is not None:
                # Separate submit button that specifically triggers voice capture
                voice_clicked = st.form_submit_button("🎤 Voice", use_container_width=True)
                if voice_clicked:
                    recognizer = sr.Recognizer()
                    try:
                        with st.spinner("🎙️ Listening..."):
                            with sr.Microphone() as source:
                                recognizer.adjust_for_ambient_noise(source, duration=1)
                                audio = recognizer.listen(source, timeout=5)
                            voice_text = recognizer.recognize_google(audio)
                            st.info(f"🎤 Captured: {voice_text}")
                            # set captured text into pending query so it processes immediately
                            st.session_state["pending_query"] = voice_text
                            st.rerun()
                    except Exception as mic_err:
                        st.error(f"❌ Microphone error: {mic_err}")

    if submitted and user_query:
        st.session_state["pending_query"] = user_query
        st.rerun()


with history_tab:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.markdown("### 📜 Conversation History")
    st.markdown("Review your previous interactions with the AI assistant")

    if st.session_state["messages"]:
        for i, msg in enumerate(st.session_state["messages"], start=1):
            role = "👤 User" if msg["role"] == "user" else "🤖 Assistant"
            css_cls = "user-msg" if msg["role"] == "user" else "bot-msg"
            st.markdown(f"<div class='stChatMessage {css_cls}'><strong>{i}. {role}:</strong> {msg['content']}</div>", unsafe_allow_html=True)

    else:
        st.info("💬 No conversation yet. Start chatting in the Chat tab!")
    st.markdown('</div>', unsafe_allow_html=True)

# Floating contact number and registration button
st.markdown(f'<div class="floating-contact">📞 {contact_number}</div>', unsafe_allow_html=True)

# Floating registration button
registration_url = "https://docs.google.com/forms/d/e/1FAIpQLSfSotUmF_68mx0tMkngYfaGsyYVf0L9jL5YyZG7DA5iLXh-bA/viewform?usp=header"
st.markdown(f'''
<div style="position: fixed; left: 16px; bottom: 16px; z-index: 9999;">
  <a href="{registration_url}" target="_blank" rel="noopener noreferrer"
     style="background: linear-gradient(135deg, #10b981, #059669);
            color: #fff;
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 999px;
            padding: 12px 18px;
            box-shadow: 0 8px 32px rgba(16, 185, 129, 0.3);
            font-weight: 700;
            font-size: 0.9rem;
            cursor: pointer;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            gap: 8px; text-decoration:none;">
    📋 Register Now
  </a>
</div>
''', unsafe_allow_html=True)
