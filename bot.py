import os
import base64
import json
from datetime import datetime
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv

# Import necessary Google/Gemini components from LangChain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Optional features (voice input / TTS / translation)
# Note: Streamlit microphone input is often tricky in web deployments.
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


# ===== Env / Setup =====
load_dotenv()
# Update to use GEMINI_API_KEY
gemini_api_key = os.getenv("GOOGLE_API_KEY", "").strip()
# Use the displayed number as the fallback for robustness
contact_number = os.getenv("CONTACT_PHONE", "+91 8179191999").strip()


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

# ===== Registration Button in Main Area (Stays) =====
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    registration_url = "https://docs.google.com/forms/d/e/1FAIpQLSfSotUmF_68mx0tMkngYfaGsyYVf0L9jL5YyZG7DA5iLXh-bA/viewform?usp=header"
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

# ===== Styles (Kept as is - they are great!) =====
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    :root {
      --primary: #6366f1; /* indigo-500 */
      --primary-dark: #4f46e5; /* indigo-600 */
      --primary-light: #a5b4fc; /* indigo-300 */
      --secondary: #ec4899; /* pink-500 */
      --accent: #06b6d4; /* cyan-500 */
      --success: #10b981; /* emerald-500 */
      --warning: #f59e0b; /* amber-500 */
      --danger: #ef4444; /* red-500 */
      
      --bg-primary: #0a0a0a; /* near black */
      --bg-secondary: #111111; /* dark gray */
      --bg-tertiary: #1a1a1a; /* lighter dark */
      --bg-card: #1e1e1e; /* card background */
      --bg-glass: rgba(30, 30, 30, 0.8); /* glass effect */
      
      --text-primary: #ffffff;
      --text-secondary: #a1a1aa; /* zinc-400 */
      --text-muted: #71717a; /* zinc-500 */
      --text-accent: #e4e4e7; /* zinc-200 */
      
      --border: #27272a; /* zinc-800 */
      --border-light: #3f3f46; /* zinc-700 */
      --shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
      --shadow-lg: 0 35px 60px -12px rgba(0, 0, 0, 0.6);
    }

    * { box-sizing: border-box; }
    
    .stApp { 
      background: linear-gradient(135deg, var(--bg-primary) 0%, #0f0f23 50%, var(--bg-primary) 100%);
      color: var(--text-primary);
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      min-height: 100vh;
    }

    /* Animated Background */
    .stApp::before {
      content: '';
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: 
        radial-gradient(circle at 20% 80%, rgba(99, 102, 241, 0.1) 0%, transparent 50%),
        radial-gradient(circle at 80% 20%, rgba(236, 72, 153, 0.1) 0%, transparent 50%),
        radial-gradient(circle at 40% 40%, rgba(6, 182, 212, 0.1) 0%, transparent 50%);
      z-index: -1;
      animation: float 20s ease-in-out infinite;
    }

    @keyframes float {
      0%, 100% { transform: translateY(0px) rotate(0deg); }
      50% { transform: translateY(-20px) rotate(180deg); }
    }


    /* Premium Cards */
    .card { 
      background: var(--bg-card); 
      border: 1px solid var(--border); 
      border-radius: 20px; 
      padding: 24px; 
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
      transition: all 0.3s ease;
    }
    
    .card:hover {
      transform: translateY(-2px);
      box-shadow: var(--shadow-lg);
      border-color: var(--border-light);
    }
    
    .card-tonal { 
      background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-tertiary) 100%);
      border: 1px solid var(--border); 
      border-radius: 20px; 
      padding: 28px; 
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }

    /* Modern Chat Interface */
    .chat-container {
      font-size: 1.2rem;
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 24px;
      padding: 24px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
      position: relative;
      overflow: hidden;
    }

    .course-container {
      font-size: 1.2rem;
      background: #8D8D9E;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 0;
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
      position: relative;
      text-align: center;
      overflow: hidden;
    }
    
    .chat-container::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 1px;
      background: linear-gradient(90deg, transparent, var(--primary), transparent);
    }

    .stChatMessage { 
      border-radius: 20px; 
      padding: 16px 20px; 
      margin: 12px 0; 
      max-width: 85%;
      backdrop-filter: blur(10px);
      transition: all 0.3s ease;
    }
    
    .stChatMessage:hover {
      transform: translateX(4px);
    }
    
    .user-msg { 
      background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
      border: 1px solid rgba(99, 102, 241, 0.3);
      color: white; 
      margin-left: auto;
      box-shadow: 0 8px 32px rgba(99, 102, 241, 0.2);
    }
    
    .bot-msg { 
      background: linear-gradient(135deg, var(--bg-tertiary) 0%, var(--bg-card) 100%);
      border: 1px solid var(--border);
      color: var(--text-primary);
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] { 
      background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
      border-right: 1px solid var(--border); 
      color: var(--text-primary);
    }
    .sidebar-banner {
      background: radial-gradient(1200px 200px at 0% -10%, rgba(99,102,241,0.25), transparent),
                   radial-gradient(1000px 200px at 100% -20%, rgba(236,72,153,0.25), transparent),
                   linear-gradient(135deg, rgba(30,30,30,0.9), rgba(20,20,30,0.9));
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 16px;
      margin: 0 0 16px 0;
      box-shadow: var(--shadow);
      position: relative;
      overflow: hidden;
    }
    .sidebar-banner::after {
      content: "";
      position: absolute;
      inset: 0;
      background: linear-gradient(90deg, transparent, rgba(99,102,241,0.15), transparent);
      height: 1px;
      top: 0;
    }
    .sidebar-header {
      display: flex;
      align-items: center;
      gap: 10px;
      margin: 0 0 16px 0;
    }
    .sidebar-header h2 {
      font-size: 1.4rem;
      font-weight: 800;
      margin: 0;
      background: linear-gradient(135deg, var(--primary), var(--secondary), var(--accent));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      letter-spacing: 0.3px;
    }
    .sidebar-icon {
      width: 36px;
      height: 36px;
      border-radius: 10px;
      display: grid;
      place-items: center;
      background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(6,182,212,0.15));
      border: 1px solid var(--border);
    }
    .sidebar-subtext {
      font-size: 0.9rem;
      color: var(--text-secondary);
      margin: -4px 0 12px 0;
    }
    .sidebar-card {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 12px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
      margin-bottom: 14px;
    }
    .sidebar-card:hover {
      border-color: var(--border-light);
      transform: translateY(-1px);
      transition: all 0.2s ease;
    }
    
    .section-title { 
      font-weight: 700; 
      font-size: 1.125rem;
      color: var(--text-primary); 
      margin: 0 0 1rem 0;
      padding-bottom: 0.5rem;
      border-bottom: 2px solid var(--primary);
    }
    .section-subtitle {
      font-weight: 700;
      font-size: 0.95rem;
      color: var(--text-accent);
      margin: 0 0 10px 0;
    }
    .divider { height: 1px; background: var(--border); margin: 10px 0; }
    .muted { color: var(--text-muted); font-size: 0.9rem; }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: var(--bg-tertiary);
      font-size: 0.8rem;
      color: var(--text-accent);
    }
    .status-pill { padding: 4px 10px; border-radius: 999px; font-size: 0.75rem; border: 1px solid var(--border); }
    .status-ok { background: rgba(16,185,129,0.12); color: #10b981; border-color: rgba(16,185,129,0.35); }
    .status-warn { background: rgba(245,158,11,0.12); color: #f59e0b; border-color: rgba(245,158,11,0.35); }
    .status-err { background: rgba(239,68,68,0.12); color: #ef4444; border-color: rgba(239,68,68,0.35); }
    

    /* Buttons */
    .stButton {
      margin: 0;
      padding: 0;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    
    .stButton > button {
      background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
      color: white;
      border: none;
      border-radius: 12px;
      padding: 0.75rem 1.5rem;
      font-weight: 600;
      transition: all 0.3s ease;
      box-shadow: 0 4px 16px rgba(99, 102, 241, 0.3);
      min-height: 48px;
      display: flex;
      align-items: center;
      justify-content: center;
      margin: 0;
      width: 100%;
    }
    
    .stButton > button:hover {
      transform: translateY(-2px);
      box-shadow: 0 8px 24px rgba(99, 102, 241, 0.4);
    }

    /* Action columns alignment - ensure both elements are at same level */
    .stColumns {
      display: flex;
      align-items: stretch;
    }
    
    .stColumns > div {
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 0;
      margin: 0;
    }
    
    /* Specific alignment for action buttons */
    .stColumns > div:first-child {
      display: flex;
      align-items: center;
      justify-content: center;
    }
    
    .stColumns > div:last-child {
      display: flex;
      align-items: center;
      justify-content: center;
    }
    
    /* Form Elements */
    .stTextInput > div > div > input {
      background: var(--bg-tertiary);
      border: 1px solid var(--border);
      border-radius: 12px;
      color: var(--text-primary);
      padding: 0.75rem 1rem;
      font-size: 1rem;
    }
    
    .stTextInput > div > div > input:focus {
      border-color: var(--primary);
      box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
    }

    /* Selectbox styling to match clear chat button exactly */
    .stSelectbox {
      margin: 0;
      padding: 0;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    
    .stSelectbox > div {
      margin: 0;
      padding: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      width: 100%;
    }
    
    .stSelectbox > div > div {
      background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
      border: none;
      border-radius: 12px;
      color: white;
      font-weight: 600;
      box-shadow: 0 4px 16px rgba(99, 102, 241, 0.3);
      transition: all 0.3s ease;
      padding: 0.75rem 1.5rem;
      min-height: 48px;
      display: flex;
      align-items: center;
      justify-content: center;
      margin: 0;
      width: 100%;
    }
    
    .stSelectbox > div > div:hover {
      transform: translateY(-2px);
      box-shadow: 0 8px 24px rgba(99, 102, 241, 0.4);
    }
    
    .stSelectbox > div > div > div {
      color: white;
      padding: 0;
      display: flex;
      align-items: center;
      height: 100%;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
      gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
      background: var(--bg-tertiary);
      border: 1px solid var(--border);
      border-radius: 12px;
      color: var(--text-secondary);
      font-weight: 600;
      padding: 0.75rem 1.5rem;
    }
    
    .stTabs [aria-selected="true"] {
      background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
      color: white;
      border-color: var(--primary);
    }

    /* Links */
    a, .stMarkdown a { 
      color: var(--primary-light);
      text-decoration: none;
      transition: color 0.3s ease;
    }
    
    a:hover, .stMarkdown a:hover {
      color: var(--primary);
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
      width: 8px;
    }
    
    ::-webkit-scrollbar-track {
      background: var(--bg-secondary);
    }
    
    ::-webkit-scrollbar-thumb {
      background: var(--primary);
      border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
      background: var(--primary-dark);
    }

    /* Hide default Streamlit elements */
    .stApp > header { display: none; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stDeployButton { display: none; }
    .stStatusWidget { display: none; }
    
    /* Remove default container padding */
    .main .block-container { 
      padding-top: 0 !important; 
      padding-bottom: 0 !important; 
    }
    /* Floating contact */
    .floating-contact {
      position: fixed;
      right: 16px;
      bottom: 16px;
      background: linear-gradient(135deg, var(--primary), var(--primary-dark));
      color: #fff;
      border: 1px solid rgba(255,255,255,0.15);
      border-radius: 999px;
      padding: 10px 14px;
      box-shadow: var(--shadow-lg);
      z-index: 9999;
      font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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
    """
    Initializes and returns the Google Generative AI Embeddings model for RAG.
    """
    # Using GoogleGenerativeAIEmbeddings (default model is powerful and fast)
    return GoogleGenerativeAIEmbeddings(model="text-embedding-004")


@st.cache_data(show_spinner=False)
def load_and_split_from_url(url: str) -> List[Any]:
    """Loads and splits documents from a given URL."""
    loader = WebBaseLoader(url)
    docs = loader.load()
    # MODIFICATION: Increased chunk_overlap to 200 for better context retention
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    return splitter.split_documents(docs)


@st.cache_resource(show_spinner=False)
def build_vectordb_for_url(url: str) -> FAISS:
    """
    Builds the FAISS vector database from the content of a given URL.
    This function leverages st.cache_resource for extremely fast subsequent loads.
    """
    texts = load_and_split_from_url(url)
    embeddings = get_embeddings()
    # FAISS is used for fast, in-memory vector indexing (meets client requirement)
    return FAISS.from_documents(texts, embedding=embeddings)


def tts_to_audio_tag(text: str, lang_code: str) -> str:
    """Converts text to base64 encoded audio tag using gTTS."""
    if not gTTS:
        return ""
    try:
        tts = gTTS(text, lang=lang_code)
        tmp_path = "response.mp3"
        tts.save(tmp_path)
        with open(tmp_path, "rb") as f:
            b64_audio = base64.b64encode(f.read()).decode()
        return f"""
        <audio controls autoplay style="width: 100%;">
            <source src="data:audio/mp3;base64,{b64_audio}" type="audio/mp3" />
            Your browser does not support the audio element.
        </audio>
        """
    except Exception:
        return ""


def maybe_translate(text: str, target_lang_code: str) -> str:
    """Translates the text if the target language is not English."""
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
    
    # Use the history of the currently active course for export
    active_course_key = st.session_state.get("active_course_name", "the selected course")
    history_to_export = st.session_state.get("all_messages", {}).get(active_course_key, [])
    has_msgs = bool(history_to_export)

    try:
        # Include current URL for context in export
        history_data = history_to_export + [
            {"role": "system", "content": f"Context URL: {st.session_state.get('active_url', 'N/A')}"}
        ]
        history_json_str = json.dumps(history_data, ensure_ascii=False, indent=2)
    except Exception:
        history_json_str = "[]"

    lines = [f"# NareshIT Course Assistant Transcript ({datetime.now().strftime('%Y-%m-%d')})\n"]
    for i, msg in enumerate(history_to_export, start=1):
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
        st.markdown(f'<div class="muted">No messages yet for **{active_course_key}**. Start a chat to enable export.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Removed Registration Form Button (Client Requirement 3)
    # st.markdown('<div class="sidebar-card">📝 Course Registration</div>', unsafe_allow_html=True)
    # [Removed button and descriptive markdown]

# ===== Session State =====
if "all_messages" not in st.session_state:
    # MODIFICATION 1: Use a dictionary to store messages per course
    st.session_state["all_messages"] = {} 

if "retriever_ready" not in st.session_state:
    st.session_state["retriever_ready"] = False
    
if "active_url" not in st.session_state:
    st.session_state["active_url"] = ""

if "active_course_name" not in st.session_state:
    st.session_state["active_course_name"] = "Select Course (Click to Load)"


# Get or initialize the currently active message list (points to a specific list inside all_messages)
current_course_key = st.session_state.get("active_course_name", "Select Course (Click to Load)")
if current_course_key not in st.session_state["all_messages"]:
    st.session_state["all_messages"][current_course_key] = []
st.session_state["messages"] = st.session_state["all_messages"][current_course_key]


# ===== Actions (Clear Chat and Course Selection) =====
action_clear, action_adv = st.columns([1,1])

with action_clear:
    # MODIFICATION 1: Clear only the active course's chat history
    def clear_active_chat():
        st.session_state["messages"].clear()
        st.toast(f"💬 Chat history for **{st.session_state['active_course_name']}** cleared")
        
    if st.button("🗑️ Clear Chat", use_container_width=True):
        clear_active_chat()

with action_adv:
    course_options: Dict[str, str] = {
        "Select Course (Click to Load)":"",
        "Full Stack Python Online Training": "https://nareshit.com/courses/full-stack-python-online-training",
        "Full Stack Data Science & AI": "https://nareshit.com/courses/full-stack-data-science-ai-online-training",
        "Full Stack Software Testing" : "https://nareshit.com/courses/full-stack-software-testing-online-training",
        "UI Full Stack Web Development With React":"https://nareshit.com/courses/ui-full-stack-web-development-with-react-online-training",
        "Full Stack Dot Net Core":"https://nareshit.com/courses/full-stack-dot-net-core-online-training",
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
        
        # Switch chat history to the newly selected course name
        st.session_state["active_course_name"] = name
        st.session_state["active_url"] = url
        
        # Re-link the messages list to the new course's history
        if name not in st.session_state["all_messages"]:
            st.session_state["all_messages"][name] = []
        st.session_state["messages"] = st.session_state["all_messages"][name]

        if not url or name == "Select Course (Click to Load)":
            st.session_state["retriever_ready"] = False
            return
        
        # Load the vector DB only if the URL is valid
        st.toast(f"🤖 Loading data for: **{name}**. Please wait...", icon="⏳")

        with st.spinner(f"🧠 Processing content for {name} with AI..."):
            try:
                vectordb = build_vectordb_for_url(url)
                st.session_state["vectordb"] = vectordb
                st.session_state["retriever_ready"] = True
                st.toast(f"✅ AI Assistant ready for **{name}**!", icon="🎉")
            except Exception as err:
                st.session_state["retriever_ready"] = False
                st.error(f"❌ Failed to process course content: {err}")
                st.session_state["active_url"] = ""
                st.session_state["vectordb"] = None
                st.session_state["active_course_name"] = "Select Course (Click to Load)"

    selected_course_name = st.selectbox("", list(course_options.keys()), key="selected_course_name", on_change=_on_course_change)
    
    # Manually trigger load if selected_course_name changes or if it's the first run
    if selected_course_name != "Select Course (Click to Load)" and st.session_state.get("active_course_name") != selected_course_name:
        _on_course_change()
        
    # Display current status
    if st.session_state.get("retriever_ready"):
        st.markdown(f'<div class="status-pill status-ok">Active RAG: {selected_course_name}</div>', unsafe_allow_html=True)
    elif selected_course_name != "Select Course (Click to Load)":
        st.markdown(f'<div class="status-pill status-warn">Loading RAG...</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-pill status-err">No Course Selected</div>', unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)


# ===== Chat Interface =====
chat_tab, history_tab = st.tabs(["💬 Chat", "📜 History"])

with chat_tab:
    st.markdown('<div class="chat-container"> 💬 AI Course Assistant</div>', unsafe_allow_html=True)
    
    active_course_name = st.session_state.get("active_course_name", "None")
    active_url = st.session_state.get('active_url', '')

    if active_course_name and active_course_name != "Select Course (Click to Load)":
        st.markdown(f"**Asking about:** [{active_course_name}]({active_url})")
    else:
        st.info("💡 **Select a course** above to enable the RAG assistant to answer specific questions.")
    
    # Show message history (now automatically correct for the active course)
    for msg in st.session_state["messages"]:
        css_cls = "user-msg" if msg["role"] == "user" else "bot-msg"
        st.markdown(f"<div class='stChatMessage {css_cls}'>{msg['content']}</div>", unsafe_allow_html=True)

    # Process any pending query first (so answer appears above input)
    pending_query = st.session_state.get("pending_query")
    if pending_query:
        # Display user query immediately
        st.markdown(f"<div class='stChatMessage user-msg'>{pending_query}</div>", unsafe_allow_html=True)
        
        if not st.session_state.get("retriever_ready"):
            st.info("⏳ Please wait for the course content to finish loading and indexing.")
        elif not gemini_api_key:
            st.warning("⚠️ GEMINI_API_KEY is missing. Add it to your environment to use the AI assistant.")
            st.stop()
        else:
            # Prepare RAG chain
            vectordb = st.session_state.get("vectordb")
            
            # Use increased retrieval count
            retriever = vectordb.as_retriever(search_kwargs={"k": 8}) if vectordb else None

            # Get the currently selected course name for context injection
            current_course = st.session_state.get("active_course_name", "the selected course")
            
            # MODIFICATION: Inject the course name into the user's query
            # This forces the retriever (vector search) to prioritize the correct course's documents.
            processed_query = f"About the {current_course}: {pending_query}"

            # Prepare the RAG prompt template (LLM still uses this instruction set)
            prompt_template = (
                f"You are a helpful and knowledgeable course assistant for NareshIT, specifically focused on the '{current_course}' course.\n"
                "Your goal is to answer student questions based *only* on the provided course context.\n"
                "If the context does not contain the specific information requested, you must clearly state that the information is unavailable in the document and then proactively provide the contact number for human support: "
                f"\"I couldn't find that specific detail in the course material, but you can always check the course page or call us directly at {contact_number} for the latest batch and prerequisite details.\"\n\n"
                "Context:\n{context}\n\n"
                "Question:\n{question}\n\n"
                "Answer in a clear and friendly tone:"
            )
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

            # Use ChatGoogleGenerativeAI (Gemini)
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash", 
                google_api_key=gemini_api_key, 
                temperature=0.2, 
                max_output_tokens=512
            )

            qa = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff",
                return_source_documents=False, # Set to False to prevent exposing raw chunks
                chain_type_kwargs={"prompt": prompt},
            )

            # Add user message to history
            st.session_state["messages"].append({"role": "user", "content": pending_query})
            
            with st.spinner("Thinking..."):
                try:
                    # Use the processed query for invocation
                    result = qa.invoke({"query": processed_query})
                    answer = result.get("result", "")
                except Exception as run_err:
                    # Use a general exception handler for API/network errors
                    answer = f"There was an error answering the question: {run_err}. Please check your internet connection or API key."

            # Check for empty or faulty answer and provide a robust fallback message
            if not answer or answer.strip() == "":
                answer = (
                    "I am sorry, I seem to be having trouble processing that request right now, or the content did not provide an answer. "
                    "Please try rephrasing your question or contact support directly at **"
                    f"{contact_number}** for immediate assistance."
                )

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
                    
            # Friendly save reminder
            st.toast("Don't forget to use the 'Export & Share' in the sidebar to save your chat!", icon="💾")
            
        # Clear pending query
        st.session_state["pending_query"] = None
        st.rerun() # Rerun to refresh the chat input form state

    
    st.markdown('</div>', unsafe_allow_html=True)

    # Input row (moved to bottom)
    with st.form(key="chat_form", clear_on_submit=True):
        disabled_input = not st.session_state.get("retriever_ready")
        user_query = st.text_input(
            "💭 Ask about the course content", 
            placeholder="e.g., What are the prerequisites for this course?",
            disabled=disabled_input
        )
        col1, col2 = st.columns([1, 4])
        with col1:
            submitted = st.form_submit_button("🚀 Send", type="primary", use_container_width=True, disabled=disabled_input)
        with col2:
            if enable_voice and sr is not None:
                # Use a different key/logic for voice input to avoid conflicts
                if st.form_submit_button("🎤 Voice", use_container_width=True, disabled=disabled_input):
                    recognizer = sr.Recognizer()
                    try:
                        with st.spinner("🎙️ Listening..."):
                            with sr.Microphone() as source:
                                recognizer.adjust_for_ambient_noise(source, duration=1)
                                audio = recognizer.listen(source, timeout=5)
                            voice_text = recognizer.recognize_google(audio)
                            
                            # Use voice text as the pending query
                            if voice_text:
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
    st.markdown(f"**Viewing history for: {current_course_key}**")
    
    # Show history from the specific active course
    if st.session_state["messages"]:
        for i, msg in enumerate(st.session_state["messages"], start=1):
            role = "👤 User" if msg["role"] == "user" else "🤖 Assistant"
            css_cls = "user-msg" if msg["role"] == "user" else "bot-msg"
            st.markdown(f"<div class='stChatMessage {css_cls}'><strong>{i}. {role}:</strong> {msg['content']}</div>", unsafe_allow_html=True)

    else:
        st.info("💬 No conversation yet for this course. Start chatting in the Chat tab!")
    st.markdown('</div>', unsafe_allow_html=True)

# Floating contact number (using the dynamic contact_number variable)
st.markdown(f'<div class="floating-contact">📞 {contact_number}</div>', unsafe_allow_html=True)

# Floating registration button (kept as is)
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
