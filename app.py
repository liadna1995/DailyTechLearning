import streamlit as st
import os
import sys
from dotenv import load_dotenv
import ssl

# --- DETERMINE BASE DIRECTORY ---
# We want logs, history, and config to live next to the executable, 
# even when running from the PyInstaller temp folder.
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load environment variables
load_dotenv(os.path.join(BASE_DIR, ".env"))

# --- SSL CERTIFICATE FIX FOR CORPORATE ENVIRONMENTS (Must be first) ---
# Force disable SSL verification for requests/aiohttp/urllib
os.environ["PYTHONHTTPSVERIFY"] = "0"

# Monkey-patch the ssl module to ignore verification globally
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
    if hasattr(ssl, 'create_default_context'):
        ssl.create_default_context = _create_unverified_https_context

import google.genai as genai
from google.genai import types, errors
import toml
import json
import re
import logging
import io
import asyncio
import edge_tts
import html
import streamlit.components.v1 as components

# Configure logging to write to a file and the console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "app.log")),  # Saves logs to a file next to the exe
        logging.StreamHandler()          # Prints logs to the terminal
    ]
)
logger = logging.getLogger(__name__)

# --- 1. SETUP & CONFIGURATION ---
# Load API Key from environment variable
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("❌ GEMINI_API_KEY not found in .env file. Please create one.")
    st.stop()

CONTEXT_FILE = os.path.join(BASE_DIR, "context.toml")
HISTORY_FILE = os.path.join(BASE_DIR, "history.txt")

# Initialize the Gemini Client using the API Key
try:
    # This creates the client object, replacing the old configure() function
    client = genai.Client(api_key=API_KEY)
    logger.info("Gemini Client initialized successfully.") # Log success
except Exception as e:
    logger.error(f"Failed to initialize Gemini Client: {e}", exc_info=True) # Log full error
    st.error(f"Initialization Error: Could not create Gemini Client. Check API Key or library installation. Details: {e}")
    # In a real app, you would exit here. We'll let the Streamlit error handle it for now.
    client = None


# Load Settings
try:
    config = toml.load(CONTEXT_FILE)
except:
    st.error(f"Error: Could not load {CONTEXT_FILE}. Using defaults.")
    config = {"user": {"role": "Developer", "stack": []}, "preferences": {"mode": "Architect", "model": "gemini-2.0-flash-lite", "exclude_topics": []}}


# --- 2. TOKEN-EFFICIENT HISTORY MANAGER (No changes needed here) ---

def load_history():
    """Loads history from the low-token plain text file."""
    history = {"rejected": [], "learned": []}
    if not os.path.exists(HISTORY_FILE):
        return history

    with open(HISTORY_FILE, 'r') as f:
        content = f.read()
        
    learned_start = content.find("### LEARNED_TOPICS ###") + len("### LEARNED_TOPICS ###")
    rejected_start = content.find("### REJECTED_TOPICS ###")
    
    learned_block = content[learned_start:rejected_start].strip().split('\n')
    rejected_block = content[rejected_start + len("### REJECTED_TOPICS ###"):].strip().split('\n')
    
    history['learned'] = [line.strip() for line in learned_block if line.strip() and not line.strip().startswith('#')]
    history['rejected'] = [line.strip() for line in rejected_block if line.strip() and not line.strip().startswith('#')]
    
    return history

def save_history(learned_titles, rejected_titles):
    """Saves history back to the plain text file."""
    
    learned_lines = "\n".join([t for t in learned_titles if t])
    rejected_lines = "\n".join([t for t in rejected_titles if t])
    
    new_content = f"""
### LEARNED_TOPICS ###
# Add learned titles below this line
{learned_lines}
### REJECTED_TOPICS ###
# Add rejected titles below this line
{rejected_lines}
"""
    with open(HISTORY_FILE, 'w') as f:
        f.write(new_content.strip())


# --- 3. THE AI BRAIN (MODIFIED) ---

def generate_lesson(rejected_in_session=[]):
    if not client:
        return None # Exit if client wasn't initialized successfully

    # Combine all seen topics to blacklist
    permanent_history = load_history()
    
    # Token Optimization: Only send the last 50 topics to the LLM to save context
    # We will filter out duplicates in Python after generation
    full_history = permanent_history['rejected'] + permanent_history['learned'] + rejected_in_session
    recent_history = full_history[-50:] if len(full_history) > 50 else full_history
    
    role = config['user']['role']
    mode = config['preferences']['mode']
    exclude = config['preferences'].get('exclude_topics', [])
    if config['user']['stack']:
        stack_text = f"Stack focus: {config['user']['stack']}"
    else:
        stack_text = f"Focus: Concepts, tools, or patterns that are currently gaining traction and have high potential utility for a modern {role}, but are not yet mainstream standard."
    
    # --- DYNAMIC PROMPT CONSTRUCTION ---
    
    base_json = """
        "title": "Topic Title",
        "summary": "Engaging and fun technical explanation (2 paragraphs). Use analogies.",
        "slides": [
            {"title": "Slide 1: Concept Core", "points": ["detail 1", "detail 2", "detail 3"]},
            {"title": "Slide 2: key details", "points": ["detail 1", "detail 2", "detail 3"]},
            {"title": "Slide 3: Use Cases", "points": ["detail 1", "detail 2", "detail 3"]},
            {"title": "Slide 4: Best Practices", "points": ["detail 1", "detail 2", "detail 3"]}
        ],
        "resources": [
             { "label": "Official Docs (MUST BE REAL)", "url": "https://..." },
             { "label": "Tutorial/Article (MUST BE REAL)", "url": "https://..." }
        ],
        "quiz": [
            {"question": "Q1", "options": ["A", "B", "C", "D"], "answer": "Option A", "explanation": "Why..."},
            {"question": "Q2", "options": ["A", "B", "C", "D"], "answer": "Option A", "explanation": "Why..."},
            {"question": "Q3", "options": ["A", "B", "C", "D"], "answer": "Option A", "explanation": "Why..."},
            {"question": "Q4", "options": ["A", "B", "C", "D"], "answer": "Option A", "explanation": "Why..."},
            {"question": "BONUS", "options": ["A", "B", "C", "D"], "answer": "Option A", "explanation": "Why..."}
        ]
    """

    if mode == "Reporter":
        task_instruction = "Task: Act as a Tech Journalist. Provide a news-style deep dive on a trending technology, release, or shift in the industry. Focus on the 'What', 'Why', and 'Who' (adoption). Include a 'Industry Pulse' section covering adoption trends and major companies using it. Do NOT provide code snippets or diagrams."
        mode_specific_json = """,
        "news_pulse": ["Bullet point 1 on adoption/news", "Bullet point 2", "Bullet point 3"],
        """  
        
    elif mode == "Hacker":
        task_instruction = "Task: Act as a Senior Dev. Teach me a new concept with a heavy focus on implementation. You MUST provide a substantive, real-world code example AND a conceptual diagram."
        mode_specific_json = """,
        "code_snippet": { 
            "language": "C# (or relevant stack language)", 
            "code": "A PRACTICAL, REAL-WORLD implementation example. No 'Hello World'. Show how to actually use the concept in production code.", 
            "description": "Explanation of the implementation." 
        },
        "diagram": "Simple Mermaid JS code (graph TD/flowchart) explaining the flow or structure. Keep it simple and clean. NO markdown backticks."
        """

    elif mode == "Architect":
        task_instruction = "Task: Act as a System Architect. Explain a design pattern, architectural concept, or system structure. You MUST provide a Mermaid JS diagram AND a Trade-offs analysis."
        mode_specific_json = """,
        "diagram": "Simple Mermaid JS code (graph TD/flowchart). Keep it simple and clean. NO markdown backticks.",
        "tradeoffs": {
            "pros": ["Pro 1", "Pro 2"],
            "cons": ["Con 1", "Con 2"]
        }
        """
    
    else: # Fallback
        task_instruction = "Task: Teach me ONE new, advanced concept."
        mode_specific_json = ""

    prompt = f"""
    Act as a tech mentor for a {role}.
    Context: {stack_text}.
    Mode: {mode}.
    Avoid these topics: {recent_history} + {exclude}.

    {task_instruction}
    Make the tone engaging and fun, using analogies where appropriate.
    
    Output strictly valid JSON:
    {{
        {base_json}
        {mode_specific_json}
    }}
    IMPORTANT: 
    1. The quiz MUST have exactly 5 questions.
    2. Resources MUST be valid, existing URLs.
    """
    
    # Define the configuration for the API call
    config_params = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.8
    )

    # Retry loop for unique topics (Python-side filtering)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Call the API using the client object
            # Using gemini-2.0-flash-lite as the most cost-effective option available
            model_name = config['preferences'].get('model', 'gemini-2.0-flash-lite')
            response = client.models.generate_content(
                model=model_name, 
                contents=prompt,
                config=config_params
            )
            # The new SDK structure uses response.text for the content string
            lesson_data = json.loads(response.text)
            
            # Check if topic was actually in the full history (since we only sent partial history)
            if lesson_data['title'] in full_history:
                logger.warning(f"Duplicate topic generated: {lesson_data['title']}. Retrying... ({attempt+1}/{max_retries})")
                continue # Try again
                
            return lesson_data
            
        except errors.ClientError as e:
            if e.code == 429:
                 logger.warning("Quota Exceeded (429): Please wait a moment before trying again.")
                 st.warning("⚠️ High Traffic: We hit the AI's rate limit. Please wait 1 minute and try again.")
                 return None
            else:
                 logger.error(f"Client Error: {e}", exc_info=True)
                 st.error(f"AI Error: {e}")
                 return None
        except Exception as e:
            logger.error(f"Content Generation Failed: {e}", exc_info=True) # Captures traceback
            st.error(f"AI Error: Could not generate content. Check your context/key. Details: {e}")
            return None
            
    st.warning("Could not generate a unique topic after multiple attempts. Try clearing some history or changing focus.")
    return None

# --- 4. THE UI (No changes needed here) ---

st.set_page_config(page_title="Daily Tech", layout="wide")

def extract_mermaid_code(text):
    """
    Robustly extracts Mermaid code from a string, handling various AI output formats.
    """
    # Pattern to find the start of the diagram
    # Common starting keywords for Mermaid diagrams
    keywords = ["graph", "flowchart", "sequenceDiagram", "classDiagram", "stateDiagram", 
                "erDiagram", "gantt", "pie", "journey", "mindmap"]
    
    lines = text.split('\n')
    start_index = -1
    
    # 1. Try to find a code block first
    match = re.search(r'```mermaid\s*\n(.*?)\n\s*```', text, re.DOTALL)
    if match:
        text = match.group(1)
        lines = text.split('\n')

    # 2. Look for the first valid keyword line
    for i, line in enumerate(lines):
        line = line.strip()
        # Check if line starts with any keyword (ignoring case usually, but mermaid is case sensitive for some)
        # graph TD is standard.
        for kw in keywords:
            if line.startswith(kw):
                start_index = i
                break
        if start_index != -1:
            break
            
    if start_index != -1:
        return "\n".join(lines[start_index:])
    
    # Fallback: simple cleanup if no keyword found (though it will likely fail)
    clean_code = re.sub(r'```[a-zA-Z]*', '', text)
    clean_code = clean_code.replace("```", "").strip()
    return clean_code

def render_mermaid(code):
    """Renders Mermaid diagrams using a custom HTML component for better visuals."""
    # Robust cleanup: Extract only the mermaid code
    clean_code = extract_mermaid_code(code)

    # Dynamic height calculation based on line count to minimize scrolling
    line_count = len(clean_code.split('\n'))
    estimated_height = max(600, line_count * 40) # Estimate ~40px per line

    # Improved HTML/JS for robust Mermaid rendering
    # We escape the code to ensure HTML safety, although Mermaid usually handles it.
    escaped_code = html.escape(clean_code)

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/mermaid/10.9.0/mermaid.min.js"></script>
    <style>
        .mermaid {{
            display: flex;
            justify-content: center;
        }}
    </style>
    </head>
    <body style="background-color: white; margin: 0; padding: 20px;">
    <div class="mermaid">
    {escaped_code}
    </div>
    <script>
        document.addEventListener('DOMContentLoaded', function() {{
            try {{
                mermaid.initialize({{
                    startOnLoad: true,
                    theme: 'base',
                    themeVariables: {{ 'primaryColor': '#ff4b4b', 'edgeLabelBackground':'#ffffff', 'tertiaryColor': '#f0f2f6' }},
                    securityLevel: 'loose',
                }});
            }} catch (e) {{
                const err = document.createElement('div');
                err.textContent = 'Mermaid Error: ' + e.message;
                err.style.color = 'red';
                document.body.appendChild(err);
            }}
        }});
    </script>
    </body>
    </html>
    """
    components.html(html_code, height=estimated_height, scrolling=True)
    return clean_code

# Session state: temporary storage for current lesson and session rejects
if 'lesson' not in st.session_state:
    st.session_state.lesson = None
if 'session_rejected' not in st.session_state:
    st.session_state.session_rejected = []
if 'current_slide_index' not in st.session_state:
    st.session_state.current_slide_index = 0

st.title("🚀 Daily Tech")
# Gamification: Show XP / Learned Count
history = load_history()
learned_count = len(history['learned'])
st.markdown(f"**Mode:** `{config['preferences']['mode']}` | **Role:** `{config['user']['role']}` | **XP:** `{learned_count} Topics Learned`")
st.divider()

# The "Go" Button (Only shows when no lesson is loaded)
if not st.session_state.lesson:
    if st.button("Teach Me Something New"):
        with st.spinner("Preparing materials..."):
            st.session_state.lesson = generate_lesson(st.session_state.session_rejected)
            st.session_state.current_slide_index = 0
            st.rerun()

# The Lesson View
if st.session_state.lesson:
    content = st.session_state.lesson
    
    # Top Section: Title & Summary
    st.header(content['title'])
    
    # improved readability: use full width markdown
    st.markdown(content['summary'])

    # Audio Player for Summary with edge-tts (High Quality)
    try:
        async def generate_audio_stream(text):
             # en-US-ChristopherNeural is deep and engaging
             # Remove markdown artifacts for cleaner speech
             clean_text = re.sub(r'[\*\#]', '', text)
             communicate = edge_tts.Communicate(clean_text, "en-US-ChristopherNeural") 
             audio_data = b""
             async for chunk in communicate.stream():
                 if chunk["type"] == "audio":
                     audio_data += chunk["data"]
             return audio_data

        with st.spinner("Generating audio..."):
            audio_bytes = asyncio.run(generate_audio_stream(content['summary']))
            st.audio(audio_bytes, format='audio/mp3')
            
    except Exception as e:
        logger.error(f"Audio generation failed: {e}")
        st.warning(f"Audio summary unavailable. Error: {e}")
    
    st.divider()
    
    # Main Content Area - Vertical Layout (Long Page)
    
    # 1. Visuals (Infographic or Code)
    st.subheader("🎨 Visual Concept")
    
    # If we have code, show it (Hacker Mode preference)
    if 'code_snippet' in content and content['code_snippet']:
        st.subheader(content['code_snippet']['description'])
        st.code(content['code_snippet']['code'], language=content['code_snippet']['language'])

    # Increase height to avoid scrolling and allow full width
    clean_code = ""
    if 'diagram' in content and content['diagram']:
        clean_code = render_mermaid(content['diagram'])
    
    # Debugging Tool for the User/Dev
    with st.expander("🛠️ Debug Visual Data"):
        st.caption("If the diagram above is broken, check the code below:")
        st.code(clean_code, language='mermaid')
        st.caption("Full JSON Response:")
        st.json(content)

    st.divider()

    # Mode Specific Extra Sections (Displayed BEFORE Slides for better flow)
    if 'news_pulse' in content and content['news_pulse']:
        st.subheader("📰 Industry Pulse")
        # Ensure it renders as a list if it's not already
        if isinstance(content['news_pulse'], list):
             for item in content['news_pulse']:
                 st.markdown(f"- {item}")
        else:
             st.markdown(content['news_pulse'])
        st.divider()
        
    if 'tradeoffs' in content and content['tradeoffs']:
        st.subheader("⚖️ Trade-off Analysis")
        c_pros, c_cons = st.columns(2)
        with c_pros:
            st.success("### ✅ Pros")
            for pro in content['tradeoffs']['pros']:
                st.write(f"- {pro}")
        with c_cons:
            st.error("### ❌ Cons")
            for con in content['tradeoffs']['cons']:
                st.write(f"- {con}")
        st.divider()

    # 2. Slides (Key Takeaways)
    st.subheader("💡 Key Takeaways")
    
    @st.fragment
    def display_slides(slides):
        total_slides = len(slides)
        
        # Callback functions to update state
        def prev_slide():
            if st.session_state.current_slide_index > 0:
                st.session_state.current_slide_index -= 1

        def next_slide():
            if st.session_state.current_slide_index < total_slides - 1:
                st.session_state.current_slide_index += 1

        # Ensure index is within bounds (safety check)
        if st.session_state.current_slide_index < 0: st.session_state.current_slide_index = 0
        if st.session_state.current_slide_index >= total_slides: st.session_state.current_slide_index = total_slides - 1
        
        current_idx = st.session_state.current_slide_index
        slide = slides[current_idx]

        # Display Current Slide
        st.markdown(f"#### {slide['title']}")
        for p in slide['points']:
            # Card-like styling for points using markdown
            st.markdown(f"""
            <div style="background-color: #f0f2f6; color: #31333F; padding: 10px; border-radius: 5px; margin-bottom: 10px; border-left: 4px solid #ff4b4b;">
                {p}
            </div>
            """, unsafe_allow_html=True)
        
        st.write("") # Spacer

        # Navigation Controls
        col_prev, col_info, col_next = st.columns([1, 2, 1])
        
        with col_prev:
            st.button("⬅️ Previous", on_click=prev_slide, disabled=current_idx == 0)

        with col_info:
            st.markdown(f"<div style='text-align: center; color: gray;'>Slide {current_idx + 1} of {total_slides}</div>", unsafe_allow_html=True)

        with col_next:
            st.button("Next ➡️", on_click=next_slide, disabled=current_idx == total_slides - 1)

    display_slides(content['slides'])
    
    st.divider()

    # Quiz Section
    if 'quiz' in content:
        st.subheader("🧠 Knowledge Check")
        
        @st.fragment
        def display_quiz(quiz_data):
            # Use full width for quiz questions
            for i, q in enumerate(quiz_data):
                st.markdown(f"##### {i+1}. {q['question']}")
                # Use a unique key for each question's radio button
                answer = st.radio(f"Select an answer for Q{i+1}:", q['options'], key=f"quiz_{i}", index=None, label_visibility="collapsed")
                
                if answer:
                    if answer == q['answer']:
                        st.success(f"Correct! {q.get('explanation', '')}")
                    else:
                        st.error(f"Incorrect. The correct answer is: {q['answer']}. \n\n{q.get('explanation', '')}")
                st.write("") # Spacer

        display_quiz(content['quiz'])

    # Deep Dive Resources
    if 'resources' in content and content['resources']:
        st.subheader("📚 Deep Dive Resources")
        for res in content['resources']:
            st.markdown(f"- [{res['label']}]({res['url']})")
        
        # Add dynamic YouTube Search Link
        search_query = content['title'].replace(" ", "+") + "+tech+tutorial"
        st.markdown(f"- [📺 Search '{content['title']}' on YouTube](https://www.youtube.com/results?search_query={search_query})")

    st.divider()
    
    # Action Buttons (Saving Action)
    c1, c2, c3 = st.columns([1, 1, 2])
    
    if c1.button("⏭️ Skip"):
        # Add to temporary session reject list and rerun to generate new topic
        st.session_state.session_rejected.append(content['title'])
        st.session_state.lesson = None
        st.rerun() 
        
    if c2.button("🚫 Exclude"):
        # Save to permanent rejected history IMMEDIATELY
        history = load_history()
        history['rejected'].append(content['title'])
        # Also save any pending session rejects to avoid losing them
        save_history(history['learned'], history['rejected'] + st.session_state.session_rejected)
        st.session_state.session_rejected = [] # Clear session list as they are now persisted
        st.session_state.lesson = None
        st.rerun()

    if c3.button("✅ Done / Learned"):
        # Save to permanent learned history
        history = load_history()
        history['learned'].append(content['title'])
        save_history(history['learned'], history['rejected'] + st.session_state.session_rejected) 
        
        st.balloons()
        st.success("Great job! Topic logged to history. Close this window now.")
        st.session_state.lesson = None
