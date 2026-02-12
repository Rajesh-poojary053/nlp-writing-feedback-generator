# app.py
import streamlit as st
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import random
import textwrap

# OPTIONAL / ENHANCEMENT IMPORTS (graceful fallback)
try:
    import textstat
    HAVE_TEXTSTAT = True
except Exception:
    HAVE_TEXTSTAT = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAVE_PLOTLY = True
except Exception:
    HAVE_PLOTLY = False

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    HAVE_WORDCLOUD = True
except Exception:
    HAVE_WORDCLOUD = False

try:
    import language_tool_python
    HAVE_LANGTOOL = True
    LT_TOOL = language_tool_python.LanguageTool('en-US')
except Exception:
    HAVE_LANGTOOL = False
    LT_TOOL = None

try:
    from spellchecker import SpellChecker
    HAVE_SPELL = True
    SPELLER = SpellChecker()
    
    # Add common technical/domain terms to avoid false positives
    technical_terms = [
        'cnn', 'rnn', 'lstm', 'gru', 'dataset', 'datasets', 'relu', 'softmax',
        'preprocessing', 'overfitting', 'underfitting', 'convolutional', 
        'resized', 'rgb', 'rgba', 'grayscale', 'workflow', 'workflows',
        'downsample', 'upsample', 'github', 'api', 'json', 'csv', 'html',
        'tensorflow', 'pytorch', 'numpy', 'pandas', 'sklearn', 'matplotlib',
        'gpu', 'cpu', 'hyperparameter', 'hyperparameters', 'backend', 'frontend',
        'boolean', 'int', 'float', 'str', 'bool', 'async', 'await'
    ]
    SPELLER.word_frequency.load_words(technical_terms)
except Exception:
    HAVE_SPELL = False
    SPELLER = None

# Sentence-BERT optional (only used if installed and user enables it)
try:
    from sentence_transformers import SentenceTransformer
    HAVE_SBERT = True
    SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    HAVE_SBERT = False
    SBERT_MODEL = None

# NLTK
nltk.download("punkt", quiet=True)

# --- Enhanced Page Config & Beautiful CSS ---
st.set_page_config(
    layout="wide", 
    page_title="AI Writing Feedback Generator",
    page_icon="✍️",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 50%, #0f1419 100%);
    }
    
    /* Beautiful card style with gradient border */
    .card {
        background: linear-gradient(145deg, #1a2332, #0f1720);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        color: #e6eef6;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 20px;
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.15);
    }
    
    /* Metric cards with gradient backgrounds */
    .metric-card {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 24px rgba(99, 102, 241, 0.3);
        color: white;
        margin-bottom: 16px;
    }
    
    .metric-card-green {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    }
    
    .metric-card-orange {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
    }
    
    .metric-card-red {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 8px 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.95;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Keywords styling */
    .kw {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        padding: 8px 16px;
        border-radius: 20px;
        margin: 4px;
        display: inline-block;
        color: white;
        font-weight: 500;
        font-size: 0.9rem;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
        transition: all 0.3s ease;
    }
    
    .kw:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.5);
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 32px 0 16px 0;
        padding-bottom: 12px;
        border-bottom: 2px solid rgba(99, 102, 241, 0.3);
    }
    
    /* Feedback box styling */
    .feedback-box {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border-left: 4px solid #6366f1;
        border-radius: 12px;
        padding: 24px;
        margin: 20px 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        color: #e2e8f0;
        line-height: 1.8;
    }
    
    /* Weak sentence highlighting */
    .weak-sentence {
        background: rgba(239, 68, 68, 0.1);
        border-left: 3px solid #ef4444;
        padding: 12px;
        margin: 12px 0;
        border-radius: 8px;
        color: #fecaca;
    }
    
    /* Progress indicator */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 32px;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(99, 102, 241, 0.6);
    }
    
    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a2332 0%, #0f1720 100%);
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Text area */
    .stTextArea textarea {
        background: #1a2332;
        border: 2px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        color: #e6eef6;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    .stTextArea textarea:focus {
        border-color: #6366f1;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.5), transparent);
        margin: 32px 0;
    }
    
    /* Icon styling */
    .icon {
        font-size: 2rem;
        margin-bottom: 8px;
    }
    
    /* Muted text */
    .muted { 
        color: #94a3b8; 
        font-size: 0.95rem;
    }
    
    /* Small text */
    .small { 
        font-size: 0.85rem; 
        color: #cbd5e1; 
    }
    
    /* Info box */
    .info-box {
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 12px;
        padding: 16px;
        margin: 16px 0;
        color: #93c5fd;
    }
    
    /* Warning box */
    .warning-box {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 12px;
        padding: 16px;
        margin: 16px 0;
        color: #fbbf24;
    }
    
    /* Success box */
    .success-box {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        padding: 16px;
        margin: 16px 0;
        color: #6ee7b7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Your original functions (kept unchanged)
# ---------------------------
def clean_sentences(text):
    raw = nltk.sent_tokenize(text)
    cleaned = []
    for s in raw:
        s2 = s.strip()
        if len(s2) < 3:
            continue
        cleaned.append(s2)
    return cleaned

def basic_stats(sentences):
    words = []
    for s in sentences:
        words.extend(nltk.word_tokenize(s))
    vocab = set(w.lower() for w in words if w.isalpha())

    avg_len = np.mean([len(nltk.word_tokenize(s)) for s in sentences]) if sentences else 0

    return {
        "Total sentences": len(sentences),
        "Total words": len(words),
        "Average sentence length": round(avg_len, 2),
        "Vocabulary size": len(vocab),
        "Vocabulary richness": round(len(vocab) / max(len(words), 1), 3),
    }

def score_sentences(sentences):
    tfidf = TfidfVectorizer(stop_words="english")
    X = tfidf.fit_transform(sentences).toarray()

    norms = np.linalg.norm(X, axis=1)
    doc_vec = X.mean(axis=0).reshape(1, -1)
    sims = cosine_similarity(X, doc_vec).ravel()

    score = 0.6 * (norms / (norms.max() + 1e-8)) + 0.4 * (sims / (sims.max() + 1e-8))
    return score

def sentence_hints(sentence):
    hints = []
    words = sentence.split()
    length = len(words)

    if length < 6:
        hints.append("The sentence is too short and may lack depth.")
    if length > 26:
        hints.append("The sentence is long and could be broken into two for clarity.")
    if "and" in sentence.lower() and length > 20:
        hints.append("Too many clauses joined with 'and'. Consider splitting.")
    if sentence.count(",") >= 3:
        hints.append("Too many commas reduce clarity. Simplify the structure.")

    if not hints:
        hints.append("The sentence is acceptable.")

    return hints

def generate_feedback(stats, keywords, weak_sentences):
    fb = []

    fb.append("WRITING QUALITY ASSESSMENT")
    fb.append(f"- The essay contains {stats['Total words']} words across {stats['Total sentences']} sentences.")
    fb.append(f"- Average sentence length is {stats['Average sentence length']} words.")

    if stats["Average sentence length"] > 22:
        fb.append("- Many sentences are too long. Shorter, clearer sentences would improve readability.")
    else:
        fb.append("- Sentence length is reasonably maintained.")

    fb.append("\nVOCABULARY ANALYSIS")
    fb.append(f"- Vocabulary size: {stats['Vocabulary size']}")
    fb.append(f"- Vocabulary richness: {stats['Vocabulary richness']}")

    if stats["Vocabulary richness"] < 0.12:
        fb.append("- Vocabulary usage is limited. Introduce more diverse and topic-specific words.")
    else:
        fb.append("- Vocabulary range is appropriate.")

    fb.append("\nCONTENT & STRUCTURE ASSESSMENT")
    if len(weak_sentences) > 0:
        fb.append(f"- {len(weak_sentences)} sentences require restructuring for clarity and coherence.")
    fb.append("- Improve paragraph transitions to ensure smoother flow.")
    fb.append("- Support key ideas with examples, explanations, or evidence.")

    fb.append("\nFINAL RECOMMENDATION")
    fb.append("The writing demonstrates understanding of the topic, but refining sentence clarity, vocabulary, and structure will strengthen the overall quality.")

    return "\n".join(fb)

# ---------------------------
# New helper functions (augmenting — original logic preserved)
# ---------------------------
def compute_readability(text):
    if not HAVE_TEXTSTAT:
        return None
    try:
        return {
            "flesch_reading_ease": round(textstat.flesch_reading_ease(text), 2),
            "gunning_fog": round(textstat.gunning_fog(text), 2),
            "dale_chall": round(textstat.dale_chall_readability_score(text), 2)
        }
    except Exception:
        return None

def grammar_and_spelling(text, top_n=10):
    grammar_matches = []
    misspelled = []
    if HAVE_LANGTOOL and LT_TOOL is not None:
        try:
            matches = LT_TOOL.check(text)
            for m in matches[:top_n]:
                grammar_matches.append({
                    "message": m.message,
                    "context": getattr(m, "context", ""),
                    "ruleId": getattr(m, "ruleId", "")
                })
        except Exception:
            grammar_matches = []
    if HAVE_SPELL and SPELLER is not None:
        try:
            words = [w for w in nltk.word_tokenize(text) if w.isalpha()]
            miss = SPELLER.unknown(words)
            
            # Filter out common false positives
            filtered_miss = []
            for word in miss:
                word_lower = word.lower()
                # Skip if it's a proper noun (capitalized) or very short
                if len(word) <= 2:
                    continue
                # Skip if it looks like an acronym (all caps and short)
                if word.isupper() and len(word) <= 5:
                    continue
                # Skip common abbreviations
                if word_lower in ['etc', 'vs', 'ie', 'eg', 'ca', 'approx']:
                    continue
                filtered_miss.append(word)
            
            misspelled = list(filtered_miss)[:top_n]
        except Exception:
            misspelled = []
    return {"grammar": grammar_matches, "misspelled": misspelled}

def sbert_sentence_ranking(sentences, top_k=5):
    if not HAVE_SBERT or SBERT_MODEL is None:
        return None
    try:
        emb = SBERT_MODEL.encode(sentences, convert_to_numpy=True)
        norms = np.linalg.norm(emb, axis=1)
        doc_vec = emb.mean(axis=0).reshape(1, -1)
        sims = cosine_similarity(emb, doc_vec).ravel()
        scores = 0.6 * (norms / (norms.max()+1e-8)) + 0.4 * (sims / (sims.max()+1e-8))
        df = pd.DataFrame({"sentence": sentences, "sbert_score": scores})
        return df.sort_values("sbert_score", ascending=False).reset_index(drop=True).head(top_k)
    except Exception:
        return None

def plot_sentence_length_hist(sentences):
    lengths = [len(s.split()) for s in sentences]
    df_len = pd.DataFrame({"length": lengths})
    if HAVE_PLOTLY:
        fig = px.histogram(
            df_len, 
            x="length", 
            nbins=20, 
            title="Sentence Length Distribution",
            color_discrete_sequence=["#6366f1"]
        )
        fig.update_layout(
            xaxis_title="Words per Sentence",
            yaxis_title="Frequency",
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6eef6", size=12),
            title_font=dict(size=18, color="#e6eef6"),
            xaxis=dict(gridcolor="rgba(99, 102, 241, 0.1)"),
            yaxis=dict(gridcolor="rgba(99, 102, 241, 0.1)")
        )
        return fig
    return None

def plot_score_distribution(df):
    if HAVE_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(df))),
            y=df["Score"],
            mode='lines+markers',
            name='Sentence Quality Score',
            line=dict(color='#8b5cf6', width=3),
            marker=dict(size=8, color='#6366f1', line=dict(width=2, color='#8b5cf6'))
        ))
        fig.update_layout(
            title="Sentence Quality Score Distribution",
            xaxis_title="Sentence Index",
            yaxis_title="Quality Score",
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6eef6", size=12),
            title_font=dict(size=18, color="#e6eef6"),
            xaxis=dict(gridcolor="rgba(99, 102, 241, 0.1)"),
            yaxis=dict(gridcolor="rgba(99, 102, 241, 0.1)"),
            hovermode='x unified'
        )
        return fig
    return None

def create_wordcloud_image(text):
    if not HAVE_WORDCLOUD:
        return None
    wc = WordCloud(
        width=1200, 
        height=500, 
        background_color="#0f1720",
        colormap="cool",
        relative_scaling=0.5,
        min_font_size=10
    ).generate(text)
    return wc

def humanize_feedback(base_feedback, stats, keywords, weak_sentences, readability, grammar_report, student_name=""):
    lines = []
    openings = [
        "Overall, solid effort. Here's a focused critique to help you tighten this.",
        "Nice work — you clearly understand the subject. A few targeted improvements will make this outstanding.",
        "Good attempt. Below are practical, teacher-style notes to strengthen clarity and argumentation.",
    ]
    lines.append(random.choice(openings))

    lines.append(f"I counted **{stats['Total words']}** words and **{stats['Total sentences']}** sentences.")
    lines.append(f"Average sentence length is **{stats['Average sentence length']}** words — this guides the tone of my suggestions.")

    if len(keywords) > 0:
        sample_kw = ", ".join(keywords[:6])
        lines.append(f"I noticed your key terms: **{sample_kw}** — make sure each appears with a clear explanation or example.")
    else:
        lines.append("I couldn't identify strong topic words — try explicitly naming concepts and defining them.")

    if readability:
        flesch = readability.get("flesch_reading_ease", None)
        if flesch is not None:
            if flesch < 50:
                lines.append("Readability is on the harder side — break long sentences and simplify vocabulary where possible.")
            elif flesch < 60:
                lines.append("Readability is moderate — rephrase a few dense sentences for smoother flow.")
            else:
                lines.append("Readability looks good — keep sentences clear and concise.")

    if grammar_report:
        gcount = len(grammar_report.get("grammar", []))
        mcount = len(grammar_report.get("misspelled", []))
        if gcount > 0:
            lines.append(f"Language checks found **{gcount}** grammar suggestions — address them, especially repeated tense or agreement issues.")
        if mcount > 0:
            lines.append(f"There are **{mcount}** possible misspellings (e.g. {', '.join(grammar_report['misspelled'][:6])}). Proofread those.")
        if gcount == 0 and mcount == 0:
            lines.append("No obvious grammar or spelling problems detected — good attention to basic mechanics.")

    if weak_sentences:
        lines.append("The following sentences would benefit most from rewriting:")
        for i, s in enumerate(weak_sentences[:3], 1):
            short = (s[:220] + "...") if len(s) > 220 else s
            hints = sentence_hints(s)
            lines.append(f"{i}. \"{short}\" — {', '.join(hints)}")
    else:
        lines.append("No standout weak sentences detected by the automatic ranking — still double-check for clarity.")

    if student_name:
        lines.append(f"{student_name}, start by tightening topic sentences and adding one specific example per paragraph — that will raise the grade significantly.")

    closes = [
        "Make these focused edits and re-run the check — you'll see measurable improvement.",
        "Tidy these areas and you'll move from 'good' to 'really good' quickly.",
        "Do the small rewrites, add an example or two, and the clarity will improve a lot."
    ]
    lines.append(random.choice(closes))

    combined = "\n\n".join([base_feedback, "SUGGESTIONS", "\n".join(lines)])
    return combined

# ---------------------------
# Streamlit UI (Beautiful Enhanced Version)
# ---------------------------

# Header with gradient
st.markdown("""
    <h1 style='text-align: center; background: linear-gradient(135deg, #6366f1, #8b5cf6); 
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
    font-size: 3rem; font-weight: 800; margin-bottom: 0;'>
        ✍️ AI Writing Feedback Generator
    </h1>
    <p style='text-align: center; color: #94a3b8; font-size: 1.2rem; margin-top: 8px;'>
        Get professional, teacher-style feedback on your essays instantly
    </p>
""", unsafe_allow_html=True)

# Sidebar with beautiful styling
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    student_name = st.text_input("👤 Student name (optional):", value="", placeholder="Enter your name...")
    
    st.markdown("---")
    st.markdown("### 🔧 Advanced Features")
    enable_sbert = st.checkbox("🧠 Enable SBERT ranking", value=False, help="Uses semantic understanding for better sentence analysis")
    show_wordcloud = st.checkbox("☁️ Show word cloud", value=True, help="Visual representation of word frequency")
    
    st.markdown("---")
    st.markdown("### 📦 Optional Dependencies")
    
    dependencies_info = []
    if not HAVE_TEXTSTAT:
        dependencies_info.append("📊 `textstat` - Readability scores")
    if not HAVE_LANGTOOL:
        dependencies_info.append("✅ `language-tool-python` - Grammar checks")
    if not HAVE_SPELL:
        dependencies_info.append("📝 `pyspellchecker` - Spell checking")
    if not HAVE_SBERT:
        dependencies_info.append("🧠 `sentence-transformers` - SBERT ranking")
    if not HAVE_PLOTLY:
        dependencies_info.append("📈 `plotly` - Interactive charts")
    if not HAVE_WORDCLOUD:
        dependencies_info.append("☁️ `wordcloud` - Word clouds")
    
    if dependencies_info:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("**Install for enhanced features:**")
        for info in dependencies_info:
            st.markdown(f"- {info}")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="success-box">✅ All optional features available!</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown('<p class="small">💡 Run from the same environment where packages are installed</p>', unsafe_allow_html=True)

# Main input area with better styling
st.markdown('<div class="card">', unsafe_allow_html=True)
text = st.text_area(
    "📄 Paste your essay here",
    height=280,
    placeholder="Enter your essay text here... The AI will analyze sentence structure, vocabulary, readability, and provide detailed feedback.",
    help="Paste plain text only. For best results, ensure your text is at least 100 words."
)
st.markdown('</div>', unsafe_allow_html=True)

# Generate button with better layout
col_left, col_center, col_right = st.columns([1, 2, 1])
with col_center:
    generate = st.button("🚀 Generate Feedback", use_container_width=True)

if generate:
    if not text.strip():
        st.markdown('<div class="warning-box">⚠️ Please paste some text before generating feedback.</div>', unsafe_allow_html=True)
        st.stop()

    # Beautiful loading animation
    progress_container = st.empty()
    with progress_container.container():
        st.markdown('<div class="info-box">🔍 Analyzing your essay... This may take a few seconds.</div>', unsafe_allow_html=True)
        my_bar = st.progress(0)
        for i in range(0, 101, 5):
            my_bar.progress(i)
            time.sleep(0.03)
        time.sleep(0.2)
    progress_container.empty()

    # Run original pipeline
    sentences = clean_sentences(text)
    stats = basic_stats(sentences)

    scores = score_sentences(sentences)
    df = pd.DataFrame({"Sentence": sentences, "Score": scores})
    ranked_df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    weak_df = df.sort_values("Score", ascending=True).head(10).reset_index(drop=True)

    try:
        vec = TfidfVectorizer(stop_words="english")
        arr = vec.fit_transform([text]).toarray()[0]
        idx = arr.argsort()[-10:][::-1]
        keywords = list(vec.get_feature_names_out()[idx])
    except Exception:
        keywords = []

    grammar_report = grammar_and_spelling(text)
    readability = compute_readability(text)

    # Beautiful metric cards at the top
    st.markdown('<h2 class="section-header">📊 Quick Overview</h2>', unsafe_allow_html=True)
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.markdown(f'''
            <div class="metric-card">
                <div class="icon">📝</div>
                <div class="metric-label">Total Words</div>
                <div class="metric-value">{stats["Total words"]}</div>
            </div>
        ''', unsafe_allow_html=True)
    
    with metric_col2:
        st.markdown(f'''
            <div class="metric-card metric-card-green">
                <div class="icon">📄</div>
                <div class="metric-label">Sentences</div>
                <div class="metric-value">{stats["Total sentences"]}</div>
            </div>
        ''', unsafe_allow_html=True)
    
    with metric_col3:
        avg_len = stats["Average sentence length"]
        card_class = "metric-card-orange" if avg_len > 22 else "metric-card-green"
        st.markdown(f'''
            <div class="metric-card {card_class}">
                <div class="icon">📏</div>
                <div class="metric-label">Avg Sentence</div>
                <div class="metric-value">{avg_len}</div>
            </div>
        ''', unsafe_allow_html=True)
    
    with metric_col4:
        vocab_rich = stats["Vocabulary richness"]
        card_class = "metric-card-red" if vocab_rich < 0.12 else "metric-card-green"
        st.markdown(f'''
            <div class="metric-card {card_class}">
                <div class="icon">📚</div>
                <div class="metric-label">Vocab Richness</div>
                <div class="metric-value">{vocab_rich}</div>
            </div>
        ''', unsafe_allow_html=True)

    # Readability and Quality Checks
    st.markdown('<h2 class="section-header">📈 Readability & Quality Checks</h2>', unsafe_allow_html=True)
    
    read_col1, read_col2 = st.columns(2)
    
    with read_col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📖 Readability Scores")
        if readability:
            flesch = readability['flesch_reading_ease']
            st.metric("Flesch Reading Ease", flesch, help="Higher is easier (0-100)")
            st.metric("Gunning Fog Index", readability['gunning_fog'], help="Years of education needed")
            st.metric("Dale-Chall Score", readability['dale_chall'], help="Grade level difficulty")
            
            if flesch >= 60:
                st.markdown('<div class="success-box">✅ Easy to read - Good job!</div>', unsafe_allow_html=True)
            elif flesch >= 50:
                st.markdown('<div class="warning-box">⚠️ Moderately difficult - Consider simplifying</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box">🔴 Difficult to read - Simplify your language</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box">Install `textstat` for readability analysis</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with read_col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🔍 Grammar & Spelling")
        gcount = len(grammar_report.get("grammar", []))
        mcount = len(grammar_report.get("misspelled", []))
        
        st.metric("Grammar Issues", gcount, help="Detected grammar problems")
        st.metric("Spelling Issues", mcount, help="Possible misspellings")
        
        if gcount == 0 and mcount == 0:
            st.markdown('<div class="success-box">✅ No issues detected - Great work!</div>', unsafe_allow_html=True)
        elif gcount + mcount <= 5:
            st.markdown('<div class="warning-box">⚠️ Minor issues found - Quick fixes needed</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box">🔴 Multiple issues - Careful proofreading needed</div>', unsafe_allow_html=True)
        
        if mcount > 0:
            st.markdown("**Potential misspellings:**")
            misspelled_display = ", ".join(grammar_report["misspelled"][:10])
            st.markdown(f'<p class="small">{misspelled_display}</p>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Visualizations Section
    st.markdown('<h2 class="section-header">📊 Visual Analysis</h2>', unsafe_allow_html=True)
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if HAVE_PLOTLY:
            fig1 = plot_sentence_length_hist(sentences)
            if fig1:
                st.plotly_chart(fig1, use_container_width=True)
        else:
            st.markdown('<div class="info-box">Install `plotly` for interactive visualizations</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with viz_col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if HAVE_PLOTLY:
            fig2 = plot_score_distribution(df)
            if fig2:
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.markdown('<div class="info-box">Install `plotly` for score visualization</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Word Cloud
    if show_wordcloud:
        st.markdown('<h2 class="section-header">☁️ Word Cloud</h2>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if HAVE_WORDCLOUD:
            wc = create_wordcloud_image(text)
            if wc:
                fig_wc, ax = plt.subplots(figsize=(14, 6))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                fig_wc.patch.set_facecolor('#0f1720')
                st.pyplot(fig_wc)
                plt.close()
        else:
            st.markdown('<div class="info-box">Install `wordcloud` and `matplotlib` for word cloud visualization</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Keywords Section
    st.markdown('<h2 class="section-header">🔑 Key Terms & Concepts</h2>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if keywords:
        st.markdown("**Most important words in your essay:**")
        kw_html = " ".join([f"<span class='kw'>{k}</span>" for k in keywords])
        st.markdown(f"<div style='margin-top: 16px;'>{kw_html}</div>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="warning-box">Not enough content for keyword extraction. Add more text.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Sentence Analysis
    st.markdown('<h2 class="section-header">✨ Sentence Quality Analysis</h2>', unsafe_allow_html=True)
    
    sent_col1, sent_col2 = st.columns([1.5, 1])
    
    with sent_col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🌟 Top 10 Strongest Sentences")
        st.markdown('<p class="muted">These sentences are well-structured and contribute significantly to your essay.</p>', unsafe_allow_html=True)
        
        display_df = ranked_df.head(10).copy()
        display_df["Score"] = display_df["Score"].round(3)
        display_df.index = range(1, len(display_df) + 1)
        st.dataframe(display_df, use_container_width=True, height=400)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with sent_col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### ⚠️ Sentences Needing Work")
        st.markdown('<p class="muted">Focus on improving these for better clarity.</p>', unsafe_allow_html=True)
        
        weak_display = weak_df.copy()
        weak_display["Score"] = weak_display["Score"].round(3)
        weak_display.index = range(1, len(weak_display) + 1)
        st.dataframe(weak_display, use_container_width=True, height=400)
        st.markdown('</div>', unsafe_allow_html=True)

    # Detailed Weak Sentence Analysis
    st.markdown('<h2 class="section-header">🔧 Detailed Improvement Suggestions</h2>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    for i, row in weak_df.iterrows():
        s = row["Sentence"]
        score = row["Score"]
        
        st.markdown(f'<div class="weak-sentence">', unsafe_allow_html=True)
        st.markdown(f"**Sentence #{i+1}** (Quality Score: {score:.3f})")
        st.markdown(f'<p style="font-style: italic; margin: 12px 0;">"{s}"</p>', unsafe_allow_html=True)
        
        hints = sentence_hints(s)
        st.markdown("**Suggestions:**")
        for hint in hints:
            st.markdown(f"• {hint}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if i >= 4:  # Show first 5 weak sentences in detail
            break
    
    st.markdown('</div>', unsafe_allow_html=True)

    # SBERT Analysis (Optional)
    if enable_sbert:
        st.markdown('<h2 class="section-header">🧠 Semantic Analysis (SBERT)</h2>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        if HAVE_SBERT:
            with st.spinner("Computing semantic rankings..."):
                sbert_top = sbert_sentence_ranking(sentences, top_k=7)
                if sbert_top is not None:
                    st.markdown("**Most semantically important sentences:**")
                    st.markdown('<p class="muted">These sentences carry the most meaning in context.</p>', unsafe_allow_html=True)
                    sbert_display = sbert_top.copy()
                    sbert_display["sbert_score"] = sbert_display["sbert_score"].round(4)
                    sbert_display.index = range(1, len(sbert_display) + 1)
                    st.dataframe(sbert_display, use_container_width=True)
                else:
                    st.error("SBERT ranking failed. Please check your installation.")
        else:
            st.markdown('<div class="info-box">Install `sentence-transformers` to enable semantic analysis</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Generate Feedback
    base_feedback = generate_feedback(stats, keywords, weak_df["Sentence"].tolist())
    human_feedback = humanize_feedback(
        base_feedback, stats, keywords, 
        weak_df["Sentence"].tolist(), 
        readability, grammar_report, student_name
    )

    # Final Feedback Section
    st.markdown('<h2 class="section-header">📝 Comprehensive Teacher Feedback</h2>', unsafe_allow_html=True)
    
    # Show humanized feedback in a beautiful box
    st.markdown('<div class="feedback-box">', unsafe_allow_html=True)
    st.markdown(human_feedback)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Expandable section for technical feedback
    with st.expander("📋 View Technical Analysis Details"):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.code(base_feedback, language=None)
        st.markdown('</div>', unsafe_allow_html=True)

    # Success message
    st.markdown('<div class="success-box" style="text-align: center; font-size: 1.1rem;">✅ Analysis complete! Review the feedback above and make improvements to your essay.</div>', unsafe_allow_html=True)
    
    # Download button for feedback
    feedback_text = f"""
ESSAY FEEDBACK REPORT
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
Student: {student_name if student_name else 'Anonymous'}

{'='*60}

{human_feedback}

{'='*60}

STATISTICS:
- Total Words: {stats['Total words']}
- Total Sentences: {stats['Total sentences']}
- Average Sentence Length: {stats['Average sentence length']}
- Vocabulary Size: {stats['Vocabulary size']}
- Vocabulary Richness: {stats['Vocabulary richness']}

{'='*60}

TECHNICAL ANALYSIS:

{base_feedback}
"""
    
    st.download_button(
        label="📥 Download Feedback Report",
        data=feedback_text,
        file_name=f"essay_feedback_{time.strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        use_container_width=True
    )

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #94a3b8; padding: 20px;'>
        <p>💡 <strong>Pro Tip:</strong> Run multiple iterations to track your improvement over time</p>
        <p style='font-size: 0.9rem; margin-top: 8px;'>Made with ❤️ using Streamlit • Powered by AI & NLP</p>
    </div>
""", unsafe_allow_html=True)