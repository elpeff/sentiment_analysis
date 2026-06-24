import streamlit as st
from transformers import pipeline
import spacy
from spacy.cli import download as spacy_download
import requests  

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentence Analyzer",
    page_icon="🔬",
    layout="centered"
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;600&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .stage-card {
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.6rem;
    border-left: 4px solid transparent;
    font-size: 0.88rem;
  }
  .stage-pass { background: rgba(52,211,153,0.08); border-color: #34d399; color: #d1fae5; }
  .stage-fail { background: rgba(248,113,113,0.08); border-color: #f87171; color: #fee2e2; }
  .stage-skip { background: rgba(107,114,128,0.08); border-color: #4b5563; color: #9ca3af; }

  .result-positive { background: rgba(52,211,153,0.1); border: 1px solid #34d399; border-radius: 14px; padding: 1.4rem 1.8rem; margin-top: 0.5rem; }
  .result-negative { background: rgba(248,113,113,0.1); border: 1px solid #f87171; border-radius: 14px; padding: 1.4rem 1.8rem; margin-top: 0.5rem; }
  .result-invalid  { background: rgba(251,191,36,0.08);  border: 1px solid #fbbf24; border-radius: 14px; padding: 1.4rem 1.8rem; margin-top: 0.5rem; }

  .result-title { font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; font-weight: 600; margin-bottom: 0.3rem; }
  .pos-color  { color: #34d399; }
  .neg-color  { color: #f87171; }
  .warn-color { color: #fbbf24; }
  .meta { font-size: 0.8rem; color: #9ca3af; margin-top: 0.2rem; }

  .pipeline-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 0.4rem;
    margin-top: 1.2rem;
  }
</style>
""", unsafe_allow_html=True)


# ── Load spaCy model (cached) ──────────────────────────────────────────────────
@st.cache_resource
def load_spacy():
    model_name = "en_core_web_sm"
    try:
        return spacy.load(model_name)
    except OSError:
        spacy_download(model_name)
        return spacy.load(model_name)


# ── Load both transformer models (cached) ─────────────────────────────────────
@st.cache_resource
def load_models():
    validity = pipeline(
        "text-classification",
        model="textattack/distilbert-base-uncased-CoLA"
    )
    return validity


nlp = load_spacy()
validity_model = load_models()


# ── Stage 1: spaCy-powered rule filter ────────────────────────────────────────
def rule_based_check(text: str) -> tuple[bool, str]:
    text = text.strip()

    # ── 1a. Empty input ────────────────────────────────────────────────────────
    if not text:
        return False, "Input is empty."

    # ── 1b. Pure numbers / mostly numeric ─────────────────────────────────────
    alpha_chars = sum(c.isalpha() for c in text)
    if alpha_chars / max(len(text), 1) < 0.5:
        return False, "Input contains too many non-letter characters (numbers, symbols, etc.)."

    # ── 1c. Run spaCy tokenisation & POS tagging ───────────────────────────────
    doc = nlp(text)
    tokens = [t for t in doc if not t.is_space]

    # ── 1d. Too few tokens ─────────────────────────────────────────────────────
    if len(tokens) < 2:
        return False, "Too short — needs at least 2 words."

    # ── 1e. Vowel density (catches random consonant strings) ──────────────────
    vowels = set("aeiouAEIOU")
    words = [t.text for t in tokens]
    words_with_vowels = [w for w in words if any(c in vowels for c in w)]
    if len(words_with_vowels) / len(words) < 0.5:
        return False, "Too many words without vowels — looks like random characters."

    # ── 1f. Average word length (catches long gibberish blobs) ────────────────
    avg_len = sum(len(w) for w in words) / len(words)
    if avg_len > 20:
        return False, "Words are unusually long — may be gibberish."

    # ── 1g. spaCy: known-word ratio via is_oov (out-of-vocabulary) ────────────
    #   spaCy marks tokens it has no vector/entry for as out-of-vocabulary.
    #   We allow up to 60 % OOV before rejecting (proper nouns are often OOV).
    alpha_tokens = [t for t in tokens if t.is_alpha]
    if alpha_tokens:
        x_ratio = sum(1 for t in alpha_tokens if t.pos_ == "X") / len(alpha_tokens)
        if x_ratio > 0.60:
            return False, "Most words are unrecognised — input looks like gibberish."

    # ── 1h. spaCy POS: at least one real content POS must be present ──────────
    #   Catches strings like "the the the" (all DET) or pure punctuation runs.
    content_pos = {"NOUN", "VERB", "ADJ", "ADV", "PROPN", "NUM"}
    pos_tags = {t.pos_ for t in tokens}
    if not pos_tags & content_pos:
        return False, "No recognisable content words (nouns, verbs, adjectives…) found."

    # ── 1i. spaCy: reject if every alpha token is tagged X (unknown/foreign) ──
    alpha_pos = [t.pos_ for t in tokens if t.is_alpha]
    if alpha_pos and all(p == "X" for p in alpha_pos):
        return False, "All tokens tagged as unknown/foreign — not valid English."

    return True, "Passed basic checks."


# ── Stage 2: CoLA linguistic validity ─────────────────────────────────────────
def validity_check(text: str) -> tuple[bool, str, float]:
    result = validity_model(text)[0]
    label  = result["label"].lower()
    score  = round(result["score"] * 100, 1)
    is_valid     = label == "label_1" and score >= 90.0
    display_label = "acceptable" if is_valid else "unacceptable"

    return is_valid, display_label, score


# ── Stage 3: Sentiment ─────────────────────────────────────────────────────────
def sentiment_check(text: str) -> tuple[str, float]:
    response = requests.post(
        "http://localhost:8000/predict",
        json={"sentence": text}
    )
    data = response.json()
    return data["label"], data["score"]


# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown("## 🔬 Sentence Analyzer")
st.caption("Three-stage pipeline · Rule filter (spaCy) → Linguistic validity (CoLA) → Sentiment")
st.divider()

sentence = st.text_area(
    "Enter a sentence",
    placeholder="e.g. I really enjoyed the concert last night.",
    height=110
)

clicked = st.button("Analyze", type="primary", use_container_width=True)

if clicked:
    text = sentence.strip()

    if not text:
        st.warning("Please enter a sentence first.")
    else:
        st.markdown('<div class="pipeline-label">Pipeline stages</div>', unsafe_allow_html=True)

        # ── Stage 1 ──────────────────────────────────────────────────────────
        rule_ok, rule_msg = rule_based_check(text)

        if rule_ok:
            st.markdown(f"""
            <div class="stage-card stage-pass">
              ✅ <strong>Stage 1 — Rule filter (spaCy)</strong> &nbsp;·&nbsp; {rule_msg}
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="stage-card stage-fail">
              ❌ <strong>Stage 1 — Rule filter (spaCy)</strong> &nbsp;·&nbsp; {rule_msg}
            </div>
            <div class="stage-card stage-skip">
              ⏭ <strong>Stage 2 — CoLA validity</strong> &nbsp;·&nbsp; Skipped
            </div>
            <div class="stage-card stage-skip">
              ⏭ <strong>Stage 3 — Sentiment</strong> &nbsp;·&nbsp; Skipped
            </div>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="result-invalid">
              <div class="result-title warn-color">⚠️ Invalid input</div>
              <div class="meta">{rule_msg} — try a proper English sentence.</div>
            </div>""", unsafe_allow_html=True)
            st.stop()

        # ── Stage 2 ──────────────────────────────────────────────────────────
        with st.spinner("Checking linguistic validity…"):
            cola_ok, cola_label, cola_score = validity_check(text)

        if cola_ok:
            st.markdown(f"""
            <div class="stage-card stage-pass">
              ✅ <strong>Stage 2 — CoLA validity</strong> &nbsp;·&nbsp;
              Linguistically acceptable &nbsp;·&nbsp; confidence {cola_score}%
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="stage-card stage-fail">
              ❌ <strong>Stage 2 — CoLA validity</strong> &nbsp;·&nbsp;
              Linguistically unacceptable &nbsp;·&nbsp; confidence {cola_score}%
            </div>
            <div class="stage-card stage-skip">
              ⏭ <strong>Stage 3 — Sentiment</strong> &nbsp;·&nbsp; Skipped
            </div>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="result-invalid">
              <div class="result-title warn-color">⚠️ Not a valid sentence</div>
              <div class="meta">The sentence doesn't appear grammatically meaningful — sentiment analysis skipped.</div>
            </div>""", unsafe_allow_html=True)
            st.stop()

        # ── Stage 3 ──────────────────────────────────────────────────────────
        with st.spinner("Analyzing sentiment…"):
            sent_label, sent_score = sentiment_check(text)

        is_pos     = sent_label == "POSITIVE"
        sent_emoji = "😊" if is_pos else "😞"
        sent_color = "pos-color" if is_pos else "neg-color"
        sent_css   = "result-positive" if is_pos else "result-negative"

        st.markdown(f"""
        <div class="stage-card stage-pass">
          ✅ <strong>Stage 3 — Sentiment</strong> &nbsp;·&nbsp;
          {sent_label.capitalize()} &nbsp;·&nbsp; confidence {sent_score}%
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="pipeline-label">Result</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="{sent_css}">
          <div class="result-title {sent_color}">{sent_emoji} {sent_label.capitalize()}</div>
          <div class="meta">Sentiment confidence: {sent_score}%</div>
        </div>""", unsafe_allow_html=True)

        st.progress(sent_score / 100, text=f"Model confidence: {sent_score}%")
