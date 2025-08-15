import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"  # Suppress warnings
os.environ["TORCH_USE_SDPA"] = "0"  # Disable SDPA
import streamlit as st
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
import wikipediaapi
import re

# Initialize Wikipedia API with user agent
wiki_wiki = wikipediaapi.Wikipedia(user_agent='FakeNewsDetection/1.0 (aswin@example.com)', language='en')

# Load pre-trained fake news classifier (BERT-tiny, fine-tuned for fake news)
classifier = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection", device=-1)  # Force CPU

# Load local LLM for explanation generation (distilgpt2, runs locally, no API key)
generator = pipeline("text-generation", model="distilgpt2", device=-1)  # Force CPU, lighter model

st.title("Fake News Detection with Explanation")

# Input form: Text or URL
input_type = st.radio("Input Type", ("Text", "URL"))
article = ""

if input_type == "Text":
    article = st.text_area("Enter the article text here")
else:
    url = st.text_input("Enter the article URL")
    if url and st.button("Fetch Article from URL"):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            # Extract main text (simple heuristic: all paragraphs)
            article = ' '.join([p.text for p in soup.find_all('p')])
            st.text_area("Fetched Article Text", article, height=200)
        except Exception as e:
            st.error(f"Error fetching URL: {e}")

if st.button("Classify and Explain") and article:
    # Step 1: Classify (truncate to 512 tokens max for model)
    result = classifier(article[:512])
    label = result[0]['label']  # Outputs 'LABEL_1' (FAKE) or 'LABEL_0' (REAL)
    score = result[0]['score']
    label = "FAKE" if label == "LABEL_1" else "REAL"
    st.write(f"**Classification Result:** {label} (Confidence: {score:.2f})")

    # Step 2: RAG - Retrieve relevant context from verified source (Wikipedia)
    st.write("**Retrieving context from verified sources (Wikipedia)...**")
    try:
        # Improved keyword extraction: select longest non-stopword
        stop_words = {'the', 'is', 'are', 'in', 'of', 'to', 'and', 'a', 'an', 'there'}
        words = re.findall(r'\b\w+\b', article[:100].lower())
        search_terms = max([w for w in words if w not in stop_words], key=len, default='solar system')
        page = wiki_wiki.page(search_terms)
        if page.exists():
            snippet = page.summary[:500]  # Short snippet
            st.write("**Snippet from Verified Source (Wikipedia):**")
            st.write(snippet)
        else:
            snippet = "No relevant verified information found."
            st.write(snippet)
    except Exception as e:
        snippet = f"Error retrieving context: {e}"
        st.error(snippet)

    # Step 3: Generate explanation using local LLM
    prompt = f"Explain why the claim '{article[:100]}' is {label} in 50-100 words. Use this fact: {snippet[:200]}."
    try:
        explanation = generator(prompt, max_length=150, min_length=50, num_return_sequences=1, do_sample=True, temperature=0.7, top_k=30, top_p=0.85, no_repeat_ngram_size=2)[0]['generated_text'].strip()
        explanation = explanation.replace(prompt, "").strip()[:300]  # Clean and truncate
        if len(explanation.split()) < 20 or prompt.lower() in explanation.lower() or not any(word in explanation.lower() for word in ["contradict", "because", "reason", "evidence"]):
            explanation = f"The claim '{article[:100]}' is flagged as {label} because it contradicts Wikipedia's evidence: {snippet[:200]}."
    except Exception as e:
        explanation = f"The claim '{article[:100]}' is flagged as {label} because it contradicts Wikipedia's evidence: {snippet[:200]}."
    st.write("**Generated Explanation:**")
    st.write(explanation)

    # Button to copy/download explanation (as text file for moderation report)
    st.download_button(
        label="Copy Explanation to Moderation Report",
        data=explanation,
        file_name="moderation_report.txt",
        mime="text/plain"
    )