import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.schema import Document
import os
from time import perf_counter
from dotenv import load_dotenv
import yt_dlp
import requests
import json

load_dotenv()

llm = ChatOllama(model="llama3.1:8b", temperature=0)

template = """
Provide a concise summary of the following in 500 words:
Content: {text}
"""

prompt = PromptTemplate(
    input_variables=["text"],
    template=template
)

def get_youtube_transcript(url):
    ydl_opts = {
        'skip_download': True,
        'writesubtitles': True,
        'subtitlesformat': 'vtt',
        'quiet': True,
        'forcejson': True,
        'extract_flat': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        subs = info.get('subtitles') or info.get('automatic_captions')
        if not subs:
            return None
        # Prefer English
        for lang in ['en', 'en-US', 'en-GB']:
            if lang in subs:
                sub_url = subs[lang][0]['url']
                ext = subs[lang][0].get('ext', '')
                break
        else:
            lang = next(iter(subs))
            sub_url = subs[lang][0]['url']
            ext = subs[lang][0].get('ext', '')

        # Handle VTT
        if ext == 'vtt' or sub_url.endswith('.vtt'):
            vtt = requests.get(sub_url).text
            lines = [
                line.strip() for line in vtt.splitlines()
                if line
                and not line.startswith(('WEBVTT', 'X-TIMESTAMP', 'NOTE'))
                and '-->' not in line
                and not line.replace('.', '').isdigit()
            ]
            transcript = ' '.join(lines)
            return transcript
        # Handle json3
        elif ext == 'json3' or sub_url.endswith('.json3'):
            resp = requests.get(sub_url)
            data = resp.json()
            texts = []
            for event in data.get('events', []):
                segs = event.get('segs')
                if segs:
                    texts.append(''.join(seg['utf8'] for seg in segs if 'utf8' in seg))
            transcript = ' '.join(texts)
            return transcript
        else:
            return "Subtitle format not supported."
        

st.set_page_config(page_title="GenAI Summarization", page_icon=":book:", layout="wide")
st.title("GenAI Summarization App")
st.subheader("Summarize URLs, and YouTube Videos")

url = st.text_input("Enter a URL or YouTube Video Link", label_visibility="collapsed")

if st.button("Summarize"):
    if not url.strip():
        st.error("Please enter a valid URL or YouTube link.")
    elif not validators.url(url):
        st.error("Invalid URL format. Please enter a valid URL or YouTube link.")
    else:
        try:
            with st.spinner("Loading documents..."):
                timer_start = perf_counter()
                if "youtube.com" in url or "youtu.be" in url:
                    transcript = get_youtube_transcript(url)
                    if not transcript:
                        st.error("No transcript found for this video.")
                        st.stop()
                    documents = [Document(page_content=transcript)]
                else:
                    loader = UnstructuredURLLoader(
                        urls=[url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.4 Safari/605.1.15"}
                    )
                    documents = loader.load()

                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                summary = chain.run(documents)
                timer_end = perf_counter()
                st.success(summary)
                st.write(f"Time taken: {timer_end - timer_start:.2f} seconds")
        except Exception as e:
            import traceback
            print(traceback.print_exc())
            st.error(f"An error occurred while processing the URL: {e}")