import yt_dlp
import requests
import json

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

# Example usage:
print(get_youtube_transcript("https://www.youtube.com/watch?v=3Gcm27l-uyQ"))