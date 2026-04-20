from flask import Flask, render_template, request, jsonify, redirect, session
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import os

app = Flask(__name__)
app.secret_key = "secret123"

# ================== LOAD DATASET SAFELY ==================
try:
    data = pd.read_csv('tcc_ceds_music_sample.csv')
except:
    data = pd.DataFrame()

if not data.empty:
    for col in ['genre', 'artist_name', 'track_name', 'release_date']:
        data[col] = data[col].fillna('')

    data['combined_features'] = data['genre'] + ' ' + data['artist_name'] + ' ' + data['track_name']

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
else:
    cosine_sim = []

# ================== GET SONG IMAGE ==================
def get_song_data(song, artist):
    try:
        url = f"https://itunes.apple.com/search?term={song} {artist}&limit=1"
        res = requests.get(url).json()
        if res['resultCount'] > 0:
            item = res['results'][0]
            return item['artworkUrl100'], item.get('previewUrl', "")
    except:
        pass
    return "https://via.placeholder.com/300", ""

# ================== RECOMMEND ==================
def get_recommendations(song_title):
    if data.empty:
        return []

    matches = data[data['track_name'].str.lower().str.contains(song_title.lower(), na=False)]

    if matches.empty:
        return []

    idx = matches.index[0]
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:9]

    results = []
    for i, score in sim_scores:
        row = data.iloc[i]
        image, preview = get_song_data(row['track_name'], row['artist_name'])

        results.append({
            "name": row['track_name'],
            "artist": row['artist_name'],
            "genre": row['genre'],
            "year": row['release_date'],
            "image": image,
            "preview": preview,
            "score": round(score * 100, 2)
        })
    return results

# ================== ROUTES ==================

@app.route('/')
def home():
    return render_template('index.html', recommendations=[], top_songs=[])

@app.route('/search')
def search():
    q = request.args.get('q', '')
    if data.empty or not q:
        return jsonify([])

    res = data[data['track_name'].str.lower().str.contains(q.lower(), na=False)]
    return jsonify(res['track_name'].head(5).tolist())

@app.route('/recommend', methods=['POST'])
def recommend():
    song = request.form.get('song')
    recs = get_recommendations(song)
    return render_template('index.html', recommendations=recs, top_songs=[])

# ================== RUN ==================
if __name__ == '__main__':
    app.run()
