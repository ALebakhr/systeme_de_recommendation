import pandas as pd
from flask import Flask, render_template_string, request
import pickle
from surprise import SVD

# Load data
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Load the trained SVD model (ensure you have saved it as 'svd_model.pkl')
try:
    with open('svd_model.pkl', 'rb') as f:
        svd_model = pickle.load(f)
except Exception as e:
    svd_model = None
    print("SVD model not loaded:", e)

# Prepare genre options
genres_set = set()
for genre_list in movies['genres'].dropna():
    for genre in genre_list.split('|'):
        genres_set.add(genre.strip())
genres = sorted(list(genres_set))

# Prepare year options (if year in title, e.g. "Movie Title (1999)")
def extract_year(title):
    import re
    match = re.search(r'\((\d{4})\)$', str(title))
    return int(match.group(1)) if match else None
movies['year'] = movies['titles'].apply(extract_year)
years = sorted(movies['year'].dropna().unique())

# Flask app
app = Flask(__name__)

HTML_FORM = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Movie Recommender</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://fonts.googleapis.com/css?family=Montserrat:400,700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Montserrat', Arial, sans-serif;
            margin: 0;
            background: linear-gradient(120deg, #e0eafc 0%, #cfdef3 100%);
            min-height: 100vh;
        }
        .container {
            background: #fff;
            padding: 40px 30px 30px 30px;
            border-radius: 16px;
            max-width: 600px;
            margin: 40px auto;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        }
        h2 {
            color: #2d3a4b;
            text-align: center;
            margin-bottom: 30px;
            font-weight: 700;
        }
        label {
            display: block;
            margin-top: 18px;
            color: #2d3a4b;
            font-weight: 600;
        }
        select, input[type=number], input[type=text] {
            width: 100%;
            padding: 10px 12px;
            margin-top: 7px;
            border: 1px solid #bfc9d1;
            border-radius: 6px;
            font-size: 1em;
            background: #f7fafc;
            transition: border 0.2s;
        }
        select:focus, input:focus {
            border: 1.5px solid #0078d7;
            outline: none;
            background: #fff;
        }
        button {
            margin-top: 28px;
            padding: 12px 0;
            width: 100%;
            background: linear-gradient(90deg, #0078d7 0%, #00c6fb 100%);
            color: #fff;
            border: none;
            border-radius: 6px;
            font-size: 1.1em;
            font-weight: 700;
            cursor: pointer;
            box-shadow: 0 2px 8px #b3c6e0;
            transition: background 0.2s, box-shadow 0.2s;
        }
        button:hover {
            background: linear-gradient(90deg, #005fa3 0%, #009ec3 100%);
            box-shadow: 0 4px 16px #b3c6e0;
        }
        table {
            width: 100%;
            margin-top: 32px;
            border-collapse: collapse;
            background: #f7fafc;
            border-radius: 8px;
            overflow: hidden;
        }
        th, td {
            padding: 12px 8px;
            border-bottom: 1px solid #e3e8ee;
            text-align: left;
        }
        th {
            background: #eaf1fb;
            color: #2d3a4b;
            font-weight: 700;
        }
        tr:last-child td {
            border-bottom: none;
        }
        tr:hover td {
            background: #e3f2fd;
        }
        @media (max-width: 700px) {
            .container { padding: 18px 5vw; }
            th, td { font-size: 0.98em; }
        }
    </style>
</head>
<body>
<div class="container">
    <h2>🎬 Recommandez-moi un film !</h2>
    <form method="post">
        <label>Genres préférés :</label>
        <select name="genres" multiple size="5">
            {% for genre in genres %}
            <option value="{{genre}}">{{genre}}</option>
            {% endfor %}
        </select>
        <label>Mots-clés dans le titre :</label>
        <input type="text" name="title_keywords" placeholder="Ex: Star, Love, War ...">
        <label>Année minimale :</label>
        <input type="number" name="min_year" min="1900" max="2100">
        <label>Année maximale :</label>
        <input type="number" name="max_year" min="1900" max="2100">
        <label>Note moyenne minimale :</label>
        <input type="number" name="min_rating" min="0" max="5" step="0.1">
        <label>Nombre minimal de notes :</label>
        <input type="number" name="min_num_ratings" min="0" step="1">
        <label>Nombre de films à afficher :</label>
        <input type="number" name="num_results" min="1" max="100" value="10">
        <button type="submit">Recommander</button>
    </form>
    {% if results is not none %}
    {% if results|length == 0 %}
    <div style="margin-top:36px; color:#d32f2f; text-align:center; font-weight:600; font-size:1.1em;">Aucun film ne correspond à vos critères.</div>
    {% else %}
    <h3 style="margin-top:36px; color:#0078d7; text-align:center;">Top {{results|length}} films recommandés :</h3>
    <table>
        <tr><th>Titre</th><th>Genres</th><th>Année</th><th>Note Moyenne</th><th>Nombre de notes</th></tr>
        {% for movie in results %}
        <tr>
            <td>{{movie['titles']}}</td>
            <td>{{movie['genres']}}</td>
            <td>{{movie['year']}}</td>
            <td>{{'%.2f' % movie['AvgRating'] if movie['AvgRating'] else ''}}</td>
            <td>{{movie['RatingsNum']}}</td>
        </tr>
        {% endfor %}
    </table>
    <div style="display:flex; flex-wrap:wrap; gap:18px; justify-content:center; margin-top:32px;">
        {% for movie in results %}
        <div style="background:#eaf1fb; border-radius:10px; box-shadow:0 2px 8px #b3c6e0; padding:18px 20px; min-width:220px; max-width:260px; margin-bottom:10px;">
            <div style="font-size:1.08em; font-weight:700; color:#2d3a4b; margin-bottom:6px;">{{movie['titles']}}</div>
            <div style="color:#0078d7; font-size:0.98em; margin-bottom:4px;">{{movie['genres']}}</div>
            <div style="color:#555; font-size:0.95em;">Année : <b>{{movie['year']}}</b></div>
            <div style="color:#555; font-size:0.95em;">Note : <b>{{'%.2f' % movie['AvgRating'] if movie['AvgRating'] else ''}}</b> ({{movie['RatingsNum']}} notes)</div>
        </div>
        {% endfor %}
    </div>
    {% endif %}
    {% endif %}
</div>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def recommend():
    results = None
    html_table = None
    if request.method == 'POST':
        selected_genres = request.form.getlist('genres')
        title_keywords = request.form.get('title_keywords', '').strip().lower()
        min_year = request.form.get('min_year', type=int)
        max_year = request.form.get('max_year', type=int)
        min_rating = request.form.get('min_rating', type=float)
        min_num_ratings = request.form.get('min_num_ratings', type=int)
        num_results = request.form.get('num_results', type=int) or 10
        filtered = movies.copy()
        if selected_genres:
            filtered = filtered[filtered['genres'].apply(lambda g: any(genre in g.split('|') for genre in selected_genres))]
        if title_keywords:
            for kw in title_keywords.split(','):
                kw = kw.strip()
                if kw:
                    filtered = filtered[filtered['titles'].str.lower().str.contains(kw, na=False)]
        if min_year:
            filtered = filtered[filtered['year'] >= min_year]
        if max_year:
            filtered = filtered[filtered['year'] <= max_year]
        # Popularity & rating
        pop = ratings.groupby('movieId').agg(RatingsNum=('rating', 'size'), AvgRating=('rating', 'mean')).reset_index()
        filtered = filtered.merge(pop, on='movieId', how='left')
        if min_rating is not None:
            filtered = filtered[filtered['AvgRating'] >= min_rating]
        if min_num_ratings is not None:
            filtered = filtered[filtered['RatingsNum'] >= min_num_ratings]
        filtered = filtered.sort_values(['AvgRating', 'RatingsNum'], ascending=[False, False])
        results = filtered.head(num_results).to_dict('records')

        # Add image URLs for each movie (using MovieLens ID)
        def image_formatter(movie_id):
            return f'<img src="https://liangfgithub.github.io/MovieImages/{movie_id}.jpg?raw=true" width="70" height="105">'
        filtered = filtered.head(num_results).copy()
        filtered['Image'] = filtered['movieId'].apply(image_formatter)
        html_table = filtered.to_html(
            escape=False,
            index=False,
            columns=["movieId", "Image", "titles", "AvgRating", "RatingsNum"]
        )
    return render_template_string(HTML_FORM + '''
    {% if html_table %}
    <div style="margin-top:36px;">
        <h3 style="color:#0078d7; text-align:center;">Aperçu visuel des films recommandés</h3>
        {{ html_table|safe }}
    </div>
    {% endif %}
    ''', genres=genres, results=results, html_table=html_table)

if __name__ == '__main__':
    app.run(debug=True)
