# CineMatch — Movie Recommendation System

> **Content-Based Filtering · Cosine Similarity · SVD Collaborative**  
> Built by [Modabbir Hossen](https://github.com/Modabbir-Hossen) · Mar 2026 – Apr 2026

A full-stack movie recommendation engine powered by the TMDB 5000 Movies dataset, featuring both content-based and collaborative filtering approaches with a cinematic dark-mode UI.

---

## Live Demo

**[→ Try it on GitHub Pages](https://modabbir-hossen.github.io/Movie-recommender/)**

---

## Features

- **Content-Based Filtering** — CountVectorizer + cosine similarity on genre, cast, crew, and keywords
- **Collaborative Filtering** — Surprise library SVD on user rating patterns  
- **Hybrid Mode** — 55/45 weighted blend of both approaches
- **80+ movies** in the live demo (4,803 via Python backend)
- **Real-time autocomplete** search across the dataset
- **Similarity score bars** for each recommendation
- **Cinematic dark UI** with smooth animations

---

## 🗂 Project Structure

```
movie-recommender/
├── index.html                 # Frontend (GitHub Pages)
├── static/
│   ├── css/style.css          # Cinematic dark-mode stylesheet
│   └── js/
│       ├── movies.js          # Dataset + genre color map
│       └── app.js             # Recommendation engine + UI
├── backend/
│   ├── app.py                 # Flask API server
│   ├── recommender.py         # Core ML recommendation logic
│   ├── requirements.txt       # Python dependencies
│   └── notebooks/
│       └── exploration.ipynb  # EDA and model development
├── data/
│   └── .gitkeep               # Download TMDB dataset here
└── README.md
```

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | HTML5, CSS3, Vanilla JS |
| Backend | Python 3.11, Flask |
| Data | Pandas, NumPy |
| ML — Content | scikit-learn (CountVectorizer, cosine_similarity) |
| ML — Collab | Surprise library (SVD) |
| Dataset | TMDB 5000 Movies (Kaggle) |
| Deployment | GitHub Pages (frontend) |

---

## Quick Start

### Frontend Only (GitHub Pages)
The `index.html` runs entirely in the browser — just open it or push to GitHub Pages.

### Full Backend Setup

```bash
# 1. Clone the repository
git clone https://github.com/Modabbir-Hossen/movie-recommender.git
cd movie-recommender

# 2. Install Python dependencies
pip install -r backend/requirements.txt

# 3. Download the TMDB 5000 dataset
# → https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
# Place tmdb_5000_movies.csv and tmdb_5000_credits.csv in /data/

# 4. Run the Flask server
python backend/app.py

# 5. Open http://localhost:5000
```

---

## How It Works

### Content-Based Pipeline

```
Raw Data (TMDB 5000)
       ↓
Feature Engineering (genres + cast + crew + keywords → tags)
       ↓
CountVectorizer → Sparse TF Matrix (4803 × vocab_size)
       ↓
Pairwise Cosine Similarity Matrix (4803 × 4803)
       ↓
Top-N Nearest Neighbours for any query film
```

### Collaborative Filtering (SVD)

```
User Ratings Matrix (users × movies)
       ↓
SVD Decomposition → Latent User/Item Factors
       ↓
Predicted Ratings for unseen (user, movie) pairs
       ↓
Ranked recommendations per user
```

### Hybrid Blend
```
final_score = 0.55 × content_similarity + 0.45 × svd_score
```

---

## Key Results

| Metric | Value |
|--------|-------|
| Movies indexed | 4,803 |
| Vocabulary size | ~12,000 terms |
| Similarity matrix | 4803 × 4803 |
| SVD RMSE (test set) | ~0.89 |
| Query response time | < 50ms |

---

## Dataset

This project uses the **[TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)** from Kaggle:
- `tmdb_5000_movies.csv` — Movie metadata, genres, keywords, overview
- `tmdb_5000_credits.csv` — Cast and crew information

Download and place both files in the `/data/` directory.

---

## Future Improvements

- [ ] Deep learning embeddings (BERT / Sentence Transformers)
- [ ] Real-time TMDB API integration for live poster images
- [ ] User session tracking for personalized collaborative scores
- [ ] Matrix factorization with implicit feedback (ALS)
- [ ] Deploy Flask backend to Railway / Render

---

## Author

**Modabbir Hossen**  
[github.com/Modabbir-Hossen](https://github.com/Modabbir-Hossen)

---

## License

MIT License — free to use, modify, and distribute.
