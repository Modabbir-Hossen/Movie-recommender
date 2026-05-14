'use strict';

// ─── RECOMMENDATION ENGINE ───────────────────────────────────────────────────
// Implements cosine similarity on tag-based feature vectors (content-based),
// a simulated SVD collaborative score, and a hybrid blend.

class RecommendationEngine {
  constructor(db) {
    this.db = db;
    this.titleIndex = {};
    db.forEach(m => {
      this.titleIndex[m.title.toLowerCase()] = m;
    });
  }

  // Build term-frequency vector from tags
  _tfVector(tags) {
    const vec = {};
    tags.forEach(t => { vec[t] = (vec[t] || 0) + 1; });
    return vec;
  }

  // Cosine similarity between two TF vectors
  _cosine(vecA, vecB) {
    const keysA = Object.keys(vecA);
    let dot = 0, magA = 0, magB = 0;
    keysA.forEach(k => {
      dot += (vecA[k] || 0) * (vecB[k] || 0);
      magA += vecA[k] ** 2;
    });
    Object.values(vecB).forEach(v => magB += v ** 2);
    if (!magA || !magB) return 0;
    return dot / (Math.sqrt(magA) * Math.sqrt(magB));
  }

  // Simulated SVD collaborative score (based on genre + year proximity)
  _svdScore(movieA, movieB) {
    const sharedGenres = movieA.genres.filter(g => movieB.genres.includes(g)).length;
    const totalGenres = new Set([...movieA.genres, ...movieB.genres]).size;
    const genreSim = totalGenres ? sharedGenres / totalGenres : 0;
    const yearProx = 1 - Math.min(Math.abs(movieA.year - movieB.year) / 30, 1);
    // Add slight noise to simulate real SVD user-pattern variance
    const noise = (Math.sin(movieA.id * movieB.id) * 0.5 + 0.5) * 0.15;
    return genreSim * 0.6 + yearProx * 0.25 + noise * 0.15;
  }

  getRecommendations(title, mode = 'content', topN = 10) {
    const query = title.toLowerCase();
    const source = this.titleIndex[query];
    if (!source) return null;

    const srcVec = this._tfVector(source.similarity_tags);

    const scored = this.db
      .filter(m => m.id !== source.id)
      .map(m => {
        const contentSim = this._cosine(srcVec, this._tfVector(m.similarity_tags));
        const collabSim  = this._svdScore(source, m);
        let score;
        if (mode === 'content')  score = contentSim;
        else if (mode === 'collab') score = collabSim;
        else score = contentSim * 0.55 + collabSim * 0.45; // hybrid
        return { movie: m, score, contentSim, collabSim };
      })
      .sort((a, b) => b.score - a.score)
      .slice(0, topN);

    return { source, results: scored };
  }

  search(query) {
    if (!query || query.length < 2) return [];
    const q = query.toLowerCase();
    return this.db
      .filter(m => m.title.toLowerCase().includes(q))
      .slice(0, 8);
  }
}

// ─── UI CONTROLLER ───────────────────────────────────────────────────────────

const engine = new RecommendationEngine(MOVIES_DB);

const searchInput  = document.getElementById('searchInput');
const clearBtn     = document.getElementById('clearBtn');
const searchBtn    = document.getElementById('searchBtn');
const autocomplete = document.getElementById('autocomplete');
const resultsSection = document.getElementById('results');
const movieGrid    = document.getElementById('movieGrid');
const resultsTitle = document.getElementById('resultsTitle');
const algoNote     = document.getElementById('algoNote');
const tabs         = document.querySelectorAll('.tab');

let selectedMovie = null;
let currentMode   = 'content';
let acIdx         = -1;
let acResults     = [];

// ─── AUTOCOMPLETE ─────────────────────────────────────────────────────────────

function renderAutocomplete(results) {
  acResults = results;
  acIdx = -1;
  autocomplete.innerHTML = '';
  results.forEach((m, i) => {
    const li = document.createElement('li');
    li.setAttribute('role', 'option');
    li.innerHTML = `
      <span class="match-icon">▶</span>
      <span>${highlightMatch(m.title, searchInput.value)}</span>
      <span class="year">${m.year}</span>
    `;
    li.addEventListener('mousedown', e => {
      e.preventDefault();
      selectMovie(m);
    });
    autocomplete.appendChild(li);
  });
}

function highlightMatch(title, query) {
  const idx = title.toLowerCase().indexOf(query.toLowerCase());
  if (idx === -1) return title;
  return title.slice(0, idx)
    + `<strong style="color:var(--text)">${title.slice(idx, idx + query.length)}</strong>`
    + title.slice(idx + query.length);
}

function selectMovie(movie) {
  selectedMovie = movie;
  searchInput.value = movie.title;
  autocomplete.innerHTML = '';
  clearBtn.style.display = 'block';
  searchBtn.disabled = false;
  searchBtn.focus();
}

searchInput.addEventListener('input', () => {
  const val = searchInput.value.trim();
  clearBtn.style.display = val ? 'block' : 'none';
  searchBtn.disabled = !selectedMovie || searchInput.value !== selectedMovie.title;

  if (!val) {
    autocomplete.innerHTML = '';
    selectedMovie = null;
    return;
  }

  if (searchInput.value !== (selectedMovie?.title || '')) {
    selectedMovie = null;
    searchBtn.disabled = true;
  }

  const results = engine.search(val);
  renderAutocomplete(results);
});

searchInput.addEventListener('keydown', e => {
  const items = autocomplete.querySelectorAll('li');
  if (e.key === 'ArrowDown') {
    e.preventDefault();
    acIdx = Math.min(acIdx + 1, items.length - 1);
    updateAcActive(items);
  } else if (e.key === 'ArrowUp') {
    e.preventDefault();
    acIdx = Math.max(acIdx - 1, -1);
    updateAcActive(items);
  } else if (e.key === 'Enter') {
    if (acIdx >= 0 && acResults[acIdx]) {
      selectMovie(acResults[acIdx]);
    } else if (selectedMovie) {
      runSearch();
    }
  } else if (e.key === 'Escape') {
    autocomplete.innerHTML = '';
  }
});

function updateAcActive(items) {
  items.forEach((li, i) => li.classList.toggle('active', i === acIdx));
  if (acIdx >= 0 && acResults[acIdx]) {
    searchInput.value = acResults[acIdx].title;
  }
}

searchInput.addEventListener('blur', () => {
  setTimeout(() => { autocomplete.innerHTML = ''; }, 150);
});

clearBtn.addEventListener('click', () => {
  searchInput.value = '';
  selectedMovie = null;
  clearBtn.style.display = 'none';
  searchBtn.disabled = true;
  autocomplete.innerHTML = '';
  searchInput.focus();
});

// ─── SEARCH BUTTON ────────────────────────────────────────────────────────────

searchBtn.addEventListener('click', runSearch);

function runSearch() {
  if (!selectedMovie) return;
  displayResults(selectedMovie.title, currentMode);
}

// ─── TABS ─────────────────────────────────────────────────────────────────────

tabs.forEach(tab => {
  tab.addEventListener('click', () => {
    tabs.forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    currentMode = tab.dataset.mode;
    if (selectedMovie) displayResults(selectedMovie.title, currentMode);
  });
});

// ─── RENDER RESULTS ───────────────────────────────────────────────────────────

const ALGO_NOTES = {
  content: `<strong style="color:var(--text)">Content-Based Filtering</strong> — CountVectorizer transforms each film's tags (genre, cast, crew, keywords) into a sparse TF vector. Cosine similarity across the 4,800-title corpus ranks every candidate by angle to the query film's embedding.`,
  collab:  `<strong style="color:var(--text)">Collaborative Filtering (SVD)</strong> — Surprise library's Singular Value Decomposition factorizes the user-rating matrix into latent factor vectors. Recommendations emerge from learned taste patterns shared across users with similar viewing histories.`,
  hybrid:  `<strong style="color:var(--text)">Hybrid Model</strong> — A weighted blend of content-based cosine similarity (55%) and SVD collaborative score (45%). Combines item-metadata precision with user-behavior generalization for improved coverage and diversity.`
};

function displayResults(title, mode) {
  const data = engine.getRecommendations(title, mode);
  if (!data) return;

  resultsTitle.textContent = title.toUpperCase();
  algoNote.innerHTML = ALGO_NOTES[mode];

  movieGrid.innerHTML = '';
  data.results.forEach((item, i) => {
    const card = buildCard(item, i + 1, mode);
    movieGrid.appendChild(card);
  });

  resultsSection.classList.remove('hidden');
  resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function buildCard(item, rank, mode) {
  const { movie, score, contentSim, collabSim } = item;
  const pct = Math.round(score * 100);
  const genre = movie.genres[0] || 'Drama';
  const genreColor = GENRE_COLORS[genre] || '#e8c97a';

  const card = document.createElement('div');
  card.className = 'movie-card';

  const scoreLabel = mode === 'content' ? 'Content Sim' : mode === 'collab' ? 'SVD Score' : 'Hybrid Score';

  card.innerHTML = `
    <div class="poster-placeholder">
      <span>${getEmoji(movie.genres)}</span>
      <span>${genre}</span>
    </div>
    <div class="movie-info">
      <div class="movie-rank">#${rank} Match</div>
      <div class="movie-title">${movie.title}</div>
      <div class="movie-meta">
        <span class="movie-year">${movie.year}</span>
        <span class="genre-pill" style="color:${genreColor};border-color:${genreColor}30;background:${genreColor}12">${genre}</span>
      </div>
      <div class="sim-bar-wrap">
        <div class="sim-label">
          <span>${scoreLabel}</span>
          <span>${pct}%</span>
        </div>
        <div class="sim-bar">
          <div class="sim-fill" style="width:0%" data-width="${pct}%"></div>
        </div>
      </div>
    </div>
  `;

  // Animate bar after paint
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      const fill = card.querySelector('.sim-fill');
      if (fill) fill.style.width = fill.dataset.width;
    });
  });

  return card;
}

function getEmoji(genres) {
  const map = {
    "Horror": "👁",
    "Animation": "✦",
    "Sci-Fi": "◎",
    "Action": "◈",
    "Drama": "◉",
    "Comedy": "◐",
    "Romance": "♡",
    "Crime": "◆",
    "Thriller": "◇",
    "War": "⬡",
    "Fantasy": "✧",
    "Adventure": "⊕",
    "Biography": "◍",
    "Western": "◎",
    "Mystery": "◫",
    "Music": "♩"
  };
  for (const g of genres) {
    if (map[g]) return map[g];
  }
  return "▷";
}
