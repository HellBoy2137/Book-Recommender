# 📚 Book Recommender System

### Overview
This project is a **Book Recommendation Web App** built using **Python**, **Machine Learning**, and **Streamlit**.  
It uses **only a single `books.csv` file** as its dataset — no external ratings or users file required.  
The system detects title, author, and rating columns automatically and provides:
- **Content-based Recommendations** using text similarity (book title + author)
- **Popularity-based Ranking** using average rating or rating count
- **Interactive Charts** to explore dataset insights

---

## 🧠 System Architecture

### 1. **Data Source**
- A single file: `data/books.csv`
- Contains metadata such as book title, author(s), publication year, and rating information
- Automatically analyzed to detect important columns:
  - `title` → Book title  
  - `authors` → Author(s)  
  - `book_id` → Unique ID or ISBN  
  - `__rating__` → Average rating extracted from your file  

---

### 2. **Core Algorithms**
#### 🧩 Content-Based Filtering
- Uses **CountVectorizer** (or TF-IDF) from `scikit-learn`
- Computes **Cosine Similarity** between books based on textual metadata (`title + author`)
- Recommends the most similar books to a given title

#### 🔥 Popularity-Based Ranking
- If a rating column exists, the app ranks books by average rating or rating count
- Used as a fallback or standalone "Top Popular Books" mode

#### 🧮 Fuzzy Matching
- Uses the **RapidFuzz** library to handle partial title searches (case-insensitive)
- Allows user to type partial or slightly misspelled titles and still get recommendations

---

### 3. **Visualization & Insights**
Powered by **Matplotlib**, the dashboard provides quick data exploration:
- 📊 Rating Distribution Histogram  
- 👩‍💻 Top Authors Bar Chart  
- ⭐ Rating vs. Ratings Count Scatter Plot  
- 🕰 Publication Year Histogram  

These can be toggled using **“Show data & charts”** in the sidebar.

---

### 4. **Frameworks & Libraries Used**
| Category | Libraries |
|-----------|------------|
| Web Framework | `Streamlit` |
| Machine Learning | `scikit-learn` |
| Data Handling | `pandas`, `numpy` |
| Text Processing | `CountVectorizer`, `cosine_similarity` |
| Visualization | `matplotlib` |
| Matching | `rapidfuzz` |
| Model Serialization | `joblib` |

---

### 5. **Project Structure**
```
book-recommender-single-books/
│
├── app.py                     # Main Streamlit application
├── requirements.txt           # Dependencies
├── README.md                  # This file
│
├── data/
│   └── books.csv              # Your uploaded dataset
│
├── src/
│   ├── utils.py               # Data loading helpers
│   ├── preprocessing.py       # Cleans & standardizes book data
│   ├── content_model.py       # CountVectorizer + Cosine recommender
│   ├── popularity.py          # Ranking by rating or count
│
└── models/                    # (Optional) Saved models for future extension
```

---

### 6. **How to Run Locally**
#### 🖥️ Prerequisites
- Python 3.9+  
- Installed packages listed in `requirements.txt`

#### 🧩 Setup
```bash
python -m venv venv
.env\Scriptsctivate      # (on Windows)
pip install -r requirements.txt
```

#### 🚀 Run the App
```bash
python -m streamlit run app.py
```

Then open the URL shown in your terminal (usually http://localhost:8501).

---

### 7. **Modes in the App**
| Mode | Description |
|------|--------------|
| **Content (Title)** | Enter any book title → returns top similar books |
| **Top Popular** | Displays highest-rated books based on dataset ratings |
| **Charts / Data** | Optional “Show charts” sidebar section for visual insights |

---

### 8. **Future Enhancements**
- Integrate a **Collaborative Filtering (CF)** model using synthetic or real user ratings  
- Add **book cover images** via Open Library API  
- Implement a **Hybrid Recommendation System** (Content + CF)  
- Deploy on **Streamlit Cloud** or **Render** with dataset caching  

---

### 9. **Tech Highlights**
- Uses **machine learning without requiring external ratings**
- Fully **dynamic column detection** from any `books.csv` format
- Supports **fuzzy title search**, **lightweight deployment**, and **visual analytics**
- Modular design (each ML component in `src/` folder)
- Clean, interactive web UI made with Streamlit

---

### 💡 Example Usage
1. Upload your `books.csv` file into the `data/` folder  
2. Run the app  
3. In sidebar → choose **“Content (title)”** mode  
4. Enter a book title → view similar books instantly  
5. Toggle **“Show data & charts”** to explore the dataset visually  

---

### 🏁 Author & License
**Developed by:** *Akanshu*  
**Frameworks:** Python · Streamlit · scikit-learn · pandas  
**License:** MIT  
