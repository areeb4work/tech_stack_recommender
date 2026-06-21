# 🧭 Tech Stack Matchmaker

A content-based recommendation engine that maps a person's skills to the tech job roles they best fit, built with **TF-IDF vectorization** and **Cosine Similarity**, wrapped in a simple interactive web UI.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Backend-000000?logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-4F6B4A)

---

## 📌 What it does

Type in at least 3 skills you have (e.g. `Python`, `SQL`, `Automation`), and the engine compares them against a dataset of job roles, returning the **Top 3 best-fitting roles** each with a percentage match score.

This is a practical implementation of **content-based filtering**, the same family of logic behind recommendation engines like Netflix or Amazon's "you might also like" applied here to career/skill matching instead of media or products.

> Built as **Project 3 - AI Recommendation Logic** for DecodeLabs.

---

## ⚙️ How it works

| Step | What happens |
|---|---|
| **1. Input** | User submits 3+ skills through the web interface |
| **2. Vectorize** | Skills (user's + every job role's) are converted into TF-IDF weighted vectors in a shared vocabulary space |
| **3. Score** | Cosine Similarity measures the angle between the user's vector and each role's vector - closer alignment = higher score |
| **4. Rank & Filter** | Roles are sorted by score, and only the Top 3 are returned to avoid choice overload |

This is the classic **Input → Process → Output (IPO)** pipeline for recommendation systems.

---

## 🗂️ Project structure

```
tech_stack_recommender/
├── app.py              # Flask server - routes & API
├── recommender.py       # Core engine: TF-IDF + Cosine Similarity logic
├── raw_skills.csv       # Dataset: job roles mapped to their skill profiles
├── templates/
│   └── index.html       # Frontend UI (chip-based input, animated results)
└── README.md
```

---

## 🚀 Getting started (this step is specifically for someone who is just getting started with GitHub & AI)

### Prerequisites
- Python 3.10 or higher
- pip

### Installation

```bash
git clone https://github.com/<your-username>/tech-stack-recommender.git
cd tech-stack-recommender
pip install flask scikit-learn pandas
```

### Run it

```bash
python app.py
```

Then open your browser to:

```
http://localhost:5000
```

Type or click at least 3 skills → click **"Find my matches"** → see your Top 3 career matches with live-animated match scores.

---

## 🛠️ Tech stack

- **Python** - core logic
- **scikit-learn** - `TfidfVectorizer`, `cosine_similarity`
- **pandas** - dataset handling
- **Flask** - backend server & API routing
- **HTML / CSS3 / vanilla JS** - interactive frontend (no frameworks)

---

## 🧩 Customizing the dataset

Want to add more roles or tune the matching? Edit `raw_skills.csv` - one row per role, with space-separated skills in the second column:

```csv
Role,Skills
Data Scientist,"Python SQL Machine Learning Statistics Pandas"
```

Restart the server after editing to reload the dataset.

> ⚠️ Keep skill naming consistent across rows (e.g. always "Cloud Computing", never "Cloud Computng") - TF-IDF can only match terms it recognizes as identical strings.

---

## 🧪 Edge cases handled

- **Cold start detection** - if a user's skills have zero overlap with the dataset's vocabulary, the app flags this honestly instead of returning meaningless 0% matches as if they were real recommendations.
- **Input validation** requires a minimum of 3 skills before scoring, for sufficient data density.

---

## 📸 Demo

<img width="1895" height="887" alt="image" src="https://github.com/user-attachments/assets/d9aa68b2-42c9-4cfe-aa0e-5f2bf73bb3cc" />

---

## 👤 Author

**Areeb Ahsan**
DecodeLabs Artificial Intelligence Series - Project 3

## 📄 License

This project is open for learning and personal portfolio use.
