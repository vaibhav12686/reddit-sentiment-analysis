# Reddit Comment Sentiment Analysis

This project analyzes sentiment, topics, and engagement patterns in 1 million Reddit comments from May 2019, collected from 40 subreddits. The analysis uses TextBlob, VADER, TF‑IDF, topic modeling, and visualization tools to understand how people talk about different subreddits and how sentiment and engagement are related.

---

## Overview

This repository contains a complete sentiment‑analysis pipeline for Reddit comments. It includes:

- Text preprocessing and cleaning  
- Sentiment analysis using TextBlob and VADER  
- TF‑IDF and count‑vectorization for top words and topic modeling  
- Wordclouds and statistical charts  
- Subreddit‑level risk and opportunity metrics  
- A lightweight machine‑learning model to predict comment score from sentiment and controversiality  

The code is written in Python and is designed to run in Jupyter Notebook or Google Colab.

---

## Features

- Text preprocessing pipeline using `nltk.punkt` and `nltk.stopwords` to tokenize and clean comments.  
- Dual sentiment analysis:  
  - `TextBlob.polarity` for a simple polarity score.  
  - `VADER` (Valence Aware Dictionary and sEntiment Reasoner) for compound, positive, negative, and neutral scores and label buckets.  
- Feature extraction using:  
  - `CountVectorizer` for term frequency and n‑grams (unigrams and bigrams).  
  - `TfidfVectorizer` for TF‑IDF‑weighted features and topic‑modeling input.  
- `WordCloud` visualizations for global, positive, and negative term patterns.  
- Subreddit‑level statistics: mean sentiment, positive/negative/neutral rates, and engagement‑related metrics.  
- Subreddit‑similarity graph built with `networkx` and `cosine_similarity`.  
- Topic modeling using `LatentDirichletAllocation` with 6 interpretable topics and per‑topic performance metrics.  
- RandomForest‑based score prediction model to explain how `sentiment` and `controversiality` relate to comment score.  
- Correlation analysis and heatmaps for numeric features.  

---

## Libraries / Python packages

The project uses the following core libraries:

```python
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import networkx as nx
```

---

## Dataset

The project uses the **"1 Million Reddit Comments from 40 Subreddits"** dataset from Kaggle:

- Dataset page: [1 Million Reddit Comments from 40 Subreddits – Kaggle](https://www.kaggle.com/datasets/smagnan/1-million-reddit-comments-from-40-subreddits)

Download the file `kaggle_RC_2019-05.csv` from Kaggle and place it in your project’s `data/` folder (or mount it in Google Colab) before running the notebook.

The dataset contains 1 million rows and the following columns:
- `subreddit`: name of the subreddit.  
- `body`: text of the comment.  
- `score`: upvote/downvote score of the comment.  
- `controversiality`: a measure of how controversial the comment is.

---

## Installation and setup

1. Install the required Python packages:

```bash
pip install pandas textblob nltk scikit-learn matplotlib seaborn wordcloud plotly
```

2. Download the dataset  
   - Go to: [https://www.kaggle.com/datasets/smagnan/1-million-reddit-comments-from-40-subreddits](https://www.kaggle.com/datasets/smagnan/1-million-reddit-comments-from-40-subreddits)  
   - Download `kaggle_RC_2019‑05.csv`.  
   - Place it in your project directory, for example under `data/kaggle_RC_2019‑05.csv`.

3. If you are using Google Colab:
   - Mount Google Drive or upload the file directly.  
   - Update the data path in your notebook (for example: `data_path = "data/kaggle_RC_2019‑05.csv"`).

4. Run the notebook top‑to‑bottom in order:
   - Data loading → cleaning → sentiment → vectorization → topic modeling → visualization → modeling.

---

## Key insights

- Overall sentiment:  
  - Average VADER compound score ≈ 0.09.  
  - About 45% of comments are positive, 30% negative, and 25% neutral.  

- Subreddit‑level sentiment:  
  - Most positive subreddits: `gonewild`, `Market76`, `aww`, `relationship_advice`.  
  - Most negative subreddits: `news`, `freefolk`, `asoiaf`, `The_Donald`.  

- Common terms:  
  - High‑frequency words: `like`, `people`, `would`, `one`, `get`, `think`, `good`, `really`.  
  - Positive terms include: `good`, `please`, `time`, `thanks`.  
  - Negative terms include: `people`, `like`, `shit`, `bad`.  

- Topic modeling (6 topics):  
  - Topics include: memes and casual banter, generic discussion, gaming, TV/episode‑talk, bot‑/mod‑related communication, and niche lifestyle or internet‑culture content.  

- Score prediction:  
  - A RandomForest model shows that `sentiment` is far more important than `controversiality` for predicting comment score.  
  - The model explains only a small amount of variance in practice, indicating that score is noisy and depends on many other factors.


---

## How to use this repository

- For learning:  
  - Open the notebook and step through each section to understand how text preprocessing, sentiment, vectorization, and visualization fit together.  

- For production‑like use:  
  - Wrap the main pipeline into a small module or script that:  
    - Loads the data  
    - Produces sentiment and topic labels  
    - Exports summarized tables and charts  
