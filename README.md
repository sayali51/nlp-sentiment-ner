# Week 3 — Sentiment Analysis & Named Entity Recognition (NER)

## Overview
Two-module NLP project: (1) a sentiment classifier achieving **91% accuracy** on IMDb reviews using Logistic Regression + TF-IDF, and (2) a named entity recognition pipeline using spaCy's `en_core_web_sm` model to extract PERSON, ORG, GPE, and DATE entities from news articles.

## Datasets
- **Sentiment:** [IMDb Movie Reviews](https://ai.stanford.edu/~amaas/data/sentiment/) — 50,000 reviews, binary labels
- **NER:** News articles corpus (publicly available)

## Results

### Sentiment Analysis
| Metric | Score |
|--------|-------|
| Accuracy | **91%** |
| Precision | 90% |
| Recall | 89% |
| F1-Score | **90%** |

**Model:** Logistic Regression + TF-IDF features

### Named Entity Recognition
| Entity Type | % of Entities Found |
|-------------|-------------------|
| PERSON | 34% |
| ORG | 28% |
| GPE / Location | 22% |
| DATE | 12% |
| Other | 4% |

**Model:** spaCy `en_core_web_sm`

## Example NER Output
```
Input: "Elon Musk announced OpenAI's expansion in India in January 2024."

Entities detected:
  Elon Musk      → PERSON
  OpenAI         → ORG
  India          → GPE
  January 2024   → DATE
```

## Known Limitation
Sarcasm handling is a known edge case — e.g. *"Oh great, another delay..."* gets misclassified as positive. Contextual models (BERT) are recommended for sarcasm-aware sentiment analysis.

## Project Structure
```
week3/
├── Sentiment_NER.ipynb    # Main notebook
├── requirements.txt
└── README.md
```

## Setup & Run
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
jupyter notebook Sentiment_NER.ipynb
```

## Skills Demonstrated
`spaCy` · `NLTK` · `Logistic Regression` · `TF-IDF` · `NER` · `F1-score` · entity extraction
