# Used Car Price Estimator & AI Buying Advisory (Deep Learning)

A full-stack AI web application that estimates used car market valuations in India using a **deep neural network** (TensorFlow / Keras), performs **Computer Vision visual photo inspections**, provides an **AI Buying Recommendation & Advisory**, and generates complete **Financial Order Deal Sheets** with history stored in **MongoDB**.

| Part | Technology |
|---|---|
| Deep Learning Model | Keras Deep Neural Network with embeddings (48,801 weights, $R^2 = 0.978$) |
| Computer Vision | PIL & NumPy image analysis (sharpness, luster, panel condition, condition rating) |
| Backend | Flask + Flask-CORS REST API |
| Database | MongoDB (falls back to local JSON file if MongoDB is offline) |
| Frontend | Modern Semantic HTML, Vanilla CSS, JavaScript, Chart.js (responsive, print-ready) |

---

## Features

1. **AI Valuation & Confidence Interval**
   - Rolling odometer display for predicted fair market value.
   - Monte-Carlo Dropout 80% confidence interval calibrated on validation data.
   - Year-wise and kilometre-wise depreciation curves.

2. **AI Buying Recommendation & Buy Index (0–100)**
   - Clear purchasing verdict: *Highly Recommended Buy*, *Fair Purchase - Negotiate*, *Buy With Repair Contingency*, or *Overpriced / High Risk*.
   - 4-Pillar Evaluation Scores: Value Score, Reliability Score, Usage Risk, and Condition Score.
   - *"What Matters Most For This Car"*: Prioritized technical and regulatory highlights (e.g. 10-year diesel rule in NCR, major 80,000 km service milestones, low annual mileage equity).

3. **Car Photo Upload & Visual AI Inspector**
   - Drag-and-drop car photo dropzone (Front, Sides, Cabin, Tyres).
   - Computer vision evaluation of sharpness, paint luster, and cosmetic condition.
   - Live AI rating badge (1–10) with automatic valuation adjustment factor.

4. **360° Pre-Purchase Condition Checklist**
   - Exhaustive checkpoints across 5 categories: Mechanical & Engine, Body & Exterior, Cabin & Electronics, Tyres & Suspension, and Legal/RTO.
   - Dynamic real-time refurbishment expense tally (e.g. tyre replacement, AC service, brake pads).

5. **Systematic 6-Step Valuation Waterfall Analysis**
   - Transparent step-by-step breakdown: Showroom Benchmark → Age Depreciation → Odometer Usage → Ownership History → Photo & Inspection Modifier → Final AI Market Valuation.

6. **Car Order Details & Financial Deal Sheet**
   - Comprehensive on-road acquisition cost breakdown (Asking Price vs Fair Value + Refurbishment Needed + RTO Transfer + Comprehensive Insurance).
   - Recommended Counter-Offer and Maximum Walk-Away Price.

7. **Buyer's Negotiation Playbook & 3-Year TCO**
   - Ready-to-copy bargaining scripts tailored to the car and detected flaws.
   - 3-Year Total Cost of Ownership (TCO) projection (Fuel, Scheduled Service, Insurance, Retained Equity).
   - One-click printable deal reports.

---

## Quick Start

### Prerequisites
Python 3.10 – 3.12 and Git.

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/swayam200545/<your-repo-name>.git
cd used_car_ai
python -m venv venv
venv\Scripts\activate      # On macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Web Application
```bash
python app.py
```
Open **http://127.0.0.1:5000** in your browser.

> Trained deep learning model weights and preprocessing metadata are already included in `artifacts/`.

---

## Retraining the Model (Optional)

To regenerate dataset and retrain from scratch:
```bash
python -m ml.make_dataset --rows 15000    # Generates data/used_cars.csv
python -m ml.train                        # Trains deep neural network (~1 min on CPU)
```

---

## Project Structure

```
├── app.py                # Flask application & REST API endpoints
├── db.py                 # MongoDB storage with local JSON fallback
├── requirements.txt      # Python dependencies
├── .gitignore            # Git exclusion rules
├── artifacts/            # Trained model.keras, metrics.json, preprocess.json
├── data/                 # Dataset files (used_cars.csv)
├── ml/
│   ├── advisory.py       # AI recommendation engine, pillars, deal sheet, playbook
│   ├── features.py       # Preprocessor and feature transformations
│   ├── image_analyzer.py # Computer vision image inspection & condition scoring
│   ├── make_dataset.py   # Synthetic Indian automotive market data generator
│   ├── predictor.py      # Inference, Monte-Carlo dropout, validation
│   └── train.py          # Neural network training and metric evaluation
├── static/               # CSS styles, JavaScript, and local fonts
└── templates/
    └── index.html        # Web application template
```

---

## Authors & Credits
Department of Information Technology · 2026-27  
* **Himanshu Chhapparghare**
* **Swayam Dhokte**
* **Aditya Dubey**  
Guide: **Mrs. Natasha Naik**
