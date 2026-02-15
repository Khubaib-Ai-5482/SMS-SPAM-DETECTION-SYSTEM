# 📧 SMS Spam Detection (TF-IDF + Naive Bayes)

## 📌 Overview

This project classifies SMS messages as **Ham** (legitimate) or **Spam** using:

- Text preprocessing and cleaning  
- TF-IDF feature extraction (unigrams + bigrams)  
- Multinomial Naive Bayes classifier  

The project also visualizes:

- Class distribution  
- Message length distribution  
- Confusion matrix  
- Most indicative words for spam messages  

Users can input any SMS text to get a real-time prediction.

---

## 🚀 Key Features

✔ Text cleaning (lowercase, remove special characters)  
✔ TF-IDF feature extraction  
✔ Multinomial Naive Bayes classifier  
✔ Accuracy evaluation  
✔ Confusion matrix visualization  
✔ Top words indicative of spam  
✔ Interactive SMS prediction  

---

## 🛠 Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  
- Regex  

---

## 📂 Dataset

The script uses the [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection):

- CSV format  
- Columns:  
  - `v1` → Label: `ham` or `spam`  
  - `v2` → SMS message text  

The script maps labels to integers:

- `ham` → 0  
- `spam` → 1  

---

## 🔎 Project Workflow

### 1️⃣ Text Cleaning

- Convert text to lowercase  
- Remove numbers and special characters  
- Keep only alphabetic characters  

```python
clean_text()
```

---

### 2️⃣ Data Visualization

- Bar plot of class distribution (Ham vs Spam)  
- Histogram of message lengths per class  

---

### 3️⃣ Train-Test Split

- 80% training  
- 20% testing  
- Stratified to preserve class balance  

---

### 4️⃣ TF-IDF Vectorization

```python
TfidfVectorizer(
    max_features=3000,
    ngram_range=(1,2)
)
```

- Uses unigrams + bigrams  
- Limits vocabulary to 3000 features  

---

### 5️⃣ Model Training

Model used:

```
Multinomial Naive Bayes
```

- Well-suited for text classification  
- Predicts probability of spam vs ham  

---

### 6️⃣ Evaluation

Metrics:

- Accuracy  
- Confusion matrix (visualized as heatmap)  
- Top 15 words indicative of spam messages  

---

## 🔮 Interactive Prediction

Type any SMS text to predict:

```
Enter SMS text:
```

Returns:

- `"Ham"` → legitimate message  
- `"Spam"` → spam message  

Example:

- Input: `"Congratulations! You have won a free iPhone"` → `"Spam"`  
- Input: `"Kal class kis time hai?"` → `"Ham"`  

---

## 📦 Installation

Install required packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## ▶️ How to Run

```bash
python your_script_name.py
```

Ensure `spam.csv` is in the same directory.

---

## 🎯 Use Cases

- Spam message detection  
- SMS or chat filtering  
- NLP text classification learning project  
- Keyword analysis for spam detection  

---

## 📈 What This Project Demonstrates

- Text preprocessing for SMS messages  
- TF-IDF feature engineering  
- Multinomial Naive Bayes classifier  
- Model interpretability using top indicative words  
- Interactive system for real-time message prediction  

---

## 👨‍💻 Author

Built as part of NLP experimentation for spam detection.

If this project is helpful, consider giving it a ⭐ on GitHub!
