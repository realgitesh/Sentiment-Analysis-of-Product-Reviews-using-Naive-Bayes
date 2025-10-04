# 📊 Sentiment Analysis of Product Reviews using Naive Bayes

This project performs **Sentiment Analysis** on product reviews (positive, negative, neutral) using **Natural Language Processing (NLP)** techniques and a **Naive Bayes classifier**.  
It includes text preprocessing, model training, evaluation, and visualization of results.  

---

## 🚀 Features
- Text preprocessing (cleaning, stopword removal, lemmatization)  
- TF-IDF vectorization for feature extraction  
- **Naive Bayes (MultinomialNB)** for sentiment classification  
- Accuracy, confusion matrix, and classification report for evaluation  
- Word clouds for positive, negative, and neutral reviews  
- Bar chart showing accuracy per class  
- Test with custom examples  

---

## 🛠️ Tech Stack / Libraries
- **Python 3.x**
- **Pandas, NumPy** → data handling  
- **NLTK** → stopwords, lemmatizer  
- **Scikit-learn** → ML model, TF-IDF, metrics  
- **Matplotlib, Seaborn** → visualization  
- **WordCloud** → word cloud generation  

---

## 📂 Project Structure
├── reviews.csv # Dataset (must be in same folder)
├── sentiment_analysis.py # Main script
├── README.md # Project documentation

yaml
Copy code

---

## ⚙️ Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/sentiment-analysis.git
   cd sentiment-analysis
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Ensure NLTK resources are downloaded:

python
Copy code
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
Place reviews.csv in the same folder as the script.

▶️ Usage
Run the script:

bash
Copy code
python sentiment_analysis.py
Example predictions:

vbnet
Copy code
Input: "The camera quality is fantastic and battery lasts long!"
Output: Positive
📊 Results
Model accuracy displayed after training

Confusion Matrix heatmap

Word clouds for each sentiment class

Bar chart for class-wise accuracy

🔍 Problems Faced
Handling missing dataset (reviews.csv)

NLTK resource download errors

Cleaning noisy text (HTML, URLs, special chars)

Class imbalance affecting accuracy

Installing required libraries

📌 Future Improvements
Use deep learning models (LSTM, BERT) for better accuracy

Add sentiment intensity score instead of just 3 classes

Build a simple web interface using Flask/Streamlit

📜 License
This project is licensed under the MIT License.
