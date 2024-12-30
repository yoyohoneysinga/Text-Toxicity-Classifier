import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

nltk.download('stopwords')

usual_messages_file_path = "Text Toxicity Classification/usual_comments.csv"
toxic_messages_file_path = "Text Toxicity Classification/data_raw_classified_toxic_comments.csv"

usual_messages_df = pd.read_csv(usual_messages_file_path)
toxic_messages_df = pd.read_csv(toxic_messages_file_path)

usual_messages_df = usual_messages_df.drop(usual_messages_df.columns[0], axis=1)
toxic_messages_df = toxic_messages_df.drop(toxic_messages_df.columns[0], axis=1)

usual_messages_df['label'] = 0
toxic_messages_df['label'] = 1
df = pd.concat([usual_messages_df, toxic_messages_df], ignore_index=True)

def preprocess_text(text):
    text = text.lower() 
    tokens = text.split() 
    tokens = [word for word in tokens if word not in stopwords.words('english')]  
    return ' '.join(tokens)

print(df.head())

df['comment_text'] = df['comment_text'].apply(preprocess_text)

vectorizer = TfidfVectorizer(max_features=1000)

X = vectorizer.fit_transform(df['comment_text'])
Y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

joblib.dump(model, 'Text Toxicity Classification/text_toxicity_model.pkl')
joblib.dump(vectorizer, 'Text Toxicity Classification/tfidf_vectorizer.pkl') 


