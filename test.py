from train_data import preprocess_text
import joblib

model = joblib.load('Text Toxicity Classification/text_toxicity_model.pkl')
vectorizer = joblib.load('Text Toxicity Classification/tfidf_vectorizer.pkl')

def process_comment(comment):
    comment_preprocessed = [preprocess_text(comment)]
    comment_vectorized = vectorizer.transform(comment_preprocessed)
    prediction = model.predict(comment_vectorized)
    return prediction

comment = input("Enter a comment: ")
result = process_comment(comment)

if result == 1:
    print("This comment is toxic.")
else:
    print("This comment is not toxic.")
