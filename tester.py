import joblib

# Load the saved model and vectorizer
loaded_model = joblib.load('trained_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Function to classify sentiment of input text
def classify_sentiment(text):
    # Transform the input text using the loaded vectorizer
    text_vec = vectorizer.transform([text])
    # Predict the sentiment label
    sentiment = loaded_model.predict(text_vec)
    return sentiment[0]

# Ask the user for a message input
user_input = input("Enter a message: ")

# Classify the sentiment of the user input
sentiment_label = classify_sentiment(user_input)

# Print the predicted sentiment
print("Predicted Sentiment:", sentiment_label)

