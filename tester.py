import joblib
import tkinter as tk
from tkinter import messagebox

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

# Function to handle button click event
def classify():
    user_input = entry.get()
    if user_input:
        sentiment_label = classify_sentiment(user_input)
        messagebox.showinfo("Predicted Sentiment", f"Sentiment: {sentiment_label}")
    else:
        messagebox.showwarning("Warning", "Please enter a message!")

# Create main window
root = tk.Tk()
root.title("Sentiment Classifier")
root.configure(bg="#f0f0f0")

# Set window size and position
window_width = 400
window_height = 150
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2
root.geometry(f"{window_width}x{window_height}+{x}+{y}")

# Create label and entry widget
label = tk.Label(root, text="Enter a message:", bg="#f0f0f0", font=("Arial", 12))
label.pack(pady=5)
entry = tk.Entry(root, width=50, font=("Arial", 12))
entry.pack(pady=5)

# Create classify button
button = tk.Button(root, text="Classify", command=classify, bg="#007bff", fg="#ffffff", font=("Arial", 12))
button.pack(pady=5)

# Run the GUI
root.mainloop()
