from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the pre-trained model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling sentiment prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the review from the form
        review = request.form['review']
        
        # Transform the input using the loaded vectorizer
        review_vectorized = vectorizer.transform([review])
        
        # Predict the sentiment using the loaded model
        prediction = model.predict(review_vectorized)
        sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
        
        # Return result to HTML page
        return render_template('index.html', review=review, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)

