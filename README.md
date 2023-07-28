# Sentiment Analysis with Naive Bayes Classifier

## Description

Sentiment analysis is a natural language processing (NLP) task that involves determining the sentiment or emotional tone expressed in a piece of text. In this project, we perform sentiment analysis on a movie review dataset using a Naive Bayes classifier. The dataset consists of movie reviews labeled as either "positive" or "negative" sentiment, and our goal is to build a model that can accurately predict the sentiment of new reviews.


## License

This project is licensed under the MIT License - a permissive open-source license that allows you to freely use, modify, and distribute the code as long as you include the original copyright notice and disclaimer. The MIT License provides flexibility for developers and encourages collaboration and contributions to the project.

## Installation

To run this project, you need to have Python installed on your system. Additionally, ensure you have the required packages installed:

- nltk
- wordcloud
- pandas
- numpy
- matplotlib
- scikit-learn

You can install the necessary packages using the following pip commands:

```bash
!pip install nltk
!pip install wordcloud
!pip install pandas
!pip install numpy
!pip install matplotlib
!pip install scikit-learn
```

## Usage

1. Clone or download the project repository from the [GitHub page](https://github.com/rohitpoddar/sentiment-analysis-naive-bayes).
2. Navigate to the project directory and open the Jupyter Notebook or Python script containing the code.
3. Make sure you have the "IMDB Dataset.csv" file in the same directory as the code.
4. Run the code provided in the script to perform sentiment analysis on the movie reviews dataset.

The script follows these steps:

1. Imports necessary libraries, including pandas, numpy, nltk, matplotlib, and scikit-learn.
2. Loads the movie review dataset into a pandas DataFrame and displays the first 10 rows.
3. Checks for missing values and duplicates in the dataset.
4. Preprocesses the text data by converting it to lowercase, removing punctuation, and performing tokenization and stopword removal.
5. Applies stemming to reduce words to their base forms for better generalization.
6. Converts the text data into numerical feature vectors using the TF-IDF (Term Frequency-Inverse Document Frequency) technique.
7. Splits the dataset into training and testing sets to train the Naive Bayes classifier.
8. Trains the Multinomial Naive Bayes classifier on the training data.
9. Evaluates the performance of the classifier using accuracy, confusion matrix, and classification report on the testing data.
10. Defines a function `predict_sentiment(review)` to predict the sentiment of new movie reviews using the trained model.
11. Generates word cloud visualizations for positive and negative reviews to gain insights into the most common words in each sentiment category.

## Example

```python
# Importing necessary libraries and functions
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud

# Function to preprocess and predict sentiment
def predict_sentiment(review):
    # Preprocess the review
    review = review.lower().replace('[^\w\s]', '')
    review = ' '.join([word for word in review.split() if word not in stop_words])
    review = ' '.join([stemmer.stem(word) for word in review.split()])

    # Convert the review into numerical feature vector using TF-IDF
    review_tfidf = vectorizer.transform([review]).toarray()

    # Predict the sentiment
    prediction = naive_bayes_classifier.predict(review_tfidf)

    return prediction[0]

# Example usage of the predict_sentiment function
review_text = "This movie was amazing! I loved every bit of it."
sentiment_prediction = predict_sentiment(review_text)
print("Sentiment Prediction:", sentiment_prediction)  # Output: 'positive'

# Generating word cloud visualizations
generate_word_cloud(positive_reviews, "Word Cloud - Positive Reviews")
generate_word_cloud(negative_reviews, "Word Cloud - Negative Reviews")
```

## Credits

This sentiment analysis project is a result of the hard work and dedication of Rohit Poddar. It incorporates various NLP techniques and the Naive Bayes classifier to achieve accurate sentiment predictions from text data.

## Feedback and Contributions

Feedback, bug reports, and contributions are welcome and encouraged! If you find any issues or have suggestions for improvements, please feel free to create a GitHub issue or submit a pull request. This project is open to collaboration and aims to improve the understanding of sentiment analysis methods within the community.