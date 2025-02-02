from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import string
from gensim.models import KeyedVectors
import spacy
import streamlit as st

# Load the dataset
data = pd.read_csv(r'movies.csv')

# Strip any leading/trailing whitespace characters from the genre column
data['genre'] = data['genre'].str.strip()

# Data cleaning: Remove punctuation and convert to lowercase
data['description'] = data['description'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)).lower())

# Create TF-IDF vectors
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['description'])

# Encode the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['genre'])


# Ensure y is a 1D array
y = np.ravel(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the neural network model
model = Sequential()
model.add(Dense(512, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
y_pred = (model.predict(X_test) > 0.5).astype("int32")
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Function to classify a new movie description
def classify_description(description):
    description = description.translate(str.maketrans('', '', string.punctuation)).lower()
    description_vector = vectorizer.transform([description]).toarray()
    prediction = (model.predict(description_vector) > 0.5).astype("int32")
    return label_encoder.inverse_transform(prediction)[0]

# Streamlit app
st.title("Movie Genre Classifier")
st.write("Paste a movie description and determine whether it looks like a horror or a romantic movie.")

user_input = st.text_area("Movie Description")

if st.button("Classify"):
    if user_input:
        genre = classify_description(user_input)
        st.write(f"The movie description looks like a **{genre}** movie.")
    else:
        st.write("Please enter a movie description.")
