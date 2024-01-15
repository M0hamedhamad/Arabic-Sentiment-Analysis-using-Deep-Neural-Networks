import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, Concatenate, Dropout, LayerNormalization
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

# Load training data
train_data = pd.read_excel('train.xlsx')


# Preprocessing function for text data
def preprocess_text(text):
    stop_words = set(stopwords.words('arabic'))
    stemmer = SnowballStemmer("arabic")

    cleaned_texts = []
    for txt in text:
        tokens = word_tokenize(txt)
        cleaned_tokens = [stemmer.stem(token.lower()) for token in tokens if
                          token.isalpha() and token.lower() not in stop_words]
        cleaned_texts.append(' '.join(cleaned_tokens))
    return cleaned_texts


# Preprocess training and validation texts
text_data = train_data['review_description'].values
val_labels = train_data['rating'].values
nltk.download('stopwords')
nltk.download('punkt')
train_texts = preprocess_text(text_data)

# Tokenization with Keras
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_texts)
train_sequences = tokenizer.texts_to_sequences(train_texts)

word_index = tokenizer.word_index
max_sequence_length = 1000  # Define your sequence length

# Padding sequences
train_data_padded = pad_sequences(train_sequences, maxlen=max_sequence_length)

# Map labels from (-1, 0, 1) to (0, 1, 2)
transformed_labels = val_labels + 1  # Shifts (-1, 0, 1) to (0, 1, 2)

embed_dim = 128
input_vocab_size = len(word_index) + 1

# Convert trinary labels to categorical for LSTM model
categorical_labels = to_categorical([x + 1 for x in val_labels], num_classes=3)  # Shift to have (0, 1, 2)

# LSTM Model
lstm_model = Sequential()
lstm_model.add(Embedding(input_vocab_size, embed_dim, input_length=max_sequence_length))
lstm_model.add(LSTM(128))
lstm_model.add(Dense(3, activation='softmax'))  # Output layer for trinary classification

# Compile the LSTM model with categorical cross-entropy loss
lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the LSTM model
lstm_model.fit(train_data_padded, np.array(categorical_labels), epochs=10, batch_size=32)

# Load test data
test_data = pd.read_csv('test_no_label.csv')

# Preprocessing test data
test_texts = test_data['review_description'].values
test_texts = preprocess_text(test_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)
test_data_padded = pad_sequences(test_sequences, maxlen=max_sequence_length)

# Make predictions using the LSTM model
lstm_predictions = lstm_model.predict(test_data_padded)

# Convert predictions to trinary classes
trinary_predictions = np.argmax(lstm_predictions, axis=1) - 1  # Shift to have (-1, 0, 1)

# Create a DataFrame with 'ID' and 'rating'
prediction_df = pd.DataFrame({'ID': range(1, len(lstm_predictions) + 1), 'rating': trinary_predictions})

# Save DataFrame to a CSV file
prediction_df.to_csv('lstm_prediction.csv', index=False)
