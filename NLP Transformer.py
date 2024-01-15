import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, GlobalMaxPooling1D, Dense, Dropout, SpatialDropout1D, Input, Attention, GlobalAveragePooling1D

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
categorical_labels = to_categorical(transformed_labels, num_classes=3)

# Define the Transformer Encoder
num_layers = 4
embed_dim = 128
num_heads = 8
ff_dim = 512
input_vocab_size = len(word_index) + 1
maximum_position_encoding = max_sequence_length

inputs = Input(shape=(max_sequence_length,))
embedding_layer = Embedding(input_dim=max_sequence_length, output_dim=embed_dim)(inputs)
spatial_dropout = SpatialDropout1D(0.5)(embedding_layer)

attention = Attention(use_scale=True)([spatial_dropout, spatial_dropout])

ffn = Dense(64, activation='relu')(attention)
ffn = Dense(embed_dim)(ffn)

add_norm = tf.keras.layers.Add()([embedding_layer, ffn])
add_norm = tf.keras.layers.LayerNormalization()(add_norm)

# Global Average Pooling
pooled_output = GlobalAveragePooling1D()(add_norm)

# Final classification layer
final_output = Dense(3, activation='softmax')(pooled_output)

# Define the Transformer model
transformer_model = Model(inputs=[inputs], outputs=final_output)

# Compile the model
transformer_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = transformer_model.fit(train_data_padded, categorical_labels, batch_size=32, epochs=10, validation_split=0.2)

# Load test data
test_data = pd.read_csv('test_no_label.csv')

# Preprocess test data
test_texts = test_data['review_description'].values
test_texts = preprocess_text(test_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)
test_data_padded = pad_sequences(test_sequences, maxlen=max_sequence_length)

# Make predictions using the Transformer model
transformer_predictions = transformer_model.predict(test_data_padded)

# Convert predictions to trinary classes
trinary_predictions = np.argmax(transformer_predictions, axis=1) - 1  # Shift to have (-1, 0, 1)

# Create a DataFrame with 'ID' and 'rating'
prediction_df = pd.DataFrame({'ID': range(1, len(transformer_predictions) + 1), 'rating': trinary_predictions})

# Write the DataFrame to a CSV file
prediction_df.to_csv('testing.csv', index=False)