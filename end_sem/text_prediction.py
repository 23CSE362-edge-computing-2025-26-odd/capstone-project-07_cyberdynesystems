import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import logging

logger = logging.getLogger(__name__)

class DataHandler:
    """
    Handles data preprocessing using a 'sliding window' method.
    It expects a pre-fitted tokenizer.
    """
    def __init__(self, 
                 corpus_text: str,
                 shared_tokenizer: Tokenizer,
                 shared_total_words: int,
                 shared_max_seq_len: int):
        
        self.corpus = corpus_text.lower()
        self.tokenizer = shared_tokenizer
        self.total_words = shared_total_words
        
        # This is now our FIXED window size, e.g., 10
        self.window_size = shared_max_seq_len
        
        self.X = []
        self.y = []

    def preprocess(self):
        """
        Create input/output sequences using a sliding window.
        X = [word_1, word_2, ..., word_10]
        y = word_11
        """
        logger.info(f"Preprocessing with window size {self.window_size}...")
        
        # Convert the entire corpus to a flat list of word indices
        # We use texts_to_sequences on a list of lines to preserve structure
        token_list = []
        for line in self.corpus.split('\n'):
            if line.strip():
                # Get sequence for one line
                line_tokens = self.tokenizer.texts_to_sequences([line])[0]
                token_list.extend(line_tokens)

        if len(token_list) <= self.window_size:
            logger.warning("Corpus is too small for the window size. No data generated.")
            return

        # Create sliding window sequences
        for i in range(len(token_list) - self.window_size):
            # Input sequence (the window)
            seq_in = token_list[i : i + self.window_size]
            
            # Output sequence (the *next* word)
            seq_out = token_list[i + self.window_size]
            
            self.X.append(seq_in)
            self.y.append(seq_out)

        if not self.X:
            logger.warning("No sequences were generated.")
            return

        logger.info(f"Total sequences generated: {len(self.X)}")
        
        # Convert to numpy arrays
        self.X = np.array(self.X)
        
        # --- CHANGE 1: DO NOT one-hot encode. Just make a numpy array ---
        self.y = np.array(self.y)

    def get_data(self):
        # Ensure data is not empty before returning
        if len(self.X) == 0:
            return np.array([]), np.array([])
        return self.X, self.y

    def get_params(self):
        return self.total_words, self.window_size


class LSTMModel:
    """
    Builds the Keras LSTM model.
    """
    def __init__(self, embedding_dim=50, lstm_units_1=100):
        self.embedding_dim = embedding_dim
        self.lstm_units_1 = lstm_units_1
        self.model = None

    def build_model(self, total_words, max_seq_len):
        """
        Builds the model. max_seq_len is our window size.
        """
        self.model = Sequential()
        
        # The input_length is now our fixed window size
        self.model.add(Embedding(input_dim=total_words, 
                                 output_dim=self.embedding_dim, 
                                 input_length=max_seq_len))
        
        self.model.add(LSTM(self.lstm_units_1, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(self.lstm_units_1))
        self.model.add(Dropout(0.2))
        
        # Output layer predicts one of the total_words
        self.model.add(Dense(total_words, activation='softmax'))
        
        # --- NEW: Define an optimizer with a 10x smaller learning rate ---
        adam_optimizer = Adam(learning_rate=0.0001) 

        self.model.compile(loss='sparse_categorical_crossentropy', 
                           optimizer=adam_optimizer, # <-- Use the new optimizer
                           metrics=['accuracy'])
        self.model.summary()
