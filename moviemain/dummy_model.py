import pandas as pd
from sklearn.neighbors import NearestNeighbors

class MovieRecommender:
    def __init__(self):
        self.model = NearestNeighbors(n_neighbors=5)
        self.genres = None  # Will hold the genre columns used during fitting

    def fit(self, df):
        # One-hot encode the 'genre' column
        genre_matrix = pd.get_dummies(df['genre'])
        self.genres = genre_matrix.columns  # Save the column names
        self.model.fit(genre_matrix)

    def predict(self, genre):
        # One-hot encode the input genre
        genre_data = pd.get_dummies([genre])
        # Ensure all columns used during fit are present in the new data
        genre_data = genre_data.reindex(columns=self.genres, fill_value=0)

        # Use the KNN model to find the nearest neighbors
        distances, indices = self.model.kneighbors(genre_data)
        return indices.flatten()  # Return as a 1D array
