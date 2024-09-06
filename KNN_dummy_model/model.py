# placeholder for our actual model, for now i use KNN

from sklearn.neighbors import NearestNeighbors
import pandas as pd
from KNN_dummy_model.load_data import load_data
from KNN_dummy_model.preprocess import preprocess_features

class MovieRecommender:
    """
    KNN model. Accepts a feature matrix corresponding to the movies
    """
    def __init__(self):
        """
        instantiator
        """
        self.model = NearestNeighbors(n_neighbors=2, algorithm='auto')


    def fit(self, movie_features_df: pd.DataFrame):
        """
        fit method
        """
        self.movie_features_df = movie_features_df
        self.model.fit(movie_features_df.drop("movieId", axis = 1))   # dropping movieId for irrelevance


    def predict(self, user_input: pd.DataFrame):    # for the setup that i have now, user_input is a df that contains the user id, the number of recommendations
                                                    # and the genre they'd like. This is what will be provided as parameters to the API
        """
        predict method
        """
        user_id = user_input['user_Id'][0]          # extracting user id
        genre = user_input['genre'][0]              # extracting desired genre by the user

        # turning user input genre into a feture vector
        # concantenating with a 0 because i can't figure out where i am losing one feature for now, DELETE LATER!
        genre_features = [int(genre == g) for g in ['Action', 'Comedy', 'Drama', 'Romance']] + [0]

        # Find closest movies based on the genre
        distances, indices = self.model.kneighbors([genre_features])
        similar_movies = self.movie_features_df.iloc[indices[0]]['movieId'].values

        return similar_movies[:user_input['numb_of_recommendations'][0]].tolist()

def load_model(input_data):
    """
    loads model to cache it through the API
    """
    # loading the enriched dataframe, called data here, and the feature matrix of all movies
    full_df = preprocess_features(input_data)
    data, movie_features_df = load_data(full_df)

    # instantiating recommender
    recommender = MovieRecommender()

    # fitting
    recommender.fit(movie_features_df)

    return recommender
