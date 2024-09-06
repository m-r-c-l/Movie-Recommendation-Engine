import pandas as pd

# # dummy dataframe
# dummy_df = {
#         'userId': [1, 2, 3, 1, 2, 1],
#         'movieId': [101, 102, 103, 104, 101, 103],
#         'genres': ['Action,Comedy,Romance', 'Drama', 'Action,Drama', 'Comedy,Romance', 'Action,Comedy,Romance', 'Action,Drama'],
#         'ratings': [4.0, 5.0, 4.0, 3.0, 4.0, 4.5],
#         'Action': [1, 0, 1, 0, 1, 1],
#         'Comedy': [1, 0, 0, 1, 1, 0],
#         'Drama': [0, 1, 1, 0, 0, 1],
#         'Romance': [1, 0, 0, 1, 1, 0]
#     }

# placeholder function that will be loading a csv and turning it into a df
# or possibly a tensorflow tensor on this step. For now it just loads dummy data
def load_data(full_df: pd.DataFrame):
    """
    Loads data and does minimal feature engineering to find the per movie average rating
    as well as creating a feature matrix for each movie. A movie's feature vector will
    look something like:
    movieId: 112
    Action: 1
    Comedy: 1
    Drama: 0

    Returns the original loaded df and also the feature matrix of all movies.
    """

    # enriching input with average rating of movie
    average_ratings_df = full_df.groupby("movieId")["ratings"].mean().reset_index()
    average_ratings_df.columns = ['movieId', 'average_rating']

    # creating feature matrix
    movie_features_df = full_df[['movieId', 'Action', 'Comedy', 'Drama', 'Romance']].drop_duplicates()
    movie_features_df = movie_features_df.merge(average_ratings_df, on='movieId')

    return full_df, movie_features_df
