import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np

# importing model
from moviemain.model_logic.registry import load_model, load_recommender
from moviemain.interface.main import *

import sys
sys.setrecursionlimit(1000)

# importing fastapi
from fastapi import FastAPI



# # function that predicts movies based on the model loaded below
# def predict_movie(model, user_id: int, top_n: int, movies):
#     # get predictions from the model (assuming it predicts top N movies for the user)
#     predicted_scores =predict(user_id, top_n)


#     top_movie_indices = np.argpartition(predicted_scores, -top_n)[-top_n:]

# #    top_movie_indices = top_movie_indices[np.argsort(-predicted_scores[top_movie_indices])]

#     # retrieve movie information from the preloaded movies dataset
#     recommendations = [movies[i] for i in top_movie_indices]

#     return recommendations



# instantiating api
app = FastAPI()


# placeholder for preloading the model
# it is accessed faster from the second try and on
       # training can be adjusted here
model = load_recommender()

if model is None:
    model = load_model()

app.state.model = model
# app.state.movies = movies
print("Model and movies cached successfully!")

# greeting endpoint
@app.get("/")
def root():
    # returns dictionary so that it can be in JSON form
    return dict(greeting = "Hello there general Kenobi!")



# Predict endpoint
@app.get("/predict")
def predict(user_id: int, top_n: int = 3):
    """
    Take a user_id and number of recommendations as input and return movie recommendations.
    """
    model = app.state.model
#    movies = app.state.movies



    assert model is not None, "Model is not loaded"

    # Call the prediction function
    recommendations = predict_from_storage(user_id = 1337).to_dict()

    return {"recommendations": recommendations}
