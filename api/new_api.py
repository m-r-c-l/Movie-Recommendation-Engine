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
    return dict(greeting = "Hello there. This is the root page of MRE!")



# Predict endpoint
@app.get("/predict")
def predict(user_id: int, top_n: int = 3):
    """
    Take a user_id and number of recommendations as input and return movie recommendations.
    """
    model = app.state.model
#    movies = app.state.movies



    assert model is not None, "Model is not loaded"

    # # OLD API THAT WORKED
    # # call the prediction function
    # recommendations = predict_from_storage(user_id).to_dict()

    # recom_dict = recommendations.get("Title")
    # title_list = [title for title in recom_dict.values()]

    # return title_list

    # call the prediction function
    cleaned_recommendations, watched_recommendations = get_recommendations_without_already_watched_and_user_history(user_id)

    # using this to counter error 500 i am getting for some reason
    cleaned_recommendations = cleaned_recommendations.replace([np.inf, -np.inf], np.nan).dropna()

    recommendations = cleaned_recommendations.to_dict()
    recom_dict = recommendations.get("Title")
    tmdb_movieid_dict = recommendations.get("tmdbId")

    title_list = [title for title in recom_dict.values()]
    tmdb_id_list = [id for id in tmdb_movieid_dict.values()]

    list_to_disp = title_list[:top_n]
    list_of_ids = tmdb_id_list[:top_n]


    return list_to_disp, list_of_ids
