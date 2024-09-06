import pandas as pd

# importing model
from moviemain.model_logic.registry import load_model


# importing fastapi
from fastapi import FastAPI



# function that predicts movies based on the model loaded below
def predict_movie(model, user_id: int, top_n: int, movies):
    """
    Generates top N movie recommendations for a given user.

    Args:
    - model: Preloaded MovieModel.
    - user_id: The ID of the user to make predictions for.
    - top_n: Number of recommendations to return.
    - movies: The movie metadata used for the model (preloaded).

    Returns:
    - List of movie recommendations.
    """
    # Get predictions from the model (assuming it predicts top N movies for the user)
    # Here, you would use the model's prediction logic for the user_id
    predicted_scores = model.predict(user_id)  # Example of prediction logic


    top_movie_indices = predicted_scores.argsort()[-top_n:][::-1]

    # Retrieve movie information from the preloaded movies dataset
    recommendations = [movies[i] for i in top_movie_indices]

    return recommendations








# instantiating api
app = FastAPI()


# placeholder for preloading the model
# it is accessed faster from the second try and on
model, movies = load_model(epochs=5)        # training can be adjusted here

app.state.model = model
app.state.movies = movies
print("Model and movies cached successfully!")

# greeting endpoint
@app.get("/")
def root():
    # returns dictionary so that it can be in JSON form
    return dict(greeting = "Hello there general Kenobi")



# # placeholder predict endpoint
# @app.get("/predict")
# def predict(user_id: int,
#              top_n: int):

#     return dict(user = user_id)



# Predict endpoint
@app.get("/predict")
def predict(user_id: int, top_n: int = 3):
    """
    Take a user_id and number of recommendations as input and return movie recommendations.
    """
    model = app.state.model
    movies = app.state.movies

    assert model is not None, "Model is not loaded"

    # Call the prediction function
    recommendations = predict_movie(model=model, user_id=user_id, top_n=top_n, movies=movies)

    return {"recommendations": recommendations}
