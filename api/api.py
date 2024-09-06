import pandas as pd

# placeholder imports for our model
from KNN_dummy_model.preprocess import preprocess_features
from KNN_dummy_model.model import MovieRecommender, load_model

# importing fastapi
from fastapi import FastAPI


# dummy data
dummy_data = {
    'userId': [1, 2, 3, 4, 1, 1],
    'movieId': [101, 102, 103, 104, 102, 103],
    'genres': ['["Action","Comedy","Romance"]', '["Drama"]', '["Action","Drama"]',
               '["Comedy","Romance"]', '["Drama"]', '["Action","Drama"]'],
    'ratings': [4, 3, 4, 5, 2, 5]
}

# instantiating api
app = FastAPI()



# preloading the model. The model will be cached into memory so that
# it is accessed faster from the second try and on
app.state.model = load_model(preprocess_features(dummy_data))



# greeting endpoint
@app.get("/")
def root():
    # returns dictionary so that it can be in JSON form
    return dict(greeting = "Hello")



# # placeholder predict endpoint
# @app.get("/predict")
# def predict(user_Id: int,
#             numb_of_recommendations: int,
#             genre: str):

#     return dict(user = user_Id)



# predict endpoint
@app.get("/predict")
def predict(user_Id: int,
            numb_of_recommendations: int,
            genre: str):
    """
    taking an integer user Id as input, the number of desired recommendations
    and the desired genre for them to watch
    """

    # for some reason does not work as i expected
    # X_pred = pd.DataFrame(locals())  # i believe this is supposed to be getting (see solution)
    #                                  # the parameters from the predict func
    #                                  # and the given values and making them
    #                                  # into a dictionary, basically saves time

    X_pred = pd.DataFrame([{
    'user_Id': user_Id,
    'numb_of_recommendations': numb_of_recommendations,
    'genre': genre
    }])

    # to debug, DELETE LATER!
    print("Input data for prediction:", X_pred)

    # placeholder for preprocessing if we need so
    # X_processed = preprocess_features(X_pred)

    model: MovieRecommender = app.state.model
    assert model is not None

    # placeholder for recommendations
    y_pred = model.predict(X_pred) # switch to X_processed if we preprocess

    return dict(predictions=y_pred)
