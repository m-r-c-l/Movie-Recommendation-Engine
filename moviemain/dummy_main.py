from fastapi import FastAPI
from pydantic import BaseModel
from dummy_model import MovieRecommender
from fastapi.responses import JSONResponse
import pandas as pd

app = FastAPI()

# Dummy data to simulate a trained model
data = {
    "UserId": [1, 2, 3, 4, 5],
    "MovieId": [101, 102, 103, 104, 105],
    "rating": [5, 4, 3, 5, 4],
    "genre": ["Action", "Drama", "Comedy", "Action", "Drama"]
}
df = pd.DataFrame(data)

# instead of the above dummy df we could import the preprocessed df as follows

# path = <something_meaningful>
# data_input = load_data(path)

# # preprocessing
# df = preprocess_data(data_input)


# Initialize and fit the recommender
recommender = MovieRecommender()
recommender.fit(df)

# Pydantic model to validate input data
class UserInput(BaseModel):
    genre: str

@app.post("/predict/")    # decorator that registers a POST endpoint at /predict/ and it executes the predict below
def predict(input: UserInput):    # input here is an instance of the UserInput class, which validates the input
    try:
        indices = recommender.predict(input.genre)    # calls the predict method of the MovieRecommender, with the input genre as input
                                                      # indices is a np.array of the nearest neighbors
        return {"recommended_movies": indices.tolist()}    # converting indices to a list

    # catches errors but I am not sure what it catches exactly
    except ValueError as e:
        return JSONResponse(status_code=400, content={"message": str(e)})
