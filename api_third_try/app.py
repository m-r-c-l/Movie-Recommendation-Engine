import streamlit as st
import requests

# Title for the Streamlit app
st.title("Movie Recommender System")

list_of_genres = ["Action","Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama",
              "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

# Dropdown to select a genre
genre = st.selectbox("Choose a genre", list_of_genres)

if st.button('Recommend'):          # creates a button labeled Recommend
    url = "http://localhost:8000/predict/"      # local endpoint
    # Call FastAPI backend
    response = requests.post(url, json={"genre": genre})    # this sends and HTTP POST request, as json in the form {"genre": <whatever picked>}

    # Check if the response is successful
    if response.status_code == 200:
        recommended_movies = response.json().get("recommended_movies", [])  # converts the response from json to a dictionary, accessing the recommended_movies key
#                                                                           # if the key does not exist it returns the list (in this case empty)
        st.write('Recommended Movies:', recommended_movies)
    else:
        st.write("Error:", response.json().get("message", "Unknown error")) # returns error message
