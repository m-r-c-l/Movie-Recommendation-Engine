from dotenv import load_dotenv
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import tensorflow_datasets as tfds

from colorama import Fore, Style


from moviemain.model_logic.basic_model import MovieModel, compile_model, train_model, evaluate_model, predict_movie
from moviemain.model_logic.registry import load_model, save_model, save_results, load_recommender ## NEW
from moviemain.model_logic.registry import mlflow_run, mlflow_transition_model ## NEW

load_dotenv()


def preprocess(dataset=os.getenv('DATA_SIZE')) -> None:
    """
    - ✅ Load the data set (100k by default for development purposes)
        - Other arguments to pass:
            - ✅ '1m' for 1 million user ratings: tested and works.
            - 📌 Overview of other TF Movielens datasets available can be found
                    here: https://www.tensorflow.org/datasets/catalog/movielens
    - ✅ Preprocess data for basic model (i.e. creating mappings)
    - ❌ Preprocess data for advanced model (i.e. feature eng, scaling)
    - ❌ Caching/storing of data
    """

    ## Get data
    ratings_path = "movielens/"+dataset+"-ratings"
    movies_path = "movielens/"+dataset+"-movies"
    ratings = tfds.load(ratings_path, split="train")
    movies = tfds.load(movies_path, split="train")

    ## Process data (for basic model)
    ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
    "user_rating": float(x["user_rating"])})

    movies = movies.map(lambda x: x["movie_title"])


    ## Process data (for advanced model)
    # tba

    ## Cache data
    # tba
    #=> For now, we just return the data and don't store it.
    #=> The data is instead fetched via process() in train()
    #=> However, it looks like the data is cached by default.
    #=> So if process() is run twice, it doesn't download the data again
    #=> Instead it loads it from cache (?)

    print("\n✅ Preprocess() completed sucessfully! \n")

    return ratings, movies

def train(seed=42,
          split=0,
          batch_size=1_024,
          epochs=3,
          learning_rate = 0.05,
          rating_weight=1.0,
          retrieval_weight=1.0,
          reshuffle_each_iteration = False,
          ) -> float:
    """
    - ✅ Splits dataset in train and test
    - ✅ Creates batches
    - ✅ Creates vocabulary for tokenization
    - ✅ Caches
    - ✅ Compiles model via model_logic.basic_model.py compile()
    - ✅ Trains model via model_logic.basic_model.py train()
    - ✅ Returns history (and other output needed for upstream functions
            (e.g. evaluate() and predict() as saving is not implemented yet)
    - ✅ Exposes first set of hyper parameters for tuning
    - ❌ Exposes all necessary hyper parameters for tuning
            => requires investigation (e.g. gridsearch?) and implementation
    - ❌ Loads data from storage if available, else run process()
    - ❌ Loads model from storage if available, else instanciate()
    - ❌ Saves model to storage
    - ❌ Saves results to storage
    """

    ## Load data (via preprocess() defined above) ##
    #❓For later, load it from some storage?
    ratings, movies = preprocess()

    ## Create (X_train_processed, y_train, X_val_processed, y_val) ##
    #❓How could we introduce a valiation set into our model?

    # Params
    num_obs = len(ratings)
    shuffle_size = num_obs # the whole dataset is shuffled at once
    train_size = int(num_obs * (1 - split))
    test_size = num_obs - train_size

    print('Total_Data: {}'.format(num_obs))
    print('Shuffle_Size: {}'.format(num_obs))
    print('Train_Size: {}'.format(train_size))
    print('Test_Size: {}'.format(test_size))

    # Define random seed
    tf.random.set_seed(seed)
    print(f'Seed: {seed}')

    # Shuffle all observations based on shuffle_size and seed
    # => should prevent the model from learning the order of the data
    #shuffled = ratings.shuffle(shuffle_size, seed=seed, reshuffle_each_iteration=reshuffle_each_iteration) # removed for now
    shuffled = ratings

    # Split dataset in train and test
    train = shuffled.take(train_size)
    test = shuffled.skip(train_size).take(test_size)

    ## Create batches
    movie_titles = movies.batch(batch_size)
    user_ids = ratings.batch(batch_size).map(lambda x: x["user_id"])

    ## Get vocabulary for tokenization
    unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
    unique_user_ids = np.unique(np.concatenate(list(user_ids)))

    print('Unique Movies: {}'.format(len(unique_movie_titles)))
    print('Unique users: {}'.format(len(unique_user_ids)))



    ### Train model using 'model.py' ###

    ## Loading model from local registry path ##

    #model = load_model() ## NEW

    #if model is None: ## NEW
    #    model = compile_model(movies,
    #                      unique_movie_titles,
    #                      unique_user_ids,
    #                      learning_rate=learning_rate,
    #                      rating_weight=rating_weight,
    #                      retrieval_weight=retrieval_weight
    #                      )

    model = compile_model(movies,
                          unique_movie_titles,
                          unique_user_ids,
                          learning_rate=learning_rate,
                          rating_weight=rating_weight,
                          retrieval_weight=retrieval_weight
                          )

    ## Cache train and test
    cached_train = train.batch(batch_size).cache()
    cached_test = test.batch(batch_size).cache()

    ## Train model
    model, history = train_model(
        model=model,
        cached_train=cached_train,
        epochs=epochs)


    params = dict(
        context="train",
        training_set_size=train_size, # not sure if needed, kept in for testing for now
        row_count=train_size # not sure if needed, kept in for testing for now
    )

    training_metrics = {
    'training_rmse': history.history['root_mean_squared_error'][-1],
    'training_loss': history.history['loss'][-1],
    'top_1_accuracy': history.history['factorized_top_k/top_1_categorical_accuracy'][-1],
    'top_5_accuracy': history.history['factorized_top_k/top_5_categorical_accuracy'][-1],
    'top_10_accuracy': history.history['factorized_top_k/top_10_categorical_accuracy'][-1],
    'top_50_accuracy': history.history['factorized_top_k/top_50_categorical_accuracy'][-1],
    'top_100_accuracy': history.history['factorized_top_k/top_100_categorical_accuracy'][-1],
    }

    # Save results on the hard drive using taxifare.ml_logic.registry
    save_results(params=params, metrics=training_metrics)

    # Save model weight on the hard drive (and optionally on GCS too!)
    #save_model(model=model)

    # The latest model should be moved to staging
    if os.getenv('MODEL_TARGET') == 'mlflow':
        mlflow_transition_model(current_stage="None", new_stage="Staging")

    print("✅ Train() completed successfully! \n")
    print(history.history)

    return model, cached_test, history, movies, train_size, test_size


def evaluate(#model, #pass like this or call via train()?
            #cached_test,# pass like this or call via train()?
            stage: str = "Production" ## Why is this needed?

    ) -> tuple[dict, pd.DataFrame]:
    """
    Evaluate the performance of the (latest production) model on processed data
    Return factorized_top_100 and rmse as a tuple
    """
    print(Fore.MAGENTA + "\n⭐️ Use case: Evaluate" + Style.RESET_ALL)

    ## Using load_model instead of calling train()
    #model = load_model(stage=stage)
    #assert model is not None
    #
    #=> get cached test????
    #metrics_dict = evaluate_model(model=model, cached_test=cached_test)


    ## To circumvent not having load_model yet ##
    model, cached_test, history, movies, train_size, test_size = train()

    metrics_dict = evaluate_model(model=model, cached_test=cached_test)


    training_metrics = { ### this can be deleted later and we fetch training_metrics instead returned from train()

        'Training RMSE': history.history['root_mean_squared_error'][-1],
        'Training Loss': history.history['loss'][-1],
        'Top 1 Accuracy': history.history['factorized_top_k/top_1_categorical_accuracy'][-1],
        'Top 5 Accuracy': history.history['factorized_top_k/top_5_categorical_accuracy'][-1],
        'Top 10 Accuracy': history.history['factorized_top_k/top_10_categorical_accuracy'][-1],
        'Top 50 Accuracy': history.history['factorized_top_k/top_50_categorical_accuracy'][-1],
        'Top 100 Accuracy': history.history['factorized_top_k/top_100_categorical_accuracy'][-1],
    }

    evaluation_metrics = {
        'Evaluation RMSE': metrics_dict['root_mean_squared_error'],
        'Evaluation Loss': metrics_dict['loss'],
        'Top 1 Accuracy (Eval)': metrics_dict['factorized_top_k/top_1_categorical_accuracy'],
        'Top 5 Accuracy (Eval)': metrics_dict['factorized_top_k/top_5_categorical_accuracy'],
        'Top 10 Accuracy (Eval)': metrics_dict['factorized_top_k/top_10_categorical_accuracy'],
        'Top 50 Accuracy (Eval)': metrics_dict['factorized_top_k/top_50_categorical_accuracy'],
        'Top 100 Accuracy (Eval)': metrics_dict['factorized_top_k/top_100_categorical_accuracy'],
    }


    params = dict(
        context="evaluate", # Package behavior
        training_set_size=test_size, # not sure if needed, kept in for testing for now
        row_count=test_size  # not sure if needed, kept in for testing for now
    )

    save_results(params=params, metrics=metrics_dict)


    # Combine both sets of metrics into a DataFrame
    eval_vs_train_df = pd.DataFrame({
        'Metric': training_metrics.keys(),
        'Training': training_metrics.values(),
        'Evaluation': evaluation_metrics.values()
    })

    print("\n✅ evaluate() done \n")
    print("📊🔍 Comparison of Training and Evaluation Results:")
    print(eval_vs_train_df.to_string(index=False))

    return metrics_dict


def predict(user_id=123, top_n=50) -> np.ndarray:
    """
    Make a prediction using the (latest) trained model
    """

    print("\n⭐️ Use case: Predict")

    #model = load_model()
    #assert model is not None

    ## To circumvent not having load_model yet

    model, cached_test, history, movies, train_size, test_size = train()

    ## Predict
    recommendations = predict_movie(model=model,
                                    user_id=user_id,
                                    top_n=top_n,
                                    movies=movies)

    print(f"\n✅ Prediction completed successfully!\n")
    print(f"🎯 Top {top_n} recommendations for user {user_id}:\n")
    print("\n".join(f"🔹 {rec}" for rec in recommendations))

    return recommendations

def predict_from_storage(user_id=123) -> pd.DataFrame:
    recommender = load_recommender()

    scores, titles = recommender([str(user_id)])

    df_recommendations = pd.DataFrame({
        'Title': titles.numpy().astype(str)[0],
        'Score': scores.numpy()[0]
    })

    print(f"\n✅ Prediction from loaded recommender completed successfully!\n")
    print(f"🎯 Top {len(df_recommendations)} recommendations for user {user_id}:\n")
    for index, row in df_recommendations.iterrows():
        print(f"🔹 {row['Title']} with a score of {row['Score']:.4f}")

    return df_recommendations


def predict_from_storage_and_get_user_history(user_id=123) -> pd.DataFrame:
    recommender = load_recommender()

    scores, titles = recommender([str(user_id)])

    df_recommendations = pd.DataFrame({
        'Title': titles.numpy().astype(str)[0],
        'Score': scores.numpy()[0]
    })

    print(f"\n✅ Prediction from loaded recommender completed successfully!\n")
    print(f"🎯 Top {len(df_recommendations)} recommendations for user {user_id}:\n")
    for index, row in df_recommendations.iterrows():
        print(f"🔹 {row['Title']} with a score of {row['Score']:.4f}")

    return df_recommendations


def get_users_viewing_and_rating_history(user_id=123) -> pd.DataFrame:

    ## Get the movies the user watched and his ratings for the movies ##
    ratings, movies = preprocess()

    user_viewing_history = ratings.filter(lambda x: x['user_id'] == tf.constant(str(user_id)))

    # Initialize empty lists for movie titles and user ratings
    movie_titles = []
    user_ratings = []

    # Iterate through the filtered dataset to access the records
    for record in user_viewing_history:
        movie_titles.append(record['movie_title'].numpy().decode('utf-8'))
        user_ratings.append(record['user_rating'].numpy())

    # Create the DataFrame for the user viewing history
    df_user_viewing_history = pd.DataFrame({
        'Title': movie_titles,
        'User Rating': user_ratings
    })

    return df_user_viewing_history



def get_recommendations_without_already_watched_and_user_history(user_id=123) -> pd.DataFrame:
        # Load the recommender and get recommendations
    recommender = load_recommender()
    scores, titles = recommender([str(user_id)])

    # Convert the recommendations into a DataFrame
    df_recommendations = pd.DataFrame({
        'Title': titles.numpy().astype(str)[0],
        'Score': scores.numpy()[0]
    })

    # Get the user's viewing history
    df_user_viewing_history = get_users_viewing_and_rating_history(user_id)

    # Filter out movies that the user has already watched
    df_filtered_recommendations = df_recommendations[~df_recommendations['Title'].isin(df_user_viewing_history['Title'])]

    print(f"\n✅ Prediction from loaded recommender completed successfully!\n")
    print(f"🎯 Top {len(df_filtered_recommendations)} filtered recommendations for user {user_id}:\n")
    for index, row in df_filtered_recommendations.iterrows():
        print(f"🔹 {row['Title']} with a score of {row['Score']:.4f}")

    return df_filtered_recommendations, df_user_viewing_history

if __name__ == '__main__':
    #preprocess()
    #train()
    #evaluate()
    predict() ## right now predict runs through all steps as save/load is not activated yet
    predict_from_storage()
