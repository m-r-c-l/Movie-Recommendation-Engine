import tensorflow as tf
import pandas as pd
import numpy as np
import tensorflow_datasets as tfds

from colorama import Fore, Style
from moviemain.model_logic.basic_model import MovieModel, compile_model, train_model, evaluate_model, predict_movie


def preprocess(dataset='100k') -> None:
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
    "user_ratings": float(x["user_rating"])
    })
    movies = movies.map(lambda x: x["movie_title"])


    ## Process data (for advanced model)
    # tba

    ## Cache data
    # tba
    #=> For now, we just return the data and don't store it.
    #=> The data is instead fetched via process() in train()

    print("\n✅ Preprocess() completed sucessfully! \n")

    return ratings, movies

def train(seed=42,
          split=0.2,
          batch_size=1024,
          epochs=1,
          learning_rate = 0.1,
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
    shuffled = ratings.shuffle(shuffle_size, seed=seed, reshuffle_each_iteration=reshuffle_each_iteration)

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
    # => This needs work to set up. See load_model() in registry.py
    # => This needs even more work to set up when we want to load non-local models
    #model = load_model()

    #if model is None:
    #    model = initialize_model(input_shape=X_train_processed.shape[1:])
                    # Instead =>? compile_model(MovieModel, learning_rate=0.1, rating_weight=1.0, retrieval_weight=1.0)

    ## Compile model
    model = compile_model(movies,
                          unique_movie_titles,
                          unique_user_ids,
                          learning_rate=learning_rate,
                          rating_weight=rating_weight,
                          retrieval_weight=retrieval_weight
                          )

    ## Cache shuffled train and test
    cached_train = train.shuffle(train_size).batch(batch_size).cache()
    cached_test = test.batch(batch_size).cache()

    ## Train model
    model, history = train_model(
        model=model,
        cached_train=cached_train,
        epochs=epochs)

    print("✅ Train() completed successfully! \n")
    print(history.history)

    return model, cached_test, history, movies


## Evaluate the model ##
# Potential to-dos:
    # 1) Fix setup to make it work ✅
    # 2) Load and save model
def evaluate(#model, #pass like this or call via train()?
            #cached_test,# pass like this or call via train()?
            #stage: str = "Production"
    ) -> tuple[dict, pd.DataFrame]:
    """
    Evaluate the performance of the (latest production) model on processed data
    Return factorized_top_100 and rmse as a tuple
    """
    print(Fore.MAGENTA + "\n⭐️ Use case: Evaluate" + Style.RESET_ALL)




    #model = load_model(stage=stage)
    #assert model is not None



    ## To circumvent not having load_model yet
    model, cached_test, history, movies = train()

    metrics_dict = evaluate_model(model=model, cached_test=cached_test)

    #save_results(params=params, metrics=metrics_dict)

    training_metrics = {
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

    # Combine both sets of metrics into a DataFrame
    eval_vs_train_df = pd.DataFrame({
        'Metric': training_metrics.keys(),
        'Training': training_metrics.values(),
        'Evaluation': evaluation_metrics.values()
    })

    print("\n✅ evaluate() done \n")
    print("📊🔍 Comparison of Training and Evaluation Results:")
    print(eval_vs_train_df.to_string(index=False))

    ## How about returning a table that compares training vs. evaluating data? From history

    return metrics_dict


def predict(user_id=1337, top_n=10) -> np.ndarray:
    """
    Make a prediction using the (latest) trained model
    """

    print("\n⭐️ Use case: Predict")

    #model = load_model()
    #assert model is not None

    ## To circumvent not having load_model yet
    model, cached_test, history, movies = train()

    ## Predict
    recommendations = predict_movie(model=model,
                                    user_id=user_id,
                                    top_n=top_n,
                                    movies=movies)

    print(f"\n✅ Prediction completed successfully!\n")
    print(f"🎯 Top {top_n} recommendations for user {user_id}:\n")
    print("\n".join(f"🔹 {rec}" for rec in recommendations))

    return recommendations




















if __name__ == '__main__':
    #preprocess(min_date='2009-01-01', max_date='2015-01-01')
    #train(min_date='2009-01-01', max_date='2015-01-01')
    #evaluate(min_date='2009-01-01', max_date='2015-01-01')
    #pred()
    print("Test")
    pass
