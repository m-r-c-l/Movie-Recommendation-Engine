import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

from colorama import Fore, Style
from api_third_try.model_logic.basic_model import MovieModel, compile_model, train_model, evaluate_model



## Get preprocessed data ##
# Potential to-do's:
    # 1) define what dataset to load 100k, 1m, 20m, etc. ✅
    # 2) caching/storing data in the cloud?
    # 3) preprocessing for enriched dataset
def preprocess(dataset='100k') -> None:
    """
    - Load the data set (100k by default for development purposes)
        - Other options:
            - '1m': tested already and it works (but takes a bit longer)
            - Overview of TF Movielens datasets can be found here:
                https://www.tensorflow.org/datasets/catalog/movielens
    - Preprocess data
    """

    ## Get data
    ratings_path = "movielens/"+dataset+"-ratings"
    movies_path = "movielens/"+dataset+"-movies"
    ratings = tfds.load(ratings_path, split="train")
    movies = tfds.load(movies_path, split="train")

    ## Process data (for basic model)
    # For advanced model it becomes more complex)
    ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
    "user_ratings": float(x["user_rating"])
    })
    movies = movies.map(lambda x: x["movie_title"])

    ## Caching ???
    # For now, we just return the data and don't store it.
    # The data is fetched in train() again

    print("✅ preprocess() done \n")

    return ratings, movies



## Train the model ##
# Potential to-dos:
    # 1) Load data from storage?
    # 2) Load model from storage?
    # 3) Return statement (metric?)
    # 4) Hyper-parameters to pass/expose (e.g. learning_rate, patience)
def train(seed=42,
          split=0.2,
          batch_size=1024,
          epochs=1,
          learning_rate = 0.1,
          rating_weight=1.0,
          retrieval_weight=1.0
          ) -> float:
    '''
    --
    '''

    ## Load data (via preprocess() defined above) ##
    # For later, load it from some storage?
    ratings, movies = preprocess()

    ## Create (X_train_processed, y_train, X_val_processed, y_val) ##

    # Params
    num_obs = len(ratings)
    #shuffle_size = num_obs # all observations are going to be shuffeled (see comment below)
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
    ## shuffled = ratings.shuffle(shuffle_size, seed=seed, reshuffle_each_iteration=False)
    ## Check: Why is this line in the code? It's not actually used anywhere?

    # Split dataset in train and test
    train = ratings.take(train_size)
    test = ratings.skip(train_size).take(test_size)


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

    print("✅ train() done \n")
    print(history.history)

    return history


## Evaluate the model ##
# Potential to-dos:
    # 1) Fix setup to make it work
    # 2) Load and save model
def evaluate(#model, pass like this or call via train()?
            #cached_test, pass like this or call via train()?
            #stage: str = "Production"
    ) -> tuple[float, float]:
    """
    Evaluate the performance of the (latest production) model on processed data
    Return factorized_top_100 and rmse as a tuple
    """
    print(Fore.MAGENTA + "\n⭐️ Use case: evaluate" + Style.RESET_ALL)

    #model = load_model(stage=stage)
    #assert model is not None



    ## To circumvent not having load_model yet
    #model = train(blablabla)

    #metrics_dict = evaluate_model(model=model, cached_test=cached_test)
    #factorized_top_100 = metrics_dict['factorized_top_k/top_100_categorical_accuracy']
    #rmse = metrics_dict['root_mean_squared_error']

    #save_results(params=params, metrics=metrics_dict)

    #print("✅ evaluate() done \n")

    #return (factorized_top_100, rmse)
    pass


def pred(X_pred) -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """

    print("\n⭐️ Use case: predict")
    pass

    """if X_pred is None:
        X_pred = pd.DataFrame(dict(
        pickup_datetime=[pd.Timestamp("2013-07-06 17:18:00", tz='UTC')],
        pickup_longitude=[-73.950655],
        pickup_latitude=[40.783282],
        dropoff_longitude=[-73.984365],
        dropoff_latitude=[40.769802],
        passenger_count=[1],
    ))"""

    #model = load_model()
    #assert model is not None

    #X_processed = preprocess_features(X_pred)
    #y_pred = model.predict(X_processed)

    #print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")
    #return y_pred


if __name__ == '__main__':
    #preprocess(min_date='2009-01-01', max_date='2015-01-01')
    #train(min_date='2009-01-01', max_date='2015-01-01')
    #evaluate(min_date='2009-01-01', max_date='2015-01-01')
    #pred()
    print("Test")
    pass
