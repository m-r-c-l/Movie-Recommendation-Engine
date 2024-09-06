
import tensorflow as tf
import tensorflow_recommenders as tfrs

from typing import Dict, Text
from colorama import Fore, Style

from moviemain.model_logic.registry import save_recommender


class MovieModel(tfrs.models.Model):

    def __init__(self, movies, unique_movie_titles, unique_user_ids, rating_weight: float, retrieval_weight: float) -> None:
        # We take the loss weights in the constructor: this allows us to instantiate
        # several model objects with different loss weights.

        super().__init__()

        embedding_dimension = 32

        # User and movie models.
        self.movie_model: tf.keras.layers.Layer = tf.keras.Sequential([
          tf.keras.layers.StringLookup(
            vocabulary=unique_movie_titles, mask_token=None),
          tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
        ])
        self.user_model: tf.keras.layers.Layer = tf.keras.Sequential([
          tf.keras.layers.StringLookup(
            vocabulary=unique_user_ids, mask_token=None),
          tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
        ])

        # A small model to take in user and movie embeddings and predict ratings.
        # We can make this as complicated as we want as long as we output a scalar
        # as our prediction.
        self.rating_model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(1),
        ])

        # The tasks.
        self.rating_task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )
        self.retrieval_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=movies.batch(128).map(self.movie_model)
            )
        )

        # The loss weights.
        self.rating_weight = rating_weight
        self.retrieval_weight = retrieval_weight

    def call(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
        # We pick out the user features and pass them into the user model.
        user_embeddings = self.user_model(features["user_id"])
        # And pick out the movie features and pass them into the movie model.
        movie_embeddings = self.movie_model(features["movie_title"])

        return (
            user_embeddings,
            movie_embeddings,
            # We apply the multi-layered rating model to a concatentation of
            # user and movie embeddings.
            self.rating_model(
                tf.concat([user_embeddings, movie_embeddings], axis=1)
            ),
        )

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:

        ratings = features.pop("user_rating")

        user_embeddings, movie_embeddings, rating_predictions = self(features)

        # We compute the loss for each task.
        rating_loss = self.rating_task(
            labels=ratings,
            predictions=rating_predictions,
        )
        retrieval_loss = self.retrieval_task(user_embeddings, movie_embeddings)

        # And combine them using the loss weights.
        return (self.rating_weight * rating_loss
                + self.retrieval_weight * retrieval_loss)




    ## How do I get unique_movie_titles, unique_user_ids, ratings, and movies in?
def compile_model(movies, unique_movie_titles, unique_user_ids, learning_rate=0.1, rating_weight=1.0, retrieval_weight=1.0) -> MovieModel:
    """
    Compile the Neural Network
    """
    model = MovieModel(movies, unique_movie_titles, unique_user_ids, rating_weight, retrieval_weight)

    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate), run_eagerly=True)


    print("✅ Model compiled")

    return model


def train_model(
        model: MovieModel,
        cached_train,
        epochs
    ) -> tuple[MovieModel, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    #es = EarlyStopping(
    #    monitor="val_loss",
    #    patience=patience,
    #    restore_best_weights=True,
    #    verbose=1
    #)

    history = model.fit(
        cached_train,
        epochs=epochs
    )

    #print(f"✅ Model trained on {len(X)} rows with min val MAE: {round(np.min(history.history['val_mae']), 2)}")

    return model, history


def evaluate_model(
        model: MovieModel,
        cached_test: tf.data.Dataset,
    ) -> dict:
    """
    Evaluate trained model performance on the training dataset
    """

    #print(Fore.BLUE + f"\nEvaluating model on {len(X)} rows..." + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None, None, None

    metrics_dict = model.evaluate(cached_test, return_dict=True)

    print(f"✅ Model evaluated")
    print(f"\nRetrieval top-100 accuracy: {metrics_dict['factorized_top_k/top_100_categorical_accuracy']:.3f}")
    print(f"Ranking RMSE: {metrics_dict['root_mean_squared_error']:.3f}")

    return metrics_dict


def predict_movie(
    model,
    user_id,
    top_n=3,
    movies = None
    ):

    assert movies is not None, "Movies dataset must be provided."

    # Create a model that takes in raw query features, and

    recommender = tfrs.layers.factorized_top_k.BruteForce(model.user_model)

    # recommends movies out of the entire movies dataset.
    recommender.index_from_dataset(

      tf.data.Dataset.zip((movies.batch(100), movies.batch(100).map(model.movie_model)))
    )

    # Get recommendations.

    _, titles = recommender(tf.constant([str(user_id)]))


    recommendations = []
    for i, title in enumerate(titles[0, :top_n].numpy()):
        recommendations.append('{}. {}'.format(i+1, title.decode("utf-8")))

    save_recommender(recommender)

    return recommendations



#def predict_rating(user, movie):
    #trained_movie_embeddings, trained_user_embeddings, predicted_rating = model({
    #      "userId": np.array([str(user)]),
    #      "original_title": np.array([movie])
    #  })
    #print("Predicted rating for {}: {}".format(movie, predicted_rating.numpy()[0][0]))
