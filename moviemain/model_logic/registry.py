
import glob
import os
import time
import pickle

from colorama import Fore, Style
from tensorflow import keras
import tensorflow as tf
from google.cloud import storage

from moviemain.params import *
import mlflow
from mlflow.tracking import MlflowClient

def save_results(params: dict, metrics: dict) -> None:
    """
    Persist params & metrics locally on the hard drive at
    "{LOCAL_REGISTRY_PATH}/params/{current_timestamp}.pickle"
    "{LOCAL_REGISTRY_PATH}/metrics/{current_timestamp}.pickle"
    - (unit 03 only) if MODEL_TARGET='mlflow', also persist them on MLflow
    """
    if MODEL_TARGET == "mlflow":
        if params is not None:
            mlflow.log_params(params)
        if metrics is not None:
            mlflow.log_metrics(metrics)
        print("✅ Results saved on MLflow")

    timestamp = time.strftime("%Y%m%d-%H%M%S")


    params_dir = os.path.join(LOCAL_REGISTRY_PATH, "params")
    metrics_dir = os.path.join(LOCAL_REGISTRY_PATH, "metrics")

    os.makedirs(params_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Save params locally
    if params is not None:
        params_path = os.path.join(params_dir, timestamp + ".pickle")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # Save metrics locally
    if metrics is not None:
        metrics_path = os.path.join(metrics_dir, timestamp + ".pickle")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    print("✅ Results saved locally")

def save_recommender(recommender = None) -> None:
    """
    ✅ Persist fitted recommender model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/recommender_models/{timestamp}/"
    ✅ if MODEL_TARGET='gcs', also persist it in your bucket on GCS at "recommender_models/{timestamp}/ (NEEDS TESTING)
    ❌ if MODEL_TARGET='mlflow', also persist it on MLflow instead of GCS
    """

    # Save locally
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    recommender_path = os.path.join(LOCAL_REGISTRY_PATH, "recommender_models", f"{timestamp}")

    recommender.save(recommender_path)

    print("✅ Recommender saved locally")

    # Upload to GCS if required
    if MODEL_TARGET == "gcs":  ### needs testing
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)

        # Upload all files from the saved model directory to GCS
        for root, dirs, files in os.walk(recommender_path):
            for file in files:
                local_file_path = os.path.join(root, file)

                # Create relative path for GCS
                relative_path = os.path.relpath(local_file_path, recommender_path)
                blob_path = f"recommender_models/{timestamp}/{relative_path}"

                # Upload each file to the appropriate GCS path
                blob = bucket.blob(blob_path)
                blob.upload_from_filename(local_file_path)
                print(f"✅ Uploaded {local_file_path} to {blob_path} in GCS")

        print("✅ Recommender saved to GCS")

        return None



def load_recommender() -> None:
    """
    Return a saved model:
    ✅ locally (latest one in alphabetical order)
    ✅ or from GCS (most recent one) if MODEL_TARGET=='gcs' (NEEDS TESTING)
    ❌ or from MLFLOW (by "stage") if MODEL_TARGET=='mlflow'

    ✅ Return None (but do not Raise) if no model is found

    """

    if MODEL_TARGET == "local":
        print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

        # Get the latest model version name by the timestamp on disk
        local_recommender_directory = os.path.join(LOCAL_REGISTRY_PATH, "recommender_models")
        local_recommender_paths = glob.glob(f"{local_recommender_directory}/*")

        if not local_recommender_paths:
            return None

        most_recent_recommender_path_on_disk = sorted(local_recommender_paths)[-1]

        print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)

        latest_recommender = tf.saved_model.load(most_recent_recommender_path_on_disk)

        print("✅ Latest recommender loaded from local disk")

        return latest_recommender

    elif MODEL_TARGET == "gcs":
        print(Fore.BLUE + f"\nLoad latest model from GCS..." + Style.RESET_ALL)

        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)

        # List all blobs with the "recommender_model" prefix
        blobs = list(bucket.list_blobs(prefix="recommender_models"))

        try:
            # Find the latest updated blob (this might be part of the model directory structure)
            latest_blob_prefix = max(blobs, key=lambda x: x.updated).name.split("/")[1]

            # Create a local path to save the model directory
            latest_recommender_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, "recommender_models", latest_blob_prefix)
            os.makedirs(latest_recommender_model_path_to_save, exist_ok=True)

            # Download all blobs (files) that belong to the latest model directory
            for blob in blobs:
                if blob.name.startswith(f"recommender_models/{latest_blob_prefix}"):
                    local_file_path = os.path.join(LOCAL_REGISTRY_PATH, blob.name)
                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                    blob.download_to_filename(local_file_path)
                    print(f"✅ Downloaded {blob.name} to {local_file_path}")

            # Load the model from the downloaded directory
            latest_recommender = tf.saved_model.load(latest_recommender_model_path_to_save)

            print("✅ Latest model downloaded from cloud storage")

            return latest_recommender
        except Exception as e:
            print(f"\n❌ No model found in GCS bucket {BUCKET_NAME} or an error occurred: {e}")

            return None
    else:
        return None



def save_model(model: keras.Model = None) -> None:
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    - if MODEL_TARGET='gcs', also persist it in your bucket on GCS at "models/{timestamp}.h5" --> unit 02 only
    - if MODEL_TARGET='mlflow', also persist it on MLflow instead of GCS (for unit 0703 only) --> unit 03 only
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}")
    model.save(model_path, save_format='tf')

    print("✅ Model saved locally")

    if MODEL_TARGET == "gcs":

        model_filename = model_path.split("/")[-1] # e.g. "20230208-161047.h5" for instance
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"models/{model_filename}")
        blob.upload_from_filename(model_path)

        print("✅ Model saved to GCS")

        return None

    if MODEL_TARGET == "mlflow":
        mlflow.tensorflow.log_model(
            model=model,
            artifact_path="model",
            registered_model_name=MLFLOW_MODEL_NAME
        )

        print("✅ Model saved to MLflow")

        return None

    return None


def load_model(stage="Production") -> keras.Model:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'  --> for unit 02 only
    - or from MLFLOW (by "stage") if MODEL_TARGET=='mlflow' --> for unit 03 only

    Return None (but do not Raise) if no model is found

    """

    if MODEL_TARGET == "local":
        print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

        # Get the latest model version name by the timestamp on disk
        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
        local_model_paths = glob.glob(f"{local_model_directory}/*")

        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

        print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)

        latest_model = keras.models.load_model(most_recent_model_path_on_disk)

        print("✅ Model loaded from local disk")

        return latest_model

    elif MODEL_TARGET == "gcs":
        print(Fore.BLUE + f"\nLoad latest model from GCS..." + Style.RESET_ALL)

        client = storage.Client()
        blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="model"))

        try:
            latest_blob = max(blobs, key=lambda x: x.updated)
            latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, latest_blob.name)
            latest_blob.download_to_filename(latest_model_path_to_save)

            latest_model = keras.models.load_model(latest_model_path_to_save)

            print("✅ Latest model downloaded from cloud storage")

            return latest_model
        except:
            print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")

            return None

    elif MODEL_TARGET == "mlflow":
        print(Fore.BLUE + f"\nLoad [{stage}] model from MLflow..." + Style.RESET_ALL)

        # Load model from MLflow
        model = None
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

        try:
            model_versions = client.get_latest_versions(name=MLFLOW_MODEL_NAME, stages=[stage])
            model_uri = model_versions[0].source

            assert model_uri is not None
        except:
            print(f"\n❌ No model found with name {MLFLOW_MODEL_NAME} in stage {stage}")

            return None

        model = mlflow.tensorflow.load_model(model_uri=model_uri)

        print("✅ Model loaded from MLflow")
        return model
    else:
        return None



def mlflow_transition_model(current_stage: str, new_stage: str) -> None:
    """
    Transition the latest model from the `current_stage` to the
    `new_stage` and archive the existing model in `new_stage`
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = MlflowClient()

    version = client.get_latest_versions(name=MLFLOW_MODEL_NAME, stages=[current_stage])

    if not version:
        print(f"\n❌ No model found with name {MLFLOW_MODEL_NAME} in stage {current_stage}")
        return None

    client.transition_model_version_stage(
        name=MLFLOW_MODEL_NAME,
        version=version[0].version,
        stage=new_stage,
        archive_existing_versions=True
    )

    print(f"✅ Model {MLFLOW_MODEL_NAME} (version {version[0].version}) transitioned from {current_stage} to {new_stage}")

    return None


def mlflow_run(func):
    """
    Generic function to log params and results to MLflow along with TensorFlow auto-logging

    Args:
        - func (function): Function you want to run within the MLflow run
        - params (dict, optional): Params to add to the run in MLflow. Defaults to None.
        - context (str, optional): Param describing the context of the run. Defaults to "Train".
    """
    def wrapper(*args, **kwargs):
        mlflow.end_run()
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT)

        with mlflow.start_run():
            mlflow.tensorflow.autolog()
            model, cached_test, history, movies = func(*args, **kwargs) ## change later when passing everything is not necessary anymore

        print("✅ mlflow_run auto-log done")

        return model, cached_test, history, movies
    return wrapper
