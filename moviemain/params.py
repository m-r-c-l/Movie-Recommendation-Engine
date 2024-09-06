import os


##################  VARIABLES  ##################
# Environment Parameters
DATA_SIZE = os.environ.get("DATA_SIZE")
MODEL_TARGET = os.environ.get("MODEL_TARGET")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE")) ## This can be deleted. Not used anyhwere.

# GCP Project
GCP_PROJECT_WAGON = os.environ.get("GCP_PROJECT_WAGON") ## This can be deleted I guess. Not used anywhere.

# Your personal GCP project for this bootcamp
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")

# Cloud Storage
BUCKET_NAME = os.environ.get("BUCKET_NAME") ## TO DO

# BigQuery
BQ_DATASET = os.environ.get("BQ_DATASET") ## Not sure we'll use BQ
BQ_REGION = os.environ.get("BQ_REGION") ## Not sure we'll use BQ

# Compute Engine
INSTANCE = os.environ.get("INSTANCE") ## TO DO

# Model Lifecycle
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME")

PREFECT_FLOW_NAME = os.environ.get("PREFECT_FLOW_NAME") # Not sure we'll use?
PREFECT_LOG_LEVEL = os.environ.get("PREFECT_LOG_LEVEL") # Not sure we'll use?

# Docker
GAR_IMAGE = os.environ.get("GAR_IMAGE") # Stelios
GAR_MEMORY = os.environ.get("GAR_MEMORY") # Stelios


##################  CONSTANTS  #####################
LOCAL_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath('__file__')),
    "raw_data"
)
LOCAL_REGISTRY_PATH = os.path.join(
    os.path.dirname(os.path.abspath('__file__')),
    "training_outputs"
)


################## VALIDATIONS #################

env_valid_options = dict(
    DATA_SIZE=["latest-small", "100k", "1m", "20m", "25m"],
    MODEL_TARGET=["local", "gcs", "mlflow"],
)

def validate_env_value(env, valid_options):
    env_value = os.environ[env]
    if env_value not in valid_options:
        raise NameError(f"Invalid value for {env} in `.env` file: {env_value} must be in {valid_options}")


for env, valid_options in env_valid_options.items():
    validate_env_value(env, valid_options)
