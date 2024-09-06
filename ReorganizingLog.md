Deleted Stelios KNNmodel_2.ipynb

Updated requirements.txt with what is needed for the API, namely (no other changes on it):
      fastapi
      uvicorn
      pandas
      scikit-learn (this is actually useless if we don't use KNN but keeping it for now)
      requests

Makefile is in my version, no changes required

Dockerfile is in my version, no changes required

.gitignore no changes

.env no changes

.envrc no changes

raw_data no changes, now includes Tim's combined df

Deleted empty Untitled notebook in folder notebooks, no other change

Added folder ./api/ (no updates in functions yet)

Folder moviemain renamed to KNN_dummy_model, will later delete

Folder api_third_try renamed to moviemain, leading to:
      changed import in main.py

Added tensorflow-datasets==4.9.2 to requirements.txt because it was not recognized by VScode

Deleted app.py, dummy_main.py, dummy_model.py, requirements.txt that were in moviemain (old api_third_try) because they were obsolete

Created ./api/new_api.py for the actual model

Changed makefile version of run_api to match new one
