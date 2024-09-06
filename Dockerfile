# FROM sets the base layer of the image we create
# what is the buster for?
FROM python:3.10.6-buster

# COPY fills the image we create with content
# the first one is the source and the second one is the destination in the container
COPY requirements.txt /requirements.txt
COPY api /api
COPY moviemain /moviemain

# RUN specifies commands that will be executed inside the image
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# CMD is the last instruction and it specifies which command the container
# runs once it has started
# host 0.0.0.0 tells uvicorn to listen to all ports
# uvicorn filename:variable is the syntax (the name of the file: the name of the variable containing the FastAPI)
# the last part updates the port  for GCP because it needs a specific one
CMD uvicorn api.api:app --host 0.0.0.0 --port $PORT
