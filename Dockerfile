# syntax=docker/dockerfile:1.2
FROM python:latest
# put you docker configuration here
# Working directory por api
WORKDIR /mlediegohiriart/code
# Copy requirements to be installed in working dir
COPY ./requirements.txt /mlediegohiriart/code/requirements.txt
# Install requirements
RUN pip install --no-cache-dir --upgrade -r /mlediegohiriart/code/requirements.txt
# Copy model, API, and __init__ code (challenge files) to working dir
COPY ./challenge/model.py /mlediegohiriart/code/challenge/model.py
COPY ./challenge/api.py /mlediegohiriart/code/challenge/api.py
COPY ./challenge/__init__.py /mlediegohiriart/code/challenge/__init__.py
# Copy data file one level up from the working dir (following dev and test structure)
COPY ./data /mlediegohiriart/data
# Copy trained model to working dir
COPY ./delay_model.json /mlediegohiriart/code/delay_model.json
# Run api server
# First parameter tells uvicorn to look for app in challenge.api on the working dir, like __init__.py does
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "80"]