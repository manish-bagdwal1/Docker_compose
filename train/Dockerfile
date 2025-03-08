FROM python:3.9

WORKDIR /train

COPY train.py .
RUN pip install --upgrade pip
RUN pip install --default-timeout=100 scikit-learn pandas  # Increased timeout

VOLUME /model

CMD ["python", "train.py"]
