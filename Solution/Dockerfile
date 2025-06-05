FROM python:3.10

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements_app.txt

RUN python -m nltk.downloader punkt averaged_perceptron_tagger averaged_perceptron_tagger_eng

EXPOSE 8000

CMD ["python", "scripts/app.py"]
