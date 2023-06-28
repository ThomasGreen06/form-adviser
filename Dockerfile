FROM python:buster

WORKDIR /app

ADD . /app

RUN apt update && apt upgrade -y && apt install -y libgl1-mesa-dev
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

ENV FLASK_ENV="docker"
ENV FLASK_APP=app.py

EXPOSE 5000

ENTRYPOINT ["python3", "app.py"]