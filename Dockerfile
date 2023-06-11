FROM python:3.9-slim-buster

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --index-url=https://pypi.org/simple/ -r requirements.txt


COPY . .

CMD [ "python", "app.py" ]
