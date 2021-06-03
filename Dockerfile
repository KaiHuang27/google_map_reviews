FROM python:3.8.6
ENV PYTHONUNBUFFERED 1
RUN mkdir /code
WORKDIR /code
COPY . /code/
RUN pip install --no-cache-dir -r requirements.txt
