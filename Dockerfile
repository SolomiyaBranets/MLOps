FROM python:3.6-buster

RUN apt-get update -y \
    && apt-get install -y python3-dev python3-pip build-essential \
    && apt-get install gcc -y \
    && apt-get install sudo -y \ 
    && apt-get clean

RUN pip install fastai==1.0.61
RUN python -m pip install flask

WORKDIR /data

ENTRYPOINT [ "python" ]
CMD [ "app.py" ]