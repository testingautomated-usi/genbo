FROM tensorflow/tensorflow:2.3.4-gpu

RUN apt-get install -y libsm6 libxext6 libxrender1

COPY requirements_docker.txt $HOME

RUN pip install -r requirements_docker.txt

RUN rm requirements_docker.txt

WORKDIR /home

ENTRYPOINT bash


