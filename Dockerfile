FROM jjanzic/docker-python3-opencv

ADD scripts/ .
ADD requirements.txt . 
ADD config.json .
ADD result_pretrain/ .

RUN apt-get update && apt-get install -y  build-essential cmake pkg-config && \
	apt-get install -y libx11-dev libatlas-base-dev && \
	apt-get install -y libgtk-3-dev libboost-python-dev
RUN mkdir /data && pip install -U setuptools
RUN pip install chainer==3.1.0 dlib==19.7.0 Flask==0.12.2 Flask-SocketIO==2.9.3 matplotlib==2.1.1 numpy

ADD sample_images/ /data/

VOLUME /data
