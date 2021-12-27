FROM pytorch/pytorch

COPY ./requirements.txt ./

RUN apt-get -y update && \
    pip install -r requirements.txt && \
    apt-get -y autoremove

WORKDIR /home

COPY . /home

ENTRYPOINT ["python", "segment_video.py"]
