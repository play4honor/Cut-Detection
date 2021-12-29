FROM pytorch/pytorch

COPY ./requirements.txt ./

RUN apt-get -y update && \
    pip install -r requirements.txt && \
    apt-get -y autoremove

COPY ./segment_video.py \
     ./setup.py \
     ./frameID/ \
     ./

WORKDIR /

RUN pip install -e .

RUN mkdir -p /sources

ENTRYPOINT ["python", "segment_video.py"]
