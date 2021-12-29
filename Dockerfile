FROM nvidia/cuda:10.2-base-ubuntu18.04

COPY ./requirements.txt ./

RUN apt-get update &&  \
    apt-get install -y --no-install-recommends \
      git                   \
      locales               \
      make                  \
      python3               \
      python3-pip           \
      python3-setuptools && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get -y autoremove

RUN locale-gen "en_US.UTF-8"
ENV LC_CTYPE="en_US.UTF-8"

RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

WORKDIR /home

COPY ./segment_video.py \
     ./setup.py \
     ./frameID/ \
     /home/

RUN pip3 install -e .

RUN mkdir sources

ENTRYPOINT ["python3", "segment_video.py"]
