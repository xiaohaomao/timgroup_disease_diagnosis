FROM continuumio/anaconda3:2024.02-1
ENV DEBIAN_FRONTEND=noninteractive
ENV VERSION=1.0

# install gcc 
RUN apt-get update
RUN apt-get install -y gcc g++

# install java
ADD ./openjdk.deb /tmp/openjdk.deb
RUN apt-get update
RUN apt install -y libxtst6 libasound2
RUN dpkg -i /tmp/openjdk.deb || apt-get install -f -y
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# create conda environment
COPY ./condarc.txt /root/.condarc
RUN conda create -y -n phenobrain python==3.6.12
COPY ./requirements.txt /home/requirements.txt

# setting environment
RUN echo 'RAREDIS_PATH="/home/core/"' >> /root/.bashrc \
 && echo 'CORE_PATH="${RAREDIS_PATH}/core"' >> /root/.bashrc \
 && echo 'BERT_SYN_PRJ_PATH="${RAREDIS_PATH}/bert_syn_project"' >> /root/.bashrc \
 && echo 'export PYTHONPATH=$CORE_PATH:$BERT_SYN_PRJ_PATH:$PYTHONPATH' >> /root/.bashrc

# create file folder & copy project
RUN mkdir -p /home/core/
COPY ./core /home/core/

WORKDIR /home
