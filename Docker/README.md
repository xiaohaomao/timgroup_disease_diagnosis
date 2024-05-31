# Phenobrain QuickStart Tutorial

Suppose you have already installed docker successfully, you shall run phenobrain by the following steps.

## Docker image create/pull

Switch to project directory and create docker image `phenobrain:1.0`

```bash
cd /path/to/phenobrain # customize your path
docker build -t phenobarin:1.0 .
```

Or you can pull docker image from our repository



## Create and run docker container

Next, create container using `phenobrain:1.0` docker image with name `pb_ins` (or any other name you prefer).

```bash
docker run -it --name pb_ins phenobrain:1.0 /bin/bash
```

You should be in docker container afterwards.

## Preparing environment for phenobrain

Before running phenobrain, first install python dependency under `phenobrain` conda environment.

```bash
conda activate phenobrain
pip install -r /home/requirements.txt
```



## Run quick test

```bash
cd /home/core
python core/script/test/test_optimal_model.py
```

The result shall be in `/home/core/result`



## Copy result from docker container 

Optionally, use another terminal session to copy result from container.

```bash
docker cp pb_ins:/home/core/result ./
```

