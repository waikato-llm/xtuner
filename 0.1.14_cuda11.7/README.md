# XTuner

Docker image for [XTuner](https://github.com/InternLM/xtuner) 0.1.14.

Uses PyTorch 2.0.1, CUDA 11.7.

## Quick start

### Inhouse registry

* Log into registry using *public* credentials:

  ```bash
  docker login -u public -p public public.aml-repo.cms.waikato.ac.nz:443 
  ```
  
* Create the following directories:

  ```bash
  mkdir cache triton
  ```

* Launch docker container

  ```bash
  docker run \
    -u $(id -u):$(id -g) -e USER=$USER \
    --gpus=all \
    --shm-size 8G \
    -v `pwd`:/workspace \
    -v `pwd`/cache:/.cache \
    -v `pwd`/triton:/.triton \
    -it public.aml-repo.cms.waikato.ac.nz:443/pytorch/pytorch-xtuner:0.1.14_cuda11.7
  ```

### Docker hub
  
* Create the following directories:

  ```bash
  mkdir cache triton
  ```

* Launch docker container

  ```bash
  docker run \
    -u $(id -u):$(id -g) -e USER=$USER \
    --gpus=all \
    --shm-size 8G \
    -v `pwd`:/workspace \
    -v `pwd`/cache:/.cache \
    -v `pwd`/triton:/.triton \
    -it waikatodatamining/pytorch-xtuner:0.1.14_cuda11.7
  ```

### Build local image

* Build the image from Docker file (from within /path_to/huggingface-transformers/0.1.14_cuda11.7)

  ```bash
  docker build -t hf .
  ```
  
* Run the container

  ```bash
  docker run --gpus=all --shm-size 8G -v /local/dir:/container/dir -it hf
  ```
  `/local/dir:/container/dir` maps a local disk directory into a directory inside the container


## Publish images

### Build

```bash
docker build -t pytorch-xtuner:0.1.14_cuda11.7 .
```

### Inhouse registry  
  
* Tag

  ```bash
  docker tag \
    pytorch-xtuner:0.1.14_cuda11.7 \
    public-push.aml-repo.cms.waikato.ac.nz:443/pytorch/pytorch-xtuner:0.1.14_cuda11.7
  ```
  
* Push

  ```bash
  docker push public-push.aml-repo.cms.waikato.ac.nz:443/pytorch/pytorch-xtuner:0.1.14_cuda11.7
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```bash
  docker login public-push.aml-repo.cms.waikato.ac.nz:443
  ```

### Docker hub  
  
* Tag

  ```bash
  docker tag \
    pytorch-xtuner:0.1.14_cuda11.7 \
    waikatodatamining/pytorch-xtuner:0.1.14_cuda11.7
  ```
  
* Push

  ```bash
  docker push waikatodatamining/pytorch-xtuner:0.1.14_cuda11.7
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```bash
  docker login
  ```


### Requirements

```bash
docker run --rm --pull=always \
  -it public.aml-repo.cms.waikato.ac.nz:443/pytorch/pytorch-xtuner:0.1.14_cuda11.7 \
  pip freeze > requirements.txt
```

## Scripts

* `xtuner` - the command-line tool that comes with XTuner, e.g. for interactive chats
* `xtuner_redis` - for making models available via Redis


## Permissions

When running the docker container as regular use, you will want to set the correct
user and group on the files generated by the container (aka the user:group launching
the container):

```bash
docker run -u $(id -u):$(id -g) -e USER=$USER ...
```


## Formats

### Prompt

```json
{
  "text": "the text to use as input.",
  "history": "previous prompts concatenated",
  "turns": 0
}
```

The (optional) `history` text and the number of `turns` are used as additional inputs to the model.

Using `RESET` as text in the prompt will reset the history.

### Response

```json
{
  "text": "the generated text.",
  "history": "previous input texts concatenated",
  "turns": 0
}
```

`history` and `turns` can be used for the next prompt.

If the `--no_history` flag is used, then these two fields will get omitted in the response.


## Huggingface

In case models or datasets require being logged into Huggingface, you achieve give your 
Docker container access via an access token.

### Create access token

In order to create an access token, do the following:
- Log into https://huggingface.co
- Go to *Settings* -> *Access tokens*
- Create a token (*read* access is sufficient, unless you want to push models back to huggingface)
- Copy the token onto the clipboard
- Save the token in a [.env file](https://hexdocs.pm/dotenvy/0.2.0/dotenv-file-format.html), using `HF_TOKEN` as the variable name

### Provide token to container

Add the following parameter to make all the environment variables stored in the `.env` file in 
the current directory available to your Docker container:

```
--env-file=`pwd`/.env
```

### Log into Huggingface

With the `HF_TOKEN` environment variable set, you can now log into Huggingface inside your Docker 
container using the following command:

```
huggingface-cli login --token=$HF_TOKEN
```