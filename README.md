# xtuner
Docker images for the [XTuner library](https://github.com/InternLM/xtuner) for tuning large language models.

Available versions:

* [0.1.15 (CUDA 11.7)](0.1.15_cuda11.7)
* [2024-02-19 (CUDA 11.7)](2024-02-19_cuda11.7)
* [0.1.14 (CUDA 11.7)](0.1.14_cuda11.7)



## Huggingface restricted access

In case models or datasets require being logged into Huggingface, you can give your 
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
