---
title: Mining in the Subnet
parent: Mining
nav_order: 3
layout: page
---
# Mining in the SoundsRight Subnet

## Overview 

Generally, mining on the subnet looks like this:

1. Miner fine-tunes a model. We recommend visiting the [website](https://www.soundsright.ai) and basing your model off of the best model from the previous competition, but ultimately it is up to you.
2. Miner uploads the model to HuggingFace and makes it publicly available. 
3. Miner ensures that their model is compatible with the validator script used to benchmark their model. See the [model tutorial doc](model_tutorial.md) for more details.
4. Miner updates their .env file with the model's data and restarts their miner neuron. The miner will automatically trigger the process of communicating the model data with validators upon restarting.

Note that there is **no fine-tuning script contained within the miner neuron itself**--all miners are responsible for fine-tuning their models externally. Miner neurons are only used to communicate model data to validators. 

However this repository does contain scripts which can be used to generate fine-tuning datasets. Note that miners will need to have an OpenAI API key in order for this to work. Please reference the [dataset generation docs](generate_data.md) for more information.

Also, **each miner can only submit models for one specific task and sample rate**. If you wish to provide models for multiple tasks and/or sample rates, you will need to register multiple miners.

## Running a Miner

### 1. Machine Specifications

As the miner's only function is to upload model metadata to the Bittensor chain and send model information to validators, it is quite lightweight and should work on most hardware configurations.

We have been testing miners on machines running on both **Ubuntu 24.04** and **Python 3.12** with the following hardware configurations:

- 16 GB RAM
- 4 vCPU
- 50 GB SSD

We also highly recommend that you use a dedicated server to run the SoundsRight miner.

### 2. Installation of Mandatory Packages

The following sections will assume you are running as root.

#### 2.1 Install Docker Engine for Ubuntu 
For installing the Docker Engine for Ubuntu, follow the official instructions: [Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/).

#### 2.2 Validate installation
After installation is done, validate the docker engine has been installed correctly:
```
docker run hello-world
```

#### 2.3 Install the mandatory packages

Run the following command:
```
apt-get install python3.12-venv
```

### 3. Preparation

#### 3.1 Setup the GitHub repository and python virtualenv
To clone the repository and setup the Python virtualenv, execute the following commands:
```
git clone https://github.com/synapsec-ai/soundsright-subnet.git
cd soundsright-subnet
python3 -m venv .venv
source .venv/bin/activate
pip install bittensor-cli==9.3.0
```

#### 3.2 Regenerate the miner wallet

The private portion of the coldkey is not needed to run the subnet miner. **Never have your private miner coldkey or hotkeys not used to run the miner stored on the server**.

To regenerate the keys on the host, execute the following commands:
```
btcli wallet regen_coldkeypub
btcli wallet regen_hotkey
```

#### 3.3 Setup .env 

Create the .env from the .env.sample file provided with the following:

```
cp .miner-env.sample .env
```

The contents of the .env file must then be adjusted. The following variables apply for miners:

| Variable | Meaning |
| :------: | :-----: |
| NETUID | The subnet's netuid. For mainnet this value is 105, and for testnet this value is 271. |
| SUBTENSOR_CHAIN_ENDPOINT | The Bittensor chain endpoint. Please make sure to always use your own endpoint. For mainnnet, the default endpoint is: wss://finney.opentensor.ai:443 and for testnet the default endpoint is: wss://test.finney.opentensor.ai:443 |
| WALLET | The name of your coldkey. |
| HOTKEY | The name of your hotkey. |
| LOG_LEVEL | Specifies the level of logging you will see on the validator. Choose between INFO, INFOX, DEBUG. DEBUGX, TRACE, and TRACEX. |
| OPENAI_API_KEY | Your OpenAI API key. This is not needed to run the miner, only to generate training datasets. |

In addition to this, the model being submitted to the competition must be specified in the .env file. Specifically, the model namespace, name, and revision must be specified in the .env for the particular competition being entered in by the miner. 

For example, if we want to submit the `main` branch of the HuggingFace model `synapsecai/my_speech_enhancement_model` to be evalauted, we designate the following: 

| Variable | Designation |
| :------: | :-----: |
| HF_MODEL_NAMESPACE | synapsecai |
| HF_MODEL_NAME | my_speech_enhancement_model |
| HF_MODEL_REVISION | main |

.env example:
```
NETUID=105
SUBTENSOR_CHAIN_ENDPOINT=wss://entrypoint-finney.opentensor.ai:443
WALLET=coldkey_name
HOTKEY=hotkey_name

# Available: INFO, INFOX, DEBUG, DEBUGX, TRACE, TRACEX
LOG_LEVEL=INFO

# Necessary for dataset generation
OPENAI_API_KEY=

# Miner model specification by task and sample rate. 
# If you have not fine-tuned a model for a specific task and sample rate, just leave it blank.
# NOTE: EACH MINER CAN ONLY RESPOND FOR ONE TASK AND ONE SAMPLE RATE. 
# PLEASE REGISTER ANOTHER MINER IF YOU HAVE ANOTHER MODEL FOR ANOTHER TASK OR SAMPLE RATE.
# 16kHz Sample Rate, Denoising Task
DENOISING_16000HZ_HF_MODEL_NAMESPACE=synapsecai
DENOISING_16000HZ_HF_MODEL_NAME=mymodel
DENOISING_16000HZ_HF_MODEL_REVISION=main

# 16kHz Sample Rate, Dereverberation Task
DEREVERBERATION_16000HZ_HF_MODEL_NAMESPACE=
DEREVERBERATION_16000HZ_HF_MODEL_NAME=
DEREVERBERATION_16000HZ_HF_MODEL_REVISION=

# HealthCheck API
HEALTHCHECK_API_HOST=0.0.0.0
HEALTHCHECK_API_PORT=6000
```

### 4. Running the Miner

Run the miner with this command:

```
docker compose up soundsright-miner -d
```
To see the logs, execute the following command: 

```
docker compose logs soundsright-miner -f
``` 

### 5. Updating the Miner

Updating the miner is done by re-launching the docker compose with the `--force-recreate` flag enabled after the git repository has been updated. This will re-create the containers.

```
cd soundsright-subnet
git pull
docker compose up soundsright-miner -d --force-recreate
```
