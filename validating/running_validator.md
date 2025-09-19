---
title: Validating in the Subnet 
parent: Validating
nav_order: 1
layout: page
---
# Validating in the SoundsRight Subnet 

## Summary
Running a validator the in Subnet requires **a subnet stake-weight of at least 1,000**, and that **you are one of the top 64 validators in the subnet, ranked by stake weight**. Please reference the [official Bittensor docs](https://docs.bittensor.com/validators/) for more information.

**We also implore validators to run:**
1. **In a separate environment dedicated to validating for only the SoundsRight subnet.**
2. **Using a child hotkey.**

## Validator deployment 

### 1. Virtual machine deployment
The subnet requires **Ubuntu 24.04**, **Python 3.12** with at least the following hardware configuration:

- 48 GB VRAM
- 60 GB RAM
- 500 GB storage (1000 IOPS)
- 5 gbit/s network bandwidth
- 10 CPU 

**CUDA 12.6** is also highly recommended.

When running the subnet validator, we are highly recommending that you run the subnet validator with DataCrunch.io using the **1x RTX A6000** instance type with **Ubuntu 24.04** and **CUDA 12.6**. 

This is the setup we are performing our testing and development with; as a result, they are being used as the performance baseline for the subnet validators.

Running the validator with DataCrunch.io is not mandatory and the subnet validator should work on other environments as well, though the exact steps for setup may vary depending on the service used. This guide assumes you're running Ubuntu 24.04 provided by DataCrunch.io, and thus skips steps that might be mandatory in other environments (for example, installing the NVIDIA and CUDA drivers).

### 2. Installation of mandatory packages

Note that for the following steps, it will be assumed that you will be running the validator fully as root and as such, any action that needs to be performed as root will not be denoted with sudo.

#### 2.1 Install Podman for Ubuntu 

For installing Podman for Ubuntu, run the following command:
```
apt-get update
apt-get -y install podman
```

#### 2.2 Install the mandatory packages

Run the following commands:
```
apt update 
apt-get install python3.12-venv
apt install jq 
apt install npm 
npm install pm2 -g 
pm2 update 
apt install -y python3.12-dev build-essential gcc g++
apt-get update
apt-get install git-lfs
git lfs install
```

#### 2.3 Configure NVIDIA Container Toolkit and CDI

Follow the instructions to download the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) with Apt.

Next, follow the instructions for [generating a CDI specification](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html).

Verify that the CDI specification was done correctly with:
```
nvidia-ctk cdi list
```
You should see this in your output:
```
nvidia.com/gpu=all
nvidia.com/gpu=0
```

### Configure Git LFS

Git LFS must be configured for the validator to work properly. The commands below cover the configuration:

```
apt-get update
apt-get install git-lfs
git lfs install

```

#### 2.5 Configure pm2 logrotate

pm2-logrotate is highly recommended for the validator. The commands below cover installation alongside the recommended settings.

```
pm2 install pm2-logrotate
pm2 set pm2-logrotate:max_size 5G
pm2 set pm2-logrotate:compress true
pm2 set pm2-logrotate:retain 7
```

### 3. Preparation

This section covers setting up the repository, virtual environment, regenerating wallets, and setting up environmental variables.

#### 3.1 Setup the GitHub repository and python virtualenv
To clone the repository and setup the Python virtualenv, execute the following commands:
```
git clone https://github.com/synapsec-ai/soundsright-subnet.git
cd soundsright-subnet
python3 -m venv .venv
source .venv/bin/activate
pip install bittensor-cli==9.9.0
```

#### 3.2 Regenerate the validator wallet

The private portion of the coldkey is not needed to run the subnet validator. **Never have your private validator coldkey or hotkeys not used to run the validator stored on the server**. Please use a dedicated server for each subnet to minimize impact of potential security issues.

To regenerate the keys on the host, execute the following commands:
```
btcli wallet regen_coldkeypub
btcli wallet regen_hotkey
```

#### 3.3 Setup the environmental variables
The subnet repository contains a sample validator env (`.env.sample`) file that is used to pass the correct parameters to the docker compose file.

Create a new file in the root of the repository called `.env` based on the given sample.
```
cp .validator-env.sample .env
```
The contents of the `.env` file must be adjusted according to the validator configuration. Below is a table explaining what each variable in the .env file represents (note that the .env variables that do not apply for validators are not listed here):

| Variable | Meaning |
| :------: | :-----: |
| NETUID | The subnet's netuid. For mainnet this value is 105, and for testnet this value is 271. |
| SUBTENSOR_CHAIN_ENDPOINT | The Bittensor chain endpoint. Please make sure to always use your own endpoint. For mainnnet, the default endpoint is: wss://finney.opentensor.ai:443, and for testnet the default endpoint is: wss://test.finney.opentensor.ai:443. |
| WALLET | The name of your coldkey. |
| HOTKEY | The name of your hotkey. |
| LOG_LEVEL | Specifies the level of logging you will see on the validator. Choose between INFO, INFOX, DEBUG. DEBUGX, TRACE, and TRACEX. |
| OPENAI_API_KEY | Your OpenAI API key. |
| CUDA_DIRECTORY | Path that points to the CUDA directory. |

.env example:
```
NETUID=105
SUBTENSOR_CHAIN_ENDPOINT=wss://entrypoint-finney.opentensor.ai:443
WALLET=my_coldkey
HOTKEY=my_hotkey
OPENAI_API_KEY=THIS-IS-AN-OPENAI-API-KEY-wfhwe78r78frfg7e8ghrveh78ehrg
CUDA_DIRECTORY=/usr/local/cuda-12.6

# Available: INFO, INFOX, DEBUG, DEBUGX, TRACE, TRACEX
LOG_LEVEL=TRACE

# HealthCheck API
HEALTHCHECK_API_HOST=0.0.0.0
HEALTHCHECK_API_PORT=6000
```

#### 3.4 Installing Python Dependencies

Run the following commands:

```
pip install --use-pep517 pesq==0.0.4
pip install -e .[validator]
pip install httpx==0.27.2
```

### 4. Running the validator

Run the validator with this command: 
```
bash scripts/run_validator.sh --name soundsright-validator --max_memory_restart 500G --branch main
```
To see the logs, execute the following command: 
```
pm2 logs <process-name-or-id>
``` 

### 5. Updating validator

To update the validator, pull the newest changes to main and restart the pm2 process:

```
cd soundsright-subnet
git pull 
pm2 restart
```

### 6. Assessing validator health 

A HealthCheck API is built into the validator, which can be queried for an assessment of the validator's performance. Note that the commands in this section assume default values for the `healthcheck_host` and `healthcheck_port` arguments of `0.0.0.0` and `6000` respectively. The following endpoints are available: 

#### 6.1 Metrics 

This endpoint offers a view of all of the metrics tabulated by the Healthcheck API. It can be queried with:
```
curl http://127.0.0.1:6000/healthcheck/metrics | jq
```

#### 6.2 Events 

This endpoint offers insight into WARNING, SUCCESS and ERROR logs in the validator. It can be queried with:
```
curl http://127.0.0.1:6000/healthcheck/events | jq
```

#### 6.3 Models for Current Competitions
This endpoint offers insight into the best models known by the validator for the previous competition. It can be queried with:
```
curl http://127.0.0.1:6000/healthcheck/current_models | jq
```

#### 6.4 Best Models by Competition

This endpoint offers insight into the best models known by the validator for the previous competition. It can be queried with:
```
curl http://127.0.0.1:6000/healthcheck/best_models | jq
```

#### 6.5 Competitions

This endpoint lists the comptitions currently run by the validator. It can be queried with:
```
curl http://127.0.0.1:6000/healthcheck/competitions | jq
```

#### 6.6 Scores by Competition

This endpoint offers insight into the previous miner scores for each competition. It can be queried with:
```
curl http://127.0.0.1:6000/healthcheck/competition_scores | jq
```

#### 6.7 Overall Scores

This endpoint offers insight into the previous overall miner scores. It can be queried with:
```
curl http://127.0.0.1:6000/healthcheck/scores | jq
```

#### 6.8 Next Competition Timestamp

This endpoint offers insight into the next competition timestamp. It can be queried with:
```
curl http://127.0.0.1:6000/healthcheck/next_competition | jq
```
