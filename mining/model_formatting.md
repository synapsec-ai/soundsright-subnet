---
title: Model Formatting
parent: Mining
nav_order: 2
layout: page
---
# Model Formatting

Models submitted to validators must follow a few formatting guidelines, and we have provided a [template](https://huggingface.co/synapsecai/soundsright-template) for miners to use. Your model will not be scored by validators unless it follows the guidelines exactly.

The `main` branch of this template is what should be modified by miners to create their own models. The branches `DENOISING_16000HZ`, `DEREVERBERATION_16000HZ`, `DENOISING_48000HZ`, and `DEREVERBERATION_48000HZ` serve as tutorials, being fitted with different pretrained checkpoints of [SGMSE+](https://huggingface.co/sp-uhh/speech-enhancement-sgmse). 

For detailed instructions on how to format your model, please reference the `README.md` in the `main` branch of the model template.

# Model Submission

When you are ready to submit your model to be benchmarked by validators, run the miner with the .env file updated with the model metadata.

There are a couple things to note:

1. Model metadata must be uploaded to the chain before the start of a competition.
2. Model revision must be specified as a commit hash.

# Model Suggestions

If you are looking for an existing model to fine-tune, here are a few suggestions:

- [SGMSE+](https://huggingface.co/sp-uhh/speech-enhancement-sgmse)
- [StoRM](https://github.com/sp-uhh/storm/)
- [Conv-TasNet](https://github.com/JusperLee/Conv-TasNet) (denoising only)
- [CDiffuSE](https://github.com/neillu23/CDiffuSE) (denoising only)

# Model Testing

A script has been provided to test that your model is compatible with the validator architecture. 

To run the script, first make sure you have configured your environment:

### 1. Virtual machine deployment
The model verficiation requires **Ubuntu 24.04** and **Python 3.12**. **CUDA 12.6** is also recommended--if you are using another version, make sure to modify the `cuda_directory` flag when running the script.

### 2. Installation of mandatory packages

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
apt install -y python3.12-dev build-essential gcc g++
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

#### 3 Setup the GitHub repository and python virtualenv
To clone the repository and setup the Python virtualenv, execute the following commands:

```
git clone https://github.com/synapsec-ai/soundsright-subnet.git
cd soundsright-subnet
python3 -m venv .venv
source .venv/bin/activate
pip install --use-pep517 pesq==0.0.4
pip install -e .[validator]
pip install httpx==0.27.2
```

Once the installation is complete, run your script with the following command:
```
(.venv) $ python3 scripts/verify_miner_model.py --model_namespace <your_namespace_here> --model_name <your_model_name_here> --model_revision <your_model_revision_here> --sample_rate <either 16000 or 48000> --cuda_directory <cuda_directory_here> 
```

For the `cuda_directory` flag, the default is `/usr/local/cuda-12.6`. If you are running a different version of CUDA you will need to adjust this accordingly.

If `MODEL VERIFICATION SUCCESSFUL.` appears in the logs, then your model is ready to be submitted to validators! 

Note that this may take a while depending on the machine you run the script with. Please reference the documentation on running a validator if you wish to mirror the hardware exactly.