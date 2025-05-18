---
title: Model Formatting
parent: Mining
nav_order: 2
layout: page
---
# Model Formatting

Models submitted to validators must follow a few formatting guidelines, and we have provided a [template](https://huggingface.co/synapsecai/soundsright-template) for miners to use. Your model will not be scored by validators unless it follows the guidelines exactly.

The `main` branch of this template is what should be modified by miners to create their own models. The branches `DENOISING_16000Hz` and `DEREVERBERATION_16000HZ` serve as tutorials, being fitted with different pretrained checkpoints of [SGMSE+](https://huggingface.co/sp-uhh/speech-enhancement-sgmse). 

For detailed instructions on how to format your model, please reference the `README.md` in the `main` branch of the model template.

# Model Suggestions

If you are looking for an existing model to fine-tune, here are a few suggestions:

- [SGMSE+](https://huggingface.co/sp-uhh/speech-enhancement-sgmse)
- [StoRM](https://github.com/sp-uhh/storm/)
- [Conv-TasNet](https://github.com/JusperLee/Conv-TasNet) (denoising only)
- [CDiffuSE](https://github.com/neillu23/CDiffuSE) (denoising only)

# Model Testing

A script has been provided to test that your model is compatible with the validator architecture. 

To run the script, first make sure you have completed the following installations:

1. Podman 

```
$ apt-get update
$ apt-get -y install podman
```

2. Python venv
```
$ cd soundsright-subnet
$ python3 -m venv .venv
$ source .venv/bin/activate
```

3. Python dependencies
```
(.venv) $ pip install --use-pep517 pesq==0.0.4 && pip install -e .[validator] && pip install httpx==0.27.2
```

Once the installation is complete, run your script with the following command:
```
(.venv) $ python3 scripts/verify_miner_model.py --model_namespace <your_namespace_here> --model_name <your_model_name_here> --model_revision <your_model_revision_here>
```

If `MODEL VERIFICATION SUCCESSFUL.` appears in the logs, then your model is ready to be submitted to validators! 

Note that this may take a while depending on the machine you run the script with (especially if you do not have a GPU). Please reference the documentation on running a validator if you wish to mirror the hardware exactly.