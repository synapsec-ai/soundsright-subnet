---
title: Dataset Generation 
parent: Mining
nav_order: 1
layout: page
---
# Fine-Tuning Dataset Generation 

Miners are able to generate datasets of their own to fine-tune their models, though you will need an ElevenLabs API key to do so.

You will also need a bit of storage to download the noise/reverb datasets. To download the WHAM (noise) dataset requires 76GB of storage, and to download the ARNI (reverb) dataset requires 51 GB of storage.

First, create a .env file with:
```
cp .miner-env.sample .env
```
Next, add your ElevenLabs API key to the ELEVENLABS_API_KEY variable in the .env file you have created. 

Then, navigate to the scripts directory in the SoundsRight repository with:
```
cd scripts
```
From there, use the `generate_dataset.py` script to generate your dataset with the following command line arguments:

| Argument | Description | Always Required |
| :------: | :---------: | :------: |
| --clean_dir | Path of directory where you want your clean data to go. | Yes |
| --sample_rate | Sample rate of the dataset either 16000 or 48000. Defaults to 48000. | Yes |
| --n | Dataset size. | Yes |
| --task | What task you want the dataset for. One of: 'denoising', 'dereverberation', 'both' | Yes |
| --noise_dir | The directory where the noisy dataset will be stored. You only need to input this if you want to generate a dataset for the denoising task. | No |
| --noise_data_dir | The directory where the data to generate noisy datasets will be stored. You only need to input this if you want to generate a dataset for the denoising task. | No |
| ---reverb_dir | The directory where the reverberation dataset will be stored. You only need to input this if you want to generate a dataset for the dereverberation task. | No |
| --reverb_data_dir | The directory where data to generate reverberation datasets will be stored. You only need to input this if you want to generate a dataset for the dereverberation task. | No |
| --output_format | If your ElevenLabs subscription is below that of a Pro plan, you will need to adjust this to output MP3 (this will automatically get converted to .wav). Your input will be one of: mp3_44100_32, mp3_44100_64, mp3_44100_96, mp3_44100_128 or mp3_44100_192. The last portion of the input signifies the bitrate of the output audio. If you have a Pro Plan or above, ignore this section. The default is: pcm_44100 | No |

Note that the ARNI and WHAM datasets will be downloaded automatically to the directories provided if they have not already been downloaded.

Here is an example of how to call the script:
```
python3 generate_dataset.py --clean_dir my_clean_dir --sample_rate 16000 --n 5000 --task denoising --noise_dir my_noise_dir --noise_data_dir my_noise_data_dir
```