# Fine-Tuning Dataset Generation 

Miners are able to generate datasets of their own to fine-tune their models, though you will need an OpenAI API key to do so.

You will also need a bit of storage to download the noise/reverb datasets. To download the WHAM (noise) dataset requires 76GB of storage, and to download the ARNI (reverb) dataset requires 51 GB of storage.

First, create a .env file with:
```
cp .env.sample .env
```
Next, add your OpenAI API key to the OPENAI_API_KEY variable in the .env file you have created. 

Then, navigate to the scripts directory in the SoundsRight repository with:
```
cd scripts
```
From there, use the `generate_dataset.py` script to generate your dataset with the following command line arguments:

| Argument | Description | Type | Always Required |
| :------: | :---------: | :--: | :------: |
| --clean_dir | Path of directory where you want your clean data to go. | str | Yes |
| --sample_rate | Sample rate of the dataset, defaults to 16000. | int | Yes |
| --n | Dataset size. | int | Yes |
| --task | What task you want the dataset for. One of: 'denoising', 'dereverberation', 'both' | str | Yes |
| --noise_dir | The directory where the noisy dataset will be stored. You only need to input this if you want to generate a dataset for the denoising task. | str | No |
| --noise_data_dir | The directory where the data to generate noisy datasets will be stored. You only need to input this if you want to generate a dataset for the denoising task. | str | No |
| ---reverb_dir | The directory where the reverberation dataset will be stored. You only need to input this if you want to generate a dataset for the dereverberation task. | str | No |
| --reverb_data_dir | The directory where data to generate reverberation datasets will be stored. You only need to input this if you want to generate a dataset for the dereverberation task. | str | No |

Note that the ARNI and WHAM datasets will be downloaded automatically to the directories provided if they have not already been downloaded.

Here is an example of how to call the script:
```
python3 generate_dataset.py --clean_dir my_clean_dir --sample_rate 16000 --n 5000 --task denoising --noise_dir my_noise_dir --noise_data_dir my_noise_data_dir
```