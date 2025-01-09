import argparse
from soundsright.base.data import generate_dataset_for_miner

parser = argparse.ArgumentParser()
parser.add_argument(
    "--clean_dir",
    type=str,
    help="Path of directory where you want your clean data to go.",
    required=True
)
parser.add_argument(
    "--sample_rate",
    type=int,
    help="Sample rate must be an int, default is 16000.",
    required=False,
    choices=[16000],
    default=16000
)
parser.add_argument(
    "--n",
    type=int,
    help="The number of data files you want to generate.",
    required=True,
)
parser.add_argument(
    "--task",
    type=str,
    help="The task you want to generate a dataset for. One of: 'denoising', 'dereverberation' or 'both'.",
    required=True,
    choices=['denoising', 'dereverberation', 'both'],
)   
parser.add_argument(
    "--noise_dir",
    type=str,
    help="The directory where the noisy dataset will be stored. You only need to input this if you want to generate a dataset for the denoising task.",
    default=None,
)
parser.add_argument(
    "--noise_data_dir",
    type=str,
    help="The directory where the data to generate noisy datasets will be stored. You only need to input this if you want to generate a dataset for the denoising task.",
    default=None,
)
parser.add_argument(
    "--reverb_dir",
    type=str,
    help="The directory where the reverberation dataset will be stored. You only need to input this if you want to generate a dataset for the dereverberation task.",
    default=None,
)
parser.add_argument(
    "--reverb_data_dir",
    type=str,
    help="The directory where data to generate reverberation datasets will be stored. You only need to input this if you want to generate a dataset for the dereverberation task.",
    default=None,
)

args = parser.parse_args()

generate_dataset_for_miner(
    clean_dir=args.clean_dir,
    sample_rate=args.sample_rate,
    n=args.n,
    task=args.task,
    reverb_data_dir=args.reverb_data_dir,
    noise_data_dir=args.noise_data_dir,
    reverb_dir=args.reverb_dir,
    noise_dir=args.noise_dir,
)