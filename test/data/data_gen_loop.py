import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from soundsright.base.data import create_noise_and_reverb_data_for_all_sampling_rates, TTSHandler, reset_all_data_directories, dataset_download

base_path = os.path.join(os.path.expanduser("~"), ".SoundsRight")
tts_base_path = os.path.join(base_path,'data/tts')
noise_base_path = os.path.join(base_path,'data/noise')
reverb_base_path = os.path.join(base_path,'data/reverb')
arni_path = os.path.join(base_path,'data/rir_data')
wham_path = os.path.join(base_path,'data/noise_data')
sample_rates = [16000]

for directory in [tts_base_path, noise_base_path, reverb_base_path, arni_path, wham_path]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

reset_all_data_directories(tts_base_path=tts_base_path, reverb_base_path=reverb_base_path, noise_base_path=noise_base_path)

if False:
    print("Downloading datasets")
    dataset_download(wham_path=wham_path, arni_path=arni_path, partial=True)

tts_handler = TTSHandler(tts_base_path=tts_base_path, sample_rates=sample_rates)
print("TTSHandler initialized")
for sr in sample_rates:
    print("Creating TTS dataset")
    tts_handler.create_openai_tts_dataset_for_all_sample_rates(
        n=3
    )

create_noise_and_reverb_data_for_all_sampling_rates(
    tts_base_path = tts_base_path,
    arni_dir_path = arni_path,
    reverb_base_path=reverb_base_path,
    wham_dir_path=wham_path,
    noise_base_path=noise_base_path,
    tasks=['denoising', 'dereverberation']
)

remove=input("remove all files? y/n")
if remove=='y':
    reset_all_data_directories(tts_base_path=tts_base_path, reverb_base_path=reverb_base_path, noise_base_path=noise_base_path)