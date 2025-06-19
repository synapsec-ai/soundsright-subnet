import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from soundsright.base.data import create_noise_and_reverb_data_for_all_sampling_rates, TTSHandler, reset_all_data_directories, dataset_download

base_path = os.path.join(os.path.expanduser("~"), ".SoundsRight")
tts_base_path = os.path.join(base_path,'test_data/tts')
noise_base_path = os.path.join(base_path,'test_data/noise')
reverb_base_path = os.path.join(base_path,'test_data/reverb')
arni_path = os.path.join(base_path,'test_data/arni')
wham_path = os.path.join(base_path,'test_data/wham')
sample_rates = [16000]

for directory in [tts_base_path, noise_base_path, reverb_base_path, arni_path, wham_path]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

reset_all_data_directories(tts_base_path=tts_base_path, reverb_base_path=reverb_base_path, noise_base_path=noise_base_path, log_level="TRACE")

if False:
    print("Downloading datasets")
    dataset_download(wham_path=wham_path, arni_path=arni_path, partial=True)

tts_handler = TTSHandler(tts_base_path=tts_base_path, sample_rates=sample_rates, print_text=True)
print("TTSHandler initialized")
for sr in sample_rates:
    print("Creating TTS dataset")
    tts_handler.create_openai_tts_dataset_for_all_sample_rates(
        n=10,
        seed=100
    )

create_noise_and_reverb_data_for_all_sampling_rates(
    tts_base_path = tts_base_path,
    arni_dir_path = arni_path,
    reverb_base_path=reverb_base_path,
    wham_dir_path=wham_path,
    noise_base_path=noise_base_path,
    tasks=['DENOISING', 'DEREVERBERATION'],
    log_level="TRACE",
    seed=100
)

remove=input("remove all files? y/n")
if remove=='y':
    reset_all_data_directories(tts_base_path=tts_base_path, reverb_base_path=reverb_base_path, noise_base_path=noise_base_path, log_level="TRACE")