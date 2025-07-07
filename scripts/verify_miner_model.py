import argparse
import logging
import os
import shutil
import time
import glob

import soundsright.base.utils as Utils
import soundsright.base.models as Models
import soundsright.base.benchmarking as Benchmarking

logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def validate_all_reverb_files_are_enhanced(reverb_dir, enhanced_dir):
    reverb_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(reverb_dir, '*.wav'))])
    enhanced_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(enhanced_dir, '*.wav'))])
    return reverb_files == enhanced_files

def initialize_run_and_benchmark_model(model_namespace, model_name, model_revision, cuda_directory):

    file_dir = os.path.dirname(os.path.abspath(__file__))
    clean_dir = os.path.join(file_dir, "assets", "clean")
    reverb_dir = os.path.join(file_dir, "assets", "reverb")
    
    output_base_path = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(output_base_path, "model")
    model_output_dir = os.path.join(output_base_path, "model_output")
    for d in [model_dir, model_output_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
            
    logging.info(f"{model_dir} exists: {os.path.exists(model_dir)}\n{model_output_dir} exists: {os.path.exists(model_output_dir)}")

    logging.info("Downloading model:")
    try:
        model_hash = Models.get_model_content_hash(
            model_id = f"{model_namespace}/{model_name}",
            revision=model_revision,
            local_dir=model_dir,
            log_level="TRACE"
        )
        logging.info(f"Model downloaded. Model hash:")
        print(model_hash)
    except Exception as e:
        logging.error(f"Model download failed because: {e}")
        return False

    logging.info("Updating CUDA_HOME in Dockerfile")
    if not Utils.update_dockerfile_cuda_home(directory=model_dir, cuda_directory=cuda_directory, log_level="TRACE"):
        logging.error("Could not update CUDA_HOME in Dockerfile.")
        shutil.rmtree(model_dir)
        return False
        
    logging.info("Validating container configuration:")
    if not Utils.validate_container_config(model_dir):
        logging.error("Container config validation failed.")
        Utils.delete_container(log_level="TRACE")
        shutil.rmtree(model_dir)
        shutil.rmtree(model_output_dir)
        return False
    logging.info("Container validation succeeded.")
    
    Utils.delete_container(log_level="TRACE")

    logging.info("Starting container:")    
    if not Utils.start_container(directory=model_dir, log_level="TRACE", cuda_directory=cuda_directory):
        logging.error("Container could not be started.")
        Utils.delete_container(log_level="TRACE")
        shutil.rmtree(model_dir)
        shutil.rmtree(model_output_dir)
        return False
    logging.info("Container started.")
    
    time.sleep(10)
    
    logging.info("Checking container status:")
    if not Utils.check_container_status(port=6500, log_level="TRACE"):
        logging.error("Container status check failed. Please check your /status/ endpoint.")
        Utils.delete_container(log_level="TRACE")
        shutil.rmtree(model_dir)
        shutil.rmtree(model_output_dir)
        return False
    logging.info("Container status check successful.")
    
    time.sleep(1)
    
    logging.info("Preparing model:")
    if not Utils.prepare(port=6500, log_level="TRACE"):
        logging.error("Model preparation failed. Please check your /prepare/ endpoint.")
        Utils.delete_container(log_level="TRACE")
        shutil.rmtree(model_dir)
        shutil.rmtree(model_output_dir)
        return False
    logging.info("Container preparation successful.")
    
    time.sleep(10)
    
    logging.info("Uploading audio:")
    if not Utils.upload_audio(port=6500, noisy_dir=reverb_dir, log_level="TRACE"):
        logging.error("Reverb audio upload failed. Please check your /upload-audio/ endpoint.")
        Utils.delete_container(log_level="TRACE")
        shutil.rmtree(model_dir)
        shutil.rmtree(model_output_dir)
        return False
    logging.info("Audio upload successful.")
    
    time.sleep(5)
    
    logging.info("Enhancing audio:")
    if not Utils.enhance_audio(port=6500, log_level="TRACE"):
        logging.error("Audio enhancement failed. Please check your /enhance/ endpoint.")
        Utils.delete_container(log_level="TRACE")
        shutil.rmtree(model_dir)
        shutil.rmtree(model_output_dir)
        return False
    logging.info("Audio enhancement successful.")
    
    time.sleep(5)
    
    logging.info("Downloading enhanced files:")
    if not Utils.download_enhanced(port=6500, enhanced_dir=model_output_dir,log_level="TRACE"):
        logging.error("Could not download enhanced files. Please check your /download-enhanced/ endpoint.")
        Utils.delete_container(log_level="TRACE")
        shutil.rmtree(model_dir)
        shutil.rmtree(model_output_dir)
        return False
    logging.info("Enhanced audio download successful.")
    
    Utils.delete_container(log_level="TRACE")
    
    logging.info("Checking to make sure that all files were enhanced:")
    if not validate_all_reverb_files_are_enhanced(reverb_dir=reverb_dir, enhanced_dir=model_output_dir):
        logging.error("Mismatch between reverb files and enhanced files. Your model did not return all of the audio files it was expected to.")
        shutil.rmtree(model_dir)
        shutil.rmtree(model_output_dir)
        return False 
    logging.info("File validation successful.")
    
    clean_files = sorted([f for f in os.listdir(clean_dir) if f.lower().endswith('.wav')])
    enhanced_files = sorted([f for f in os.listdir(model_output_dir) if f.lower().endswith('.wav')])
    noisy_files = sorted([f for f in os.listdir(reverb_dir) if f.lower().endswith('.wav')])
    
    logging.info(f"Clean files: {clean_files}\nNoisy files: {noisy_files}\nEnhanced files: {enhanced_files}")
    
    logging.info("Calculating metrics:")
    try:
        metrics_dict = Benchmarking.calculate_metrics_dict(
            sample_rate=16000,
            clean_directory=clean_dir,
            enhanced_directory=model_output_dir,
            noisy_directory=reverb_dir,
            log_level="TRACE",
        )
        logging.info(f"Calculated model performance benchmarks: {metrics_dict}")
    except Exception as e:
        logging.error(f"Benchmarking metrics could not be calculated because: {e}")
    
    shutil.rmtree(model_dir)
    shutil.rmtree(model_output_dir)
    
    return True 

def verify_miner_model(model_namespace, model_name, model_revision, cuda_directory): 
    
    logging.info(f"Starting verificaiton for model: huggingface.co/{model_namespace}/{model_name}/tree/{model_revision}")
    
    if not initialize_run_and_benchmark_model(model_namespace=model_namespace, model_name=model_name, model_revision=model_revision, cuda_directory=cuda_directory):
        logging.critical(f"MODEL VERIFICATION FAILED.")
        return

    logging.info("\n\nMODEL VERIFICATION SUCCESSFUL.")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model_namespace",
        type=str,
        help="HuggingFace namespace (user/org name).",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="HuggingFace model name.",
        required=True,
    )
    parser.add_argument(
        "--model_revision",
        type=str,
        help="HuggingFace model revision (branch name).",
        required=True,
    )

    parser.add_argument(
        "--cuda_directory",
        type=str,
        help="The path to the CUDA directory",
        default="/usr/local/cuda-12.6"
    )
    
    args = parser.parse_args()
    
    verify_miner_model(
        model_namespace=args.model_namespace,
        model_name=args.model_name,
        model_revision=args.model_revision,
        cuda_directory=args.cuda_directory,
    )