import os
import sys
import time
import glob
import subprocess
from huggingface_hub import snapshot_download
import asyncio
import soundsright.base.utils as Utils
from soundsright.base.data import create_noise_and_reverb_data_for_all_sampling_rates, dataset_download, TTSHandler

class AsyncModelRunTester:

    def __init__(self):

        self.seed = 69

        self.base_path = os.path.join(os.path.expanduser("~"), ".SoundsRight")
        self.tts_base_path = os.path.join(self.base_path,'test_data','tts')
        self.noise_base_path = os.path.join(self.base_path,'test_data','noise')
        self.reverb_base_path = os.path.join(self.base_path,'test_data','reverb')
        self.arni_path = os.path.join(self.base_path,'test_data','arni')
        self.wham_path = os.path.join(self.base_path,'test_data','wham')
        self.enhanced1 = os.path.join(self.base_path, 'test_data','enhanced1')
        self.enhanced2 = os.path.join(self.base_path, 'test_data','enhanced2')
        self.enhanced3 = os.path.join(self.base_path, 'test_data','enhanced3')
        self.enhanced4 = os.path.join(self.base_path, 'test_data','enhanced4')
        self.enhanced5 = os.path.join(self.base_path, 'test_data','enhanced5')
        self.model_path_1 = os.path.join(self.base_path,'models','model1')
        self.model_path_2 = os.path.join(self.base_path,'models','model2')
        self.model_path_3 = os.path.join(self.base_path,'models','model3')
        self.model_path_4 = os.path.join(self.base_path,'models','model4')
        self.model_path_5 = os.path.join(self.base_path,'models','model5')

        self.tags = [
            "model1",
            "model2",
            "model3",
            "model4",
            "model5",
        ]

        self.paths = [
            self.model_path_1, 
            self.model_path_2, 
            self.model_path_3, 
            self.model_path_4, 
            self.model_path_5
        ]

        self.output_paths = [
            self.enhanced1,
            self.enhanced2,
            self.enhanced3,
            self.enhanced4,
            self.enhanced5,
        ]

        self.ports = [
            6501,
            6502,
            6503,
            6504,
            6505,
        ]

        for directory in [self.tts_base_path, self.noise_base_path, self.reverb_base_path, self.arni_path, self.wham_path, self.enhanced_path, self.model_path_1, self.model_path_2, self.model_path_3, self.model_path_4, self.model_path_5]:
            if not os.path.exists(directory):
                os.makedirs(directory)

        for path, port in zip(self.paths, self.ports):
            snapshot_download(repo_id="synapsecai/SoundsRightModelTemplate", local_dir=path, revision="DENOISING_16000HZ")
            Utils.replace_string_in_directory(directory=path, old_string="6500", new_string=str(port))

        self.sample_rates = [16000]

        dataset_download(wham_path=self.wham_path, arni_path=self.arni_path, partial=True)

        self.TTSHandler = TTSHandler(
            tts_base_path=self.tts_base_path, 
            sample_rates=self.sample_rates
        )

        self.TTSHandler.create_openai_tts_dataset_for_all_sample_rates(n=30, seed=self.seed)

        create_noise_and_reverb_data_for_all_sampling_rates(
            tts_base_path = self.tts_base_path,
            arni_dir_path = self.arni_path,
            reverb_base_path=self.reverb_base_path,
            wham_dir_path=self.wham_path,
            noise_base_path=self.noise_base_path,
            tasks=['DENOISING', 'DEREVERBERATION'],
            log_level="TRACE",
            seed=100
        )

        if not self.run_async_build():
            print("model build failed")
            sys.exit()

    async def build_container_async(tag_name: str, directory: str) -> bool:
        """
        Build one miner model image async, return True if operation was successful and False otherwise
        """

        dockerfile_path = None

        # Search for docker-compose.yml in the directory and its subdirectories
        for root, _, files in os.walk(directory):
            for file in files:
                if file == "Dockerfile":
                    dockerfile_path = os.path.join(root, file)
                    break
            if dockerfile_path:
                break

        if not dockerfile_path:
            return False
        
        if not os.path.isfile(dockerfile_path):
            return False

        try:
            
            process = await asyncio.create_subprocess_exec(
                "podman", "build",
                "-t", tag_name,
                "--file", dockerfile_path,
                "--no-cache",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.path.dirname(dockerfile_path),
            )

            try:
                stdout, stderr = await asyncio.wait_for(process.communicate())
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return False

            if process.returncode != 0:
                return False

        except Exception as e:
            return False
        
        return True

    async def build_containers_async(self):

        paths = [
            self.model_path_1,
            self.model_path_2,
            self.model_path_3,
            self.model_path_4,
            self.model_path_5,
        ]

        tasks = []

        for tag, path in zip(self.tags, paths):

            task = self.build_container_async(tag_name=tag,directory=path)
            tasks.append(task)

        outputs = await asyncio.gather(*tasks)

        return outputs
    
    def run_async_build(self):

        outputs = asyncio.run(self.build_containers_async())

        return len(outputs) == 5
    
    def validate_all_noisy_files_are_enhanced(self, enhanced_path):
        noisy_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(self.noise_base_path, "16000", '*.wav'))])
        enhanced_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(enhanced_path, '*.wav'))])
        return noisy_files == enhanced_files
    
    async def run_model_evaluation(self, tag, port, enhanced_path):

        start_time = time.time()

        status1 = Utils.start_container_replacement(
            tag_name=tag,
            port=port,
            log_level="TRACE"
        )

        if not status1:
            return False
        
        status2 = Utils.check_container_status(port=port, log_level="TRACE")

        if not status2:
            return False 

        status3 = Utils.prepare(port=port, log_level="TRACE")

        if not status3:
            return False
        
        status4 = Utils.upload_audio(noisy_dir=os.path.join(self.noise_base_path, "16000"), port=port, log_level="TRACE")

        if not status4:
            return False
        
        status5 = Utils.enhance_audio(port=port, log_level="TRACE")

        if not status5:
            return False
        
        status6 = Utils.download_enhanced(port=port, enhanced_dir=enhanced_path)

        if not status6:
            return False 
        
        status7 = self.validate_all_noisy_files_are_enhanced(enhanced_path=enhanced_path)

        if not status7:
            return False 
        
        completion_time = time.time() - start_time

        subprocess.run(
            ["podman", "rm", "-f", tag],
            check=True,
            capture_output=True
        )
        
        return completion_time
    
    async def run_eval_group(self, count):

        tasks = []

        for i in range(count):
            
            task = self.run_model_evaluation(
                tag=self.tags[i],
                port=self.ports[i],
                enhanced_path=self.output_paths[i]
            )

        start_time = time.time()

        outputs = asyncio.gather(*tasks)

        completion_time = time.time() - start_time

        return outputs, completion_time
    
    def save_lines_to_file(self, lines):
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for line in lines:
                print(line)
                f.write(f"{line}\n")
    
    def run_eval_test(self):

        lines = []

        for i in range(5):

            outputs, completion_time = asyncio.run(self.run_eval_group(i))

            count = i + 1
            lines.append(f"Total completion time for {count} models: {completion_time}. Individual completion times: {outputs}")

        self.save_lines_to_file(lines=lines)


if __name__ == "__main__":

    tester = AsyncModelRunTester()
    tester.run_eval_test()

        

    
