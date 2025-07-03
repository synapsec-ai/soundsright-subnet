import os
import random
import asyncio
import subprocess
import time
from huggingface_hub import snapshot_download

import soundsright.base.utils as Utils

class AsyncImageBuildTester:

    def __init__(self):

        self.denoising_path=f"{os.path.expanduser('~')}/.SoundsRight/image_test/denoising"
        self.dereverb_path=f"{os.path.expanduser('~')}/.SoundsRight/image_test/dereverberation"
        self.output_path = f"{os.path.expanduser('~')}/.SoundsRight/outputs"
        self.output_txt_path = os.path.join(self.output_path,'build_results.txt')
        for path in [self.denoising_path, self.dereverb_path, self.output_path]:
            if not os.path.exists(path):
                os.makedirs(path)

        if not os.listdir(self.denoising_path):
            snapshot_download(repo_id="synapsecai/SoundsRightModelTemplate", local_dir=self.denoising_path, revision="DENOISING_16000HZ")

        if not os.listdir(self.dereverb_path):
            snapshot_download(repo_id="synapsecai/SoundsRightModelTemplate", local_dir=self.dereverb_path, revision="DEREVERBERATION_16000HZ")

        self.cpu_count = Utils.get_cpu_core_count()

        self.output_path = f"{os.path.expanduser('~')}/.soundsright/image_test/output.txt"

    def clear_podman_cache(self):
        try:
            # Remove all images
            subprocess.run(
                ["podman", "rmi", "-a", "-f"],
                check=True,
                capture_output=True
            )
            # System prune
            subprocess.run(
                ["podman", "system", "prune", "-a", "-f"],
                check=True,
                capture_output=True
            )
            return True

        except Exception as e:
            return False

    async def build_container_async(self, directory: str) -> bool:
        """
        Build one miner model image async, return True if operation was successful and False otherwise
        """
        start_time = time.time()

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
            random_integer = random.randint(1,1000000000000000000000000000)
            tag_name = str(random_integer)
            print(f"building container with tag name: {tag_name} for directory: {directory}")

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
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10000)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return False

            if process.returncode != 0:
                print("build failed")
                return False

        except Exception as e:
            print(f"build failed because: {e}")
            return False
        
        return time.time() - start_time
    
    async def build_containers_async(self, images_per_cpu):

        total_image_count = self.cpu_count * images_per_cpu

        tasks = []
        for _ in range(total_image_count):

            task = asyncio.create_task(self.build_container_async(directory=self.denoising_path))
            tasks.append(task)
        
        start_time = int(time.time())

        outputs = await asyncio.gather(*tasks)

        completion_time = int(time.time()) - start_time

        return completion_time, outputs
    
    def run_async_build(self, images_per_cpu):

        completion_time, outputs = asyncio.run(self.build_containers_async(images_per_cpu=images_per_cpu))

        tot, length, tot_len = 0, 0, 0

        for k in outputs:

            if isinstance(k, float):
                tot += k
                length += 1
            
            tot_len += 1

        avg_comp_time = tot/length

        success_rate =  len(outputs) - length

        return completion_time, avg_comp_time, success_rate, tot_len
    
    def save_lines_to_file(self, lines):
        
        with open(self.output_txt_path, 'w', encoding='utf-8') as f:
            for line in lines:
                print(line)
                f.write(f"{line}\n")

    def run_build_time_test(self):

        completion_times = []
        avg_comp_times = []
        success_rates = []
        total_lengths = []
        ipcs = [1,2,3,4]
        lines = []

        for ipc in ipcs:

            total_image_count = ipc * self.cpu_count
            print(f"now building {total_image_count} images--{ipc} images per cpu.")

            completion_time, avg_comp_time, success_rate, tot_len = self.run_async_build(images_per_cpu=ipc)

            completion_times.append(completion_time)
            avg_comp_times.append(avg_comp_time)
            success_rates.append(success_rate)
            total_lengths.append(tot_len)

            if ipcs != 4:
                self.clear_podman_cache()

        for ct, act, sr, ipc, tl in zip(completion_times, avg_comp_times, success_rates, ipcs, total_lengths):

            line = f"# of Images per CPU: {ipc}. Number of attempted image builds: {tl} Total completion time: {ct}. Average completion time: {act}. Success rate: {sr}."
            print(line)
            lines.append(line)
        
        self.save_lines_to_file(lines=lines)
        print(lines)

if __name__ == "__main__":

    tester = AsyncImageBuildTester()
    tester.run_build_time_test()