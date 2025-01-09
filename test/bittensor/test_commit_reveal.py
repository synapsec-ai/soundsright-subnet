import bittensor as bt
import secrets
import numpy as np
import os
import time
from dotenv import load_dotenv
load_dotenv() 

class CommitMachine:

    def __init__(self):
        
        # Setup Bittensor objects
        self.subtensor = bt.subtensor(network="test")
        self.metagraph = self.subtensor.metagraph(netuid=os.environ.get("NETUID"))
        self.wallet = bt.wallet(name="testnet_validator", hotkey="validator")

        self.commit_reveal_weights_interval = self.subtensor.get_subnet_hyperparameters(netuid=os.environ.get("NETUID")).commit_reveal_weights_interval

        # Store salts to be revealed
        self.weight_objects = []

    def _get_random_weights(self, metagraph) -> np.ndarray:
        
        # Generate random weights
        weights = np.array([secrets.randbelow(1025) for _ in range(metagraph.size)])

        return weights.tolist()

    def _store_weight_metadata(self, salt, uids, weights, block):
        
        # Construct weight object
        data = {
            "salt": np.array([salt]),
            "uids": uids,
            "weights": weights,
            "block": block
        }

        # Store weight object
        self.weight_objects.append(data)

    def reveal_weights(self, weight_object):

        status, msg = self.subtensor.reveal_weights(
            wallet=self.wallet,
            netuid=38,
            uids=weight_object["uids"],
            weights=weight_object["weights"],
            salt=weight_object["salt"],
            max_retries=5
        )

        print(f'Weight reveal status: {status} - Status message: {msg}')
        
        return status

    def commit_weights(self):

        # Generate metadata
        salt = secrets.randbelow(2**16)
        uids = self.metagraph.uids.tolist()
        weights = self._get_random_weights(self.metagraph.uids)

        # Store metadata
        self._store_weight_metadata(salt, uids, weights, self.subtensor.block)

        # Commit weights
        status, msg = self.subtensor.commit_weights(
            wallet=self.wallet,
            netuid=38,
            salt=[salt],
            uids=uids,
            weights=weights,
            max_retries=5
        )

        print(f'Weight commit status: {status} - Status message: {msg}')
        
        return status

def test_commit_reveal():
    commit_machine = CommitMachine()

    # Add few commits to the list, sleep to separate the commits into different blocks
    assert commit_machine.commit_weights(), "Commit weights failed"
    time.sleep(bt.BLOCKTIME * 2)
    assert commit_machine.commit_weights(), "Commit weights failed"
    time.sleep(bt.BLOCKTIME * 2)
    assert commit_machine.commit_weights(), "Commit weights failed"

    # Reveal all weight commits
    while(len(commit_machine.weight_objects) > 0):

        # Get current block
        current_block = commit_machine.subtensor.block
        print(f'Current block: {current_block}')

        # Iterate all weight objects
        for i,obj in enumerate(commit_machine.weight_objects):
            diff = (current_block - obj["block"])
            print(f'Difference is {diff} blocks')
            if diff > commit_machine.commit_reveal_weights_interval:
                print(f'Revealing: {obj}')
                assert commit_machine.reveal_weights(obj), "Reveal weights failed"

                # Remove from objects list
                commit_machine.weight_objects.pop(i)

        # Sleep for blocktime before next iteration
        time.sleep(bt.BLOCKTIME)


if __name__ == "__main__":
    test_commit_reveal()