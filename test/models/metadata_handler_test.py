import bittensor as bt 
import pytest
import argparse
import os
import hashlib
import time
import random
import string
import asyncio
from dotenv import load_dotenv
load_dotenv() 

import soundsright.base.models as Models

def init_metadata_handler(subnet_netuid = 38):
    parser = argparse.ArgumentParser()
    parser.add_argument("--netuid", type=int, default=subnet_netuid)
    parser.add_argument("--subtensor.network", type=str, default="test")
    config = bt.config(parser=parser)
    subtensor = bt.subtensor(config=config, network = "test")
    ck=os.environ.get("WALLET")
    hk=os.environ.get("HOTKEY")
    wallet=bt.wallet(name=ck,hotkey=hk)
    
    return Models.ModelMetadataHandler(
        subtensor=subtensor,
        subnet_netuid=subnet_netuid,
        log_level="INFO",
        wallet=wallet
    ), wallet.hotkey.ss58_address

@pytest.mark.parametrize("competition_id, expected", [
    (1, "DENOISING_16000HZ"),
    (2, "DEREVERBERATION_16000HZ"),
    ("1", "DENOISING_16000HZ"),
    ("2", "DEREVERBERATION_16000HZ"),
    (5, None),
    ("5", None),
    (None, None),
    ({}, None),
    (True, None),
    (False, None)
])

def test_get_competition_name_from_competition_id(competition_id, expected):
    metadata_handler, _ = init_metadata_handler()
    assert metadata_handler.get_competition_name_from_competition_id(competition_id) == expected
    
@pytest.mark.parametrize("competition_name, expected", [
    ("DENOISING_16000HZ", 1),
    ("DEREVERBERATION_16000HZ", 2),
])

def test_get_competition_id_from_competition_name(competition_name, expected):
    metadata_handler, _ = init_metadata_handler()
    assert metadata_handler.get_competition_id_from_competition_name(competition_name) == expected
    
def generate_random_string(N):
    """Generate a random alphanumeric string of length N."""
    if N < 1:
        raise ValueError("Length N must be at least 1.")
    
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(N))    

def test_chain_metadata_upload_round_trip():
    metadata_handler, hk = init_metadata_handler()
    
    metadata_str = f"{generate_random_string(10)}:{generate_random_string(25)}:{generate_random_string(5)}:{generate_random_string(2)}:{generate_random_string(48)}:1"
    hashed_metadata_str = hashlib.sha256(metadata_str.encode()).hexdigest()
    
    upload_outcome = asyncio.run(metadata_handler.upload_model_metadata_to_chain(metadata=hashed_metadata_str))
    assert upload_outcome == True
    block_uploaded = metadata_handler.subtensor.get_current_block()
    
    assert block_uploaded 
    
    download_outcome = asyncio.run(metadata_handler.obtain_model_metadata_from_chain(hotkey=hk))
    assert download_outcome == True
    assert block_uploaded >= metadata_handler.metadata_block
    assert hashlib.sha256(metadata_str.encode()).hexdigest() == metadata_handler.metadata 