import bittensor as bt 
import hashlib
import random
import asyncio
from async_substrate_interface import AsyncSubstrateInterface
import soundsright.base.data as Data

async def get_seed() -> int:
    """
    Obtains a seed based on a hash of the extrinsics the most 
    recent block with a block number divisible by 100.
    """
    subtensor = bt.subtensor(network="finney")
    current_block = subtensor.get_current_block()
    remainder = current_block % 100
    query_block = current_block - remainder
    async_substrate = AsyncSubstrateInterface(url="wss://entrypoint-finney.opentensor.ai:443")
    async with async_substrate:
        block_data = await async_substrate.get_block(block_number=query_block)
        block_extrinsics = block_data["extrinsics"]
        extrinsics_string = "".join([str(extrinsic) for extrinsic in block_extrinsics])
        hash_obj = hashlib.sha256(extrinsics_string.encode("utf-8"))
        seed = int(hash_obj.hexdigest(), 16)
        return seed
    
def test_seeded_random_sentence():

    seed = asyncio.run(get_seed())
    rs1 = Data.RandomSentence(rng=random.Random(seed))
    rs2 = Data.RandomSentence(rng=random.Random(seed))

    assert rs1.simple_sentence() == rs2.simple_sentence()
    assert rs1.sentence() == rs2.sentence()
    assert rs1.bare_bone_with_adjective() == rs2.bare_bone_with_adjective()
    assert rs1.bare_bone_sentence() == rs2.bare_bone_sentence()
    
