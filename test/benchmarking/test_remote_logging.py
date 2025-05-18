import pytest 
import os
import time
import bittensor as bt

import soundsright.base.benchmarking as Benchmarking 

from dotenv import load_dotenv
load_dotenv()

def get_hk():
    ck_name = os.getenv("WALLET")
    hk_name = os.getenv("HOTKEY")
    wallet = bt.wallet(name=ck_name, hotkey=hk_name)
    return wallet.hotkey

def test_miner_models_remote_logging():
    
    hk = get_hk()
    
    miner_models = {
        "category": "current",
        "validator":hk.ss58_address,
        "timestamp":int(time.time()*1000),
        "models": {
            "DENOISING_16000HZ":[
            {
                "hotkey":"miner_hotkey_ss58adr",
                "hf_model_name":"SoundsRightModelTemplate",
                "hf_model_namespace":"synapsecai",
                "hf_model_revision":"main",
                "model_hash":"aaaaaaaaaaaaaaaaa11111aaaaa",
                "block":17873471294,
                "metrics":{
                "PESQ":{
                    "scores":[1.0,1.1,1.2],
                    "average":1.1,
                    "confidence_interval":[1.05,1.15],
                },
                "ESTOI":{
                    "scores":[1.0,1.1,1.2],
                    "average":1.1,
                    "confidence_interval":[1.05,1.15],
                },
                "SI_SDR":{
                    "scores":[1.0,1.1,1.2],
                    "average":1.1,
                    "confidence_interval":[1.05,1.15],
                },
                "SI_SAR":{
                    "scores":[1.0,1.1,1.2],
                    "average":1.1,
                    "confidence_interval":[1.05,1.15],
                },
                "SI_SIR":{
                    "scores":[1.0,1.1,1.2],
                    "average":1.1,
                    "confidence_interval":[1.05,1.15],
                },
                }
            }
            ],
            "DEREVERBERATION_16000HZ":[
            {
                "hotkey":"miner_hotkey_ss58adr",
                "hf_model_name":"SoundsRightModelTemplate",
                "hf_model_namespace":"synapsecai",
                "hf_model_revision":"main",
                "model_hash":"aaaaaaaaaaaaaaaaa11111aaaaa",
                "block":17873471294,
                "metrics":{
                "PESQ":{
                    "scores":[1.0,1.1,1.2],
                    "average":1.1,
                    "confidence_interval":[1.05,1.15],
                },
                "ESTOI":{
                    "scores":[1.0,1.1,1.2],
                    "average":1.1,
                    "confidence_interval":[1.05,1.15],
                },
                "SI_SDR":{
                    "scores":[1.0,1.1,1.2],
                    "average":1.1,
                    "confidence_interval":[1.05,1.15],
                },
                "SI_SAR":{
                    "scores":[1.0,1.1,1.2],
                    "average":1.1,
                    "confidence_interval":[1.05,1.15],
                },
                "SI_SIR":{
                    "scores":[1.0,1.1,1.2],
                    "average":1.1,
                    "confidence_interval":[1.05,1.15],
                },
                }
            }
            ],
        }
    }
    
    logging_outcome = Benchmarking.miner_models_remote_logging(
        hotkey=hk, 
        current_miner_models=miner_models,
        log_level="TRACE"
    )
    
    assert logging_outcome, "Miner model logging failed."
    
def test_sgmse_remote_logging():
     
    hk = get_hk()
     
    sgmse_benchmark = {
        "category": "sgmse",
        "validator":hk.ss58_address,
        "timestamp":int(time.time()*1000),
        "models": {
            "DENOISING_16000HZ":{
            "PESQ":{
                "scores":[1.0,1.1,1.2],
                "average":1.1,
                "confidence_interval":[1.05,1.15],
            },
            "ESTOI":{
                "scores":[1.0,1.1,1.2],
                "average":1.1,
                "confidence_interval":[1.05,1.15],
            },
            "SI_SDR":{
                "scores":[1.0,1.1,1.2],
                "average":1.1,
                "confidence_interval":[1.05,1.15],
            },
            "SI_SAR":{
                "scores":[1.0,1.1,1.2],
                "average":1.1,
                "confidence_interval":[1.05,1.15],
            },
            "SI_SIR":{
                "scores":[1.0,1.1,1.2],
                "average":1.1,
                "confidence_interval":[1.05,1.15],
            },
            },
            "DEREVERBERATION_16000HZ":{
            "PESQ":{
                "scores":[1.0,1.1,1.2],
                "average":1.1,
                "confidence_interval":[1.05,1.15],
            },
            "ESTOI":{
                "scores":[1.0,1.1,1.2],
                "average":1.1,
                "confidence_interval":[1.05,1.15],
            },
            "SI_SDR":{
                "scores":[1.0,1.1,1.2],
                "average":1.1,
                "confidence_interval":[1.05,1.15],
            },
            "SI_SAR":{
                "scores":[1.0,1.1,1.2],
                "average":1.1,
                "confidence_interval":[1.05,1.15],
            },
            "SI_SIR":{
                "scores":[1.0,1.1,1.2],
                "average":1.1,
                "confidence_interval":[1.05,1.15],
            },
            },
        }
    }
    
    logging_outcome = Benchmarking.sgmse_remote_logging(
        hotkey=hk,
        sgmse_benchmarks=sgmse_benchmark,
        log_level="TRACE"
    )
    
    assert logging_outcome, "SGMSE+ benchmark failed"