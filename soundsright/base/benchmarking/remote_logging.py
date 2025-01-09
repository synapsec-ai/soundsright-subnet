import secrets 
import requests
import time
import json
import bittensor as bt 

import soundsright.base.utils as Utils 

def requests_post(url, headers: dict, data: dict, log_level: str, timeout: int = 12) -> dict:
    """Handles sending remote logs to SYNAPSEC remote logging API"""
    try:
        # Get prompt
        res = requests.post(url=url, headers=headers, data=json.dumps(data), timeout=timeout)
        # Check for correct status code
        if res.status_code == 201:
            return res
        
        Utils.subnet_logger(
            severity="ERROR",
            message=f"Unable to connect to remote host: {url}: HTTP/{res.status_code} - {res.json()}",
            log_level=log_level,
        )
        
        return res
    
    except requests.exceptions.ReadTimeout as e:
        Utils.subnet_logger(
            severity="ERROR",
            message=f"Remote API request timed out: {e}",
            log_level=log_level,
        )
    except requests.exceptions.JSONDecodeError as e:
        Utils.subnet_logger(
            severity="ERROR",
            message=f"Unable to read the response from the remote API: {e}",
            log_level=log_level,
        )
    except requests.exceptions.ConnectionError as e:
        Utils.subnet_logger(
            severity="ERROR",
            message=f"Unable to connect to the remote API: {e}",
            log_level=log_level,
        )
    except Exception as e:
        Utils.subnet_logger(
            severity="ERROR",
            message=f'Generic error during request: {e}',
            log_level=log_level,
        )
        
    return {}
    
def miner_models_remote_logging(hotkey: bt.Keypair, current_miner_models: dict, log_level: str) -> bool:
    """
    Attempts to log the best models from current competition.
    
    Returns: 
        bool: True if logging was successful, False otherwise
    """
    nonce = str(secrets.token_hex(24))
    timestamp = str(int(time.time()))

    signature = Utils.sign_data(hotkey=hotkey, data=f'{nonce}-{timestamp}')

    headers = {
        "X-Hotkey": hotkey.ss58_address,
        "X-Signature": signature,
        "X-Nonce": nonce,
        "X-Timestamp": timestamp,
        "X-API-Key":hotkey.ss58_address
    }
    
    Utils.subnet_logger(
        severity="DEBUG",
        message=f"Sending current models to remote logger. Model data: {current_miner_models}. Headers: {headers}",
        log_level=log_level,
    )

    body = {
        "models":current_miner_models,
        "category":"current"
    }

    res = requests_post(url="https://logs.soundsright.ai/", headers=headers, data=body, log_level=log_level)

    if res and res.status_code == 201:
        
        Utils.subnet_logger(
            severity="DEBUG",
            message="Current model remote logging successful.",
            log_level=log_level,
        )
        
        return True
    
    Utils.subnet_logger(
        severity="ERROR",
        message="Current model remote logging unsuccessful. Please contact subnet owners if issue persists.",
        log_level=log_level,
    )
    
    return False

def sgmse_remote_logging(hotkey: bt.Keypair, sgmse_benchmarks: dict, log_level: str) -> bool:
    
    """
    Attempts to log the best models from current competition.
    
    Returns: 
        bool: True if logging was successful, False otherwise
    """
    nonce = str(secrets.token_hex(24))
    timestamp = str(int(time.time()))

    signature = Utils.sign_data(hotkey=hotkey, data=f'{nonce}-{timestamp}')

    headers = {
        "X-Hotkey": hotkey.ss58_address,
        "X-Signature": signature,
        "X-Nonce": nonce,
        "X-Timestamp": timestamp,
        "X-API-Key":hotkey.ss58_address
    }
    
    Utils.subnet_logger(
        severity="DEBUG",
        message=f"Sending SGMSE+ benchmarks for all competitions on new dataset to remote logger. Model data: {sgmse_benchmarks}. Headers: {headers}",
        log_level=log_level,
    )

    body = {
        "models":sgmse_benchmarks,
        "category":"sgmse"
    }

    res = requests_post(url="https://logs.soundsright.ai/", headers=headers, data=body, log_level=log_level)

    if res and res.status_code == 201:
        
        Utils.subnet_logger(
            severity="DEBUG",
            message="SGMSE+ benchmark remote logging successful.",
            log_level=log_level,
        )
        
        return True
    
    Utils.subnet_logger(
        severity="ERROR",
        message="SGMSE+ benchmark remote logging unsuccessful. Please contact subnet owners if issue persists.",
        log_level=log_level,
    )
    
    return False