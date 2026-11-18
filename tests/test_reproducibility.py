import os
import hashlib
import pytest
import yaml
import pandas as pd
from src.data.stream_loader import ElectricityStreamLoader

def test_frozen_config_hash_integrity():
    """
    Confirms that configs/frozen.yaml matches the exact SHA-256 recorded in configs/frozen_hash.txt.
    Prevents silent test-stream overfitting or configuration drift.
    """
    config_path = "configs/frozen.yaml"
    hash_path = "configs/frozen_hash.txt"
    
    assert os.path.exists(config_path), "configs/frozen.yaml is missing!"
    assert os.path.exists(hash_path), "configs/frozen_hash.txt is missing!"
    
    with open(config_path, "rb") as f:
        computed_hash = hashlib.sha256(f.read()).hexdigest()
        
    with open(hash_path, "r", encoding="utf-8") as f:
        line = f.read().strip()
        expected_hash = line.replace("SHA-256:", "").strip()
        
    assert computed_hash == expected_hash, (
        f"Frozen config hash mismatch! Expected: {expected_hash}, Computed: {computed_hash}. "
        "Do not edit frozen.yaml silently!"
    )

def test_dataset_and_split_integrity():
    """
    Confirms the dataset dimensions and exact frozen 30% / 70% stream splits.
    """
    with open("configs/frozen.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    csv_path = cfg["data"]["csv_path"]
    assert os.path.exists(csv_path), f"Dataset {csv_path} does not exist!"
    
    df = pd.read_csv(csv_path)
    assert len(df) == 35064, f"Dataset must have exactly 35,064 rows, got {len(df)}"
    
    loader = ElectricityStreamLoader(csv_path)
    full_stream = list(loader.iter_stream(warmup_rows=24))
    assert len(full_stream) == 35040, f"Full stream must have 35,040 instances, got {len(full_stream)}"
    
    val_instances = cfg["data"]["val_instances"]
    test_instances = cfg["data"]["test_instances"]
    assert val_instances + test_instances == len(full_stream), (
        f"Split mismatch: {val_instances} + {test_instances} != {len(full_stream)}"
    )
    assert val_instances == 10512, f"Validation split must be 10,512, got {val_instances}"
    assert test_instances == 24528, f"Test split must be 24,528, got {test_instances}"



def test_frozen_v2_config_hash_integrity():
    """
    Confirms that configs/frozen_v2.yaml matches the exact SHA-256 recorded in configs/frozen_v2_hash.txt.
    """
    config_path = "configs/frozen_v2.yaml"
    hash_path = "configs/frozen_v2_hash.txt"
    
    assert os.path.exists(config_path), "configs/frozen_v2.yaml is missing!"
    assert os.path.exists(hash_path), "configs/frozen_v2_hash.txt is missing!"
    
    with open(config_path, "rb") as f:
        computed_hash = hashlib.sha256(f.read()).hexdigest()
        
    with open(hash_path, "r", encoding="utf-8") as f:
        line = f.read().strip()
        expected_hash = line.replace("SHA-256:", "").strip()
        
    assert computed_hash == expected_hash, (
        f"Frozen v2 config hash mismatch! Expected: {expected_hash}, Computed: {computed_hash}."
    )


def test_frozen_v3_config_hash_integrity():
    """
    Confirms that configs/frozen_v3.yaml matches the exact SHA-256 recorded in configs/frozen_v3_hash.txt.
    """
    config_path = "configs/frozen_v3.yaml"
    hash_path = "configs/frozen_v3_hash.txt"
    
    assert os.path.exists(config_path), "configs/frozen_v3.yaml is missing!"
    assert os.path.exists(hash_path), "configs/frozen_v3_hash.txt is missing!"
    
    with open(config_path, "rb") as f:
        computed_hash = hashlib.sha256(f.read()).hexdigest()
        
    with open(hash_path, "r", encoding="utf-8") as f:
        line = f.read().strip()
        expected_hash = line.replace("SHA-256:", "").strip()
        
    assert computed_hash == expected_hash, (
        f"Frozen v3 config hash mismatch! Expected: {expected_hash}, Computed: {computed_hash}."
    )
