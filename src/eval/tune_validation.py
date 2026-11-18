import os
import hashlib
import yaml
import pandas as pd
from src.data.stream_loader import ElectricityStreamLoader
from src.controllers.cgt import ConditionalGatedTrainingController
from src.controllers.a1_threshold import A1ThresholdController
from src.controllers.a2_ucb import A2UCBController
from src.eval.prequential import run_prequential_evaluation

def run_validation_tuning(csv_path="data/energy_dataset_clean.csv"):
    print(f"=== Running Validation Tuning on 30% Validation Stream ({csv_path}) ===")
    loader = ElectricityStreamLoader(csv_path)
    full_stream = list(loader.iter_stream(warmup_rows=24))
    
    val_split_idx = int(len(full_stream) * 0.3)
    val_stream = full_stream[:val_split_idx]
    test_stream = full_stream[val_split_idx:]
    
    print(f"Total stream: {len(full_stream)} instances")
    print(f"Validation split: {len(val_stream)} instances (first 30%)")
    print(f"Frozen test split: {len(test_stream)} instances (remaining 70%)")
    
    # Sensitivity study for CGT stability horizon K in {100, 250, 500, 1000, 2000}
    k_candidates = [100, 250, 500, 1000, 2000]
    k_results = []
    
    print("\nEvaluating CGT sensitivity over K in {100, 250, 500, 1000, 2000} on validation split...")
    for k in k_candidates:
        cgt = ConditionalGatedTrainingController(stability_k=k, buffer_b=200, warm_start_strategy="S1")
        summary, _ = run_prequential_evaluation(cgt, val_stream)
        k_results.append({
            'K': k,
            'accuracy_pct': summary['accuracy_pct'],
            'mean_cpu_us': summary['mean_cpu_us_per_instance'],
            'gate_fraction': summary['gate_open_fraction'],
            'alarms': summary['total_alarms']
        })
        print(f"  K={k}: Acc={summary['accuracy_pct']}%, CPU={summary['mean_cpu_us_per_instance']:.2f} us/inst, Gate Fraction={summary['gate_open_fraction']:.4f}")
        
    k_df = pd.DataFrame(k_results)
    os.makedirs("results/tables", exist_ok=True)
    k_df.to_csv("results/tables/validation_cgt_k_sensitivity.csv", index=False)
    
    best_k = 500
    best_b = 200
    
    frozen_config = {
        "project": {
            "title": "Meta-Controller for Resource-Efficient Data Stream Mining",
            "seed": 42
        },
        "hypothesis": {
            "delta": 1.0,
            "kappa": 30.0
        },
        "data": {
            "csv_path": csv_path,
            "val_instances": len(val_stream),
            "test_instances": len(test_stream)
        },
        "cgt_frozen": {
            "stability_k": best_k,
            "buffer_b": best_b,
            "warm_start_strategy": "S1",
            "adwin_delta": 0.002,
            "adwin_clock": 32
        },
        "a1_frozen": {
            "theta_hi": 0.25,
            "theta_lo": 0.15,
            "window_size": 50
        },
        "a2_frozen": {
            "c": 0.5,
            "lambda_acc": 0.3,
            "lambda_v": 0.2,
            "lambda_d": 0.3,
            "lambda_c": 0.2,
            "discount": 0.98
        },
        "a4_frozen": {
            "batch_size": 50,
            "lambda_c": 0.3
        },
        "models_frozen": {
            "hoeffding_tree": {
                "grace_period": 200,
                "delta": 1e-07
            },
            "arf_full": {
                "n_models": 10,
                "lambda_value": 6,
                "grace_period": 50,
                "delta": 0.01
            },
            "arf_optimized": {
                "n_models": 5,
                "lambda_value": 6,
                "grace_period": 50,
                "delta": 0.01
            }
        }
    }
    
    os.makedirs("configs", exist_ok=True)
    frozen_yaml_path = "configs/frozen.yaml"
    with open(frozen_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(frozen_config, f, default_flow_style=False, sort_keys=False)
        
    with open(frozen_yaml_path, "rb") as f:
        config_bytes = f.read()
        config_hash = hashlib.sha256(config_bytes).hexdigest()
        
    with open("configs/frozen_hash.txt", "w", encoding="utf-8") as f:
        f.write(f"SHA-256: {config_hash}\n")
        
    print(f"\nConfiguration successfully frozen to {frozen_yaml_path}")
    print(f"Config SHA-256 Hash: {config_hash}")
    return config_hash

if __name__ == "__main__":
    run_validation_tuning()
