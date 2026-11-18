import os
import sys
import subprocess
from src.data.audit_data import run_data_audit
from src.eval.tune_validation import run_validation_tuning
from src.eval.run_benchmark import run_benchmark
from src.stats.statistical_tests import run_statistical_tests
from src.plots.generate_plots import generate_all_plots

def main():
    print("==========================================================")
    print("REPRODUCIBILITY PIPELINE: RE-RUNNING ALL EXPERIMENTS")
    print("==========================================================")
    
    # 1. Audit and prepare data
    print("\n[Step 1/5] Running Data Audit and Negative Solar Clipping...")
    run_data_audit()
    
    # 2. Automated tests
    print("\n[Step 2/5] Running Automated Leakage & Reproducibility Tests...")
    test_res = subprocess.run([sys.executable, "-m", "pytest", "tests/"], capture_output=True, text=True)
    print(test_res.stdout)
    if test_res.returncode != 0:
        print("ERROR: Tests failed!")
        sys.exit(1)
        
    # 3. Validation tuning and config freeze verification
    print("\n[Step 3/5] Verifying Frozen Configurations...")
    run_validation_tuning()
    
    # 4. Final frozen benchmark
    print("\n[Step 4/5] Running Final Frozen Test Stream Benchmark...")
    run_benchmark()
    
    # 5. Statistical testing and publication plots
    print("\n[Step 5/5] Generating Statistical Tests & Publication Figures...")
    run_statistical_tests()
    generate_all_plots()
    
    print("\n==========================================================")
    print("ALL EXPERIMENTS COMPLETED REPRODUCIBLY. SEE RESULTS.md")
    print("==========================================================")

if __name__ == '__main__':
    main()
