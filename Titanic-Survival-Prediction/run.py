import subprocess
import sys
import os

def run_step(script, label):
    print(f"\n{'='*60}")
    print(f"  RUNNING: {label}")
    print(f"{'='*60}\n")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"\n❌ ERROR in {label}. Stopping.")
        sys.exit(1)
    print(f"\n✅ {label} complete!")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    run_step("src/data_preprocessing.py", "Step 1: Data Preprocessing")
    run_step("src/model_training.py",     "Step 2: Model Training")
    run_step("src/predict.py",            "Step 3: Predictions")

    print("\n" + "="*60)
    print("  ALL DONE! Check outputs/ folder for results.")
    print("="*60)
