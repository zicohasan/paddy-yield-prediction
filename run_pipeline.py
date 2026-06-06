import os
import subprocess
import time
import sys

# Workspace base directory
WORKSPACE_DIR = r"C:\Users\DELL LATITUDE E5420\Documents\ITS"

# Sequence of scripts to run in order
PIPELINE_SCRIPTS = [
    "extract_coordinates.py",
    "fetch_weather.py",
    "query_satellite_weather.py",
    "build_combined_dataset.py",
    "train_combined_models.py",
    "generate_satellite_figure.py"
]

def run_script(script_name):
    script_path = os.path.join(WORKSPACE_DIR, script_name)
    print(f"\n============================================================")
    print(f"STEP: Running {script_name}...")
    print(f"============================================================")
    
    t0 = time.time()
    # Execute python process and stream stdout to console
    # Using sys.executable to ensure we use the current conda environment python
    process = subprocess.Popen([sys.executable, script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Stream stdout line-by-line
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
            
    # Capture remaining stderr
    rc = process.poll()
    if rc != 0:
        print("\nERROR:")
        stderr = process.stderr.read()
        print(stderr)
        print(f"\n[FAILED] {script_name} terminated early with exit code {rc}")
        return False
        
    print(f"\n[SUCCESS] Completed {script_name} in {time.time() - t0:.1f}s.")
    return True

def main():
    t_start = time.time()
    print("=" * 60)
    print("MASTER WORKFLOW COORDINATOR PIPELINE FOR YIELD PREDICTION")
    print("=" * 60)
    print(f"Workspace directory: {WORKSPACE_DIR}")
    
    # Run each step in order
    for idx, script in enumerate(PIPELINE_SCRIPTS, 1):
        print(f"\n[Pipeline Progress] Running Step {idx}/{len(PIPELINE_SCRIPTS)}")
        if not run_script(script):
            print("\nPipeline failed early due to errors in a preceding step.")
            sys.exit(1)
            
    print("\n" + "=" * 60)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {time.time() - t_start:.1f}s!")
    print("=" * 60)
    print("All results, CSV outputs, comparison plots, and study area NDVI maps are generated and saved.")

if __name__ == "__main__":
    main()
