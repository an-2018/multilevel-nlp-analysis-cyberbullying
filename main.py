import subprocess
import sys
import os
from pathlib import Path

def run_script(script_path, args=None):
    """
    Runs a Python script using subprocess, handling arguments and checking for errors.

    Args:
        script_path (Path): The path to the Python script to run.
        args (list, optional): A list of command-line arguments for the script. Defaults to None.

    Returns:
        bool: True if the script ran successfully, False otherwise.
    """
    command = [sys.executable, str(script_path)] # Use the same Python interpreter
    if args:
        command.extend(args)

    print(f"\n----- Running {script_path.name} -----")
    print(f"Command: {' '.join(command)}")

    try:
        # Use check=True to automatically raise CalledProcessError on non-zero exit codes
        # Capture output for better debugging if needed (optional)
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        print(f"Output from {script_path.name}:\n{result.stdout}")
        if result.stderr:
            print(f"Stderr from {script_path.name}:\n{result.stderr}", file=sys.stderr)
        print(f"----- {script_path.name} finished successfully -----")
        return True
    except FileNotFoundError:
        print(f"Error: Script not found at {script_path}", file=sys.stderr)
        return False
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path.name}:", file=sys.stderr)
        print(f"Return code: {e.returncode}", file=sys.stderr)
        print(f"Output:\n{e.stdout}", file=sys.stderr)
        print(f"Stderr:\n{e.stderr}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"An unexpected error occurred while running {script_path.name}: {e}", file=sys.stderr)
        return False

def main():
    """
    Orchestrates the execution of the entire pipeline.
    """
    # --- Configuration ---
    # Assuming the scripts are in the 'src' directory relative to this pipeline script
    # Adjust base_dir if your structure is different
    base_dir = Path(__file__).parent # Directory containing this pipeline script
    src_dir = base_dir / "src"
    data_dir = base_dir / "data" / "processed"
    results_dir = base_dir / "results"
    models_dir = base_dir / "models"

    # Ensure necessary directories exist (optional, scripts might create them)
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Define script paths
    scripts = {
        "phase1": src_dir / "phase1-tfidf.py",
        "phase2": src_dir / "phase2-feature-extraction.py",
        "phase3": src_dir / "phase3-model-training.py",
        "phase4": src_dir / "phase4-feature-fusion.py",
        "phase8": src_dir / "phase8-xai.py",
    }

    # Define arguments for scripts that need them
    # Phase 2 needs input (output of phase 1) and output paths
    phase2_input = data_dir / "phase1_output.csv"
    phase2_output = data_dir / "phase2_output.csv" # Phase 3 expects this name
    phase2_args = [str(phase2_input), str(phase2_output)]

    # --- Execution Pipeline ---
    pipeline_steps = [
        ("phase1", scripts["phase1"], None), # Assumes phase1 uses relative paths internally
        ("phase2", scripts["phase2"], phase2_args),
        ("phase3", scripts["phase3"], None), # Assumes phase3 uses relative paths internally
        ("phase4", scripts["phase4"], None), # Assumes phase4 uses relative paths internally
        ("phase8", scripts["phase8"], None), # Assumes phase8 uses relative paths internally
    ]

    for name, script_path, script_args in pipeline_steps:
        if not run_script(script_path, script_args):
            print(f"\nPipeline stopped due to error in {name}.")
            sys.exit(1) # Exit the pipeline script with an error code

    print("\n===== Pipeline completed successfully! =====")

if __name__ == "__main__":
    main()