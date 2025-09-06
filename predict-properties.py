import shutil
import warnings
import subprocess, sys, os

import pandas as pd
import torch
from bionemo.core.data.load import load
import argparse
import valohai

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description='Predict protein properties using a pre-trained model.')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--precision', type=str, default='fp16')
    parser.add_argument('--micro-batch-size', type=int, default=8)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    work_dir = os.environ.get("WORK_DIR")

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
        print(f"Directory '{work_dir}' created.")

    # Extract zipped dataset
    data_archive = valohai.inputs('dataset').path(process_archives=False)
    extract_dir = os.path.join(os.path.dirname(data_archive), "extracted_data")

    # Unzip the dataset
    shutil.unpack_archive(data_archive, extract_dir, format='zip')
    print(f"Dataset extracted to: {extract_dir}")


    # Collect sequences
    all_sequences = []

    
    # Walk through files
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                if "sequence" in df.columns:
                    all_sequences.extend(df["sequence"].dropna().tolist())

    print(f"Loaded {len(all_sequences)} sequences from CSV files")
    # Put into DataFrame
    df = pd.DataFrame(all_sequences, columns=["sequences"])




    checkpoint_path = load("esm2/650m:2.0")
    print(checkpoint_path)



    # Save the DataFrame to a CSV file
    data_path = os.path.join(work_dir, "sequences.csv")
    df.to_csv(data_path, index=False)

    ckpt_path = load("esm2/650m:2.0")
    results_path = os.path.join(work_dir, "results")

    # Build the command for using esm2 inference script
    cmd = ["infer_esm2",
        "--checkpoint-path", str(ckpt_path),
        "--data-path", str(data_path),
        "--results-path", str(results_path),
        "--precision", str(args.precision),
        "--num-gpus", str(args.num_gpus),
        "--micro-batch-size", str(args.micro_batch_size),
        "--include-embeddings"]

    print("Running:", " ".join(cmd))

    # Run it and stream logs to console
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")

    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"infer_esm2 failed with code {process.returncode}")
    else:
        print("Inference finished. Results are in:", results_path)