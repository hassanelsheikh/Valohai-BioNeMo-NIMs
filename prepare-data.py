import json
import shutil
from typing import Dict, Any
import valohai
from bionemo.data import UniRef50Preprocess
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare UniRef50 dataset using BioNeMo preprocessing.')
    parser.add_argument('--data_version', type=str, default='v1.0', help='Version of the dataset to use when saving preprocessed data.')
    return parser.parse_args()

def prepare_uniref_dataset(source: str, output_dir: str):
    """
    Prepare UniRef50 dataset using BioNeMo preprocessing.

    Args:
        source (str): Dataset source (e.g., "uniprot").
        output_dir (str): Directory where processed dataset will be stored.
    """
    data = UniRef50Preprocess()
    data.prepare_dataset(
        source=source,
        output_dir=output_dir,
    )



if __name__ == "__main__":
    args = parse_args()

    prepare_uniref_dataset(
        source="uniprot",
        output_dir="/valohai/outputs/uniref50",
    )


    train_zipped_path = valohai.outputs().path("uniref50_train")
    test_zipped_path = valohai.outputs().path("uniref50_test")
    val_zipped_path = valohai.outputs().path("uniref50_val")

    shutil.make_archive(train_zipped_path, 'zip', "/valohai/outputs/uniref50/train")
    shutil.make_archive(test_zipped_path, 'zip', "/valohai/outputs/uniref50/test")
    shutil.make_archive(val_zipped_path, 'zip', "/valohai/outputs/uniref50/val")

    # Save Valohai metadata
    metadata: Dict[str, Dict[str, Any]] = {
        "uniref50_train.zip": {
            "valohai.dataset-versions": [
                 f"dataset://uniref50/{args.data_version}"
             ],
        },
        "uniref50_test.zip": {
            "valohai.dataset-versions": [
                f"dataset://uniref50/{args.data_version}"
            ],
        },
        "uniref50_val.zip": {
            "valohai.dataset-versions": [
                f"dataset://uniref50/{args.data_version}"
            ],
        }
    }

    metadata_path: str = valohai.outputs().path("valohai.metadata.jsonl")
    with open(metadata_path, "w") as outfile:
        for file_name, file_metadata in metadata.items():
            json.dump({"file": file_name, "metadata": file_metadata}, outfile)
            outfile.write("\n")
