import json
import shutil

import valohai
from bionemo.data import UniRef50Preprocess


def prepare_uniref_dataset(source, output_dir):

    data = UniRef50Preprocess()
    data.prepare_dataset(
        source=source,
        output_dir=output_dir,
    )



if __name__ == "__main__":
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
    metadata = {
        "uniref50_train.zip": {
            "valohai.dataset-versions": [
                 "dataset://uniref50/version1"
             ],
        },
        "uniref50_test.zip": {
            "valohai.dataset-versions": [
                "dataset://uniref50/version1"
            ],
        },
        "uniref50_val.zip": {
            "valohai.dataset-versions": [
                "dataset://uniref50/version1"
            ],
        }
    }

    metadata_path = valohai.outputs().path("valohai.metadata.jsonl")
    with open(metadata_path, "w") as outfile:
        for file_name, file_metadata in metadata.items():
            json.dump({"file": file_name, "metadata": file_metadata}, outfile)
            outfile.write("\n")

