import math
import os
from typing import List, Dict, Any

import pandas as pd
import torch
import valohai
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

# Model and tokenizer initialization
model_name: str = "nferruz/ProtGPT2"
tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_name)

sequences: str = valohai.inputs("sequences").path()

# Load CSV from valohai input
df: pd.DataFrame = pd.read_csv(sequences)

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculatePerplexity(
    sequence: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer
) -> float:
    """
    Calculate perplexity of a given sequence using the provided model and tokenizer.

    Args:
        sequence (str): Input sequence (amino acid sequence).
        model (PreTrainedModel): HuggingFace Causal LM model.
        tokenizer (PreTrainedTokenizer): HuggingFace tokenizer.

    Returns:
        float: Perplexity score of the sequence.
    """
    input_ids: torch.Tensor = torch.tensor(tokenizer.encode(sequence)).unsqueeze(0)
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return math.exp(loss)


if __name__ == "__main__":
    model.to(device)
    results: List[Dict[str, Any]] = []

    for index, row in df.iterrows():
        seq: str = row['generated_text']
        ppl: float = calculatePerplexity(seq, model, tokenizer)
        results.append({"sequence": seq, "perplexity": ppl})
        print(f"Sequence: {seq}\nPerplexity: {ppl}\n")

    results_df: pd.DataFrame = pd.DataFrame(results)

    work_dir: str = os.environ.get("WORK_DIR", "./outputs")
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
        print(f"Directory '{work_dir}' created.")

    results_path: str = os.path.join(work_dir, "results")
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    results_csv_path: str = os.path.join(results_path, "predicted_properties.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"Predicted properties saved to {results_csv_path}")

    print("Inference completed successfully.")
