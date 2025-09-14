import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
import valohai
import os
import pandas as pd


model_name = "nferruz/ProtGPT2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

sequences = valohai.inputs("sequences").path()

#Convert the sequence to a string like this
#(note we have to introduce new line characters every 60 amino acids,
#following the FASTA file format).

#Load csv from valohai input
df = pd.read_csv(sequences)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ppl function
def calculatePerplexity(sequence, model, tokenizer):
    input_ids = torch.tensor(tokenizer.encode(sequence)).unsqueeze(0) 
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return math.exp(loss)


if __name__ == "__main__":
    model.to(device)
    results = []
    for index, row in df.iterrows():
        seq = row['generated_text']
        ppl = calculatePerplexity(seq, model, tokenizer)
        results.append({"sequence": seq, "perplexity": ppl})
        print(f"Sequence: {seq}\nPerplexity: {ppl}\n")

    results_df = pd.DataFrame(results)

    work_dir = os.environ.get("WORK_DIR")
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
        print(f"Directory '{work_dir}' created.")

    results_path = os.path.join(work_dir, "results")
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    results_csv_path = os.path.join(results_path, "predicted_properties.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"Predicted properties saved to {results_csv_path}")


    print("Inference completed successfully.")