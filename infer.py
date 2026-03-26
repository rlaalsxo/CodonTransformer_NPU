import argparse
import json
import os
from datetime import datetime

import torch

from CodonTransformer.CodonPrediction import predict_dna_sequence
from CodonTransformer.CodonUtils import ORGANISM2ID


def parse_args():
    parser = argparse.ArgumentParser(description="CodonTransformer DNA sequence prediction")

    parser.add_argument("--protein", type=str, required=True, help="Input protein sequence")
    parser.add_argument("--organism", type=str, default="Escherichia coli general", help="Organism name (default: Escherichia coli general)")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu, cuda, cuda:0, rngd:0 (default: cpu)")
    parser.add_argument("--model-path", type=str, default=None, help="Path to model file (.pt/.ckpt). If not set, loads from HuggingFace")
    parser.add_argument("--deterministic", action="store_true", default=True, help="Use greedy decoding (default: True)")
    parser.add_argument("--no-deterministic", action="store_true", help="Use nucleus sampling instead of greedy decoding")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature (default: 0.2)")
    parser.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling top-p (default: 0.95)")
    parser.add_argument("--num-sequences", type=int, default=1, help="Number of sequences to generate (default: 1)")
    parser.add_argument("--match-protein", action="store_true", help="Force predicted DNA to translate back to input protein")
    parser.add_argument("--output", type=str, default=None, help="Output file path (.json or .fasta). If not set, prints to stdout")
    parser.add_argument("--list-organisms", action="store_true", help="List all supported organisms and exit")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_organisms:
        for name, org_id in sorted(ORGANISM2ID.items()):
            print(f"  [{org_id:3d}] {name}")
        return

    deterministic = not args.no_deterministic

    device = torch.device(args.device)

    results = predict_dna_sequence(
        protein=args.protein,
        organism=args.organism,
        device=device,
        model=args.model_path,
        deterministic=deterministic,
        temperature=args.temperature,
        top_p=args.top_p,
        num_sequences=args.num_sequences,
        match_protein=args.match_protein,
    )

    if not isinstance(results, list):
        results = [results]

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

        if args.output.endswith(".fasta") or args.output.endswith(".fa"):
            with open(args.output, "w") as f:
                for i, result in enumerate(results):
                    header = f"seq_{i + 1}|{result.organism}" if len(results) > 1 else f"seq|{result.organism}"
                    f.write(f">{header}\n{result.predicted_dna}\n")
        else:
            output_data = [
                {
                    "organism": r.organism,
                    "protein": r.protein,
                    "predicted_dna": r.predicted_dna,
                }
                for r in results
            ]
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)

        print(f"Saved to {args.output}")
    else:
        for i, result in enumerate(results):
            if len(results) > 1:
                print(f"--- Sequence {i + 1} ---")
            print(f"Organism: {result.organism}")
            print(f"Protein:  {result.protein}")
            print(f"DNA:      {result.predicted_dna}")
            if len(results) > 1:
                print()


if __name__ == "__main__":
    main()
