"""Psyduck - A simple command-line tool to process CSV files using CrossEncoder from SentenceTransformers."""
#!/usr/bin/env python3

import argparse
import sys
import csv
from pathlib import Path
from sentence_transformers import CrossEncoder

def cross_encoder(queries: list[str], statement: str) -> list[dict]:
    """Function using CrossEncoder to rank queries against a statement."""
    model = CrossEncoder('cross-encoder/stsb-roberta-large')
    ranks = model.rank(statement, queries, return_documents=True)
    return ranks

def load_model(model_path: str) -> dict:
    """Function loading model from CSV file."""
    model = {}
    with open(model_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            item = row['ITEM']
            code = row['CODE']
            statement = row['STATEMENT']
            value = row['VALUE']
            if item not in model:
                model[item] = []
            model[item].append({'code': code, 'statement': statement, 'value': value})
    return model

def read_statements(statement_path: str) -> str:
    """Function reading statements from a text file."""
    with open(statement_path, mode='r', encoding='utf-8') as file:
        lines = [line.strip() for line in file if line.strip()]
    return " ".join(lines)

def run_command(model_path: str, statement_path: str):
    """Function to run the command with the given model and statement paths."""
    model_file = Path(model_path)
    if not model_file.exists():
        print(f"Error: Model file '{model_file}' does not exist.")
        sys.exit(1)
    if not str(model_file).lower().endswith('.csv'):
        print(f"Error: '{model_file}' is not a CSV file.")
        sys.exit(1)

    statement_file = Path(statement_path)
    if not statement_file.exists():
        print(f"Error: Statement file '{statement_file}' does not exist.")
        sys.exit(1)

    model = load_model(model_file)
    statement = read_statements(statement_file)

    for item, responses in model.items():
        queries = [resp['statement'] for resp in responses]
        rank = cross_encoder(queries, statement)
        argmax = max(rank, key=lambda x: x['score'])['corpus_id']
        most_similar = queries[argmax]
        print(f"{item}: {most_similar}")


def main():
    """Main function to parse arguments and run commands."""
    parser = argparse.ArgumentParser(description="Psyduck - CSV processing tool")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    run_parser = subparsers.add_parser('run', help='Process a CSV file')
    run_parser.add_argument('model', type=str, help='Path to the model file (CSV)')
    run_parser.add_argument('statement', type=str, help='Path to the statement file to process')

    args = parser.parse_args()

    if args.command == 'run':
        run_command(args.model, args.statement)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
