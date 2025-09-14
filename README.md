# Psyduck

A text similarity analysis tool that uses cross-encoder models to find the most relevant responses from a predefined set of options based on input statements.

## Overview

Psyduck processes CSV model files containing predefined statements and uses sentence transformers with cross-encoder models to find the best matching responses for given input statements. It's particularly useful for survey analysis, sentiment matching, and response categorization tasks.

## Features

- **Cross-Encoder Analysis**: Uses advanced transformer models for semantic similarity matching
- **CSV Model Processing**: Supports structured CSV files with items, codes, statements, and values
- **Text File Input**: Processes plain text statements for analysis
- **Command-Line Interface**: Easy-to-use CLI with argparse

## Get Started

### Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd psyduck
```

2. Sync the project dependencies using uv:
```bash
uv sync
```

This will automatically create a virtual environment and install all required dependencies including:
- `sentence-transformers` for cross-encoder models
- `scikit-learn` for additional ML utilities
- `numpy` for numerical operations

### Usage

Run the tool using the following command:

```bash
./main.py run <model_csv_file> <statement_file>
```

#### Example

```bash
./main.py run samples/test.csv inputs/test.txt
```

### CSV File Format

Your model CSV file **MUST** have the following structure:

```csv
ITEM,CODE,STATEMENT,VALUE
1,1_A,"Love",1
1,1_B,"Like",2
1,1_C,"Neutral",3
1,1_D,"Dislike",4
1,1_E,"Hate",5
2,2_A,"Happy",1
2,2_B,"Content",2
2,2_C,"Indifferent",3
2,2_D,"Sad",4
2,2_E,"Angry",5
```

**Columns:**
- `ITEM`: Grouping identifier for related statements
- `CODE`: Unique code for each response option
- `STATEMENT`: The actual text statement/response
- `VALUE`: Numerical value associated with the statement

### Statement File Format

The statement file should be a plain text file containing the statement you want to analyze:

```
Would I rather be feared or loved? Easy. Both. I want people to be afraid of how much they love me.
```

### How It Works

1. **Model Loading**: The tool loads your CSV file and groups statements by ITEM
2. **Statement Processing**: Reads the input statement from the text file
3. **Cross-Encoder Analysis**: Uses a pre-trained cross-encoder model (`cross-encoder/stsb-roberta-large`) to rank similarity between the input statement and all predefined statements
4. **Best Match Selection**: For each item group, finds the statement with the highest similarity score
5. **Results Output**: Displays the most similar statement for each item

### Example Output

```
1: Like
2: Content
```

This means that for item group 1, "Like" was the most similar to your input statement, and for item group 2, "Content" was the best match.