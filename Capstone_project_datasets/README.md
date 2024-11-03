# The Squishifier Data Overview

## Description of the Data
This repository includes three main datasets that are integral to the development of the Squishifier, an AI-powered tool designed to modernize linguistic form by replacing outdated terms with vibrant, meaningful alternatives. The datasets are structured to support various aspects of natural dialect processing and linguistic transformation.

## Datasets

### 1. Exuberant Synonyms for Outdated Terms
- **Source**: Self-created
- **Description**: This dataset is a collection of carefully curated synonyms designed to replace outdated or negatively connoted terms. Each word in this list embodies themes of regeneration and timelessness.
- **Structure**: A simple text file containing a list of terms and their corresponding exuberant synonyms.
- **Location**: `Capstone_project_datasets/Exuberant synonyms for outdated terms.rtf`

### 2. Neologisms
- **Source**: Self-created
- **Description**: This dataset contains newly coined terms (neologisms) that can be integrated into modern communication. These terms were specifically crafted to convey innovation and relevance.
- **Structure**: A text file listing neologisms with brief explanations of their meanings (some of the neologisms do not have definitions yet).
- **Location**: `Capstone_project_datasets/Neologisms.txt`

### 3. WordNet Data
- **Source**: [WordNet by Princeton University](https://wordnet.princeton.edu)
- **Description**: A comprehensive lexical database of English, which includes definitions, synonyms, and semantic relationships. This dataset serves as the backbone for understanding the structure and relationships of words.
- **Structure**: Contains synsets (groups of synonyms) along with definitions and relationships between words.
- **Access**: The WordNet data utilized in this project can be explored and accessed through the NLTK library in Python.

## How to Access the Data

### Exuberant Synonyms and Neologisms
- **Local Files**: These files are included in the `Capstone_project_datasets` directory of this repository:
  - `Capstone_project_datasets/Exuberant synonyms for outdated terms.rtf`
  - `Capstone_project_datasets/Neologisms.txt`
- **Structure**: Both files are simple text documents. The exuberant synonyms file lists terms alongside their replacements, while the neologisms file provides a comprehensive list of new terms.

## How to Access the WordNet Data

To utilize the WordNet data within your projects, execute the following Python commands:

```python
import nltk

# Download WordNet data
nltk.download('wordnet')
```
Documentation
For more information on WordNet, visit the official WordNet site: [WordNet by Princeton University](https://wordnet.princeton.edu)
