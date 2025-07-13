# OpenGrep Kaggle Dataset Analysis

This project performs static security analysis on code snippets from a Kaggle dataset using [OpenGrep](https://github.com/opengrep/opengrep). It processes multiple programming languages, applies language-specific security rules from the `opengrep-rules` repository, and converts results into CSV files for further analysis.

## 📦 Dataset

Code samples are sourced from the [COD-Blocks](https://www.kaggle.com/datasets/wilfriedkonan/cod-blocks) dataset, which includes:

- C  
- C#  
- CSS  
- HTML  
- Java  
- JavaScript  
- PHP  
- Python  
- SQL

The dataset is accessed using `kagglehub`, and individual code samples are saved to `files/<language>/codes/`.

## 🔧 Setup Instructions

### Run the colab version of this project, which requires no installing
https://colab.research.google.com/drive/1o1NmgBeRMV3F7y2dYRMJFrLg3r2xwHSx?usp=sharing


### 1. Clone the Repository and Submodules

```bash
git clone --recurse-submodules https://github.com/regularpooria/code_analysis_uqo
cd code_analysis_uqo

```

### 2. Run the commands in run_static_analysis_tool.ipynb or run the file run_static_analysis_tool.py



## 🗂 Directory Structure

```
files/
  └── python/
      ├── codes/
      ├── rules/
      ├── output.sarif
      └── python.csv
```

## 📑 Output Example (CSV)

Each CSV row contains:

- `conversation_hash`  
- `code_index`  
- `error_line`  
- `error_character`  
- `error_id`  
- `error_message`

| conversation_hash | code_index | error_line | error_character | error_id           | error_message               |
|-------------------|------------|------------|------------------|---------------------|-----------------------------|
| abc123            | 001        | 42         | 5                | insecure-temp-file | Temporary file is insecure |
