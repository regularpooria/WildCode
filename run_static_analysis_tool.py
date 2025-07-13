#!/usr/bin/env python
# coding: utf-8

# ### Install the necessary tools for this script

# In[ ]:


get_ipython().system("git clone https://github.com/opengrep/opengrep-rules.git")
get_ipython().system(
    "curl -fsSL https://raw.githubusercontent.com/opengrep/opengrep/main/install.sh | bash"
)
get_ipython().system("pip install kagglehub[pandas-datasets]")


# ### Importing necessary modules

# In[ ]:


import kagglehub
from kagglehub import KaggleDatasetAdapter
from pandas import DataFrame

import os
import json
import shutil
import csv


# In[ ]:


def load_dataset(file_path):
    df: DataFrame = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "wilfriedkonan/cod-blocks",
        file_path,
    )
    return df


datasets = {
    "c": "c.json",
    "csharp": "csharp.json",
    "css": "css.json",
    "html": "html.json",
    "java": "java.json",
    "javascript": "javascript.json",
    "php": "php.json",
    "python": "python.json",
    "sql": "sql.json",
}


# Looping through each language in the Kaggle dataset and turning the .json files into actual files, then saving them to files/LANGUAGE/codes

# In[ ]:


for language in datasets.keys():
    os.makedirs(f"files/{language}/codes/", exist_ok=True)
    os.makedirs(f"files/{language}/rules/", exist_ok=True)

    df = load_dataset(datasets[language])
    for index, data_point in df.iterrows():
        with open(
            f"files/{language}/codes/{data_point['filename']}", "w", encoding="utf-8"
        ) as f:
            f.write(data_point["code"])


# ### Filtering for security rules in codegrep-rules repository

# In[ ]:


def copy_security_yaml_rules(src_root: str, dst_root: str):
    """
    Walk src_root, find all .yaml files under any 'security' folder,
    and copy them to dst_root, preserving subdirectory structure.
    """
    for root, dirs, files in os.walk(src_root):
        # only consider paths that have 'security' in their hierarchy
        if "security" in root.split(os.sep):
            for file in files:
                if file.endswith(".yaml"):
                    # compute relative path under src_root
                    rel_dir = os.path.relpath(root, src_root)
                    dst_dir = os.path.join(dst_root, rel_dir)
                    os.makedirs(dst_dir, exist_ok=True)

                    src_file = os.path.join(root, file)
                    dst_file = os.path.join(dst_dir, file)
                    shutil.copy2(src_file, dst_file)
                    print(f"Copied: {rel_dir}/{file}")


# In[ ]:


for language in datasets.keys():
    if not os.path.exists(f"opengrep-rules/{language}"):
        continue
    copy_security_yaml_rules(f"opengrep-rules/{language}", f"files/{language}/rules/")


# ### Runing static analysis tool
# Looping through each language and running the codegrep static analysis tool on them, and saving the results in files/language/output.sarif

# In[ ]:


for language in datasets.keys():
    if os.path.exists(f"opengrep-rules/{language}"):
        get_ipython().system(
            "/root/.opengrep/cli/latest/opengrep scan --sarif-output=files/{language}/output.sarif -f files/{language}/rules files/{language}/codes"
        )


# ### Converting Sarif files into CSV
# (for ease of use)

# In[ ]:


for language in datasets.keys():
    if os.path.exists(f"opengrep-rules/{language}"):
        rows = []
        sarif_path = f"files/{language}/output.sarif"
        if not os.path.exists(sarif_path):
            continue

        with open(sarif_path, "r", encoding="utf-8") as f:
            data = json.loads(f.read())
            for run in data["runs"]:
                for result in run.get("results", []):
                    message = result.get("message", {}).get("text", "")
                    rule_id = result.get("ruleId", "")

                    # Some results may have multiple locations
                    for location in result.get("locations", []):
                        loc = location.get("physicalLocation", {})
                        artifact = loc.get("artifactLocation", {})
                        region = loc.get("region", {})

                        conversation_hash = (
                            artifact.get("uri", "").split("/")[-1].split("_")[0]
                        )
                        code_index = (
                            artifact.get("uri", "")
                            .split("/")[-1]
                            .split("_")[1]
                            .split(".")[0]
                        )
                        start_line = region.get("startLine", "")
                        start_column = region.get("startColumn", "")

                        rows.append(
                            [
                                conversation_hash,
                                code_index,
                                start_line,
                                start_column,
                                rule_id,
                                message,
                            ]
                        )

        # Write to CSV
        csv_path = f"files/{language}/{language}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "conversation_hash",
                    "code_index",
                    "error_line",
                    "error_character",
                    "error_id",
                    "error_message",
                ]
            )
            writer.writerows(rows)

        print(f"CSV written to: {csv_path}")
