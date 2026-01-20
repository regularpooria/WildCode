import os
import yaml

ROOT_FOLDER = "opengrep-rules/"  # replace with your path


def extract_cwes_from_yaml(yaml_path, target_ids):
    with open(yaml_path, "r") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            # print(f"Error parsing {yaml_path}: {e}")
            return {}

    cwe_map = {}

    if not data or "rules" not in data:
        return cwe_map

    for rule in data["rules"]:
        rule_id = rule.get("id")
        if rule_id in target_ids:
            # Extract CWE list if available
            cwes = []
            metadata = rule.get("metadata", {})
            if "cwe" in metadata:
                # The CWE list might be a list of strings
                cwes = metadata["cwe"]
                # Sometimes might be just a single string - normalize to list
                if isinstance(cwes, str):
                    cwes = [cwes]
            cwe_map[rule_id] = cwes
    return cwe_map


def walk_and_extract_cwe(target_ids):
    all_cwes = {}
    for dirpath, _, filenames in os.walk(ROOT_FOLDER):
        for filename in filenames:
            if filename.endswith((".yaml", ".yml")):
                path = os.path.join(dirpath, filename)
                cwe_for_file = extract_cwes_from_yaml(path, target_ids)
                for rule_id, cwes in cwe_for_file.items():
                    # accumulate results; if same rule id occurs multiple times, merge CWEs
                    if rule_id in all_cwes:
                        all_cwes[rule_id].update(cwes)
                    else:
                        all_cwes[rule_id] = set(cwes)
    # convert sets back to lists for easier use
    return {rule: list(cwes) for rule, cwes in all_cwes.items()}
