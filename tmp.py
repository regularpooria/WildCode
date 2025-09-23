import os, json

datasets = {
    "c": "c_cleaned.json",
    "csharp": "csharp.json",
    "java": "java_cleaned.json",
    "javascript": "javascript_cleaned.json",
    "php": "php.json",
    "python": "python_cleaned.json",
}


tracker = {
    "c": 0,
    "csharp": 0,
    "java": 0,
    "javascript": 0,
    "php": 0,
    "python": 0,
}

extensions = {
    "c": ".c",
    "csharp": ".cs",
    "java": ".java",
    "javascript": ".js",
    "php": ".php",
    "python": ".py",
}

df = []
with open("other_cleaned.json", "r", encoding="utf-8") as f:
    raw = f.readlines()
    for line in raw:
        df.append(json.loads(line))

for language in datasets.keys():
    os.makedirs(f"files/{language}/codes/", exist_ok=True)
    os.makedirs(f"files/{language}/rules/", exist_ok=True)

for row in df:

    language = row["language"].lower()
    if language == "c++":
        language = "c"
    elif language == "c#":
        language = "csharp"
    if language not in datasets.keys():
        continue
    tracker[language] += 1
    with open(
        f"files/{language}/codes/{row['conversation_hash']}_{row['code_index']}{extensions[language]}",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(row["code"])

print(tracker)
