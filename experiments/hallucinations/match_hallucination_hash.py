import json

lang = "python"
with open(f"tmp/{lang}_imports_list_hash.json", "r", encoding="utf-8") as f:
    hash_list = json.load(f)

with open(f"results/hallucinations_{lang}.json", "r", encoding="utf-8") as f:
    hallucinations = json.load(f)


for key in hash_list.copy().keys():
    if key not in hallucinations:
        del hash_list[key]


with open(f"results/hallucinations_{lang}_hash.json", "w", encoding="utf-8") as f:
    json.dump(hash_list, f)
