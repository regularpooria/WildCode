import pandas as pd
import numpy as np
from transformers import pipeline
import re
import os
import gc
from datetime import datetime

# Load BART zero-shot classification model
zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", multi_label=True)

import nltk
from nltk.tokenize import sent_tokenize

# Only run this once to download the sentence tokenizer
nltk.download('punkt')
from langdetect import detect


def extract_first_two_sentences(text):
    """
    Extracts the first two sentences from a given text string using NLTK's sentence tokenizer.

    Args:
        text (str): The input text.

    Returns:
        str: A string containing the first two sentences, or fewer if there aren't two.
    """
    sentences = sent_tokenize(text)
    return " ".join(sentences[:2])


# Categorization function
def normalize_text(text):
    '''
    Lower case, strip non character, Strip excess whitespace
    :param text:
    :return:
    '''
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# def categorize_followups(followup_list, candidate_labels=rich_labels, label_map=label_to_category, threshold=0.7):

def conversation_intent(message, category_labels, label_map, threshold=0.5):
    """
    Uses BART zero-shot model to classify each message into categories.
    Returns a list of best-matched categories per message.
    """
    # categories = list(label_map.values())
    if not isinstance(message, str) or not message.strip():
        return []

    cleaned = normalize_text(message)

    result = zero_shot_classifier(cleaned, category_labels)

    matched_categories = [
        (label_map[label], float(score))
        for label, score in zip(result["labels"], result["scores"])
        if score >= threshold
    ]

    if not matched_categories:
        return []

    return sorted(matched_categories, key=lambda x: x[1], reverse=True)


def conversation_intent_simple(message, category_labels, threshold=0.5):
    """
    Uses BART zero-shot model to classify each message into categories.
    Returns a list of best-matched categories per message.
    """
    # categories = list(label_map.values())
    if not isinstance(message, str) or not message.strip():
        return []

    cleaned = normalize_text(message)

    result = zero_shot_classifier(cleaned, category_labels)

    matched_categories = [
        (label, float(score))
        for label, score in zip(result["labels"], result["scores"])
        if score >= threshold
    ]

    if not matched_categories:
        return []

    return sorted(matched_categories, key=lambda x: x[1], reverse=True)


def extract_text_before_code(conversation_text):
    """
    Extracts the portion of a conversation that appears before the first code block.

    Code blocks can be fenced (```), inline (`), or indented.
    """
    if not isinstance(conversation_text, str):
        return ""

    # # Pattern to detect triple-backtick fenced code blocks or inline backticks
    # code_pattern = r"```"
    #
    # # Find first match
    # match = re.search(code_pattern, conversation_text)
    # if match:
    #     return conversation_text[:match.start()].strip()
    # Find all code block positions
    code_matches = list(re.finditer(r"```", conversation_text))
    if code_matches:
        # If there's text before the first code block, return it
        first_start = code_matches[0].start()
        if first_start > 0:
            return conversation_text[:first_start].strip()
        else:
            # Otherwise, try to return text after the last code block
            last_end = code_matches[-1].end()
            return conversation_text[last_end:].strip()

    # If no code blocks found, return entire text
    return conversation_text.strip()


def find_code_conversation(conversation):
    """
    Checks if any of the user messages in the list contains a code block using ```.

    Args:
        user_messages (str): user messages.

    Returns:
        bool: True if any message contains a code block, False otherwise.
    """
    """
    Checks if any 'content' field in a list or ndarray of dictionaries contains a code block (```).
    """
    if isinstance(conversation, np.ndarray):
        conversation = conversation.tolist()

    if not isinstance(conversation, list):
        return False

    for turn in conversation:
        content = turn.get("content", "")
        if isinstance(content, str) and "```" in content:
            return True

    return False


def extract_init_and_followups_assistant(convo):
    """
    For each conversation (list of messages), extract:
    - init_message: first message from role 'user'
    - followup_message: list of remaining user messages (if any)
    """
    if not isinstance(convo, (list, np.ndarray)):
        return pd.Series({"init_assistant": None, "followup_assistant": []})

    user_messages = [m.get("content", "") for m in convo if isinstance(m, dict) and m.get("role") == "assistant"]

    init = user_messages[0] if user_messages else None
    followups = user_messages[1:] if len(user_messages) > 1 else []

    return pd.Series({"init_assistant": init, "followup_assistant": followups})


def extract_init_and_followups(convo):
    """
    For each conversation (list of messages), extract:
    - init_message: first message from role 'user'
    - followup_message: list of remaining user messages (if any)
    """
    if not isinstance(convo, (list, np.ndarray)):
        return pd.Series({"init_message": None, "followup_message": []})

    user_messages = [extract_text_before_code(m.get("content", "")) for m in convo if
                     isinstance(m, dict) and m.get("role") == "user"]

    init = user_messages[0] if user_messages else None
    followups = user_messages[1:] if len(user_messages) > 1 else []

    return pd.Series({"init_message": init, "followup_message": followups})


def categorize_followups(followup_list, candidate_labels, label_map, threshold=0.5):
    """
    Applies semantic category matching to each follow-up user message.
    Returns a list of matched categories (list of lists).
    """
    if not isinstance(followup_list, (list, np.ndarray)):
        return []

    return [
        conversation_intent(extract_first_two_sentences(msg), candidate_labels, label_map, threshold)
        for msg in followup_list
        if isinstance(msg, str) and msg.strip()
    ]


def extract_code_df():
    # read conversation dataframe
    df = pd.read_pickle('/Wildchat/wildchat.pkl')
    # find rows with code
    df["has_code"] = df["conversation"].apply(find_code_conversation)
    df_with_code = df[df["has_code"] == True]
    df_with_code.to_pickle(
        '/Wildchat/Wildchat_code.pkl')  # 90299 conversation has code initial or in followup
    df_with_code_english = df_with_code[df_with_code['language'] == 'English']
    df_with_code_english.to_pickle(
        '/Wildchat/Wildchat_code_en.pkl')  # 48391 conversation is in english and has code initial or in followup


def extract_messages():
    df = pd.read_pickle('/Wildchat/Wildchat_code_en.pkl')
    df[["init_message", "followup_text"]] = df["conversation"].apply(extract_init_and_followups)
    df.to_pickle("/Wildchat/wildchat_code_en_init_followup.pkl")


def main():
    # Step 1-
    extract_code_df()
    # Step 2-Extract inital message and followup user messages
    extract_messages()

    df=pd.read_pickle('/Wildchat/wildchat_code_en_init_followup.pkl') #conversation containe python and has more than one conversation
    df = df.drop(columns=["conversation"])

    # Step 3- Load initial conversation by user template table
    template_df = pd.read_csv(
        "/Wildchat/Templates_with_Explanations_word.csv")  # Templates_with_Explanations.csv")#Semantic_Templates_by_Category.csv")#Improved_Semantic_Template_Categories_1.csv")#Semantic_Templates_by_Category.csv")

    # Create rich descriptive labels using examples for Semantic_Templates_by_Category.csv

    template_df["Rich_Label"] = template_df.apply(
        lambda
            row: f'This category is about {row["Explanation"]} It typically includes words like: {row["Word Explain"]}.',
        axis=1
    )
    # template_df["Rich_Label"] = template_df.apply(
    #     lambda
    #         row: f'This category typically includes words like: {row["Word Explain"]}.',
    #     axis=1
    # )

    rich_init_candidate_labels = template_df["Rich_Label"].tolist()
    # rich_init_candidate_labels = template_df["Category"].tolist()

    init_label_to_category = dict(zip(template_df["Rich_Label"], template_df["Category"]))

    '''
    # Load follow-up user request template table
    template_df = pd.read_csv("/Wildchat/Follow-Up_Request_Templates_by_Category.csv")
    template_df["Templates"] = template_df["Templates"].apply(eval)
    template_df["Rich_Label"] = template_df.apply(
        lambda row: f'This category is about {row["Category"]} and some example queries are: ' +
                    "; ".join(f'"{ex.strip()}"' for ex in row["Templates"][:10]),
        axis=1
    )

    rich_followup_candidate_labels = template_df["Rich_Label"].tolist()
    followup_label_to_category = dict(zip(template_df["Rich_Label"], template_df["Category"]))
    '''
    # Test with a user follow-up message
    # user_message = "Can you fix the bug in this function?"
    # print("User message:", user_message)
    # for cat, score in conversation_intent(user_message, followup_categories):
    #     print(f"- {cat} ({score:.2f})")

    # Step 4: Apply functions in chunks##############################################################3
    # Define base path for saving chunks
    output_dir = "/Wildchat/chunks"
    # os.makedirs(output_dir, exist_ok=True)

    # Split into chunks of 10,000
    '''
    chunks = np.array_split(df, len(df) // 1000 + 1)
    for i in range(7, len(chunks)):
        chunk = chunks[i]
        chunk.to_pickle(f"/Wildchat/chunks1/wildchat_chunk_{i + 1}.pkl")
    del df
    gc.collect()
    '''

    # Process each chunk and save
    for i in range(0, 49):
        # chunk = chunks[i]
        chunk = pd.read_pickle(f"/Wildchat/chunks1/wildchat_chunk_{i + 1}.pkl")
        print(f"Processing chunk {i + 1} at {datetime.now()}")

        # Apply init category classification
        chunk["user_init_category"] = chunk["init_message"].apply(
            lambda text: conversation_intent(
                extract_first_two_sentences(text),
                rich_init_candidate_labels,
                init_label_to_category
            )
        )

        # Apply follow-up classification
        chunk["followup_categories"] = chunk["followup_text"].apply(
            lambda lst: categorize_followups(
                lst,
                rich_init_candidate_labels,
                init_label_to_category
            )
        )

        # Save to pickle
        chunk_path = os.path.join(output_dir, f"wildchat_chunk_{i + 1}.pkl")

        chunk.to_pickle(chunk_path)

        # Clean up memory
        del chunk
        gc.collect()

    print("All chunks processed and saved.")

    # Step 5: Re-load and combine all saved chunks
    all_files = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".pkl")])
    df_final = pd.concat([pd.read_pickle(f) for f in all_files], ignore_index=True)

    # Save full result
    df_final.to_pickle('/Wildchat/wildchat_code_en_init_categorized.pkl')
    print("Final concatenated file saved.")

    ################################
    '''
    #Step 6: Apply categorization model
    df["user_init_category"] = df["init_message"].apply(lambda text: conversation_intent(extract_first_two_sentences(text), rich_init_candidate_labels, init_label_to_category))
    #df["user_init_category"] = df["init_message"].apply(lambda text: conversation_intent_simple(text, rich_init_candidate_labels))

    df.to_pickle('/Wildchat/wildchat_code_en_init_categorized.pkl')
    # Apply to the DataFrame (ensure init_message/followup_text are already extracted)
    df["followup_categories"] = df["followup_text"].apply(
        lambda lst: categorize_followups(lst, rich_init_candidate_labels, init_label_to_category)
    )
    # Save result
    df.to_pickle("/Wildchat/wildchat_code_en_init_followup_categorized.pkl")
    df.to_csv("/Wildchat/wildchat_categorized.csv")

    #
    def detect_language_safe(text):
    try:
        return detect(text)
    except:
        return 'unknown'

    # Apply to 'init_message' column
    df['language'] = df['init_message'].apply(detect_language_safe)

    # Filter only English rows
    df_english = df[df['language'] == 'en']

    df=df_english

    ##############get rows with specific features like secure coding

    df_init=df[df['user_init_category'].apply(lambda x: x != [])]
    df_init_secure = df[df['user_init_category'].apply(
        lambda lst:
        any(isinstance(item, tuple) and item[0] == 'Secure Coding' and item[1]>0.7 for item in lst)

    )] 

    df[df['followup_categories'].apply(lambda lst: any(len(sublist) > 0 for sublist in lst))]

    df[df['followup_categories'].apply(
        lambda outer: any(
            any(isinstance(item, tuple) and item[0] == 'Secure Coding Requests' for item in sub)
            for sub in outer
        )
    )]
    df[df['followup_categories'].apply(
        lambda outer: any(
            any(isinstance(item, tuple) and item[0] == 'Bug Reports' for item in sub)
            for sub in outer
        )
    )]
    '''


def detect_language_safe(text):
    try:
        return detect(text)
    except:
        return 'unknown'


def analysis():
    df = pd.read_pickle('/Wildchat/wildchat_code_en_init_categorized.pkl')
    # Apply to 'init_message' column
    df['init_language'] = df['init_message'].apply(detect_language_safe)
    df.to_pickle('/Wildchat/wildchat_code_en_categorized_lang.pkl')
    df_en = df[df['init_language'] == 'en']  # 34478  out of #df_rows=48391
    df_init = df_en[df_en['user_init_category'].apply(lambda x: x != [])]  # 31784
    df_init_secure = df_init[df_init['user_init_category'].apply(
        lambda lst:
        any(isinstance(item, tuple) and item[0] == 'Secure Coding' and item[1] > 0.7 for item in lst)
    )]  # >0.7->3171, >0.8->594, >0.9->144, >0.95 ->30
    df_init_secure = df_init[df_init['user_init_category'].apply(
        lambda lst:
        any(isinstance(item, tuple) and item[0] == 'Secure Coding' and item[1] > 0.95 for item in lst)
    )]

    df_init_secure.to_csv('/Wildchat/wildchat_code_en_init_secure_95.csv')
    df_init_secure.to_pickle('/Wildchat/wildchat_code_en_init_secure_95.pkl')

    df_followup_secure = df_init[df_init['followup_categories'].apply(
        lambda outer:
        any(
            isinstance(item, tuple) and item[0] == 'Secure Coding' and item[1] > 0.90
            for inner in outer
            for item in inner
        )
    )]  # 70-> 1886, 80->633, 90->134,95->35
    df_followup_secure = df_init[df_init['followup_categories'].apply(
        lambda outer:
        any(
            isinstance(inner, list) and len(inner) > 0 and inner[0][0] == 'Secure Coding'
            for inner in outer
        ) if isinstance(outer, list) else False
    )]  # 65
    df_followup_secure.to_pickle('/Wildchat/wildchat_code_en_init_secure_followup_first.pkl')
    df_followup_secure.to_csv('/Wildchat/wildchat_code_en_init_secure_followup_first.csv')

    #######################Analysing intent


def check_lang():
    lang_df = pd.read_csv(
        "/Wildchat/conversation_languages.csv")  # 37452      # Load the CSV containing conversation_hash and languages
    df = pd.read_pickle('/Wildchat/wildchat_code_en_init_categorized.pkl')  # 48391

    # Step 1: Clean and update the languages column
    def update_languages(langs):
        if isinstance(langs, str):
            langs = [l.strip().lower() for l in langs.split(";")]
            if "csharp" in langs and "C#" not in langs:
                langs.append("C#")
            if "c++" in langs and "cpp" not in langs:
                langs.append("cpp")
        return langs

    lang_df["languages"] = lang_df["languages"].apply(update_languages)

    # Step 2: Merge with your main df using conversation_hash
    lang_df = lang_df.merge(df, on="conversation_hash", how="left")

    # Step 3: Check if any language appears in init_message
    def find_language_mentions(text, langs):
        if not isinstance(text, str) or not isinstance(langs, list):
            return []
        found = [lang for lang in langs if lang.lower() in text.lower()]
        return found

    lang_df1 = lang_df.dropna(
        subset=['conversation'])  # 17500, including NaN conversations 37452, 19952 has NaN conversation
    # it means they have 17500 english conversation

    lang_df1["mentioned_languages"] = lang_df1.apply(
        lambda row: find_language_mentions(row["init_message"], row["languages"]),
        axis=1
    )
    lang_df1["followup_mentioned_languages"] = lang_df1.apply(
        lambda row: find_language_mentions(
            " ".join(row["followup_text"]) if isinstance(row["followup_text"], list) else "", row["languages"]),
        axis=1
    )

    # Define a function that computes the new mentions
    def find_new_followup_mentions(row):
        init_langs = set(row["mentioned_languages"]) if isinstance(row["mentioned_languages"], list) else set()
        followup_langs = set(row["followup_mentioned_languages"]) if isinstance(row["followup_mentioned_languages"],
                                                                                list) else set()
        return list(followup_langs - init_langs)

    # Add a new column with the new languages
    lang_df1["new_followup_languages"] = lang_df1.apply(find_new_followup_mentions, axis=1)

    # Filter rows where new languages were found in followup
    new_rows = lang_df1[lang_df1["new_followup_languages"].apply(lambda x: len(x) > 0)]  # 2455

    lang_df1.to_pickle('/Wildchat/conversation_languages_all.pkl')
    new_rows[["init_assistant_message", "followup_assistant_text"]] = new_rows["conversation"].apply(
        extract_init_and_followups_assistant)
    new_rows["assistant_init_contains_code"] = new_rows["init_assistant_message"].apply(
        lambda x: isinstance(x, str) and "```" in x
    )
    coded_rows = new_rows[new_rows["assistant_init_contains_code"]]

    coded_rows.to_pickle('/Wildchat/coded_rows.pkl')  # 920

    def extract_first_word_after_code_block(text):
        if isinstance(text, str):
            match = re.search(r"```([^\s]*)", text)
            if match:
                return match.group(1).strip()
        return None

    coded_rows["assistant_init_code_language"] = coded_rows["init_assistant_message"].apply(
        extract_first_word_after_code_block)

    def language_mismatch(row):
        # Normalization map
        normalize = {
            "cpp": "c++",
            "c++": "c++",
            "csharp": "c#",
            "c#": "c#"
        }
        # Normalize assistant language
        code_lang_raw = str(row.get("assistant_init_code_language", "")).lower().strip()
        code_lang = normalize.get(code_lang_raw, code_lang_raw)
        # Normalize follow-up languages
        followup_langs = [
            normalize.get(str(lang).lower().strip(), str(lang).lower().strip())
            for lang in row.get("new_followup_languages", [])
        ]
        # Ignore empty assistant language
        if not code_lang:
            return False
        return code_lang not in followup_langs

    # Filter rows
    mismatch_rows = coded_rows[coded_rows.apply(language_mismatch, axis=1)]  # 251 out of coded and english conversation
    mismatch_rows.to_pickle('/Wildchat/mismatch_rows.pkl')

    ##################with all rows containing code no matter  what the conversation language is
    df = pd.read_pickle('/Wildchat/wildchat.pkl')
    lang_df = lang_df.merge(df, on="conversation_hash", how="left")  # 37452
    del df
    lang_df["languages"] = lang_df["languages"].apply(update_languages)
    lang_df[["init_message", "followup_text"]] = lang_df["conversation"].apply(extract_init_and_followups)
    # repeat previous steps
    lang_df["mentioned_languages"] = lang_df.apply(
        lambda row: find_language_mentions(row["init_message"], row["languages"]),
        axis=1
    )
    lang_df["followup_mentioned_languages"] = lang_df.apply(
        lambda row: find_language_mentions(
            " ".join(row["followup_text"]) if isinstance(row["followup_text"], list) else "", row["languages"]),
        axis=1
    )
    lang_df["new_followup_languages"] = lang_df.apply(find_new_followup_mentions, axis=1)
    new_rows = lang_df[lang_df["new_followup_languages"].apply(
        lambda x: len(x) > 0)]  # 5788 #number of rows user name a programming language in followup
    new_rows[["init_assistant_message", "followup_assistant_text"]] = new_rows["conversation"].apply(
        extract_init_and_followups_assistant)
    new_rows["assistant_init_contains_code"] = new_rows["init_assistant_message"].apply(
        lambda x: isinstance(x, str) and "```" in x
    )
    coded_rows = new_rows[new_rows[
        "assistant_init_contains_code"]]  # those chatgpt gave a code in initial request, there might be conversations that in the followup chatgpt gave code and user asked for another one
    # ...
    # coded_rows.to_pickle('/Wildchat/coded_rows_all.pkl') #2058
    #    mismatch_rows.to_pickle('/Wildchat/mismatch_rows_all.pkl')#648


def find_keywords_in_conversations():
    df = pd.read_pickle('/Wildchat/wildchat_code_en_init_categorized.pkl')

    # Sample keyword list (normalize quotes)
    keywords = [
        "secure code",
        "deserialization attack",
        "redos",
        "weak cryptography",
        "weak randoms number generation",
        "predictable random number",
        "weak hash function"
    ]

    # Convert keywords to lowercase for case-insensitive matching
    keywords = [kw.lower() for kw in keywords]

    # Function to find which keywords appear in a given conversation string
    def find_keywords(conversation):
        text = " ".join(str(item.get("content", "")) for item in conversation if isinstance(item, dict))
        text = text.lower()
        return [kw for kw in keywords if kw in text]

    def contains_keyword(conversation, keyword):
        text = " ".join(str(item.get("content", "")) for item in conversation if isinstance(item, dict))
        text = text.lower()
        return keyword.lower() in text

    def find_violat(conversation):
        text = " ".join(str(item.get("content", "")) for item in conversation if isinstance(item, dict))
        text = text.lower()
        return 'violat' in text

    # Apply to DataFrame (assumes df['conversation'] contains strings or structured text)
    df["matched_keywords"] = df["conversation"].apply(find_keywords)
    df["contains_violat"] = df["conversation"].apply(find_violat)

    # Keep only rows where matched_keywords is not empty
    df = df[df["matched_keywords"].apply(lambda x: isinstance(x, list) and len(x) > 0)]

    df.to_csv('/Wildchat/wildchat_code_matched_keywords.csv')
    df_v = df[df["contains_violat"] == True]
    df_v.to_pickle('/Wildchat/wildchat_code_violat.pkl')
    import re

    def extract_violat_sentences(conversation):
        results = []
        for item in conversation:
            if isinstance(item, dict):
                content = item.get("content", "")
                role = item.get("role", "")
                # Split into sentences
                sentences = re.split(r'(?<=[.!?])\s+', content)
                for sentence in sentences:
                    if 'violat' in sentence.lower():
                        results.append({"role": role, "sentence": sentence.strip()})
        return results

    df["violat_sentences"] = df.apply(
        lambda row: extract_violat_sentences(row["conversation"]) if row["contains_violat"] else [],
        axis=1
    )

    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer("all-MiniLM-L6-v2")
    keyword_embeddings = model.encode(keywords, convert_to_tensor=True)

    def semantic_match(conversation):
        text = " ".join(str(item.get("content", "")) for item in conversation if isinstance(item, dict))
        text_embedding = model.encode(text, convert_to_tensor=True)
        scores = util.cos_sim(text_embedding, keyword_embeddings)[0]
        matched = [keywords[i] for i, score in enumerate(scores) if score > 0.7]
        return matched

    df["matched_keywords_semantic"] = df["conversation"].apply(semantic_match)  # run this at the end of the night
    df = df[df["matched_keywords_semantic"].apply(lambda x: isinstance(x, list) and len(x) > 0)]
    df.to_csv('/Wildchat/wildchat_code_matched_keywords_semantic.csv')




if __name__ == '__main__':
    check_lang()
    find_keywords_in_conversations()
    main()
    analysis()












