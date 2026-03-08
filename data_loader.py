import os
import re


def load_dataset(data_folder="data"):

    documents = []
    _QUOTED = re.compile(r"^\s*>.*$", re.MULTILINE)
    _SIG = re.compile(r"\n-- \n.*", re.DOTALL)
    _EMAIL = re.compile(r"\S+@\S+\.\S+")
    _URL = re.compile(r"https?://\S+|www\.\S+")

    print("Loading dataset...")

    for root, dirs, files in os.walk(data_folder):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="latin-1") as f:
                    text = f.read()

                if "\n\n" in text:
                    text = text.split("\n\n", 1)[1]

                text = _SIG.sub("", text)
                text = _QUOTED.sub("", text)
                text = _EMAIL.sub(" ", text)
                text = _URL.sub(" ", text)
                text = re.sub(r"\n{3,}", "\n\n", text)
                text = text.strip()
                text = text[:2000]

                if len(text.split()) >= 20:
                    documents.append(text)

            except Exception:
                continue

    print(f"Loaded {len(documents)} documents")
    return documents