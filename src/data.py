import re

from datasets import load_dataset
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv()

type Sequences = list[list[str]]

SPLITS = ("train", "validation", "test")


def tokenise(text: str) -> Sequences:
    """Clean one raw line and split it into sentences of whitespace tokens."""
    header = re.compile(r"^\s*=+\s*.*?\s*=+\s*$")
    # strip leading and trailing whitespace
    text = text.strip()
    # drop empty lines and "= title =" headers of any level
    if not text or header.match(text):
        return []
    # join "@-@" into a hyphen
    text = re.sub(r"\s*@-@\s*", "-", text)
    # join "@,@" into a comma
    text = re.sub(r"\s*@,@\s*", ",", text)
    # join "@.@" into a point
    text = re.sub(r"\s*@\.@\s*", ".", text)
    # collapse repeated whitespace into one space
    text = re.sub(r"\s+", " ", text)
    # lower case all characters
    text = text.lower()
    # split into sentences on "." tokens
    seqs = []
    seq = []
    for tok in text.split():
        seq.append(tok)
        if tok == ".":
            seqs.append(seq)
            seq = []
    if seq:
        seqs.append(seq)
    return seqs


def load_data() -> dict[str, Sequences]:
    """Load WikiText-2 raw and return tokenised sentences per split."""
    raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    data = {}
    for split in SPLITS:
        seqs = []
        for line in raw[split]["text"]:
            seqs.extend(tokenise(line))
        data[split] = seqs
    return data


def tokenise_news(text: str) -> list[str]:
    """Lower case one AG News article as a single sentence, dropping parenthesised words and html tags, into word and punctuation tokens."""
    token = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[.,!?;:]")
    text = re.sub(r"\([^)]*\)", " ", text.lower())
    text = re.sub(r"&lt;.*?&gt;", " ", text)
    return token.findall(text)


def load_ag_news(val_frac: float = 0.1, seed: int = 0) -> dict[str, tuple[Sequences, list[int]]]:
    """Load AG News and return (token sequences, labels) per split, holding out a stratified validation split from train."""
    raw = load_dataset("fancyzhx/ag_news")
    X = [tokenise_news(t) for t in raw["train"]["text"]]
    y = list(raw["train"]["label"])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_frac, random_state=seed, stratify=y)
    X_test = [tokenise_news(t) for t in raw["test"]["text"]]
    y_test = list(raw["test"]["label"])
    return {"train": (X_train, y_train), "validation": (X_val, y_val), "test": (X_test, y_test)}
