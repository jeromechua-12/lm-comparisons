import re

from datasets import load_dataset
from dotenv import load_dotenv

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
