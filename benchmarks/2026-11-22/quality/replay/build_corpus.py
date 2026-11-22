"""Freeze the same long Django source context for both tokenizers."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from tokenizers import Tokenizer

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("archive", type=Path)
parser.add_argument("qwen_tokenizer", type=Path)
parser.add_argument("inkling_tokenizer", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
id = "django__django-14631"
tokenizers = [
    Tokenizer.from_file(str(f)) for f in [args.qwen_tokenizer, args.inkling_tokenizer]
]


def counts(txt):
    return [len(t.encode(txt, add_special_tokens=False).ids) for t in tokenizers]


def key(name):
    return hashlib.sha256(("quality22-django-source:" + name).encode()).digest()


with tarfile.open(args.archive) as tar:
    paths = sorted(
        (
            p.name
            for p in tar
            if p.isfile()
            and p.name.startswith("django/")
            and p.name.endswith(".py")
            and p.size > 100
        ),
        key=key,
    )
    docs = sorted(
        (
            p.name
            for p in tar
            if p.isfile()
            and p.name.startswith("docs/")
            and p.name.endswith(".txt")
            and p.size > 500
        ),
        key=key,
    )
    paths = [item for triple in zip(docs[::2], paths, docs[1::2]) for item in triple]
    anchors = ["django/forms/forms.py", "django/forms/boundfield.py"]
    files = []
    sums = [0, 0]
    for name in anchors:
        cc = counts(
            f"\n--- {name} ---\n" + tar.extractfile(name).read().decode("utf-8")
        )
        sums = [s + c for s, c in zip(sums, cc)]
    for name in paths:
        if name in anchors:
            continue
        body = tar.extractfile(name).read().decode("utf-8")
        part = f"\n--- {name} ---\n" + body
        cc = counts(part)
        if min(s + c for s, c in zip(sums, cc)) > 198200:
            continue
        files.append((name, part))
        sums = [s + c for s, c in zip(sums, cc)]
        if min(sums) >= 196850:
            break
    for name, fraction in zip(anchors, [0.25, 0.70]):
        part = f"\n--- {name} ---\n" + tar.extractfile(name).read().decode("utf-8")
        files.insert(int(fraction * len(files)), (name, part))
    text = (
        "Reference snapshot from the same Django checkout. Sections are labelled by repository path.\n"
        + "".join(t for _, t in files)
    )
cc = counts(text)
assert 196608 <= min(cc) and max(cc) < 244500, cc

report = {
    "instance_id": id,
    "sha256": hashlib.sha256(text.encode()).hexdigest(),
    "source_tokens": dict(zip(("qwen", "inkling"), cc)),
    "files": [n for n, _ in files],
}
expected = json.loads(
    (Path(__file__).resolve().parents[1] / "long-source.json").read_text()
)
if report != expected:
    raise ValueError("Corpus does not match the frozen content and tokenizer counts")
args.output.write_text(text)
print(cc, len(files), flush=True)
