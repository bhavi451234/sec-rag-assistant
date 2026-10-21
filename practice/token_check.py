"""How many of our chunks are longer than the model can read?"""

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
limit = model.max_seq_length
print("model reads at most", limit, "tokens")

texts = []
for line in Path("data/chunks/chunks.jsonl").read_text(encoding="utf-8").splitlines():
    texts.append(json.loads(line)["text"])

counts = [len(ids) for ids in model.tokenizer(texts)["input_ids"]]
too_long = sum(1 for n in counts if n > limit)
print(len(texts), "chunks | too long:", too_long, "| longest:", max(counts), "tokens")
print("average tokens per chunk:", sum(counts) // len(counts))