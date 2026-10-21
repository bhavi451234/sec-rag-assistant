"""Toy demo: turn sentences into vectors and compare their meaning."""

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "Apple faces intense competition in smartphones.",
    "Rivals are launching cheaper phones and taking market share.",
    "The company repurchased shares worth $90 billion.",
    "Tesla's vehicles may be recalled over safety defects.",
    "Banana bread needs ripe bananas and flour.",
]

vectors = model.encode(sentences, normalize_embeddings=True)
print(vectors.shape)
print(vectors[0][:5])

query = vectors[0]
scores = vectors @ query
for sentence, score in zip(sentences, scores):
    print(f"{score:.2f} {sentence}")