"""English vocabulary neighbors in a local text embedding space."""

from threading import Lock

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
WORDS = """
acceptance achievement anger anxiety apology attention calm caution challenge
clarity collaboration confidence confusion curiosity deadline decision delay
determination disappointment doubt effort empathy excitement fatigue fear focus
frustration gratitude happiness help hope improvement insight joy learning loss
motivation patience planning pride progress relief resolve risk sadness safety
satisfaction setback stress success support surprise tension uncertainty urgency
verification warning worry
""".split()


class MoodEmbeddings:
    model_name = ENCODER

    def __init__(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(ENCODER, local_files_only=True)
        self.model = AutoModel.from_pretrained(ENCODER, local_files_only=True, use_safetensors=True)
        self.model.eval()
        self.lock = Lock()
        self.vectors = self.encode(WORDS)

    def encode(self, texts):
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        with self.torch.inference_mode():
            tokens = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (tokens * mask).sum(1) / mask.sum(1).clamp_min(1)
            return self.torch.nn.functional.normalize(pooled, dim=1)

    def neighbors(self, text):
        with self.lock, self.torch.inference_mode():
            scores = self.vectors @ self.encode([text])[0]
            values, indices = scores.topk(6)
            return [{"word": WORDS[i], "cosine": round(score, 3)}
                    for i, score in zip(indices.tolist(), values.tolist())]
