"""Deterministic corpus sampling without process-global random state."""

import random


class CorpusBuilder:
    """Sample source texts, validate group sizes, and retain source attribution.

    Supply already-loaded texts; network access belongs to dataset workflows.
    Short source pools fail explicitly rather than silently changing a corpus.
    """

    def __init__(self, count, seed, exclude=()):
        self.count = count
        self.random = random.Random(seed)
        self.exclude = set(exclude)
        self.records = []
        self.sources = {}

    def add(self, domain, texts, source, *, subtopic=None, min_length=81):
        texts = list(
            dict.fromkeys(
                text.strip()
                for text in texts
                if text and len(text.strip()) >= min_length and text.strip() not in self.exclude
            )
        )
        if len(texts) < self.count:
            raise ValueError(
                f"{domain}/{subtopic or ''}: need {self.count} texts, found {len(texts)}"
            )
        self.random.shuffle(texts)
        selected = texts[: self.count]
        metadata = {"domain": domain}
        key = domain
        if subtopic is not None:
            metadata["subtopic"] = subtopic
            key += "/" + subtopic
        self.records.extend({**metadata, "text": text} for text in selected)
        self.sources[key] = {"source": source, "n": len(selected)}
