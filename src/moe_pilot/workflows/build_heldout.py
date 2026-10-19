"""Build heldout pilot workflow."""


def run(config):
    # Held-out eval texts: same sources as corpus.json, disjoint texts, different seed.
    from moe_pilot.artifacts import load_corpus, write_json
    from moe_pilot.corpora import CorpusBuilder
    from moe_pilot.datasets import DatasetLoader

    loader = DatasetLoader(config)
    load_dataset = loader.load

    builder = CorpusBuilder(
        count=config.corpora.heldout_count,
        seed=config.analysis.seed + 1,
        exclude=(c["text"] for c in load_corpus(config.paths.data / "corpus.json")),
    )
    add = builder.add

    d = load_dataset("openai/openai_humaneval", split="test")
    add(
        "code",
        [r["prompt"] + r["canonical_solution"] for r in d],
        "openai/openai_humaneval test (disjoint from corpus.json)",
    )
    d = load_dataset("EleutherAI/hendrycks_math", "algebra", split="test")
    add(
        "math",
        [r["problem"] + "\n" + r["solution"] for r in d],
        "EleutherAI/hendrycks_math algebra test (disjoint)",
    )
    d = load_dataset("allenai/sciq", split="test")
    add("science", [r["support"] for r in d], "allenai/sciq test support (disjoint)")
    d = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test[:2000]")
    add(
        "prose",
        [r["article"][:1500] for r in d],
        "abisee/cnn_dailymail 3.0.0 test[:2000] (disjoint)",
    )
    d = load_dataset("papluca/language-identification", split="test")
    add(
        "multilingual",
        [r["text"] for r in d if r["labels"] != "en"],
        "papluca/language-identification test non-English (disjoint)",
    )
    corpus = config.paths.data / "heldout.json"
    sources = config.paths.data / "heldout_sources.json"
    write_json(corpus, builder.records)
    write_json(sources, {"groups": builder.sources, "datasets": loader.sources})
    return [corpus, sources]
