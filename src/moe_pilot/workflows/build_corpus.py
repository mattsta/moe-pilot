"""Build corpus pilot workflow."""


def run(config, *, device=None):
    from moe_pilot.artifacts import write_json
    from moe_pilot.corpora import CorpusBuilder
    from moe_pilot.datasets import DatasetLoader

    loader = DatasetLoader(config)
    load_dataset = loader.load

    builder = CorpusBuilder(count=config.corpora.analysis_count, seed=config.analysis.seed)
    add = builder.add

    d = load_dataset("openai/openai_humaneval", split="test")
    add(
        "code",
        [r["prompt"] + r["canonical_solution"] for r in d],
        "openai/openai_humaneval test (prompt+canonical_solution)",
    )
    d = load_dataset("EleutherAI/hendrycks_math", "algebra", split="test")
    add(
        "math",
        [r["problem"] + "\n" + r["solution"] for r in d],
        "EleutherAI/hendrycks_math algebra test (problem+solution)",
    )
    d = load_dataset("allenai/sciq", split="test")
    add("science", [r["support"] for r in d], "allenai/sciq test (support paragraphs)")
    d = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test[:2000]")
    add(
        "prose",
        [r["article"][:1500] for r in d],
        "abisee/cnn_dailymail 3.0.0 test[:2000] (article, first 1500 chars)",
    )
    d = load_dataset("papluca/language-identification", split="test")
    rows = [r for r in d if r["labels"] != "en"]
    add(
        "multilingual",
        [r["text"] for r in rows],
        "papluca/language-identification test, non-English (19 languages)",
    )
    corpus = config.paths.data / "corpus.json"
    sources = config.paths.data / "corpus_sources.json"
    write_json(corpus, builder.records)
    write_json(sources, {"groups": builder.sources, "datasets": loader.sources})
    return [corpus, sources]
