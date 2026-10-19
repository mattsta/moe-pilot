"""Build corpus pilot workflow."""


def main(model_dir="models/olmoe"):
    from datasets import load_dataset

    from moe_pilot.artifacts import write_json
    from moe_pilot.corpora import CorpusBuilder

    builder = CorpusBuilder(count=100, seed=0)
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
    rows = [r for r in d if r["labels"] not in ("en",)]
    add(
        "multilingual",
        [r["text"] for r in rows],
        "papluca/language-identification test, non-English (19 languages)",
    )
    write_json("corpus.json", builder.records)
    write_json("corpus_sources.json", builder.sources)
    print(builder.sources)
