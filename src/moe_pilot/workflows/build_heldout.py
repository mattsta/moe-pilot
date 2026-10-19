"""Build heldout pilot workflow."""


def main(model_dir="models/olmoe"):
    # Held-out eval texts: same sources as corpus.json, disjoint texts, different seed.
    from datasets import load_dataset

    from moe_pilot.artifacts import read_json, write_json
    from moe_pilot.corpora import CorpusBuilder

    builder = CorpusBuilder(
        count=40, seed=123, exclude=(c["text"] for c in read_json("corpus.json"))
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
    write_json("heldout.json", builder.records)
    write_json("heldout_sources.json", builder.sources)
    print(builder.sources)
