"""Build subtopics pilot workflow."""


def run(config, *, device=None):
    from datasets import concatenate_datasets

    from moe_pilot.artifacts import write_json
    from moe_pilot.corpora import CorpusBuilder
    from moe_pilot.datasets import DatasetLoader

    loader = DatasetLoader(config)
    load_dataset = loader.load

    builder = CorpusBuilder(count=config.corpora.subtopic_count, seed=config.analysis.seed + 2)

    def add(dom, sub, texts, source, minlen=100):
        builder.add(dom, texts, source, subtopic=sub, min_length=minlen)
        print(dom, sub, builder.count, flush=True)

    for cfgn, sub in [
        ("algebra", "algebra"),
        ("geometry", "geometry"),
        ("number_theory", "number_theory"),
        ("counting_and_probability", "counting_probability"),
    ]:
        d = load_dataset("EleutherAI/hendrycks_math", cfgn, split="test")
        add(
            "math",
            sub,
            [r["problem"] + "\n" + r["solution"] for r in d],
            f"EleutherAI/hendrycks_math {cfgn} test (problem+solution)",
        )
    for lang in ["python", "javascript", "java", "go"]:
        d = load_dataset("claudios/code_search_net", lang, split="test")
        add(
            "code",
            lang,
            [r["whole_func_string"] for r in d.select(range(min(3000, len(d))))],
            f"claudios/code_search_net {lang} test[:3000] whole_func_string",
        )
    d = load_dataset("b-mc2/sql-create-context", split="train[:5000]")
    add(
        "code",
        "sql",
        [r["context"] + "\n" + r["answer"] for r in d],
        "b-mc2/sql-create-context train[:5000] (context+answer SQL)",
        minlen=80,
    )
    for sub, subs in [
        ("biology", ["high_school_biology", "college_biology"]),
        ("chemistry", ["high_school_chemistry", "college_chemistry"]),
        ("physics", ["high_school_physics", "college_physics", "conceptual_physics"]),
    ]:
        d = concatenate_datasets([load_dataset("cais/mmlu", s, split="test") for s in subs])
        add(
            "science",
            sub,
            [r["question"] + "\n" + "\n".join(f"{c}" for c in r["choices"]) for r in d],
            f"cais/mmlu {'+'.join(subs)} test (question+choices)",
            minlen=120,
        )
    d = load_dataset("papluca/language-identification", split="test")
    for lang in ["de", "fr", "es", "zh", "ru", "ar", "hi"]:
        add(
            "multilingual",
            lang,
            [r["text"] for r in d if r["labels"] == lang],
            f"papluca/language-identification test labels=={lang}",
            minlen=60 if lang != "zh" else 30,
        )
    corpus = config.paths.data / "subtopic_corpus.json"
    sources = config.paths.data / "subtopic_sources.json"
    write_json(corpus, builder.records)
    write_json(sources, {"groups": builder.sources, "datasets": loader.sources})
    return [corpus, sources]
