"""Explicit dataset downloads pinned to repository commits with source records."""


class DatasetLoader:
    def __init__(self, config):
        self.revisions = {source.repository: source.revision for source in config.datasets}
        self.sources = []

    def load(self, repository, configuration=None, *, split):
        revision = self.revisions.get(repository)
        if revision is None:
            raise ValueError(
                f"Set datasets.{repository!r} to an immutable commit SHA before downloading"
            )
        from datasets import load_dataset

        dataset = load_dataset(
            repository, *([configuration] if configuration else []), split=split, revision=revision
        )
        self.sources.append(
            {
                "repository": repository,
                "configuration": configuration,
                "split": split,
                "revision": revision,
                "fingerprint": getattr(dataset, "_fingerprint", None),
            }
        )
        return dataset
