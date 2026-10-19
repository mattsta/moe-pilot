"""Explicit condition metadata and validation; filenames carry no scientific meaning."""

import math
import re
from dataclasses import asdict, dataclass, field

from moe_pilot.artifacts import read_json, write_json
from moe_pilot.perturbations import Intervention


@dataclass(frozen=True)
class Condition:
    name: str
    domain: str
    scope: str
    count: int
    family: str
    mask: dict[int, list[int]]
    global_mass: float
    on_domain_mass: float
    parent: str | None = None
    seed: int | None = None
    mode: str = "reroute"
    strength: float = 1.0
    donors: dict[int, dict[int, int]] = field(default_factory=dict)

    @property
    def intervention(self):
        return Intervention(self.mask, self.mode, self.strength, self.donors)


@dataclass(frozen=True)
class Conditions:
    domains: tuple[str, ...]
    layers: int
    experts: int
    top_k: int
    items: tuple[Condition, ...]

    def validate(self):
        if self.layers < 1 or not 1 <= self.top_k < self.experts or len(self.domains) < 2:
            raise ValueError("Invalid condition-set dimensions")
        by_name = {item.name: item for item in self.items}
        if not by_name or len(by_name) != len(self.items):
            raise ValueError("Conditions must be nonempty and names unique")
        for item in self.items:
            item.intervention.validate(self.layers, self.experts, self.top_k)
            if (
                not isinstance(item.name, str)
                or not re.fullmatch(r"[A-Za-z0-9_-]+", item.name)
                or item.name == "base"
            ):
                raise ValueError("Condition names must be safe identifiers; base is reserved")
            if type(item.count) is not int or item.count < 1:
                raise ValueError("Condition expert counts must be positive integers")
            if any(
                not math.isfinite(value) or value < 0
                for value in (item.global_mass, item.on_domain_mass)
            ):
                raise ValueError("Condition routing masses must be finite and nonnegative")
            if item.domain not in self.domains or item.family not in (
                "specialist",
                "global-control",
                "on-domain-control",
            ):
                raise ValueError(f"Invalid metadata for condition {item.name}")
            if not item.mask or any(
                not 0 <= layer < self.layers
                or type(layer) is not int
                or len(experts) != item.count
                or len(set(experts)) != item.count
                or any(
                    type(expert) is not int or not 0 <= expert < self.experts for expert in experts
                )
                for layer, experts in item.mask.items()
            ):
                raise ValueError(f"Invalid mask for condition {item.name}")
            if item.count > self.experts - self.top_k:
                raise ValueError(f"Condition {item.name} leaves fewer than top-k experts")
            if item.family != "specialist":
                parent = by_name.get(item.parent)
                if (
                    parent is None
                    or parent.family != "specialist"
                    or parent.domain != item.domain
                    or parent.count != item.count
                    or parent.scope != item.scope
                    or parent.mode != item.mode
                    or parent.strength != item.strength
                ):
                    raise ValueError(f"Invalid control parent for {item.name}")
                if set(parent.mask) != set(item.mask) or any(
                    set(parent.mask[layer]) & set(experts) for layer, experts in item.mask.items()
                ):
                    raise ValueError(
                        f"Control {item.name} must mask the same layers and disjoint experts"
                    )
            elif item.parent is not None:
                raise ValueError("Specialists cannot have a control parent")

    def save(self, filename):
        self.validate()
        write_json(filename, {"schema": 2, **asdict(self)})

    @classmethod
    def load(cls, filename):
        raw = read_json(filename)
        if raw.pop("schema", None) != 2:
            raise ValueError("Unsupported condition schema")
        items = tuple(
            Condition(
                **{
                    **item,
                    "mask": {int(layer): experts for layer, experts in item["mask"].items()},
                    "donors": {
                        int(layer): {int(target): donor for target, donor in donors.items()}
                        for layer, donors in item["donors"].items()
                    },
                }
            )
            for item in raw.pop("items")
        )
        result = cls(items=items, domains=tuple(raw.pop("domains")), **raw)
        result.validate()
        return result
