"""Immutable, explicit expert intervention semantics shared by every caller."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType


@dataclass(frozen=True)
class Intervention:
    """Alter expert contributions at each declared target layer.

    Fixed-routing modes preserve the router's choices on the incoming hidden
    states at that site; subsequent layers respond naturally to changed states.
    ``strength`` interpolates outputs for rerouting/substitution and attenuates
    targeted gate weights for removal/rescaling. Zero strength is a no-op.
    """

    mask: Mapping[int, Sequence[int]]
    mode: str = "reroute"
    strength: float = 1.0
    donors: Mapping[int, Mapping[int, int]] = field(default_factory=dict)

    def __post_init__(self):
        if self.mode not in ("reroute", "remove", "rescale", "substitute"):
            raise ValueError("Unknown intervention mode")
        if (
            isinstance(self.strength, bool)
            or not isinstance(self.strength, Real)
            or not math.isfinite(self.strength)
            or not 0 <= self.strength <= 1
        ):
            raise ValueError("Intervention strength must be finite and between zero and one")
        if not isinstance(self.mask, Mapping) or not isinstance(self.donors, Mapping):
            raise ValueError("Intervention mask and donors must be layer mappings")
        mask = {}
        for layer, experts in self.mask.items():
            if (
                type(layer) is not int
                or layer < 0
                or not isinstance(experts, Sequence)
                or isinstance(experts, str)
                or not experts
            ):
                raise ValueError("Intervention layers must be nonnegative with nonempty masks")
            if any(type(expert) is not int or expert < 0 for expert in experts):
                raise ValueError("Expert IDs must be nonnegative integers")
            if len(set(experts)) != len(experts):
                raise ValueError("Expert masks must be unique")
            mask[layer] = tuple(sorted(experts))
        donors = {}
        for layer, replacements in self.donors.items():
            if type(layer) is not int or layer not in mask or not isinstance(replacements, Mapping):
                raise ValueError("Donors must map affected layers and experts")
            if any(type(target) is not int for target in replacements) or set(replacements) != set(
                mask[layer]
            ):
                raise ValueError("Each substituted expert needs exactly one donor")
            if any(
                type(donor) is not int or donor < 0 or donor in mask[layer]
                for donor in replacements.values()
            ):
                raise ValueError("Donors must be nonnegative and outside the target mask")
            donors[layer] = MappingProxyType(dict(sorted(replacements.items())))
        if self.mode == "substitute" and set(donors) != set(mask):
            raise ValueError("Substitution requires a donor map for every target layer")
        if self.mode != "substitute" and donors:
            raise ValueError("Donor maps apply only to substitution")
        object.__setattr__(self, "mask", MappingProxyType(dict(sorted(mask.items()))))
        object.__setattr__(self, "donors", MappingProxyType(dict(sorted(donors.items()))))

    def __bool__(self):
        return bool(self.mask) and self.strength > 0

    def validate(self, layers, experts, top_k):
        for layer, targets in self.mask.items():
            if layer >= layers or any(expert >= experts for expert in targets):
                raise ValueError("Intervention dimensions exceed the model")
            if self.mode == "reroute" and len(targets) > experts - top_k:
                raise ValueError("Rerouting must leave top-k experts available")
            if any(donor >= experts for donor in self.donors.get(layer, {}).values()):
                raise ValueError("Donor ID exceeds the model expert count")

    def record(self):
        return {
            "mode": self.mode,
            "strength": float(self.strength),
            "mask": {layer: list(experts) for layer, experts in self.mask.items()},
            "donors": {layer: dict(donors) for layer, donors in self.donors.items()},
            "fixed_routing": "router choices on incoming states at each target site",
            "zero_mass_rescaling": "zero output when every selected contribution is removed",
        }
