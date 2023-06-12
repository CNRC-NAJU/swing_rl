from dataclasses import dataclass


@dataclass(slots=True)
class PerturbationConfig:
    """Size of the perturbation.
    One node can be assigned to have multiple perturbations"""

    size: int = 1

    def from_dict(self, config: dict[str, int]) -> None:
        for key, value in config.items():
            assert hasattr(self, key)
            setattr(self, key, value)
