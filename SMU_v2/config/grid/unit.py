import warnings
from dataclasses import dataclass
from typing import Any

# TODO:  When finding steady state, decrease mass, increase gamma
# generator_temporary_unit_mass: float = 0.1
# generator_temporary_unit_gamma: float = 10.0


@dataclass(slots=True, frozen=True, kw_only=True)
class UnitConfig:
    """
    Unit values of consumer/generator/sink node
    power should not be changed
    """

    _power: int
    mass: float
    gamma: float

    def from_dict(self, config: dict[str, Any]) -> None:
        for key, value in config.items():
            if key == "power":
                continue
            assert hasattr(self, key)
            setattr(self, key, value)

    @property
    def power(self) -> int:
        return self._power

    @power.setter
    def power(self, _: int) -> None:
        warnings.warn(f"You should not change unit power. Ignore", stacklevel=2)
        return


@dataclass(slots=True, frozen=True, kw_only=True)
class RenewableUnitConfig:
    """
    Unit values of renewable node
    power should not be changed
    """

    _power: int
    gamma_mass_ratio: float = 1.0  # gamma / mass

    def from_dict(self, config: dict[str, Any]) -> None:
        for key, value in config.items():
            assert hasattr(self, key)
            if key == "power":
                continue
            setattr(self, key, value)

    @property
    def power(self) -> int:
        return self._power

    @power.setter
    def power(self, _: int) -> None:
        warnings.warn(f"You should not change unit power. Ignore", stacklevel=2)
        return
