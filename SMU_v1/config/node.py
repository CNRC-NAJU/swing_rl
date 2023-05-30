import warnings
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class NodeConfig:
    # ---------------- Generator ----------------
    _generator_unit_power: int = 1  # This should not be changed
    _generator_unit_mass: float = 1.0
    _generator_unit_gamma: float = 1.0

    # TODO:  When finding steady state, decrease mass, increase gamma
    # generator_temporary_unit_mass: float = 0.1
    # generator_temporary_unit_gamma: float = 10.0

    # ---------------- Renewable ----------------
    _renewable_unit_power: int = 1  # This should not be changed
    _renewable_unit_gamma_mass_ratio: float = 1.0  # gamma / mass

    # ---------------- Consumer ----------------
    _consumer_unit_power: int = -1  # This should not be changed
    _consumer_unit_mass: float = 1.0
    _consumer_unit_gamma: float = 1.0

    # -------- Controllable Consumer ---------
    _sink_unit_power: int = -1  # This should not be changed
    _sink_unit_mass: float = 1.0
    _sink_unit_gamma: float = 1.0

    def __post__init__(self) -> None:
        assert self.validate_source(self._generator_unit_power)
        assert self.validate_positive(self._generator_unit_mass)
        assert self.validate_positive(self._generator_unit_gamma)

        assert self.validate_source(self._renewable_unit_power)
        assert self.validate_positive(self._renewable_unit_gamma_mass_ratio)

        assert self.validate_user(self._consumer_unit_power)
        assert self.validate_positive(self._consumer_unit_mass)
        assert self.validate_positive(self._consumer_unit_gamma)

        assert self.validate_user(self._sink_unit_power)
        assert self.validate_positive(self._sink_unit_mass)
        assert self.validate_positive(self._sink_unit_gamma)

    def from_dict(self, config: dict[str, Any]) -> None:
        for key, value in config.items():
            assert hasattr(self, key)
            setattr(self, key, value)

    # ------------- Unit powers ----------------
    @staticmethod
    def validate_source(power: int) -> bool:
        return power == 1

    @staticmethod
    def validate_user(power: int) -> bool:
        return power == -1

    @property
    def generator_unit_power(self) -> int:
        return self._generator_unit_power

    @property
    def renewable_unit_power(self) -> int:
        return self._renewable_unit_power

    @property
    def consumer_unit_power(self) -> int:
        return self._consumer_unit_power

    @property
    def sink_unit_power(self) -> int:
        return self._sink_unit_power

    @generator_unit_power.setter
    def generator_unit_power(self, value: int) -> None:
        if value != 1:
            warnings.warn(f"Generator unit power is fixed to 1. Ignore", stacklevel=2)
        return

    @renewable_unit_power.setter
    def renewable_unit_power(self, value: int) -> None:
        if value != 1:
            warnings.warn(f"Renewable unit power is fixed to 1. Ignore", stacklevel=2)
        return

    @consumer_unit_power.setter
    def consumer_unit_power(self, value: int) -> None:
        if value != -1:
            warnings.warn(f"Consumer unit power is fixed to -11. Ignore", stacklevel=2)
        return

    @sink_unit_power.setter
    def sink_unit_power(self, value: int) -> None:
        if value != -1:
            warnings.warn(f"Sink unit power is fixed to -1. Ignore", stacklevel=2)
        return

    # ------------- Other parameters ----------------
    @staticmethod
    def validate_positive(value: float) -> bool:
        return value > 0

    @property
    def generator_unit_mass(self) -> float:
        return self._generator_unit_mass

    @property
    def generator_unit_gamma(self) -> float:
        return self._generator_unit_gamma

    @property
    def renewable_unit_gamma_mass_ratio(self) -> float:
        return self._renewable_unit_gamma_mass_ratio

    @property
    def consumer_unit_mass(self) -> float:
        return self._generator_unit_mass

    @property
    def consumer_unit_gamma(self) -> float:
        return self._consumer_unit_gamma

    @property
    def sink_unit_mass(self) -> float:
        return self._sink_unit_mass

    @property
    def sink_unit_gamma(self) -> float:
        return self._sink_unit_gamma

    @generator_unit_mass.setter
    def generator_unit_mass(self, value: float) -> None:
        if not self.validate_positive(value):
            warnings.warn(
                "Generator unit mass should be positive. Ignore", stacklevel=2
            )
            return
        self._generator_unit_mass = value

    @generator_unit_gamma.setter
    def generator_unit_gamma(self, value: float) -> None:
        if not self.validate_positive(value):
            warnings.warn(
                "Generator unit gamma should be positive. Ignore", stacklevel=2
            )
            return
        self._generator_unit_gamma = value

    @renewable_unit_gamma_mass_ratio.setter
    def renewable_unit_gamma_mass_ratio(self, value: float) -> None:
        if not self.validate_positive(value):
            warnings.warn(
                "Renewable unit gamma/mass should be positive. Ignore", stacklevel=2
            )
            return
        self._renewable_unit_gamma_mass_ratio = value

    @consumer_unit_mass.setter
    def consumer_unit_mass(self, value: float) -> None:
        if not self.validate_positive(value):
            warnings.warn("Consumer unit mass should be positive. Ignore", stacklevel=2)
            return
        self._consumer_unit_mass = value

    @consumer_unit_gamma.setter
    def consumer_unit_gamma(self, value: float) -> None:
        if not self.validate_positive(value):
            warnings.warn(
                "Consumer unit gamma should be positive. Ignore", stacklevel=2
            )
            return
        self._consumer_unit_gamma = value

    @sink_unit_mass.setter
    def sink_unit_mass(self, value: float) -> None:
        if not self.validate_positive(value):
            warnings.warn("Sink unit mass should be positive. Ignore", stacklevel=2)
            return
        self._sink_unit_mass = value

    @sink_unit_gamma.setter
    def sink_unit_gamma(self, value: float) -> None:
        if not self.validate_positive(value):
            warnings.warn("Sink unit gamma should be positive. Ignore", stacklevel=2)
            return
        self._sink_unit_gamma = value


NODE_CONFIG = NodeConfig()
