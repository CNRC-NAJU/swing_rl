from abc import ABC, abstractmethod

from .type import NodeType


class Node(ABC):
    __slots__ = [
        "_max_units",
        "_active_units",
        "_unit_power",
        "_unit_mass",
        "_unit_gamma",
    ]

    def __init__(self, max_units: int) -> None:
        assert max_units >= 2, "max units should be at least 2"
        self._max_units = max_units
        self._active_units: int = 0  # 0 or [1, max_units], 0 only when grid is offline

        self._unit_power: int
        self._unit_mass: float
        self._unit_gamma: float

    def set(self, units: int) -> None:
        """Set active units manually"""
        assert isinstance(units, int) and 0 <= units <= self._max_units
        self._active_units = units

    def increase(self) -> bool:
        """Increase currently active unit.
        If the increment was failed
        i.e., already all units are active, return False"""
        if self.headroom_increase == 0:
            # All units are already active: failed
            return False

        # Number of active_units units are increased
        self._active_units += 1
        return True

    def decrease(self) -> bool:
        """Increase currently active unit.
        If the decrement was failed,
        i.e., already no units are active, return False"""
        if self.headroom_decrease == 0:
            # No units are already inactive: failed
            return False

        # Number of active_units units are decreased
        self._active_units -= 1
        return True

    def __str__(self) -> str:
        return (
            f"{self.type.name.lower()}, max: {self._max_units}, active:"
            f" {self._active_units}"
        )

    @property
    def max_units(self) -> int:
        return self._max_units

    @property
    def active_units(self) -> int:
        return self._active_units

    @property
    def headroom_increase(self) -> int:
        """Headroom for increase"""
        return self._max_units - self._active_units

    @property
    def headroom_decrease(self) -> int:
        """Headroom for decrease"""
        return self._active_units - 1

    @property
    def headroom(self) -> int:
        """Headroom for perturbation: both increase and decrease"""
        return max(self.headroom_decrease, self.headroom_increase)

    @property
    def ratio(self) -> float:
        return self._active_units / self._max_units

    @property
    def capacity(self) -> int:
        return abs(self._unit_power * self._max_units)

    @property
    @abstractmethod
    def type(self) -> NodeType:
        pass

    @property
    @abstractmethod
    def power(self) -> int:
        pass

    @property
    @abstractmethod
    def mass(self) -> float:
        pass

    @property
    @abstractmethod
    def gamma(self) -> float:
        pass
