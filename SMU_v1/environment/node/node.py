from abc import ABC, abstractmethod

import numpy as np

from .type import NodeType


class Node(ABC):
    def __init__(self, max_units: int) -> None:
        self.max_units = max_units
        self.active_units: int = 1  # [1, max_units]

        self.unit_power: int = 0
        self.unit_mass: float = 0.0
        self.unit_gamma: float = 0.0

    def increase(self) -> bool:
        """Increase currently active unit.
        If the increment was failed
        i.e., already all units are active, return False"""
        if self.full_active:
            # All units are already active: failed
            return False

        # Number of active_units units are increased
        self.active_units += 1
        return True

    def decrease(self) -> bool:
        """Increase currently active unit.
        If the decrement was failed,
        i.e., already no units are active, return False"""
        if self.full_inactive:
            # No units are already inactive: failed
            return False

        # Number of active_units units are decreased
        self.active_units -= 1
        return True

    def perturbate(self, rng: np.random.Generator) -> int:
        """Randomly increase or decrease active units
        If increased, return 1. Otherwise, return -1
        """
        # Randomly change power
        if rng.random() < 0.5:
            success = self.increase()
            if not success:
                # If increasing is failed, always do decreasing
                self.decrease()
                return -1
            return 1
        else:
            success = self.decrease()
            if not success:
                # If decreasing is failed, always do increasing
                self.increase()
                return 1
            return -1

    def __str__(self) -> str:
        return f"Type: {self.type.name}, max: {self.max_units}, active: {self.active_units}"

    @property
    def full_active(self) -> bool:
        return self.active_units == self.max_units

    @property
    def full_inactive(self) -> bool:
        return self.active_units == 1

    @property
    def ratio(self) -> float:
        return self.active_units / self.max_units

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

    @property
    @abstractmethod
    def capacity(self) -> int:
        pass
