from typing import Any, Dict, NamedTuple

import numpy as np
import pandas as pd


class Terminals(NamedTuple):
    """
    Carries different kinds of terminals.

    Args:
        ck: Container terminals.
        rail: Rail terminals.
        water: Waterway terminals.
    """
    ck: pd.DataFrame
    rail: pd.DataFrame
    water: pd.DataFrame


class Tonnes(NamedTuple):
    """
    Carries different kinds of tonnes.

    Args:
        road: Non-container road tonnes.
        rail: Non-container rail tonnes.
        water: Non-container waterway tonnes.
        ck: Container tonnes
    """
    road: pd.DataFrame
    rail: pd.DataFrame
    water: pd.DataFrame
    ck: pd.DataFrame


class Routes(NamedTuple):
    """
    Carries different kinds of container routes.

    Args:
        direct: Direct routes.
        toTerminal: Routes to a terminal.
        fromTerminal: routes from a terminal.
    """
    direct: pd.DataFrame
    toTerminal: pd.DataFrame
    fromTerminal: pd.DataFrame


class CostFigure(NamedTuple):
    """
    Carries costfigures in eur per hour and kilometer.

    Args:
        eurPerHour: .
        eurPerKilometer: .
    """
    eurPerHour: np.ndarray
    eurPerKilometer: np.ndarray


class CostFigures(NamedTuple):
    """
    Carries costfigures for vehicle types, container and non-container.

    Args:
        vehicleType: CostFigure
        nonContainer: CostFigure
        container: CostFigure
    """
    vehicleType: CostFigure
    nonContainer: CostFigure
    container: CostFigure


class ContainerStatistics(NamedTuple):
    """
    Carries costfigures for vehicle types, container and non-container.

    Args:
        vehicleType: CostFigure
        nonContainer: CostFigure
        container: CostFigure
    """
    avgShipSizeCK: np.ndarray
    emptyShareCK: Any
    avgShipSizeEmptyCK: Any


class Probabilities(NamedTuple):
    """
    Carries probabilities.

    Args:
        producer: Dict[int, Dict[int, np.ndarray]]
        consumer: Dict[int, Dict[int, np.ndarray]]
    """
    producer: Dict[int, Dict[int, np.ndarray]]
    consumerCumulative: Dict[int, Dict[int, np.ndarray]]
