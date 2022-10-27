from types import FunctionType
import numpy as np


def numpy_pivot_table(
    data: np.ndarray,
    index: int,
    column: int,
    aggfunc: FunctionType
):
    """
    Maak een pivottabel zoals pandas.pivot_table maar dan op een numpy.ndarray i.p.v.
    een pandas.DataFrame.

    Args:
        data (np.ndarray): De tweedimensionele matrix met data.
        index (int): Het kolomnummer waarvan de waardes in de index moeten.
        column (int): Het kolomnummer waarvan we de waardes willen hebben.
        aggfunc (FunctionType): De functie om toe te passen op de waardes.

    Returns:
        Dict: De gemaakte pivottabel.
    """
    values = {}
    for i in range(len(data)):
        try:
            values[data[i, index]].append(data[i, column])
        except KeyError:
            values[data[i, index]] = [data[i, column]]

    pivotTable = dict((i, aggfunc(values[i])) for i in values.keys())

    return pivotTable


def get_euclidean_skim(
    zonesX: np.ndarray, zonesY: np.ndarray
) -> np.ndarray:
    """
    Maak een skim met hemelsbrede afstand o.b.v. coordinaten
    (in meters bij gebruik van Amersfoort RD coordinaten).

    Args:
        zonesX (np.ndarray): _description_
        zonesY (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    numZones = len(zonesX)
    skimEuclidean = ((
        (
            zonesX.repeat(numZones).reshape(numZones, numZones) -
            zonesX.repeat(numZones).reshape(numZones, numZones).transpose()
        ))**2 + (
        (
            zonesY.repeat(numZones).reshape(numZones, numZones) -
            zonesY.repeat(numZones).reshape(numZones, numZones).transpose()
        ))**2) ** 0.5

    return skimEuclidean.flatten()
