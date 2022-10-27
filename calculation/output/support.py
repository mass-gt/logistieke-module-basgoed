from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
from settings import Settings


def get_length_class_shares(settings: Settings) -> Dict[Tuple[int, str], float]:
    """
    Haalt de verdeling over de lengteklassen op uit "VerdelingLengteklassen".

    Args:
        settings (Settings): _description_

    Returns:
        Dict[Tuple[int, str], float]: _description_
    """
    return dict(
        ((row['vehicle_type_lwm'], row['length_class']), row['share'])
        for row in pd.read_csv(
            settings.get_path("VerdelingLengteklassen"), sep='\t').to_dict('records'))


def get_nrm_to_vam(zones: pd.DataFrame) -> np.ndarray:
    """
    Maakt de koppeling van NRM-zone naar VAM-zone.

    Args:
        zones (pd.DataFrame): _description_

    Returns:
        np.ndarray: _description_
    """
    return np.array(zones['zone_vam'], dtype=int)


def add_vam_to_tours(tours: pd.DataFrame, arrayNRMtoVAM: np.ndarray) -> pd.DataFrame:
    """
    Bepaalt de herkomst en bestemming van elke rit in "Tours" op VAM-niveau.

    Args:
        tours (pd.DataFrame): _description_
        arrayNRMtoVAM (np.ndarray): _description_

    Returns:
        pd.DataFrame: _description_
    """
    tours['orig_vam'] = arrayNRMtoVAM[tours['orig_nrm'].values - 1]
    tours['dest_vam'] = arrayNRMtoVAM[tours['dest_nrm'].values - 1]
    return tours


def make_vam_matrices(
    tours: pd.DataFrame,
    lengthClassShares: Dict[Tuple[int, str], float],
    dimLengthClass: List[str],
    yearFac: float
) -> Dict[str, np.ndarray]:
    """
    Maakt vrachtautomatrix op VAM-zonering per lengteklasse.

    Args:
        tours (pd.DataFrame): _description_
        lengthClassShares (Dict[Tuple[int, str], float]): _description_
        dimLengthClass (List[str]): _description_
        yearFac (float): _description_

    Returns:
        Dict[str, np.ndarray]: _description_
    """
    maxVAM = max(np.r_[tours['orig_vam'].values, tours['dest_vam'].values])

    matricesVAM = dict(
        (lengthClass, np.zeros((maxVAM, maxVAM))) for lengthClass in dimLengthClass)

    for row in tours.to_dict('records'):
        origVAM = row['orig_vam']
        destVAM = row['dest_vam']
        vt = row['vehicle_type_lwm']

        for lengthClass in dimLengthClass:
            share = lengthClassShares[(vt, lengthClass)]
            if share > 0:
                matricesVAM[lengthClass][origVAM - 1, destVAM - 1] += share

    for lengthClass in dimLengthClass:
        matricesVAM[lengthClass] = np.round(matricesVAM[lengthClass] * yearFac, 6)

    return matricesVAM


def write_vam_matrices(
    settings: Settings,
    matricesVAM: Dict[str, np.ndarray],
    sep: str = '\t'
):
    """
    Schrijft de vrachtautomatrix per lengteklasse weg naar een karaktergescheiden bestand.

    Args:
        settings (Settings): _description_
        matricesVAM (Dict[str, np.ndarray]): _description_
        sep (str, optional): _description_. Defaults to '\t'.
    """
    for lengthClass, matrix in matricesVAM.items():
        numRows, numCols = matrix.shape

        with open(settings.get_path("VrachtAutoMatrix", length_class=lengthClass), 'w') as f:
            f.write(f"lms_lad{sep}lms_los{sep}aantrit\n")

            for row in range(numRows):
                for col in range(numCols):
                    value = matrix[row, col]
                    if value > 0:
                        f.write(f"{row + 1}{sep}{col + 1}{sep}{value}\n")


def make_bam_matrices(
    tripsVanConstruction: np.ndarray,
    tripsVanService: np.ndarray,
    tours: pd.DataFrame,
    parcelTours: pd.DataFrame,
    arrayNRMtoVAM: np.ndarray,
    dimVanSegment: List[str],
    dimVT: List[Dict[str, str]],
    yearFacConstruction: float,
    yearFacService: float,
    yearFacGoods: float,
    yearFacParcel: float
) -> List[np.ndarray]:
    """
    Maakt bestelautomatrix op VAM-zonering per bestelautosegment.

    Args:
        tripsVanConstruction (np.ndarray): _description_
        tripsVanService (np.ndarray): _description_
        tours (pd.DataFrame): _description_
        parcelTours (pd.DataFrame): _description_
        arrayNRMtoVAM (np.ndarray): _description_
        dimVanSegment (List[str]): _description_
        dimVT (List[str]): _description_
        yearFacConstruction (float): _description_
        yearFacService (float): _description_
        yearFacGoods (float): _description_
        yearFacParcel (float): _description_

    Returns:
        List[np.ndarray]: _description_
    """
    maxVAM = max(arrayNRMtoVAM)
    maxNRM = len(arrayNRMtoVAM)

    matricesBAM = [np.zeros((maxVAM, maxVAM)) for vanSegment in dimVanSegment]

    # Bouw en service
    enumODsNRM = np.array(np.meshgrid(
        np.arange(maxNRM) + 1, np.arange(maxNRM) + 1)).T.reshape(-1, 2)
    trips = pd.DataFrame(np.c_[
        arrayNRMtoVAM[enumODsNRM[:, 0] - 1],
        arrayNRMtoVAM[enumODsNRM[:, 1] - 1],
        tripsVanConstruction * yearFacConstruction,
        tripsVanService * yearFacService],
        columns=['orig_vam', 'dest_vam', 'construction', 'service'])

    for origVAM, row in pd.pivot_table(
        trips, index='orig_vam', columns='dest_vam', values='construction', aggfunc=sum
    ).to_dict().items():
        for destVAM, numTrips in row.items():
            matricesBAM[0][int(origVAM - 1), int(destVAM - 1)] += numTrips

    for origVAM, row in pd.pivot_table(
        trips, index='orig_vam', columns='dest_vam', values='service', aggfunc=sum
    ).to_dict().items():
        for destVAM, numTrips in row.items():
            matricesBAM[1][int(origVAM - 1), int(destVAM - 1)] += numTrips

    # Goederenvervoer
    vtVan = [i for i in range(len(dimVT)) if dimVT[i]['name'].lower() == 'bestel'][0]
    for row in tours[tours['vehicle_type_lwm'] == vtVan].to_dict('records'):
        origVAM = row['orig_vam']
        destVAM = row['dest_vam']
        matricesBAM[2][origVAM - 1, destVAM - 1] += yearFacGoods

    # Post en pakket
    for row in parcelTours[parcelTours['vehicle_type_lwm'] == 'Van'].to_dict('records'):
        origVAM = arrayNRMtoVAM[row['orig_nrm'] - 1]
        destVAM = arrayNRMtoVAM[row['dest_nrm'] - 1]
        matricesBAM[3][origVAM - 1, destVAM - 1] += yearFacParcel

    return matricesBAM


def write_bam_matrix(
    settings: Settings,
    matricesBAM: List[np.ndarray],
    sep: str = '\t'
):
    """
    Schrijft de bestelautomatrix weg naar een karaktergescheiden bestand.

    Args:
        settings (Settings): _description_
        matricesBAM (List[np.ndarray]): _description_
        sep (str, optional): _description_. Defaults to '\t'.
    """
    maxVAM = matricesBAM[0].shape[0]
    numVanSegment = len(matricesBAM)

    matrixBAM = np.array(np.meshgrid(
        np.arange(maxVAM) + 1,
        np.arange(maxVAM) + 1,
        np.arange(numVanSegment) + 1)).T.reshape(-1, 3)
    matrixBAM = matrixBAM[np.lexsort([matrixBAM[:, j] for j in range(2, -1, -1)])]
    matrixBAM = np.c_[matrixBAM, np.zeros(maxVAM * maxVAM * numVanSegment)]

    for vanSegment in range(numVanSegment):
        matrixBAM[vanSegment::numVanSegment, -1] = matricesBAM[vanSegment].flatten()

    matrixBAM = matrixBAM[matrixBAM[:, -1] > 0]

    matrixBAM = pd.DataFrame(matrixBAM, columns=['lms_lad', 'lms_los', 'kenmerk', 'aantrit'])
    matrixBAM['aantrit'] = matrixBAM['aantrit'].round(6)
    for col in ['lms_lad', 'lms_los', 'kenmerk']:
        matrixBAM[col] = matrixBAM[col].astype(int)

    matrixBAM.to_csv(settings.get_path("BestelAutoMatrix"), sep=sep, index=False)
