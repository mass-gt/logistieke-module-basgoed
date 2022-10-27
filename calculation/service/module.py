import logging

import pandas as pd
import numpy as np
from typing import Dict, Tuple

from calculation.common.io import write_mtx, get_skims

from settings import Settings


def run_module(settings: Settings, logger: logging.Logger):

    logger.debug("\tInladen en prepareren invoerdata...")

    tolerance = settings.module.service_tolerance
    maxIter = settings.module.service_max_num_iter

    yearFactorService = settings.module.year_factor_service
    yearFactorConstruction = settings.module.year_factor_construction

    # Import cost parameters
    costVan = np.array(pd.read_csv(
        settings.get_path("KostenKentallenVoertuigType", vehicle_type="Bestel"),
        sep='\t'))[:, [0, 1, 2]]
    costPerKilometer = np.average(costVan[:, 1])
    costPerHour = np.average(costVan[:, 2])

    # Import distance decay parameters
    coeffsDistDecay = get_coeffs_dist_decay(settings)

    # Import zones data
    zones = pd.read_csv(settings.get_path("ZonesNRM"), sep='\t')
    numZones = len(zones)
    zonesNL = np.where(zones['landsdeel'] <= 4)[0]

    # Import regression coefficients
    coeffsProdAttr = get_coeffs_prod_attr(settings)

    # Haal skim met reistijden en afstanden op
    skimTravTime, skimDistance, skimCostCharge = get_skims(
        settings, logger, zones, freight=False, changeZeroValues=True, returnSkimCostCharge=True)

    # Surface of DCs per zone
    surfaceDC = np.array(zones['surface_distr__meter2'], dtype=float)

    logger.debug('\tBerekenen producties en attracties')

    prodService, prodBouw = calc_productions(zones, zonesNL, coeffsProdAttr, surfaceDC)

    logger.debug('\tBepalen initiele HB-matrices..')

    # Travel costs
    skimCost = (
        costPerHour * (skimTravTime / 3600) +
        costPerKilometer * (skimDistance / 1000) +
        skimCostCharge
    ).reshape(numZones, numZones)

    # Intrazonal costs (half of costs to nearest zone in terms of travel costs)
    for i in range(numZones):
        skimCost[i, i] = 0.5 * np.min(skimCost[i, skimCost[i, :] > 0])

    matrixService, matrixBouw = calc_start_matrices(
        prodService, prodBouw, coeffsDistDecay, skimCost)

    logger.debug('\tDistributie serviceritten...')

    matrixService = distribute_trips(
        prodService, matrixService, tolerance, maxIter, yearFactorService, logger)

    logger.debug('\tDistributie bouwritten...')

    matrixBouw = distribute_trips(
        prodBouw, matrixBouw, tolerance, maxIter, yearFactorConstruction, logger)

    logger.debug('\tWegschrijven HB-matrices...')

    write_mtx(settings.get_path("RittenBestelService"), matrixService.flatten(), numZones)
    write_mtx(settings.get_path("RittenBestelBouw"), matrixBouw.flatten(), numZones)


def get_coeffs_dist_decay(settings: Settings) -> Dict[str, Dict[str, float]]:
    """
    Leest de afstandsvervalcoefficienten in voor bouw en service.

    Args:
        settings (Settings): _description_

    Returns:
        Dict[str, Dict[str, float]]: _description_
    """
    data = pd.read_csv(
        settings.get_path("CoefficientenAfstandsverval_VanSegment"), sep='\t', index_col=0)

    coeffsDistDecay = {'service': {}, 'construction': {}}
    for row in data.to_records('dict'):
        coeffsDistDecay[row['van_segment']][row['parameter']] = row['value']

    return coeffsDistDecay


def get_coeffs_prod_attr(settings: Settings) -> Dict[str, Dict[str, float]]:
    """
    Lees de productie-attractieparameters in.

    Args:
        settings (Settings): _description_

    Returns:
        Dict[str, Dict[str, float]]: _description_
    """
    coeffsProdAttr = {'service': {}, 'construction': {}}

    for row in pd.read_csv(
        settings.get_path("CoefficientenProdAttr_VanSegment"), sep='\t'
    ).to_dict('records'):
        segment = row['van_segment']
        for col in ['surface_dc', 'population', 'calib_fac']:
            coeffsProdAttr[segment][col] = row[col]

    for row in pd.read_csv(
        settings.get_path("CoefficientenProdAttr_VanSegment_EmploymentCategory"), sep='\t'
    ).to_dict('records'):
        segment = row['van_segment']
        empl = row['employment_category']
        coeffsProdAttr[segment][f'empl_{empl}'] = row['value']

    return coeffsProdAttr


def calc_productions(
    zones: pd.DataFrame, zonesNL: np.ndarray,
    coeffsProdAttr: Dict[str, Dict[str, float]],
    surfaceDC: np.ndarray
) -> Tuple[np.ndarray]:
    """
    Bepaal het aantal ritten dat elke zone produceert voor service en bouw.

    Args:
        zones (pd.DataFrame): Alle zones, met socioeconomische gegevens.
        zonesNL (np.ndarray): De indices van de Nederlandse zones.
        coeffsProdAttr (dict): De PA-coefficienten voor service en bouw.
        surfaceDC (np.ndarray): Per zone het oppervlak aan distributiecentra.

    Returns:
        tuple: Met daarin:
            - np.ndarray: Per zone de productie van serviceritten.
            - np.ndarray: Per zone de productie van bouwritten.
    """
    # Determine produced trips per zone for service and construction
    prodService = np.zeros(len(zones), dtype=float)
    prodBouw = np.zeros(len(zones), dtype=float)

    coeffsService = coeffsProdAttr['service']
    coeffsBouw = coeffsProdAttr['construction']

    # For the zones in NL
    for i in zonesNL:
        prodService[i] = coeffsService['calib_fac'] * (
            coeffsService['surface_dc'] * surfaceDC[i] +
            coeffsService['empl_landbouw'] * zones.at[i, 'empl_landbouw'] +
            coeffsService['empl_industrie'] * zones.at[i, 'empl_industrie'] +
            coeffsService['empl_detail'] * zones.at[i, 'empl_detail'] +
            coeffsService['empl_diensten'] * zones.at[i, 'empl_diensten'] +
            coeffsService['empl_overig'] * zones.at[i, 'empl_overig'] +
            coeffsService['population'] * zones.at[i, 'population'])

        prodBouw[i] = coeffsBouw['calib_fac'] * (
            coeffsBouw['surface_dc'] * surfaceDC[i] +
            coeffsBouw['empl_landbouw'] * zones.at[i, 'empl_landbouw'] +
            coeffsBouw['empl_industrie'] * zones.at[i, 'empl_industrie'] +
            coeffsBouw['empl_detail'] * zones.at[i, 'empl_detail'] +
            coeffsBouw['empl_diensten'] * zones.at[i, 'empl_diensten'] +
            coeffsBouw['empl_overig'] * zones.at[i, 'empl_overig'] +
            coeffsBouw['population'] * zones.at[i, 'population'])

    return prodService, prodBouw


def calc_start_matrices(
    prodService: np.ndarray, prodConstruction: np.ndarray,
    coeffsDistDecay: Dict[str, Dict[str, float]],
    skimCost: np.ndarray
) -> Tuple[np.ndarray]:
    """
    Maak de initiele ritmatrices aan.

    Args:
        prodService (np.ndarray): _description_
        prodConstruction (np.ndarray): _description_
        coeffsDistDecay (dict): _description_
        skimCost (np.ndarray): _description_

    Returns:
        tuple: Met daarin:
            - np.ndarray: De startmatrix voor service.
            - np.ndarray: De startmatrix voor bouw.
    """
    alphaService = coeffsDistDecay['service']['alpha']
    betaService = coeffsDistDecay['service']['beta']
    alphaConstruction = coeffsDistDecay['construction']['alpha']
    betaConstruction = coeffsDistDecay['construction']['beta']

    # Travel resistance
    matrixService = 100 / (1 + (np.exp(alphaService) * skimCost ** betaService))
    matrixConstruction = 100 / (1 + (np.exp(alphaConstruction) * skimCost ** betaConstruction))

    # Multiply by productions and then attractions to get start matrix
    # (assumed: productions = attractions)
    matrixService *= np.tile(prodService, (len(prodService), 1))
    matrixConstruction *= np.tile(prodConstruction, (len(prodConstruction), 1))

    matrixService *= np.tile(prodService, (len(prodService), 1)).transpose()
    matrixConstruction *= np.tile(prodConstruction, (len(prodConstruction), 1)).transpose()

    return matrixService, matrixConstruction


def distribute_trips(
    prod: np.ndarray, matrix: np.ndarray,
    tolerance: float, maxIter: int, yearFactor: int,
    logger: logging.Logger
) -> np.ndarray:
    """
    Voer de tripdistributie uit gegeven de zonale producties/attracties en de startmatrix.

    Args:
        prod (np.ndarray): Productie ritten per zone.
        matrix (np.ndarray): De startmatrix.
        tolerance (float): Het tolerantiecriterium (in percentage afwijking).
        maxIter (int): Het maximum aantal iteraties.
        yearFactor (int): De jaarfactor om van jaartotaal naar dagtotaal te gaan.
        logger (logging.Logger): _description_

    Returns:
        np.ndarray: De geschaalde rittenmatrix.
    """
    itern = 0
    conv = tolerance + 100

    nZones = len(matrix)

    while (itern < maxIter) and (conv > tolerance):
        itern += 1

        maxColScaleFac = 0
        totalRows = np.sum(matrix, axis=0)

        for j in range(nZones):
            total = totalRows[j]

            if total > 0:
                scaleFacCol = prod[j] / total

                if abs(scaleFacCol) > abs(maxColScaleFac):
                    maxColScaleFac = scaleFacCol

                matrix[:, j] *= scaleFacCol

        maxRowScaleFac = 0
        totalCols = np.sum(matrix, axis=1)

        for i in range(nZones):
            total = totalCols[i]

            if total > 0:
                scaleFacRow = prod[i] / total

                if abs(scaleFacRow) > abs(maxRowScaleFac):
                    maxRowScaleFac = scaleFacRow

                matrix[i, :] *= scaleFacRow

        conv = round(max(abs(maxColScaleFac - 1), abs(maxRowScaleFac - 1)), 4)
        print(f'\t\tIteratie {itern} - Convergentie {conv}')

    if conv > tolerance:
        logger.warning(
            f'Convergentie {conv} is lager dan het tolerantiecriterium {tolerance}, ' +
            'mogelijk zijn meer iteraties nodig.')

    matrix /= yearFactor

    return matrix
