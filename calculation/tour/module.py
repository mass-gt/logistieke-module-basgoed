import logging

import numpy as np
import pandas as pd
import multiprocessing as mp
import shapefile as shp
import functools
from numba import njit
from typing import Dict, List, Tuple

from calculation.common.io import get_skims
from calculation.common.params import get_num_cpu, set_seed

from settings import Settings


def run_module(settings: Settings, logger: logging.Logger):

    logger.debug("\tInladen en prepareren invoerdata...")

    # Number of CPUs over which the tour formation is parallelized
    numCPU = get_num_cpu(settings.module.tour_num_cpu, 10, logger)

    # The number of carriers that transport the shipments not going to or from a DC
    numCarriersNonDC = settings.module.tour_num_carriers_non_dc

    # Maximum number of shipments in tour
    maxNumShips = settings.module.tour_max_num_ships

    # Maximum length of empty trip
    maxEmptyTripDistance = settings.module.tour_max_empty_trip_distance

    # Dimensies
    dimGG = settings.commodities
    dimLS = settings.dimensions.logistic_segment
    dimNSTR = settings.dimensions.nstr
    dimVT = settings.dimensions.vehicle_type_lwm
    dimCombType = settings.dimensions.combustion_type

    numLS = len(dimLS)
    numNSTR = len(dimNSTR) - 1
    numGG = max(dimGG)
    numVT = len(dimVT)
    numCombType = len(dimCombType)

    # Carrying capacity in kg
    carryingCapacity = get_truck_capacities(settings, numVT)

    # Importeer de NRM-moederzones
    zones = pd.read_csv(settings.get_path("ZonesNRM"), sep='\t')
    zones.index = zones['zone_nrm']
    numZones = len(zones)

    # Add level of urbanization based on segs (# houses / jobs) and surface (square m)
    zonesSted = np.array((zones['population'] / zones['surface__meter2']) >= 2500, dtype=int)

    # Maak een DataFrame met distributiecentra(zones)
    distrCenters = zones[['zone_bg', 'zone_ck', 'zone_nrm', 'surface_distr__meter2']]
    distrCenters = distrCenters[distrCenters['surface_distr__meter2'] > 0]
    distrCenters['employment'] = (
        distrCenters['surface_distr__meter2'] * settings.module.empl_per_m2_dc)
    distrCenters.index = np.arange(len(distrCenters))
    dcZones = np.array(distrCenters['zone_nrm'], dtype=int)
    numDC = len(distrCenters)

    # Koppeltabel NRM naar BG
    dictNRMtoBG = dict((zones.at[i, 'zone_nrm'], zones.at[i, 'zone_bg']) for i in zones.index)
    BGwithOverlapNRM = np.unique(zones['zone_bg'])

    # Import shipments data
    shipments, shipmentsCK = get_shipments(settings.get_path("Zendingen"))

    # niet-container zendingen in kilogrammen
    shipments['weight__ton'] *= 1000

    # Voeg kenmerken toe m.b.t. locatietypes aan niet-container zendingen
    # (header: LOGNODE_LOADING, LOGNODE_UNLOADING, URBAN, EXTERNAL)
    shipments = add_location_types_to_shipments(shipments, zonesSted, BGwithOverlapNRM)

    # Import probability distribution of departure time (per logistic segment)
    cumProbDepTimes = get_cum_prob_deptimes(settings, numLS)

    # Import ZEZ scenario input
    if settings.module.apply_zez:
        # Which zones are ZEZ and by which UCC-zone are they served
        zonesZEZ = pd.read_csv(settings.get_path("ZonesZEZ"), sep='\t')
        zezToUCC = dict(
            (zonesZEZ.at[i, 'zone_nrm'], zonesZEZ.at[i, 'zone_nrm_ucc']) for i in zonesZEZ.index)
        uccZones = np.array(np.unique(zonesZEZ['zone_nrm_ucc']), dtype=int)
        isZEZ = np.zeros(len(zones), dtype=int)
        isZEZ[zonesZEZ['zone_nrm'].values - 1] = 1

        # Cumulatieve kansen van de energiebronnen binnen elk
        # voertuigtype en logistiek segment
        cumProbsCombConsolidated, cumProbsCombDirect = get_cum_prob_combustion(
            settings, numCombType, numLS, numVT)

    logger.debug("\tZendingen prepareren voor tourformatie...")

    set_seed(settings.module.seed_tour_assign_carriers, logger, 'seed-tour-assign-carriers')

    # Add a column with a carrierID
    # - Shipments are first grouped by DC, one carrier per DC assumed [carrierID 0 to nDC].
    # - Shipments not loaded or unloaded at DC are randomly assigned
    #   to one of the other carriers [carrierID nDC to nDC+nCarNonDC].
    if settings.module.apply_zez:
        shipments = assign_carrier_to_shipments(
            shipments, settings.module.apply_zez, numDC, numCarriersNonDC,
            isZEZ, uccZones, zezToUCC)
    else:
        shipments = assign_carrier_to_shipments(
            shipments, settings.module.apply_zez, numDC, numCarriersNonDC,
            [], [], {})

    # Place shipments in Numpy array for faster accessing of values
    shipments = shipments[[
        'shipment_id', 'orig_nrm', 'dest_nrm', 'carrier_id',
        'vehicle_type_lwm', 'nstr', 'weight__ton',
        'dc_loading', 'dc_unloading', 'urban',
        'logistic_segment',
        'orig_bg', 'dest_bg',
        'external', 'commodity', 'container']]
    shipments = np.array(shipments, dtype=object)

    # Split shipments that are larger than the vehicle capacity
    # into multiple shipments
    shipments = split_shipments_with_capacity_exceedence(shipments, carryingCapacity)

    # Sort shipments by carrierID
    shipments = shipments[shipments[:, 3].argsort()]
    shipmentDict = dict(np.transpose(np.vstack(
        (np.arange(len(shipments)), shipments[:, 0].copy()))))

    # Give the shipments a new shipmentID after ordering the
    # array by carrierID
    shipments[:, 0] = np.arange(len(shipments))

    # Make sure SHIP_ID, ORIG_NRM and DEST_NRM are integers
    shipments[:, [0, 1, 2]] = np.array(shipments[:, [0, 1, 2]], dtype=int)

    # Haal skim met reistijden en afstanden op
    skimTravTime, skimDistance = get_skims(
        settings, logger, zones, freight=True, changeZeroValues=True)

    # Concatenate time and distance arrays as a 2-column matrix
    skim = np.c_[skimTravTime, skimDistance]

    # Coefficienten End Tour model
    coeffsTour = get_coeffs_tour(settings, numNSTR, numVT)

    # Total number of shipments and carriers
    numShipments = len(shipments)
    numCarriers = len(np.unique(shipments[:, 3]))

    # At which index is the first shipment of each carrier
    carrierMarkers = [0]
    for i in range(1, numShipments):
        if shipments[i, 3] != shipments[i - 1, 3]:
            carrierMarkers.append(i)
    carrierMarkers.append(numShipments)

    # How many shipments does each carrier have?
    numShipmentsPerCarrier = [None for car in range(numCarriers)]
    if numCarriers == 1:
        numShipmentsPerCarrier[0] = numShipments
    else:
        for i in range(numCarriers):
            numShipmentsPerCarrier[i] = carrierMarkers[i + 1] - carrierMarkers[i]

    # Bepaal welke CPU welke carriers doet
    set_seed(settings.module.seed_tour_formation, logger, 'seed-tour-formation')
    chunks = [
        (np.arange(numCarriers)[i::numCPU], np.random.randint(100000))
        for i in range(numCPU)]

    if numCPU > 1:

        logger.debug("\tStart tourformatie.")

        # Initialiseer een pool object dat de taken verdeelt over de CPUs
        p = mp.Pool(numCPU)

        # Voer de tourformatie uit
        tourformationResult = p.map(functools.partial(
            form_tours,
            carrierMarkers, shipments, skim, numZones, maxNumShips, carryingCapacity, dcZones,
            numShipmentsPerCarrier, numCarriers, coeffsTour,
            numNSTR), chunks)

        # Wait for completion of parallellization processes
        p.close()
        p.join()

        # Initialization of lists with information regarding tours
        # tours: here we store the shipmentIDs of each tour
        # tourSequences: here we store the order of (un)loading locations of each tour
        # numTours:  number of tours constructed for each carrier
        tours = [[] for car in range(numCarriers)]
        tourSequences = [[] for car in range(numCarriers)]
        numTours = np.zeros(numCarriers, dtype=int)
        tourIsExternal = [[] for car in range(numCarriers)]

        # Pak de tourformatieresultaten uit
        for cpu in range(numCPU):
            for car in chunks[cpu][0]:
                tours[car] = tourformationResult[cpu][0][car]
                tourSequences[car] = tourformationResult[cpu][1][car]
                numTours[car] = tourformationResult[cpu][2][car]
                tourIsExternal[car] = tourformationResult[cpu][3][car]

        del tourformationResult

    else:
        logger.debug("\tStart tourformatie.")

        tours, tourSequences, numTours, tourIsExternal = form_tours(
            carrierMarkers, shipments, skim, numZones, maxNumShips, carryingCapacity, dcZones,
            numShipmentsPerCarrier, numCarriers, coeffsTour,
            numNSTR, chunks[0])

    logger.debug("\t\tTourformatie afgerond voor alle carriers.")

    logger.debug("\tLege ritten toevoegen...")

    tourSequences, emptyTripAdded = add_empty_trips(
        tourSequences, tourIsExternal, numCarriers, numTours, skimDistance, maxEmptyTripDistance)

    logger.debug("\tExtra tourstatistieken afleiden...")

    numShipmentsPerTour, numTripsPerTour, numTripsTotal, tripWeights = get_extra_tour_attr(
        tours, tourSequences, shipments, numCarriers, numTours)

    logger.debug("\tVertrektijden trekken...")

    set_seed(settings.module.seed_tour_departure_times, logger, 'seed-tour-departure-times')

    depTimeTour = draw_dep_times(cumProbDepTimes, tours, shipments, numCarriers, numTours)

    set_seed(settings.module.seed_tour_combustion_types, logger, 'seed-tour-combustion-types')

    logger.debug("\tAandrijvingstypes bepalen...")

    if settings.module.apply_zez:
        combTypeTour, combTypeShipmentsCK = draw_combustion_types(
            settings, cumProbsCombConsolidated, cumProbsCombDirect,
            shipments, shipmentsCK, tours, tourSequences,
            numTours, numCarriers, numDC, numCarriersNonDC, numZones)
    else:
        # In the not-UCC scenario everything is assumed to be fuel
        combTypeTour = [[0 for tour in range(numTours[car])] for car in range(numCarriers)]
        combTypeShipmentsCK = [0 for i in shipmentsCK.index]

    logger.debug("\tTours herstructureren en CK-ritten aanmaken..")

    outputTours = format_output_tours(
        tours, tourSequences, tourIsExternal,
        shipments, shipmentsCK,
        zones, dcZones, skim, dictNRMtoBG, BGwithOverlapNRM,
        cumProbDepTimes,
        numCarriers, numDC, numTours,
        numShipmentsPerTour, numTripsPerTour, numTripsTotal,
        emptyTripAdded, tripWeights, depTimeTour, combTypeTour, combTypeShipmentsCK,
        numNSTR, numGG)

    logger.debug("\tTours wegschrijven als tekstbestand..")

    outputTours.to_csv(settings.get_path("Tours"), index=None, sep='\t')

    logger.debug("\tZendingen verrijken met tourgegevens...")

    shipments = enrich_shipments(
        shipments, shipmentsCK, shipmentDict, BGwithOverlapNRM, numCarriers, numTours, tours)

    shipments.to_csv(
        settings.get_path("ZendingenNaTourformatie"), index=False, sep='\t')

    if settings.module.write_shape_tour:

        logger.debug("\tTours wegschrijven als shapefile...")

        write_tours_to_shp(outputTours, settings.get_path("ToursShape"))

    logger.debug("\tRittenmatrices maken...")

    create_and_write_trip_matrices(outputTours, numLS - 1, numVT, settings)


def get_truck_capacities(settings: Settings, numVT: int) -> np.ndarray:
    """
    Returns truck capacities (in kg).
    """
    carryingCapacity = -1 * np.ones(numVT)
    for row in pd.read_csv(settings.get_path("LaadvermogensLWM"), sep='\t').to_dict('records'):
        vt = int(row['vehicle_type_lwm'])
        weight = float(row['weight__kgram'])
        carryingCapacity[vt] = weight

    if np.any(carryingCapacity == -1):
        raise Exception('Er zijn missende voertuigtypes in "LaadvermogensLWM".')

    return carryingCapacity


def get_coeffs_tour(
    settings: Settings,
    numNSTR: int,
    numVT: int
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Lees de coefficienten voor het keuzemodel voor tourformatie in.

    De sleutels zijn:
        - first
            - constant
            - tour_duration
            - capacity_utilization
            - transshipment
            - dc_loading
            - dc_unloading
            - urban
            - const_nstr
            - const_vehicle_type_lwm
        - later
            - constant
            - tour_duration
            - capacity_utilization
            - transshipment
            - dc_loading
            - dc_unloading
            - urban
            - proximity
            - number_of_stops
            - const_nstr
            - const_vehicle_type_lwm

    Args:
        settings (Settings): _description_
        numNSTR (int): Het aantal NSTR-goederengroepen.
        numVT (int): Het aantal voertuigtypen.

    Returns:
        dict: De coefficienten.
    """
    coeffs = {'first': {}, 'later': {}}

    coeffs1 = pd.read_csv(
        settings.get_path("CoefficientenTourformatie_TourStage"), sep='\t')
    coeffs2 = pd.read_csv(
        settings.get_path("CoefficientenTourformatie_TourStage_NSTR"), sep='\t')
    coeffs3 = pd.read_csv(
        settings.get_path("CoefficientenTourformatie_TourStage_VehicleTypeLWM"), sep='\t')

    # Herformateer coefficienten met dimensie (tour_stage)
    for parameter in [
        'constant', 'tour_duration', 'capacity_utilization',
        'transshipment', 'dc_loading', 'dc_unloading', 'urban'
    ]:
        coeffs['first'][parameter] = coeffs1.loc[
            (coeffs1['parameter'] == parameter) &
            (coeffs1['tour_stage'] == 'first'), 'value'].values[0]

    for parameter in [
        'constant', 'tour_duration', 'capacity_utilization',
        'transshipment', 'dc_loading', 'dc_unloading', 'urban',
        'proximity', 'number_of_stops'
    ]:
        coeffs['later'][parameter] = coeffs1.loc[
            (coeffs1['parameter'] == parameter) &
            (coeffs1['tour_stage'] == 'later'), 'value'].values[0]

    # Herformateer coefficienten met dimensie (tour_stage, nstr)
    for tourStage in ['first', 'later']:
        coeffs[tourStage]['const_nstr'] = np.zeros(numNSTR, dtype=float)
        for row in coeffs2[coeffs2['tour_stage'] == tourStage].to_dict('records'):
            nstr = row['nstr']
            value = row['value']
            coeffs[tourStage]['const_nstr'][nstr] = value

    # Herformateer coefficienten met dimensie (tour_stage, vehicle_type_lwm)
    for tourStage in ['first', 'later']:
        coeffs[tourStage]['const_vehicle_type_lwm'] = np.zeros(numVT, dtype=float)
        for row in coeffs3[coeffs3['tour_stage'] == tourStage].to_dict('records'):
            vt = row['vehicle_type_lwm']
            value = row['value']
            coeffs[tourStage]['const_vehicle_type_lwm'][vt] = value

    return coeffs


def get_cum_prob_deptimes(
    settings: Settings,
    numLS: int,
) -> np.ndarray:
    """
    Haal de vertrektijdverdeling op en maak hem cumulatief per logistiek segment.

    Args:
        settings (Settings): _description_
        numLS (int): Het aantal logistieke segmenten.

    Returns:
        np.ndarray: _description_
    """
    path = settings.get_path("KansenVertrektijd")

    probDepTimes = np.zeros((24, numLS - 1), dtype=float)
    for row in pd.read_csv(path, sep='\t').to_dict('records'):
        ls = row['logistic_segment']
        hour = row['hour']
        probDepTimes[hour, ls] = row['share']

    cumProbDepTimes = np.zeros((24, numLS - 1), dtype=float)
    for ls in range(numLS - 1):
        cumProbDepTimes[:, ls] = np.cumsum(probDepTimes[:, ls])
        cumProbDepTimes[:, ls] /= cumProbDepTimes[-1, ls]

    return cumProbDepTimes


def get_cum_prob_combustion(
    settings: Settings,
    numCombType: int, numLS: int, numVT: int
) -> Tuple[List[np.ndarray]]:
    """
    Bepaal binnen elk voertuigtype en logistiek segment de cumulatieve
    kansen van de energiebronnen (uit 'KansenVoertuigtypeZEZ').

    Args:
        settings (Settings): _description_
        numCombType (int): _description_
        numLS (int): _description_
        numVT (int): _description_

    Returns:
        tuple: Met daarin:
            - list: Per voertuigtype een numpy.ndarray met kansen per voertuigtype (rij) en
            energiebron (kolom)
            - np.ndarray: kansen per voertuigtype (rij) en energiebron (kolom)
    """
    # Vehicle/combustion shares (for ZEZ scenario)
    scenarioZEZ = pd.read_csv(settings.get_path("KansenVoertuigtypeZEZ"), sep='\t')

    # Combustion type shares per vehicle type and logistic segment
    # (for trips between UCCs and ZEZ)
    cumProbsCombConsolidated = [
        np.zeros((numLS, numCombType), dtype=float) for vt in range(numVT)]

    for i in scenarioZEZ[scenarioZEZ['consolidated'] == 1].index:
        ls = int(scenarioZEZ.at[i, 'logistic_segment'])
        vt = int(scenarioZEZ.at[i, 'vehicle_type_lwm'])
        ct = int(scenarioZEZ.at[i, 'combustion_type'])
        prob = scenarioZEZ.at[i, 'probability']
        cumProbsCombConsolidated[vt][ls, ct] += prob

    for vt in range(numVT):
        for ls in range(numLS):
            tmpCumSum = np.cumsum(cumProbsCombConsolidated[vt][ls, :])
            tmpSum = tmpCumSum[-1]
            if tmpSum > 0:
                cumProbsCombConsolidated[vt][ls, :] = tmpCumSum / tmpSum
            else:
                cumProbsCombConsolidated[vt][ls, :] = np.arange(1, numCombType + 1) / numCombType

    # Combustion type shares per vehicle type and logistic segment
    # (for trips to/from ZEZ without consolidation in UCC)
    cumProbsCombDirect = np.zeros((numVT - 2, numCombType), dtype=float)

    for i in scenarioZEZ[scenarioZEZ['consolidated'] == 0].index:
        vt = int(scenarioZEZ.at[i, 'vehicle_type_lwm'])
        ct = int(scenarioZEZ.at[i, 'combustion_type'])
        prob = scenarioZEZ.at[i, 'probability']
        cumProbsCombDirect[vt, ct] += prob

    for vt in range(numVT - 2):
        tmpCumSum = np.cumsum(cumProbsCombDirect[vt, :])
        tmpSum = tmpCumSum[-1]
        if tmpSum > 0:
            cumProbsCombDirect[vt, :] = tmpCumSum / tmpSum
        else:
            cumProbsCombDirect[vt, :] = np.arange(1, numCombType + 1) / numCombType

    return cumProbsCombConsolidated, cumProbsCombDirect


def get_shipments(path: str) -> Tuple[pd.DataFrame]:
    """
    Haal de zendingen uit module SHIP op en splits ze naar verschijningsvorm.

    Args:
        path (str): De bestandslocatie van het zendingenbestand.

    Returns:
        tuple: Met daarin:
            - pd.DataFrame: Niet-container zendingen.
            - pd.DataFrame: Container zendingen.
    """
    shipments = pd.read_csv(path, sep='\t')

    # Container shipments
    shipmentsCK = shipments.copy()[shipments['container'] == 1]
    shipmentsCK.index = np.arange(len(shipmentsCK))

    # Non-container shipments
    shipments = shipments[shipments['container'] == 0]
    shipments.index = np.arange(len(shipments))

    return shipments, shipmentsCK


def add_location_types_to_shipments(
    shipments: pd.DataFrame,
    zonesSted: np.ndarray,
    BGwithOverlapNRM: dict
) -> pd.DataFrame:
    """
    Voeg toe of de zending is geladen of gelost op een DC

    Args:
        shipments (pd.DataFrame): _description_
        zonesSted (np.ndarray): _description_
        BGwithOverlapNRM (dict): _description_

    Returns:
        pandas.DataFrame: De zendingen met extra kolommen:
            - LOGNODE_LOADING
            - LOGNODE_UNLOADING
            - URBAN
            - EXTERNAL
    """
    # Is the shipment loaded at a distribution center?
    shipments['dc_loading'] = 0
    shipments.loc[(shipments['orig_dc'] != -9999), 'dc_loading'] = 2
    shipments.loc[(shipments['orig_tt'] != -9999), 'dc_loading'] = 1

    # Is the shipment UNloaded at a distribution center?
    shipments['dc_unloading'] = 0
    shipments.loc[(shipments['dest_dc'] != -9999), 'dc_unloading'] = 2
    shipments.loc[(shipments['dest_tt'] != -9999), 'dc_unloading'] = 1

    # Is the loading or unloading point in an urbanized region?
    origNRM = np.array(shipments['orig_nrm'], dtype=int)
    destNRM = np.array(shipments['dest_nrm'], dtype=int)
    shipments['urban'] = (
        (zonesSted[origNRM - 1]) | (zonesSted[destNRM - 1])).astype(int)

    # Gaat de zending naar buiten het bereik van de NRM-zonering of niet
    shipments['external'] = 0
    for i in shipments.index:
        isExternal = (
            (shipments.at[i, 'orig_bg'] not in BGwithOverlapNRM) or
            (shipments.at[i, 'dest_bg'] not in BGwithOverlapNRM))
        if isExternal:
            shipments.at[i, 'external'] = 1

    return shipments


def assign_carrier_to_shipments(
    shipments: pd.DataFrame,
    applyZEZ: bool,
    numDC: int, numCarriersNonDC: int,
    isZEZ: np.ndarray, uccZones: np.ndarray, zezToUCC: dict
) -> pd.DataFrame:
    """
    Wijs elke zendingen toe aan een vervoerder, voeg dit toe als extra kolom
    met header 'carrier_id' aan de zendingen.

    Args:
        shipments (pd.DataFrame): _description_
        applyZEZ (bool): _description_
        numDC (int): _description_
        numCarriersNonDC (int): _description_
        isZEZ (np.ndarray): _description_
        uccZones (np.ndarray): _description_
        zezToUCC (dict): _description_

    Returns:
        pd.DataFrame: De zendingen met extra kolom CARRIER.
    """
    shipments['carrier_id'] = 0

    whereLoadDC = (shipments['orig_dc'] != -9999)
    whereUnloadDC = (shipments['dest_dc'] != -9999)
    whereBothDC = (whereLoadDC) & (whereUnloadDC)
    whereNoDC = ~(whereLoadDC) & ~(whereUnloadDC)

    shipments.loc[whereLoadDC, 'carrier_id'] = shipments['orig_dc'][whereLoadDC]
    shipments.loc[whereUnloadDC, 'carrier_id'] = shipments['dest_dc'][whereUnloadDC]
    shipments.loc[whereBothDC, 'carrier_id'] = [
        [shipments['orig_dc'][i], shipments['dest_dc'][i]][np.random.randint(0, 2)]
        for i in shipments.loc[whereBothDC, :].index]
    shipments.loc[whereNoDC, 'carrier_id'] = (
        numDC + np.random.randint(0, numCarriersNonDC, np.sum(whereNoDC)))

    # Extra carrierIDs for shipments transported from Urban Consolidation Centers
    if applyZEZ:
        whereToUCC = np.where(shipments['to_ucc'] == 1)[0]
        whereFromUCC = np.where(shipments['from_ucc'] == 1)[0]

        for i in whereToUCC:
            origNRM = shipments.at[i, 'orig_nrm']

            if isZEZ[origNRM - 1]:
                uccIndex = np.where(uccZones == zezToUCC[origNRM])[0][0]
                shipments.at[i, 'carrier_id'] = numDC + numCarriersNonDC + uccIndex

        for i in whereFromUCC:
            destNRM = shipments.at[i, 'dest_nrm']

            if isZEZ[destNRM - 1]:
                uccIndex = np.where(uccZones == zezToUCC[destNRM])[0][0]
                shipments.at[i, 'carrier_id'] = numDC + numCarriersNonDC + uccIndex

    return shipments


def split_shipments_with_capacity_exceedence(
    shipments: np.ndarray,
    carryingCapacity: np.ndarray
) -> pd.DataFrame:
    """
    Knip zendingen die te groot zijn voor het laadvermogen van hun toegewezen
    voertuigtype op in meerdere zendingen.

    Args:
        shipments (np.ndarray): _description_
        carryingCapacity (np.ndarray): _description_

    Returns:
        numpy.ndarray: De zendingen.
    """
    weight = shipments[:, 6]
    vehicleType = shipments[:, 4]
    capacityExceedence = np.array([
        weight[i] / carryingCapacity[int(vehicleType[i])]
        for i in range(len(shipments))])
    whereCapacityExceeded = np.where(capacityExceedence > 1)[0]
    nNewShipments = int(np.sum(np.ceil(
        capacityExceedence[whereCapacityExceeded])))

    newShipments = np.zeros(
        (nNewShipments, shipments.shape[1]), dtype=object)

    count = 0
    for i in whereCapacityExceeded:
        nShipments = int(np.ceil(capacityExceedence[i]))
        newWeight = weight[i] / nShipments
        newShipment = shipments[i, :].copy()
        newShipment[6] = newWeight

        for n in range(nShipments):
            newShipments[count, :] = newShipment
            count += 1

    shipments = np.append(shipments, newShipments, axis=0)
    shipments = np.delete(shipments, whereCapacityExceeded, axis=0)
    shipments[:, 0] = np.arange(len(shipments))
    shipments = shipments.astype(int)

    return shipments


def nearest_neighbor(
    tourLocs: np.ndarray,
    dc: int,
    skim: np.ndarray,
    numZones: int
) -> np.ndarray:
    '''
    Creates a tour sequence to visit all loading and unloading locations.
    First a nearest-neighbor search is applied,then a 2-opt posterior improvement phase.

    Args:
        tourLocs (numpy.ndarray): Array with loading locations of shipments
            in column 0 and unloading locations in column 1
        dc (int): Zone ID of the DC in case the carrier is located in a DC zone

    Returns:
        numpy.ndarray: The order of visiting the zones of the tour.
    '''
    # Arrays of unique loading and unloading locations to be visited
    loading = np.unique(tourLocs[:, 0])
    unloading = np.unique(tourLocs[:, 1])

    # Total number of loading and unloading locations to visit
    numLoad = len(loading)
    numUnload = len(unloading)

    # Here we will store the sequence of locations
    tourSequence = np.zeros(numLoad + numUnload, dtype=int)

    # First visited location = first listed location
    tourSequence[0] = tourLocs[0, 0]

    # This location has already been visited,
    # remove from list of remaining loading locations
    loading = loading[loading != tourLocs[0, 0]]

    # For each shipment (except the last), decide the one that is
    # visited for loading
    for currentship in range(numLoad - 1):

        # (Re)initialize array with travel times from current to
        # each remaining shipment
        timeCurrentToRemaining = np.zeros(len(loading))

        # Go over all remaining loading locations
        for remainship in range(len(loading)):

            # Fill in travel time current location to
            # each remaining loading location
            timeCurrentToRemaining[remainship] = skim[
                (tourSequence[currentship] - 1) * numZones + (loading[remainship] - 1), 0]

        # Index of the nearest unvisited loading location
        nearestShipment = np.argmin(timeCurrentToRemaining)

        # Fill in as the next location in tour sequence
        tourSequence[currentship + 1] = loading[nearestShipment]

        # Remove this shipment from list of remaining loading locations
        loading = np.delete(loading, nearestShipment)

    # Now visit the unloading locations
    for currentship in range(numUnload):

        # (Re)initialize array with travel times from current to
        # each remaining shipment
        timeCurrentToRemaining = np.zeros(len(unloading))

        # Go over all remaining unloading locations
        for remainship in range(len(unloading)):

            # Fill in travel time current location to
            # each remaining loading location
            timeCurrentToRemaining[remainship] = skim[
                (tourSequence[numLoad - 1 + currentship] - 1) * numZones +
                (unloading[remainship] - 1), 0]

        # Index of the nearest unvisited unloading location
        nearestShipment = np.argmin(timeCurrentToRemaining)

        # Fill in as the next location in tour sequence
        tourSequence[numLoad + currentship] = unloading[nearestShipment]

        # Remove this shipment from list of remaining unloading locations
        unloading = np.delete(unloading, nearestShipment)

    # If the carrier is at a DC, start the tour here
    # (not always necessarily the first loading location)
    if dc is not None:
        if tourSequence[0] != dc:
            tourSequence = np.array([dc] + list(tourSequence), dtype=int)

    # Make sure the tour does not visit the homebase in between
    # (otherwise it's not 1 tour but 2 tours)
    nStartLoc = set(np.where(tourSequence == tourSequence[0])[0][1:])
    if len(nStartLoc) > 1:
        tourSequence = [tourSequence[i] for i in range(len(tourSequence)) if i not in nStartLoc]
        tourSequence.append(tourSequence[0])
        tourSequence = np.array(tourSequence, dtype=int)
    lenTourSequence = len(tourSequence)

    # Only do 2-opt if tour has more than 3 stops
    if lenTourSequence > 3:
        tourSequence = two_opt(tourSequence, tourLocs, skim, numZones)

    return tourSequence


@njit
def two_opt(
    tourSequence: np.ndarray,
    tourLocs: np.ndarray,
    skim: np.ndarray,
    numZones: int
) -> np.ndarray:
    """
    Verwisselt telkens twee locaties in de tour van volgorde en accepteert deze omwisseling
    als het leidt tot een kortere tour en als van elk van de zendingen de laadlocatie wordt
    bezocht voordat de loslocatie wordt aangedaan.

    Args:
        tourSequence (np.ndarray): _description_
        tourLocs (np.ndarray): _description_
        skim (np.ndarray): _description_
        numZones (int): _description_

    Returns:
        np.ndarray: _description_
    """
    startLocations = tourSequence[:-1] - 1
    endLocations = tourSequence[1:] - 1
    tourDuration = np.sum(skim[startLocations * numZones + endLocations, 0]) / 3600

    numStops = len(tourSequence)

    for shiftLocA in range(1, numStops - 1):
        for shiftLocB in range(1, numStops - 1):
            if shiftLocA != shiftLocB:
                swappedTourSequence = tourSequence.copy()
                swappedTourSequence[shiftLocA] = tourSequence[shiftLocB]
                swappedTourSequence[shiftLocB] = tourSequence[shiftLocA]

                swappedStartLocations = swappedTourSequence[:-1] - 1
                swappedEndLocations = swappedTourSequence[1:] - 1
                swappedTourDuration = np.sum(
                    skim[swappedStartLocations * numZones + swappedEndLocations, 0]) / 3600

                # Only make the swap definitive if it reduces the tour duration
                if swappedTourDuration < tourDuration:

                    # Check if the loading locations are visited before the unloading locations
                    precedence = True
                    for i in range(len(tourLocs)):
                        load = tourLocs[i, 0]
                        unload = tourLocs[i, 1]
                        whereLoad = [
                            j for j in range(numStops) if swappedTourSequence[j] == load][0]
                        whereUnload = [
                            j for j in range(numStops) if swappedTourSequence[j] == unload][-1]
                        if whereLoad > whereUnload:
                            precedence = False
                            break

                    if precedence:
                        tourSequence = swappedTourSequence.copy()
                        tourDuration = swappedTourDuration

    return tourSequence


def get_tourdur(
    tourSequence: np.ndarray,
    skim: np.ndarray,
    nZones: int
) -> float:
    """
    Calculates the tour duration of the tour (so far)

    Args:
        tourSequence (np.ndarray): Array with ordered sequence of visited locations in tour.
        skim (np.ndarray): _description_
        nZones (int): _description_

    Returns:
        float: _description_
    """
    tourSequence = np.array(tourSequence)
    tourSequence = tourSequence[tourSequence != 0]
    startLocations = tourSequence[: -1] - 1
    endLocations = tourSequence[1:] - 1
    tourDuration = np.sum(skim[startLocations * nZones + endLocations, 0]) / 3600

    return tourDuration


def cap_utilization(
    veh: int,
    weights: np.ndarray,
    carryingCapacity: np.ndarray
) -> float:
    """
    Calculates the capacity utilzation of the tour (so far).
    Assumes that vehicle type chosen for 1st shipment in tour defines the carrying capacity.

    Args:
        veh (int): _description_
        weights (np.ndarray): _description_
        carryingCapacity (np.ndarray): _description_

    Returns:
        float: _description_
    """
    cap = carryingCapacity[veh]
    weight = sum(weights)
    return weight / cap


def calc_proximity(
    tourlocs: np.ndarray,
    universal: np.ndarray,
    skim: np.ndarray,
    nZones: int
) -> np.ndarray:
    """
    Reports the proximity value [km] of each shipment in the universal choice set.

    Args:
        tourlocs (np.ndarray): Array with the locations visited in the tour so far
        universal (np.ndarray): _description_
        skim (np.ndarray): _description_
        nZones (int): _description_

    Returns:
        np.ndarray: _description_
    """
    # Unique locations visited in the tour so far
    # (except for tour starting point)
    if tourlocs[0, 0] == tourlocs[0, 1]:
        tourlocs = [x for x in np.unique(tourlocs) if x != 0]
    else:
        tourlocs = [x for x in np.unique(tourlocs) if x != 0 and x != tourlocs[0, 0]]

    # Loading and unloading locations of the remaining shipments
    otherShipments = universal[:, 1:3].astype(int)
    nOtherShipments = len(otherShipments)

    # Initialization
    distancesLoading = np.zeros((len(tourlocs), nOtherShipments))
    distancesUnloading = np.zeros((len(tourlocs), nOtherShipments))

    for i in range(len(tourlocs)):
        distancesLoading[i, :] = skim[
            (tourlocs[i] - 1) * nZones + (otherShipments[:, 0] - 1), 1] / 1000
        distancesUnloading[i, :] = skim[
            (tourlocs[i] - 1) * nZones + (otherShipments[:, 1] - 1), 1] / 1000

    # Proximity measure = distance to nearest loading and unloading
    # location summed up
    distances = np.min(distancesLoading + distancesUnloading, axis=0)

    return distances


def lognode_loading(
    tour: np.ndarray,
    shipments: np.ndarray
) -> bool:
    """
    Returns a Boolean that states whether a logistical node is visited in the tour for loading.

    Args:
        tour (np.ndarray): _description_
        shipments (np.ndarray): _description_

    Returns:
        bool: _description_
    """
    if len(tour) == 1:
        return shipments[tour][0][7] == 2
    else:
        return np.any(shipments[tour][0][7] == 2)


def lognode_unloading(
    tour: np.ndarray,
    shipments: np.ndarray
) -> bool:
    """
    Returns a Boolean that states whether a logistical node is visited in the tour for unloading.

    Args:
        tour (np.ndarray): _description_
        shipments (np.ndarray): _description_

    Returns:
        bool: _description_
    """
    if len(tour) == 1:
        return shipments[tour][0][8] == 2
    else:
        return np.any(shipments[tour][0][8] == 2)


def transship(
    tour: np.ndarray,
    shipments: np.ndarray
) -> bool:
    """
    Returns a Boolean that states whether a transshipment zone is visited in the tour.

    Args:
        tour (np.ndarray): _description_
        shipments (np.ndarray): _description_

    Returns:
        bool: _description_
    """
    if len(tour) == 1:
        isTransship = (
            (shipments[tour][0][7] == 1) or
            (shipments[tour][0][8] == 1))
    else:
        isTransship = (
            np.any(shipments[tour][0][7] == 1) or
            np.any(shipments[tour][0][8] == 1))

    return isTransship


def urbanzone(
    tour: np.ndarray,
    shipments: np.ndarray
) -> bool:
    """
    Returns a Boolean that states whether an urban zone is visited in the tour.

    Args:
        tour (np.ndarray): _description_
        shipments (np.ndarray): _description_

    Returns:
        bool: _description_
    """
    if len(tour) == 1:
        return shipments[tour][0][9]
    else:
        return np.any(shipments[tour][0][9] == 1)


def max_nstr(
    tour: np.ndarray,
    shipments: np.ndarray,
    nNSTR: int
) -> int:
    """
    Returns the NSTR goods type (0-9) of which the highest weight is
    transported in the tour (so far)

    Args:
        tour (np.ndarray): Array with the IDs of all shipments in the tour.
        shipments (np.ndarray): _description_
        nNSTR (int): _description_

    Returns:
        int: _description_
    """
    nstrWeight = np.zeros(nNSTR)

    for i in range(len(tour)):
        shipNSTR = shipments[tour[i], 5]
        shipWeight = shipments[tour[i], 6]
        nstrWeight[shipNSTR] += shipWeight

    return np.argmax(nstrWeight)


def max_bggg(
    tour: np.ndarray,
    shipments: np.ndarray,
    nGG: int
) -> int:
    """
    Returns the BasGoed goods type (1-13) of which the highest weight is
    transported in the tour (so far).

    Args:
        tour (np.ndarray): Array with the IDs of all shipments in the tour.
        shipments (np.ndarray): _description_
        nGG (int): _description_

    Returns:
        int: _description_
    """
    bgggWeight = np.zeros(nGG)

    for i in range(len(tour)):
        shipBGGG = shipments[tour[i], 14]
        shipWeight = shipments[tour[i], 6]
        bgggWeight[shipBGGG - 1] += shipWeight

    return np.argmax(bgggWeight) + 1


def sum_weight(
    tour: np.ndarray,
    shipments: np.ndarray
) -> float:
    """
    Returns the total weight of all goods that are transported in the tour

    Args:
        tour (np.ndarray): Array with the IDs of all shipments in the tour.
        shipments (np.ndarray): _description_

    Returns:
        float: _description_
    """
    sumWeight = 0

    for i in range(len(tour)):
        # The weights of the shipments in the tour are summed up
        # and converted to tonnes
        shipWeight = shipments[tour[i], 6]
        sumWeight += (shipWeight / 1000)

    return sumWeight


def endtour_first(
    tourDuration: float,
    capUt: float,
    tour: np.ndarray,
    coeffsTour: Dict[str, Dict[str, np.ndarray]],
    shipments: np.ndarray,
    nNSTR: int
) -> bool:
    """
    Returns True if we decide to end the tour, False if we decide to add another shipment.

    Args:
        tourDuration (float): _description_
        capUt (float): _description_
        tour (np.ndarray): _description_
        coeffsTour (dict): _description_
        shipments (np.ndarray): _description_
        nNSTR (int): _description_

    Returns:
        bool: _description_
    """
    # Calculate explanatory variables
    vt = shipments[tour[0], 4]
    nstr = max_nstr(tour, shipments, nNSTR)

    # Calculate utility
    coeffsEndTourFirst = coeffsTour['first']
    etUtility = (
        coeffsEndTourFirst['constant'] +
        coeffsEndTourFirst['tour_duration'] * tourDuration**0.5 +
        coeffsEndTourFirst['capacity_utilization'] * capUt**2 +
        coeffsEndTourFirst['transshipment'] * transship(tour, shipments) +
        coeffsEndTourFirst['dc_loading'] * lognode_loading(tour, shipments) +
        coeffsEndTourFirst['dc_unloading'] * lognode_unloading(tour, shipments) +
        coeffsEndTourFirst['urban'] * urbanzone(tour, shipments) +
        coeffsEndTourFirst['const_vehicle_type_lwm'][vt] +
        coeffsEndTourFirst['const_nstr'][nstr])

    # Calculate probability
    etProbability = np.exp(etUtility) / (np.exp(etUtility) + np.exp(0))

    # Monte Carlo to simulate choice based on probability
    return np.random.rand() < etProbability


def endtour_later(
    tour: np.ndarray,
    tourLocs: np.ndarray,
    tourSequence: np.ndarray,
    universal: np.ndarray,
    skim: np.ndarray,
    nZones: int,
    carryingCapacity: np.ndarray,
    coeffsTour: Dict[str, Dict[str, np.ndarray]],
    shipments: np.ndarray,
    nNSTR: int
) -> bool:
    """
    Returns True if we decide to end the tour, False if we decide to add another shipment.

    Args:
        tour (np.ndarray): _description_
        tourLocs (np.ndarray): _description_
        tourSequence (np.ndarray): _description_
        universal (np.ndarray): _description_
        skim (np.ndarray): _description_
        nZones (int): _description_
        carryingCapacity (np.ndarray): _description_
        coeffsTour (dict): _description_
        shipments (np.ndarray): _description_
        nNSTR (int): _description_

    Returns:
        bool: _description_
    """
    # Calculate explanatory variables
    tourDuration = get_tourdur(tourSequence, skim, nZones)
    proximity = np.min(calc_proximity(tourLocs, universal, skim, nZones))
    capUt = cap_utilization(shipments[tour[0], 4], shipments[tour, 6], carryingCapacity)
    numberOfStops = len(np.unique(tourLocs))
    vt = shipments[tour[0], 4]
    nstr = max_nstr(tour, shipments, nNSTR)

    # Calculate utility
    coeffsEndTourLater = coeffsTour['later']
    etUtility = (
        coeffsEndTourLater['constant'] +
        coeffsEndTourLater['tour_duration'] * tourDuration +
        coeffsEndTourLater['capacity_utilization'] * capUt +
        coeffsEndTourLater['proximity'] * proximity +
        coeffsEndTourLater['transshipment'] * transship(tour, shipments) +
        coeffsEndTourLater['number_of_stops'] * np.log(numberOfStops) +
        coeffsEndTourLater['dc_loading'] * lognode_loading(tour, shipments) +
        coeffsEndTourLater['dc_unloading'] * lognode_unloading(tour, shipments) +
        coeffsEndTourLater['urban'] * urbanzone(tour, shipments) +
        coeffsEndTourLater['const_vehicle_type_lwm'][vt] +
        coeffsEndTourLater['const_nstr'][nstr])

    # Calculate probability
    etProbability = np.exp(etUtility) / (np.exp(etUtility) + np.exp(0))

    # Monte Carlo to simulate choice based on probability
    return np.random.rand() < etProbability


def selectshipment(
    tour: np.ndarray,
    tourLocs: np.ndarray,
    universal: np.ndarray,
    skim: np.ndarray,
    nZones: int,
    carryingCapacity: np.ndarray,
    shipments: np.ndarray,
    allowanceFac: float = 1.1
) -> np.ndarray:
    """
    Returns the shipment which is to be added to the tour. If -2 is returned, there were no
    feasible shipments to add to the tour.

    Args:
        tour (np.ndarray): _description_
        tourLocs (np.ndarray): _description_
        universal (np.ndarray): _description_
        skim (np.ndarray): _description_
        nZones (int): _description_
        carryingCapacity (np.ndarray): _description_
        shipments (np.ndarray): _description_
        allowanceFac (float, optional): _description_

    Returns:
        np.ndarray: _description_
    """

    # Some tour characteristics as input for the constraint checks
    if type(tour) == int:
        bggg = shipments[tour][14]
        vt = shipments[tour][4]
        isExternal = shipments[tour][13]
        if isExternal:
            origBG = shipments[tour][11]
            destBG = shipments[tour][12]
    else:
        bggg = shipments[tour[0]][14]
        vt = shipments[tour[0]][4]
        isExternal = shipments[tour[0]][13]
        if isExternal:
            origBG = shipments[tour[0]][11]
            destBG = shipments[tour[0]][12]

    # Check capacity utilization
    tourWeight = np.sum(shipments[tour, 6])
    capUt = (tourWeight + universal[:, 6]) / carryingCapacity[vt]

    # Check proximity of other shipments to the tour
    prox = calc_proximity(tourLocs, universal, skim, nZones)

    # Which shipments belong to the same BasGoed-goods type
    # and have the same vehicle type as the tour
    sameBGGG = np.array(universal[:, 14] == bggg)
    sameVT = np.array(universal[:, 4] == vt)

    # Initialize feasible choice set, those shipments that
    # comply with constraints
    if isExternal == 1:
        selectShipConstraints = (
            (capUt < allowanceFac) & (sameBGGG) &
            (universal[:, 11] == origBG) & (universal[:, 12] == destBG))
    else:
        selectShipConstraints = (capUt < allowanceFac) & (sameBGGG) & (universal[:, 13] == 0)

    feasibleChoiceSet = universal[selectShipConstraints]

    # If there are no feasible shipments,
    # the return -2 statement is used to end the tour
    if len(feasibleChoiceSet) == 0:
        return -2

    else:
        sameVT = sameVT[selectShipConstraints]
        prox = prox[selectShipConstraints]

        # Make shipments of same vehicle type more likely to be chosen
        # (through lower proximity value)
        prox[sameVT] -= 0.001
        prox[sameVT] /= 2

        # Select the shipment with minimum distance to the tour
        # (proximity)
        ssChoice = np.argmin(prox)
        chosenShipment = feasibleChoiceSet[ssChoice]

        return chosenShipment


def form_tours(
    carrierMarkers: np.ndarray,
    shipments: np.ndarray,
    skim: np.ndarray,
    numZones: int,
    maxNumShips: int,
    carryingCapacity: np.ndarray,
    dcZones: np.ndarray,
    numShipmentsPerCarrier: np.ndarray, numCarriers: int,
    coeffsTour: Dict[str, Dict[str, np.ndarray]],
    numNSTR: int,
    carriers: np.ndarray
) -> Tuple[List[List[int]]]:
    """
    Run the tour formation procedure for a set of carriers with a set of shipments.

    Args:
        carrierMarkers (np.ndarray): _description_
        shipments (np.ndarray): _description_
        skim (np.ndarray): _description_
        numZones (int): _description_
        maxNumShips (int): _description_
        carryingCapacity (np.ndarray): _description_
        dcZones (np.ndarray): _description_
        numShipmentsPerCarrier (np.ndarray): _description_
        numCarriers (int): _description_
        coeffsTour (dict): _description_
        numNSTR (int): _description_
        carriers (np.ndarray): _description_

    Returns:
        tuple: Met daarin:
            - list: Voor elke tour de zending-ID's.
            - list: Voor elke tour de bezochte zones.
            - list: Voor elke carrier het aantal tours.
            - list: Voor elke tour of deze de scope van de NRM-zonering verlaat of niet.
    """
    seed = carriers[1]
    carriers = carriers[0]

    np.random.rand(seed)

    tours = [[] for car in range(numCarriers)]
    tourSequences = [[] for car in range(numCarriers)]
    nTours = np.zeros(numCarriers, dtype=int)
    tourIsExternal = [[] for car in range(numCarriers)]

    for car in carriers:
        print(f'\tForming tours for carrier {car+1} of {numCarriers}...', end='\r')

        tourCount = 0

        # Universal choice set = all non-allocated shipments per carrier
        universalChoiceSet = shipments[carrierMarkers[car]:carrierMarkers[car + 1], :].copy()

        if car < len(dcZones):
            dc = dcZones[car]

            # Sort by shipment distance, this will help in constructing
            # more efficient/realistic tours
            shipmentDistances = skim[
                (universalChoiceSet[:, 1] - 1) * numZones + (universalChoiceSet[:, 1] - 1)]
            universalChoiceSet = np.c_[universalChoiceSet, shipmentDistances]
            universalChoiceSet = universalChoiceSet[universalChoiceSet[:, -1].argsort()]
            universalChoiceSet = universalChoiceSet[:, :-1]

        else:
            dc = None

        while len(universalChoiceSet) != 0:
            shipmentCount = 0

            # Loading and unloading locations of the shipments in the tour
            tourLocations = np.zeros(
                (min(numShipmentsPerCarrier[car], maxNumShips) * 2, 2), dtype=int)
            tourLocations[shipmentCount, 0] = universalChoiceSet[0, 1].copy()
            tourLocations[shipmentCount, 1] = universalChoiceSet[0, 2].copy()

            # First shipment in the tour is the first listed one in
            # universal choice set
            tour = np.array([universalChoiceSet[0, 0].copy()], dtype=int)

            # Tour with only 1 shipment: go from loading to unloading
            tourSequence = np.zeros(min(numShipmentsPerCarrier[car], maxNumShips) * 2, dtype=int)
            tourSequence[shipmentCount] = universalChoiceSet[0, 1].copy()
            tourSequence[shipmentCount + 1] = universalChoiceSet[0, 2].copy()

            # Does the shipment leave the scope of the NRM-zoning system
            externalShipment = universalChoiceSet[0, 13]

            # Remove shipment from universal choice set
            universalChoiceSet = np.delete(universalChoiceSet, 0, 0)

            # If no shipments left for carrier, break out of while loop,
            # go to next carrier
            if len(universalChoiceSet) == 0:
                tourCount += 1
                nTours[car] = tourCount
                tours[car].append(tour.copy())
                tourSequences[car].append([x for x in tourSequence.copy() if x != 0])
                tourIsExternal[car].append(externalShipment)
                break

            # Shipments that goes to BG-zone outside the edges
            # of the NRM-zone system
            tourIsExternal[car].append(externalShipment)

            # Input for ET constraints and choice check
            tourLocs = tourLocations[0:(shipmentCount + 1)]
            tourDuration = get_tourdur(tourSequence, skim, numZones)
            prox = calc_proximity(tourLocs, universalChoiceSet, skim, numZones)
            capUt = cap_utilization(shipments[tour[0], 4], shipments[tour, 6], carryingCapacity)

            etConstraintCheck = (
                (shipmentCount < maxNumShips) and (tourDuration < 9) and
                (capUt < 1) and (np.min(prox) < 100))

            if not etConstraintCheck:
                tourCount += 1
                tours[car].append(tour.copy())
                tourSequences[car].append([x for x in tourSequence.copy() if x != 0])
                continue

            # End Tour? --> Yes or No
            etChoice = endtour_first(
                tourDuration, capUt, tour, coeffsTour, shipments, numNSTR)

            while (not etChoice) and len(universalChoiceSet) != 0:
                tourLocs = tourLocations[0:(shipmentCount + 1)]
                tourDuration = get_tourdur(tourSequence, skim, numZones)
                prox = calc_proximity(tourLocs, universalChoiceSet, skim, numZones)
                capUt = cap_utilization(
                    shipments[tour[0], 4], shipments[tour, 6], carryingCapacity)

                etConstraintCheck = (
                    (shipmentCount < maxNumShips) and (tourDuration < 9) and
                    (capUt < 1) and (np.min(prox) < 100))

                if not etConstraintCheck:
                    tourCount += 1
                    tours[car].append(tour.copy())
                    tourSequences[car].append([x for x in tourSequence.copy() if x != 0])
                    break

                chosenShipment = selectshipment(
                    tour, tourLocs,
                    universalChoiceSet, skim,
                    numZones, carryingCapacity, shipments)

                # If no feasible shipments
                if np.any(chosenShipment == -2):
                    tourCount += 1
                    tours[car].append(tour.copy())
                    tourSequences[car].append([x for x in tourSequence.copy() if x != 0])
                    break

                else:
                    shipmentCount += 1

                    tour = np.append(tour, int(chosenShipment[0]))
                    tourLocations[shipmentCount, 0] = int(chosenShipment[1])
                    tourLocations[shipmentCount, 1] = int(chosenShipment[2])
                    tourSequence = nearest_neighbor(
                        tourLocations[0:(shipmentCount + 1)], dc, skim, numZones)

                    shipmentToDelete = np.where(
                        universalChoiceSet[:, 0] == chosenShipment[0])[0][0]
                    universalChoiceSet = np.delete(universalChoiceSet, shipmentToDelete, 0)

                    if len(universalChoiceSet) != 0:
                        tourLocs = tourLocations[0:(shipmentCount + 1)]
                        etChoice = endtour_later(
                            tour, tourLocs, tourSequence, universalChoiceSet, skim, numZones,
                            carryingCapacity, coeffsTour, shipments, numNSTR)

                    else:
                        tourCount += 1
                        nTours[car] = tourCount
                        tours[car].append(tour.copy())
                        tourSequences[car].append([x for x in tourSequence.copy() if x != 0])
                        break

            else:
                tourCount += 1
                tours[car].append(tour.copy())
                tourSequences[car].append([x for x in tourSequence.copy() if x != 0])

    return (tours, tourSequences, nTours, tourIsExternal)


def add_empty_trips(
    tourSequences: list, tourIsExternal: list,
    numCarriers: int, numTours: list,
    skimDistance: np.ndarray,
    maxEmptyTripDist: int
) -> Tuple[List[List[int]]]:
    """
    Voeg lege ritten toe aan het einde van tours die niet uit zichzelf al terugkomen op de
    startlocatie van de tour.

    Args:
        tourSequences (list): _description_
        tourIsExternal (list): _description_
        numCarriers (int): _description_
        numTours (list): _description_
        skimDistance (np.ndarray): _description_
        maxEmptyTripDist (int): _description_

    Returns:
        tuple: Met daarin:
            - list: Geupdatete versie van tourSequences.
            - list: Per carrier en tour een boolean die aangeeft of een lege rit is toegevoegd.
    """
    numZones = int(len(skimDistance)**0.5)

    emptyTripAdded = [[None for tour in range(numTours[car])] for car in range(numCarriers)]

    for car in range(numCarriers):
        for tour in range(numTours[car]):
            tourStartZone = tourSequences[car][tour][0]
            tourEndZone = tourSequences[car][tour][-1]
            emptyTripDist = skimDistance[
                (tourEndZone - 1) * numZones + (tourStartZone - 1)] / 1000

            # Voeg alleen een lege rit terug toe als de tour niet automatisch al op de
            # startlocatie terugkomt, niet eindigt in een BG-zone buiten de scope van het NRM,
            # en als de lege rit niet langer dan 120km zou zijn
            emptyTripAdded[car][tour] = (
                (tourSequences[car][tour][0] != tourSequences[car][tour][-1]) and
                (tourIsExternal[car][tour] == 0) and
                (emptyTripDist < maxEmptyTripDist))

            if emptyTripAdded[car][tour]:
                tourSequences[car][tour].append(tourSequences[car][tour][0])

    return tourSequences, emptyTripAdded


def get_extra_tour_attr(
    tours: list, tourSequences: list,
    shipments: np.ndarray,
    numCarriers: int, numTours: list
) -> tuple:
    """
    Bepaal extra attributen van de tours.

    Args:
        tours (list): _description_
        tourSequences (list): _description_
        shipments (np.ndarray): _description_
        numCarriers (int): _description_
        numTours (list): _description_

    Returns:
        tuple: Met daarin:
            - list: Aantal zendingen per tour per carrier.
            - list: Aantal trips per tour per carrier.
            - int: Totaal aantal ritten
            - list: Het gewicht dat in elke rit in het voertuig aanwezig is per tour per carrier.
    """
    # Number of shipments and trips of each tour
    numShipmentsPerTour = [
        [len(tours[car][tour]) for tour in range(numTours[car])]
        for car in range(numCarriers)]
    numTripsPerTour = [
        [len(tourSequences[car][tour]) - 1 for tour in range(numTours[car])]
        for car in range(numCarriers)]
    numTripsTotal = np.sum([np.sum(x) for x in numTripsPerTour])

    # Weight per trip
    tripWeights = [
        [np.zeros(numTripsPerTour[car][tour], dtype=int) for tour in range(numTours[car])]
        for car in range(numCarriers)]

    for car in range(numCarriers):
        for tour in range(numTours[car]):
            origs = [tourSequences[car][tour][trip] for trip in range(numTripsPerTour[car][tour])]
            shipmentLoaded = [False for ship in range(numShipmentsPerTour[car][tour])]
            shipmentUnloaded = [False for ship in range(numShipmentsPerTour[car][tour])]

            for trip in range(numTripsPerTour[car][tour]):
                # Startzone of trip
                orig = origs[trip]

                # If it's the first trip of the tour, initialize a counter to add and subtract
                # weight from
                if trip == 0:
                    tmpTripWeight = 0

                for i in range(len(tours[car][tour])):
                    ship = tours[car][tour][i]

                    # If loading location of the shipment was the startpoint of the trip
                    # and shipment has not been loaded/unloaded yet
                    if (
                        shipments[ship][1] == orig and
                        not (shipmentLoaded[i] or shipmentUnloaded[i])
                    ):
                        # Add weight of shipment to counter
                        tmpTripWeight += shipments[ship][6]
                        shipmentLoaded[i] = True

                    # If unloading location of the shipment was the startpoint of the trip
                    # and shipment has not been unloaded yet
                    if (
                        shipments[ship][2] == orig and
                        shipmentLoaded[i] and not shipmentUnloaded[i]
                    ):
                        # Remove weight of shipment from counter
                        # (if it's not an intrazonal shipment)
                        if not (shipments[ship][1] == orig and shipments[ship][2] == orig):
                            tmpTripWeight -= shipments[ship][6]
                            shipmentUnloaded[i] = True

                tripWeights[car][tour][trip] = tmpTripWeight

                # Remove intrazonal shipment weight from the counter after updating 'tripWeights'
                for i in range(len(tours[car][tour])):
                    ship = tours[car][tour][i]
                    if (
                        shipments[ship][1] == orig and shipments[ship][2] == orig and
                        shipmentLoaded[i] and not shipmentUnloaded[i]
                    ):
                        tmpTripWeight -= shipments[ship][6]
                        shipmentUnloaded[i] = True

    return numShipmentsPerTour, numTripsPerTour, numTripsTotal, tripWeights


def draw_dep_times(
    cumProbDepTimes: np.ndarray,
    tours: list, shipments: np.ndarray,
    numCarriers: int, numTours: list
) -> List[int]:
    """
    Trek voor elke tour een vertrektijdstip.

    Args:
        cumProbDepTimes (np.ndarray): _description_
        shipments (np.ndarray): _description_
        numCarriers (int): _description_
        numTours (list): _description_

    Returns:
        list: Per tour het getrokken vertrektijdstip in uren na middernacht.
    """
    depTimeTour = [[None for tour in range(numTours[car])] for car in range(numCarriers)]

    for car in range(numCarriers):
        for tour in range(numTours[car]):
            ls = shipments[tours[car][tour][0], 10]
            depTimeTour[car][tour] = np.where(cumProbDepTimes[:, ls] > np.random.rand())[0][0]

    return depTimeTour


def draw_combustion_types(
    settings: Settings,
    cumProbsCombConsolidated: list, cumProbsCombDirect: np.ndarray,
    shipments: np.ndarray, shipmentsCK: pd.DataFrame,
    tours: list, tourSequences: list,
    numTours: list, numCarriers: int, numDC: int, numCarriersNonDC: int,
    numZones: int
) -> Tuple[List[List[int]]]:
    """
    Trek een energiebron voor tours van/naar/in de ZEZ, de rest blijft benzine/diesel.

    Args:
        settings (Settings): _description_
        cumProbsCombConsolidated (list): _description_
        cumProbsCombDirect (np.ndarray): _description_
        shipments (np.ndarray): _description_
        tours (list): _description_
        tourSequences (list): _description_
        numTours (list): _description_
        numCarriers (int): _description_
        numDC (int): _description_
        numCarriersNonDC (int): _description_
        numZones (int): _description_

    Returns:
        tuple: Met daarin:
            - list: Per niet-container tour de getrokken energiebron.
            - list: Per container zending de getrokken energiebron.
    """
    combTypeTour = [[0 for tour in range(numTours[car])] for car in range(numCarriers)]

    zonesZEZ = pd.read_csv(settings.get_path("ZonesZEZ"), sep='\t')
    isZEZ = np.zeros(numZones, dtype=int)
    isZEZ[zonesZEZ['zone_nrm'].values - 1] = 1

    for car in range(numCarriers):
        for tour in range(numTours[car]):
            ls = shipments[tours[car][tour][0], 10]
            vt = shipments[tours[car][tour][0],  4]

            # Combustion type for tours from/to UCCs
            if car >= numDC + numCarriersNonDC:
                combTypeTour[car][tour] = np.where(
                    cumProbsCombConsolidated[vt][ls, :] > np.random.rand())[0][0]

            else:
                inZEZ = [isZEZ[x - 1] for x in np.unique(tourSequences[car][tour]) if x != 0]

                # Combustion type for tours within or entering/leaving ZEZ (but not from/to UCCs)
                if np.any(inZEZ):
                    combTypeTour[car][tour] = np.where(
                        cumProbsCombDirect[vt, :] > np.random.rand())[0][0]

                # Fuel for all other tours that do not go to the ZEZ
                else:
                    combTypeTour[car][tour] = 0

    combTypeShipmentsCK = [0 for i in shipmentsCK.index]

    for i in shipmentsCK.index:
        vt = shipmentsCK.at[i, 'vehicle_type_lwm']
        origNRM = shipmentsCK.at[i, 'orig_nrm']
        destNRM = shipmentsCK.at[i, 'dest_nrm']

        if isZEZ[origNRM - 1] or isZEZ[destNRM - 1]:
            combTypeShipmentsCK[i] = np.where(cumProbsCombDirect[vt, :] > np.random.rand())[0][0]

    return combTypeTour, combTypeShipmentsCK


def format_output_tours(
    tours: list, tourSequences: list, tourIsExternal: list,
    shipments: pd.DataFrame, shipmentsCK: pd.DataFrame,
    zones: pd.DataFrame, dcZones: np.ndarray, skim: np.ndarray,
    dictNRMtoBG: dict, BGwithOverlapNRM: np.ndarray,
    cumProbDepTimes: np.ndarray,
    numCarriers: int, numDC: int, numTours: list,
    numShipmentsPerTour: list, numTripsPerTour: list, numTripsTotal: int,
    emptyTripAdded: list, tripWeights: list, depTimeTour: list,
    combTypeTour: list, combTypeShipmentsCK: list,
    numNSTR: int, numGG: int
) -> pd.DataFrame:
    """
    Herstructuur alle ritgegevens naar een DataFrame. Maak daarbij ook gelijk ritten aan voor
    de CK-zendingen.

    Args:
        tours (list): _description_
        tourSequences (list): _description_
        tourIsExternal (list): _description_
        shipments (pd.DataFrame): _description_
        shipmentsCK (pd.DataFrame): _description_
        zones (pd.DataFrame): _description_
        dcZones (np.ndarray): _description_
        skim (np.ndarray): _description_
        dictNRMtoBG (dict): _description_
        BGwithOverlapNRM (np.ndarray): _description_
        cumProbDepTimes (np.ndarray): _description_
        numCarriers (int): _description_
        numDC (int): _description_
        numTours (list): _description_
        numShipmentsPerTour (list): _description_
        numTripsPerTour (list): _description_
        numTripsTotal (int): _description_
        emptyTripAdded (list): _description_
        tripWeights (list): _description_
        depTimeTour (list): _description_
        combTypeTour (list): _description_
        combTypeShipmentsCK (list): _description_
        numNSTR (int): _description_
        numGG (int): _description_

    Returns:
        pd.DataFrame: De ritinformatie om weg te schrijven naar een tekstbestand.
    """
    zonesVAM = np.array(zones['zone_vam'], dtype=int)
    zonesX = np.array(zones['x_rd__meter'])
    zonesY = np.array(zones['y_rd__meter'])
    numZones = len(zones)

    columns = [
        'carrier_id', 'tour_id', 'trip_id',
        'orig_nrm', 'dest_nrm', 'orig_bg', 'dest_bg', 'orig_vam', 'dest_vam',
        'x_rd_orig__meter', 'x_rd_dest__meter', 'y_rd_orig__meter', 'y_rd_dest__meter',
        'vehicle_type_lwm', 'nstr',  'commodity', 'logistic_segment',
        'n_shipments',
        'dc_id',
        'tour_weight__ton', 'trip_weight__ton',
        'tour_deptime__hour', 'trip_deptime__hour', 'trip_arrtime__hour',
        'combustion_type', 'external_zone_bg', 'container']
    dataTypes = [
        str, int,  int,
        int, int, int, int, int, int,
        float, float,
        float, float,
        int, int, int, int,
        int,
        int,
        float, float,
        float, float, float,
        int, int, int]

    outputTours = np.zeros((numTripsTotal, len(columns)), dtype=object)

    tripcount = 0
    for car in range(numCarriers):
        for tour in range(numTours[car]):
            for trip in range(numTripsPerTour[car][tour]):
                outputTours[tripcount][0] = car   # carrierID
                outputTours[tripcount][1] = tour  # tourID
                outputTours[tripcount][2] = trip  # tripID

                # NRM-zones
                origNRM = tourSequences[car][tour][trip]
                destNRM = tourSequences[car][tour][trip + 1]
                outputTours[tripcount][3] = origNRM
                outputTours[tripcount][4] = destNRM

                # BG-zones
                if tourIsExternal[car][tour] == 0:
                    origBG = dictNRMtoBG[origNRM]
                    destBG = dictNRMtoBG[destNRM]
                    outputTours[tripcount][5] = origBG
                    outputTours[tripcount][6] = destBG

                # LMSVAM-zones
                outputTours[tripcount][7] = zonesVAM[origNRM - 1]
                outputTours[tripcount][8] = zonesVAM[destNRM - 1]

                # X/Y coordinates of origin and destination of trip
                orig = tourSequences[car][tour][trip] - 1
                dest = tourSequences[car][tour][trip + 1] - 1
                outputTours[tripcount][9] = zonesX[orig]
                outputTours[tripcount][10] = zonesX[dest]
                outputTours[tripcount][11] = zonesY[orig]
                outputTours[tripcount][12] = zonesY[dest]

                # Vehicle type
                outputTours[tripcount][13] = shipments[tours[car][tour][0], 4]

                # Dominant NSTR and BasGoed goods type (by weight)
                outputTours[tripcount][14] = max_nstr(tours[car][tour], shipments, numNSTR)
                outputTours[tripcount][15] = max_bggg(tours[car][tour], shipments, numGG)

                # Logistic segment of tour
                outputTours[tripcount][16] = shipments[tours[car][tour][0], 10]

                # Number of shipments transported in tour
                outputTours[tripcount][17] = numShipmentsPerTour[car][tour]

                # ID for DC zones
                outputTours[tripcount][18] = (
                    dcZones[outputTours[tripcount][0]] if car < numDC else -9999)

                # Sum of the weight carried in the tour
                outputTours[tripcount][19] = sum_weight(tours[car][tour], shipments)

                # Weight carried in the trip
                outputTours[tripcount][20] = tripWeights[car][tour][trip] / 1000

                # Departure time of tour
                outputTours[tripcount][21] = depTimeTour[car][tour]

                # Combustion type of the vehicle
                outputTours[tripcount][24] = combTypeTour[car][tour]

                # If BG-zone outside of scope of NRM-zone system
                if tourIsExternal[car][tour] == 1:
                    origBG = shipments[tours[car][tour][0], 11]
                    destBG = shipments[tours[car][tour][0], 12]
                    if origBG not in BGwithOverlapNRM:
                        outputTours[tripcount][25] = origBG
                    if destBG not in BGwithOverlapNRM:
                        outputTours[tripcount][25] = destBG
                    outputTours[tripcount][5] = origBG
                    outputTours[tripcount][6] = destBG
                else:
                    outputTours[tripcount][25] = -9999

                # Not containerized
                outputTours[tripcount][26] = 0

                # Column 18 and 19: Departure and arrival time of each trip
                if trip == 0:
                    depTime = depTimeTour[car][tour] + np.random.rand()
                    origNRM = outputTours[tripcount][3]
                    destNRM = outputTours[tripcount][4]
                    travTime = (
                        skim[(origNRM - 1) * numZones + (destNRM - 1)][0] / 3600)
                    outputTours[tripcount][22] = depTime
                    outputTours[tripcount][23] = depTime + travTime
                else:
                    dwellTime = 0.5 * np.random.rand()
                    depTime = outputTours[tripcount - 1][21] + dwellTime
                    origNRM = outputTours[tripcount][3]
                    destNRM = outputTours[tripcount][4]
                    travTime = (
                        skim[(origNRM - 1) * numZones + (destNRM - 1)][0] / 3600)
                    outputTours[tripcount][22] = depTime
                    outputTours[tripcount][23] = depTime + travTime

                # BG-GG of empty trips is -1
                if tripWeights[car][tour][trip] == 0:
                    outputTours[tripcount][15] = -1

                tripcount += 1

            # BG-GG of empty trips is -1
            if emptyTripAdded[car][tour]:
                outputTours[tripcount - 1][15] = -1

    # Containerized trips
    numTripsCK = len(shipmentsCK)
    outputToursCK = np.zeros((numTripsCK, len(columns)), dtype=object)
    origsNRM = np.array(shipmentsCK['orig_nrm'])
    destsNRM = np.array(shipmentsCK['dest_nrm'])
    origsBG = np.array(shipmentsCK['orig_bg'])
    destsBG = np.array(shipmentsCK['dest_bg'])
    for i in range(numTripsCK):
        outputToursCK[i, 0] = -9999
        outputToursCK[i, 1] = i
        outputToursCK[i, 2] = 0

        origNRM = int(origsNRM[i])
        destNRM = int(destsNRM[i])
        origBG = int(origsBG[i])
        destBG = int(destsBG[i])
        origVAM = zonesVAM[origNRM - 1]
        destVAM = zonesVAM[destNRM - 1]

        nstr = shipmentsCK.at[i, 'nstr']
        gg = shipmentsCK.at[i, 'commodity']
        ls = shipmentsCK.at[i, 'logistic_segment']

        outputToursCK[i, 3] = origNRM
        outputToursCK[i, 4] = destNRM
        outputToursCK[i, 5] = origBG
        outputToursCK[i, 6] = destBG
        outputToursCK[i, 7] = origVAM
        outputToursCK[i, 8] = destVAM
        outputToursCK[i, 9] = zonesX[origNRM - 1]
        outputToursCK[i, 10] = zonesY[origNRM - 1]
        outputToursCK[i, 11] = zonesX[destNRM - 1]
        outputToursCK[i, 12] = zonesY[destNRM - 1]
        outputToursCK[i, 13] = shipmentsCK.at[i, 'vehicle_type_lwm']
        outputToursCK[i, 14] = nstr
        outputToursCK[i, 15] = gg
        outputToursCK[i, 16] = ls
        outputToursCK[i, 17] = 1
        outputToursCK[i, 18] = -9999
        outputToursCK[i, 19] = shipmentsCK.at[i, 'weight__ton']
        outputToursCK[i, 20] = shipmentsCK.at[i, 'weight__ton']

        depTime = np.where(cumProbDepTimes[:, ls] > np.random.rand())[0][0]
        travTime = skim[(origNRM - 1) * numZones + (destNRM - 1), 0] / 3600

        outputToursCK[i, 21] = depTime
        outputToursCK[i, 22] = depTime
        outputToursCK[i, 23] = depTime + travTime

        # Combustion type
        outputToursCK[i, 24] = combTypeShipmentsCK[i]

        # If BG-zone outside of scope of NRM-zone system
        if origBG not in BGwithOverlapNRM:
            outputToursCK[i, 25] = origBG
        elif destBG not in BGwithOverlapNRM:
            outputToursCK[i, 25] = destBG
        else:
            outputToursCK[i, 25] = -9999

        # Containerized
        outputToursCK[i, 26] = 1

    outputTours = np.append(outputTours, outputToursCK, axis=0)

    # Create DataFrame object for easy formatting and exporting to csv
    outputTours = pd.DataFrame(outputTours, columns=columns)
    for i in range(len(outputTours.columns)):
        outputTours.iloc[:, i] = (
            outputTours.iloc[:, i].astype(dataTypes[i]))

    for col in [
        'tour_weight__ton', 'trip_weight__ton',
        'tour_deptime__hour', 'trip_deptime__hour', 'trip_arrtime__hour'
    ]:
        outputTours[col] = outputTours[col].round(5)

    return outputTours


def enrich_shipments(
    shipments: pd.DataFrame, shipmentsCK: pd.DataFrame,
    shipmentDict: dict,
    BGwithOverlapNRM: dict,
    numCarriers: int, numTours: list, tours: list
) -> pd.DataFrame:
    """
    Voeg extra kenmerken toe aan de zendingen, zoals de tour-ID.

    Args:
        shipments (pd.DataFrame): _description_
        shipmentsCK (pd.DataFrame): _description_
        shipmentDict (dict): _description_
        BGwithOverlapNRM (dict): _description_
        numCarriers (int): _description_
        numTours (list): _description_
        tours (list): _description_

    Returns:
        pd.DataFrame: De zendingen met extra kenmerken.
    """
    shipmentTourID = {}
    for car in range(numCarriers):
        for tour in range(numTours[car]):
            for ship in range(len(tours[car][tour])):
                shipmentTourID[shipmentDict[tours[car][tour][ship]]] = (
                    f'{car}_{tour}')

    # Place shipments in DataFrame with the right headers
    shipments = pd.DataFrame(shipments)
    shipments.columns = [
        'shipment_id', 'orig_nrm', 'dest_nrm', 'carrier_id',
        'vehicle_type_lwm', 'nstr', 'weight__ton', 'dc_loading', 'dc_unloading',
        'urban', 'logistic_segment', 'orig_bg', 'dest_bg',
        'external', 'commodity', 'container']

    shipments['carrier_id'] -= 1
    shipments['weight__ton'] /= 1000
    shipments['shipment_id'] = [shipmentDict[x] for x in shipments['shipment_id']]
    shipments['tour_id'] = [shipmentTourID[x] for x in shipments['shipment_id']]
    numShipsMS = len(shipments)

    # Extra columns also for containerized shipments
    shipmentsCK['carrier_id'] = -9999
    shipmentsCK['dc_loading'] = -9999
    shipmentsCK['dc_unloading'] = -9999
    shipmentsCK['urban'] = -9999
    shipmentsCK['shipment_id'] = numShipsMS + np.arange(len(shipmentsCK))
    shipmentsCK['tour_id'] = [f'-9999_{i}' for i in range(len(shipmentsCK))]
    shipmentsCK['external'] = 0
    for i in shipmentsCK.index:
        origBG = shipments.at[i, 'orig_bg']

        if origBG not in BGwithOverlapNRM:
            shipmentsCK.at[i, 'external'] = 1
        else:
            destBG = shipments.at[i, 'dest_bg']
            if destBG not in BGwithOverlapNRM:
                shipmentsCK.at[i, 'external'] = 1

    # Append containerized shipments
    shipments = pd.concat([shipments, shipmentsCK[list(shipments.columns)]])
    shipments.index = np.arange(len(shipments))

    shipments = shipments.sort_values('shipment_id')

    # Change order of columns
    shipments = shipments[[
        'shipment_id', 'carrier_id', 'tour_id',
        'orig_nrm', 'dest_nrm', 'orig_bg', 'dest_bg',
        'commodity', 'logistic_segment', 'nstr', 'vehicle_type_lwm',
        'weight__ton', 'container']]

    return shipments


def write_tours_to_shp(
    outputTours: pd.DataFrame,
    path: str
):
    """
    Schrijf de rondritten weg als shapefile.

    Args:
        outputTours (pd.DataFrame): _description_
        path (str): _description_
    """
    Ax = list(outputTours['x_rd_orig__meter'])
    Ay = list(outputTours['y_rd_orig__meter'])
    Bx = list(outputTours['x_rd_dest__meter'])
    By = list(outputTours['y_rd_dest__meter'])

    # Initialize shapefile fields
    w = shp.Writer(path)
    w.field('carrier_id',         'N', size=5, decimal=0)
    w.field('tour_id',            'N', size=5, decimal=0)
    w.field('trip_id',            'N', size=3, decimal=0)
    w.field('orig_nrm',           'N', size=4, decimal=0)
    w.field('dest_nrm',           'N', size=4, decimal=0)
    w.field('orig_bg',            'N', size=4, decimal=0)
    w.field('dest_bg',            'N', size=4, decimal=0)
    w.field('orig_vam',           'N', size=4, decimal=0)
    w.field('dest_vam',           'N', size=4, decimal=0)
    w.field('vehicle_type_lwm',   'N', size=2, decimal=0)
    w.field('nstr',               'N', size=2, decimal=0)
    w.field('commodity',          'N', size=2, decimal=0)
    w.field('logistic_segment',   'N', size=2, decimal=0)
    w.field('n_shipments',        'N', size=3, decimal=0)
    w.field('dc_id',              'N', size=4, decimal=0)
    w.field('tour_weight__ton',   'N', size=5, decimal=2)
    w.field('trip_weight__ton',   'N', size=5, decimal=2)
    w.field('tour_deptime__hour', 'N', size=4, decimal=2)
    w.field('trip_deptime__hour', 'N', size=5, decimal=2)
    w.field('trip_arrtime__hour', 'N', size=5, decimal=2)
    w.field('combustion_type',    'N', size=2, decimal=0)
    w.field('external_zone_bg',   'N', size=5, decimal=0)
    w.field('container',          'N', size=2, decimal=0)

    coordCols = ['x_rd_orig__meter', 'y_rd_orig__meter', 'x_rd_dest__meter', 'y_rd_dest__meter']
    dbfData = np.array(outputTours.drop(columns=coordCols), dtype=object)

    numTrips = dbfData.shape[0]

    for i in range(numTrips):
        # Add geometry
        w.line([[[Ax[i], Ay[i]], [Bx[i], By[i]]]])

        # Add data fields
        w.record(*dbfData[i, :])

    w.close()


def create_and_write_trip_matrices(
    outputTours: pd.DataFrame,
    numLS: int, numVT: int,
    settings: Settings
):
    """
    Stel HB-matrices met aantallen ritten op voor de BasGoed- en NRM-zonering en schrijf deze
    weg als tekstbestanden.

    Args:
        outputTours (pd.DataFrame): _description_
        numLS (int): _description_
        numVT (int): _description_
        settings (Settings): _description_
    """
    # Maak dummies in tours variabele per logistiek segment,
    # voertuigtype en N_TOT (altijd 1 hier)
    for ls in range(numLS):
        outputTours['n_trips_ls' + str(ls)] = (outputTours['logistic_segment'] == ls).astype(int)
    for vt in range(numVT):
        outputTours['n_trips_vt' + str(vt)] = (outputTours['vehicle_type_lwm'] == vt).astype(int)
    outputTours['n_trips'] = 1

    # NRM-tripmatrix
    cols = [
        'orig_nrm', 'dest_nrm',
        *[f'n_trips_ls{ls}' for ls in range(numLS)],
        *[f'n_trips_vt{vt}' for vt in range(numVT)],
        'n_trips']

    # Gebruik deze dummies om het aantal ritten per HB te bepalen,
    # voor elk logistiek segment, voertuigtype en totaal
    pivotTable = pd.pivot_table(
        outputTours,
        values=cols[2:],
        index=['orig_nrm', 'dest_nrm'],
        aggfunc=np.sum)
    pivotTable['orig_nrm'] = [x[0] for x in pivotTable.index]
    pivotTable['dest_nrm'] = [x[1] for x in pivotTable.index]
    pivotTable = pivotTable[cols]

    pivotTable.to_csv(settings.get_path("RittenMatrixNRM"), index=False, sep='\t')

    # BG-tripmatrix
    cols = [
        'orig_bg', 'dest_bg',
        *[f'n_trips_ls{ls}' for ls in range(numLS)],
        *[f'n_trips_vt{vt}' for vt in range(numVT)],
        'n_trips']

    # Gebruik deze dummies om het aantal ritten per HB te bepalen,
    # voor elk logistiek segment, voertuigtype en totaal
    pivotTable = pd.pivot_table(
        outputTours,
        values=cols[2:],
        index=['orig_bg', 'dest_bg'],
        aggfunc=np.sum)
    pivotTable['orig_bg'] = [x[0] for x in pivotTable.index]
    pivotTable['dest_bg'] = [x[1] for x in pivotTable.index]
    pivotTable = pivotTable[cols]

    pivotTable.to_csv(settings.get_path("RittenMatrixBG"), index=False, sep='\t')
