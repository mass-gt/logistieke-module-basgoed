import logging

import numpy as np
import pandas as pd
import shapefile as shp
from numba import njit, int32
import scipy.sparse.csgraph
from scipy.sparse import lil_matrix
import multiprocessing as mp
import functools
from typing import Dict, List, Tuple

from calculation.common.data import get_euclidean_skim
from calculation.common.io import read_mtx
from calculation.common.params import get_num_cpu

from settings import Settings


def run_module(settings: Settings, logger: logging.Logger):

    logger.debug("\tInladen en prepareren invoerdata...")

    dimLS = settings.dimensions.logistic_segment
    dimET = settings.dimensions.emission_type
    dimVT = settings.dimensions.vehicle_type_lwm
    dimCombType = settings.dimensions.combustion_type
    dimRoadType = settings.dimensions.road_type

    numLS = len(dimLS)
    numET = len(dimET)
    numVT = len(dimVT)
    numCombType = len(dimCombType)

    # To convert emissions to kilograms
    etDict = dict((i, value) for i, value in enumerate(dimET))
    etInvDict = dict((v, k) for k, v in etDict.items())

    # Which vehicle type can be used in the parcel module
    vehTypesParcels = [i for i in range(numVT) if dimVT[i]['parcels']]

    # Carrying capacity in kg
    carryingCapacity = get_truck_capacities(settings, numVT)

    # Number of CPUs over which the processes are parallelized
    numCPU = get_num_cpu(settings.module.traf_num_cpu, 8, logger)

    # Aantal routes waarover gespreid wordt per HB
    numMultiRoute = settings.module.traf_num_multiroute
    if numMultiRoute is None:
        numMultiRoute = 1

    # Importeer de NRM-moederzones
    zones = pd.read_csv(settings.get_path("ZonesNRM"), sep='\t')
    zones.index = zones['zone_nrm']
    numZones = len(zones)

    # Importeer NRM-netwerk
    nodes = pd.read_csv(settings.get_path("NetwerkKnopen"), sep='\t')
    links = pd.read_csv(settings.get_path("NetwerkLinks"), sep='\t')
    numNodes = len(nodes)
    numLinks = len(links)

    # Hercoderen nodes
    nodeDict = dict((i, nodes.iloc[i, 0]) for i in range(numNodes))
    invNodeDict = dict((nodes.iloc[i, 0], i) for i in range(numNodes))

    nodes['node'] = [invNodeDict[x] for x in nodes['node'].values]
    nodes.index = np.arange(numNodes)

    links['node_a'] = [invNodeDict[x] for x in links['node_a'].values]
    links['node_b'] = [invNodeDict[x] for x in links['node_b'].values]

    shipments = pd.read_csv(settings.get_path("ZendingenNaTourformatie"), sep='\t')
    ggSharesMS = pd.pivot_table(
        shipments[shipments['container'] == 0],
        values='weight__ton', index='commodity', aggfunc=sum)
    ggSharesCK = pd.pivot_table(
        shipments[shipments['container'] == 1],
        values='weight__ton', index='commodity', aggfunc=sum)

    ggSharesMS.index = [int(x) for x in ggSharesMS.index]
    ggSharesCK.index = [int(x) for x in ggSharesCK.index]

    del shipments

    # Calculate average cost weighted by weight per goods type and containerized/non-containerized
    costPerHrFreight, costPerKmFreight = get_cost_freight(settings, ggSharesMS, ggSharesCK)

    # Cost parameters van
    costPerHrVan, costPerKmVan = get_cost_van(settings)

    # Ophalen betekenis verschillende soorten heffingen
    chargeValues, chargeIsDistanceBased = get_charges(settings)

    # Variabele om de linkID te verkrijgen o.b.v. A- en B-knoop
    linkDict = get_link_dict(links)

    # Travel costs
    linkChargeFreight, linkChargeVan = calc_cost_charge(
        links, chargeValues, chargeIsDistanceBased)
    links['cost_freight__euro'] = (
        costPerKmFreight * (links['distance__meter'] / 1000) +
        costPerHrFreight * (links['time_freight__s'] / 3600) +
        linkChargeFreight)
    links['cost_van__euro'] = (
        costPerKmVan * (links['distance__meter'] / 1000) +
        costPerHrVan * (links['time_van__s'] / 3600) +
        linkChargeVan)

    # Set connector travel costs high so these are not chosen other
    # than for entering/leaving network
    links.loc[links['link_type'] == 99, 'cost_freight__euro'] = 10000
    links.loc[links['link_type'] == 99, 'cost_van__euro'] = 10000

    # Set travel costs for forbidden-for-freight-links high so these are not chosen for freight
    links.loc[links['user_type'] == 2, 'cost_freight__euro'] = 10000

    # Set travel costs for only-freight-links high so these are not chosen for vans
    links.loc[links['user_type'] == 3, 'cost_van__euro'] = 10000

    # Set travel times on links in ZEZ Rotterdam high so these are only used to go to UCC
    # and not for through traffic
    costFreightHybr = links['cost_freight__euro'].copy()
    costVanHybr = links['cost_van__euro'].copy()
    if settings.module.apply_zez:
        links.loc[links['zez'] == 1, 'cost_freight__euro'] += 10000
        links.loc[links['zez'] == 1, 'cost_van__euro'] += 10000

    # Initialize empty fields with emissions and traffic intensity per link
    # (also save list with all field names)
    volCols = [
        *[f'n_trips_ls{ls}' for ls in range(numLS)],
        'n_trips_van_service', 'n_trips_van_construction',
        *[f'n_trips_vt{vt}' for vt in range(numVT)],
        'n_trips_ls8_vt8', 'n_trips_ls8_vt9',
        'n_trips']
    intensityFields = []
    for et in etDict.values():
        intensityFields.append(et.lower())
    for volCol in volCols:
        intensityFields.append(volCol)
    for ls in range(numLS):
        for et in etDict.values():
            intensityFields.append(et.lower() + '_ls' + str(ls))
    for ls in ['van_service', 'van_construction']:
        for et in etDict.values():
            intensityFields.append(et.lower() + '_' + str(ls))

    # Lees de emissiefactoren in
    emissionFacs = get_emission_facs(settings, etInvDict, dimRoadType, dimVT)

    # Instellingen specifiek voor het ZEZ-scenario
    if settings.module.apply_zez:
        probsCombustionVans = get_probs_combustion_vans(settings, numCombType)
        zonesZEZ = pd.read_csv(settings.get_path("ZonesZEZ"), sep='\t')
        zoneIsZEZ = np.zeros(len(zones), dtype=int)
        zoneIsZEZ[zonesZEZ['zone_nrm'].values - 1] = 1
    else:
        probsCombustionVans = np.zeros(numCombType, dtype=float)
        probsCombustionVans[0] = 1.0
        zoneIsZEZ = np.zeros(len(zones), dtype=int)

    # Inladen tourbestanden
    allTrips, allParcelTrips = get_trips(settings, carryingCapacity)
    trips = np.array(allTrips)
    del allTrips

    # Determine linktypes (urban/rural/highway)
    stadLinkTypes, buitenwegLinkTypes, snelwegLinkTypes = [6, 7], [3, 4, 5], [1, 2]

    whereStad = [links['link_type'][i] in stadLinkTypes for i in links.index]
    whereBuitenweg = [links['link_type'][i] in buitenwegLinkTypes for i in links.index]
    whereSnelweg = [links['link_type'][i] in snelwegLinkTypes for i in links.index]

    roadtypeArray = np.zeros(numLinks)
    roadtypeArray[whereStad] = 1
    roadtypeArray[whereBuitenweg] = 2
    roadtypeArray[whereSnelweg] = 3

    stadArray, buitenwegArray, snelwegArray = [roadtypeArray == x for x in [1, 2, 3]]
    distArray = np.array(links['distance__meter']) / 1000
    ZEZarray = np.array(links['zez'] == 1, dtype=bool)
    NLarray = np.array(links['nl'] == 1, dtype=bool)

    # Bring ORIG and DEST to the front of the list of column names
    newColOrder = volCols.copy()
    newColOrder.insert(0, 'dest_nrm')
    newColOrder.insert(0, 'orig_nrm')

    # HB-matrices met aantallen ritten ophalen
    tripMatrix, tripMatrixParcels = get_trip_matrices(settings, newColOrder)
    tripMatrixOrigins = set(tripMatrix[:, 0])

    if numMultiRoute >= 2:
        np.random.seed(100)
        whereTripsByIter = [[] for r in range(numMultiRoute)]
        for i in range(len(trips)):
            whereTripsByIter[np.random.randint(numMultiRoute)].append(i)
    elif numMultiRoute == 1:
        whereTripsByIter = [np.arange(len(trips))]
    else:
        raise Exception(
            f'traf-num-multiroute should be >= 1, now it is {numMultiRoute}.')

    # Vaste seed voor spreiding reiskosten op links
    if numMultiRoute >= 2:
        linksRandArray = [[] for r in range(numMultiRoute)]
        linksA = np.array(links['node_a'])
        linksB = np.array(links['node_b'])
        for i in range(len(links)):
            np.random.seed(linksA[i] + linksB[i])
            for r in range(numMultiRoute):
                linksRandArray[r].append(np.random.rand())
    else:
        linksRandArray = []

    # Initialiseer dictionaries om de emissies per rit bij te houden
    tripsCO2, tripsCO2_NL = {}, {}
    tripsDist, tripsDist_NL = {}, {}
    parcelTripsCO2 = {}

    # From which nodes do we need to perform the shortest path algoritm
    indices = np.arange(numZones)

    # From which nodes does every CPU perform the shortest path algorithm
    indicesPerCPU = [[indices[cpu::numCPU], cpu] for cpu in range(numCPU)]
    origSelectionPerCPU = [np.arange(numZones)[cpu::numCPU] for cpu in range(numCPU)]

    # Whether a separate route search needs to be done for hybrid/clean vehicles or not
    doHybrRoutes = np.any(trips[:, 6] == 3) or settings.module.apply_zez

    # Initialize arrays for intensities and emissions
    linkTripsArray = np.zeros((numLinks, len(volCols)))
    linkEmissionsArray = [np.zeros((numLinks, numET)) for ls in range(numLS)]

    for r in range(numMultiRoute):

        logger.debug(f"\tRoutes zoeken (vracht - deel {r + 1})...")

        prevFreight, prevFreightHybr = get_all_prev(
            np.array(links[['node_a', 'node_b']], dtype=int), linksRandArray, doHybrRoutes,
            links['cost_freight__euro'].values, costFreightHybr,
            numNodes, numZones, numMultiRoute, r,
            numCPU, origSelectionPerCPU, indices, indicesPerCPU,
            logger)

        logger.debug(
            f"\tEmissies en intensiteiten berekenen (vracht - deel {r + 1})")

        freightResults = calc_emissions_freight(
            tripMatrix, trips[whereTripsByIter[r], :], tripMatrixOrigins,
            prevFreight, prevFreightHybr, linkDict,
            stadArray, buitenwegArray, snelwegArray, ZEZarray, NLarray, distArray,
            emissionFacs,
            numLinks, numET, numZones, numLS, len(volCols),
            doHybrRoutes)

        linkTripsArray += freightResults[0]

        for ls in range(numLS):
            linkEmissionsArray[ls] += freightResults[1][ls]

        tripIDs = list(freightResults[2].keys())
        for trip in tripIDs:
            tripsCO2[trip] = freightResults[2][trip]
            tripsCO2_NL[trip] = freightResults[3][trip]
            tripsDist[trip] = freightResults[4][trip]
            tripsDist_NL[trip] = freightResults[5][trip]

        del freightResults, prevFreight, prevFreightHybr

    if numMultiRoute >= 2:
        whereParcelTripsByIter = [[] for r in range(numMultiRoute)]
        for i in range(len(allParcelTrips)):
            whereParcelTripsByIter[np.random.randint(numMultiRoute)].append(i)
    elif numMultiRoute == 1:
        whereParcelTripsByIter = [np.arange(len(allParcelTrips))]
    else:
        raise Exception(f"traf-num-multiroute zou >= 1 moeten zijn, maar is {numMultiRoute}")

    # Initialize arrays for intensities and emissions of vans
    linkVanTripsArray = np.zeros((numLinks, 2))
    linkCleanVanTripsArray = np.zeros((numLinks, 2))

    for r in range(numMultiRoute):

        logger.debug(f"\tRoutes zoeken (bestel - deel {r + 1})...")

        prevVan, prevVanHybr = get_all_prev(
            np.array(links[['node_a', 'node_b']], dtype=int), linksRandArray, doHybrRoutes,
            links['cost_van__euro'].values, costVanHybr,
            numNodes, numZones, numMultiRoute, r,
            numCPU, origSelectionPerCPU, indices, indicesPerCPU,
            logger)

        logger.debug(
            f"\tEmissies en intensiteiten berekenen (bestel - deel {r + 1})")

        logger.debug('\t\tPakketsegment...')

        # Logistic segment: parcel deliveries
        ls = 8

        # Assume half of vehicle capacity (in terms of weight) used
        capUt = 0.5

        for vt in vehTypesParcels:
            parcelTrips = allParcelTrips.loc[whereParcelTripsByIter[r], :]
            parcelTrips = parcelTrips.loc[(parcelTrips['vehicle_type_lwm'] == vt), :]
            parcelTrips = np.array(parcelTrips)

            emissionFacStad = np.array([
                emissionFacs['stad_leeg'][vt, et] + capUt * (
                    emissionFacs['stad_vol'][vt, et] - emissionFacs['stad_leeg'][vt, et])
                for et in range(numET)])
            emissionFacBuitenweg = np.array([
                emissionFacs['buitenweg_leeg'][vt, et] + capUt * (
                    emissionFacs['buitenweg_vol'][vt, et] - emissionFacs['buitenweg_leeg'][vt, et])
                for et in range(numET)])
            emissionFacSnelweg = np.array([
                emissionFacs['snelweg_leeg'][vt, et] + capUt * (
                    emissionFacs['snelweg_vol'][vt, et] - emissionFacs['snelweg_leeg'][vt, et])
                for et in range(numET)])

            if len(parcelTrips) > 0:

                parcelResults = calc_emissions_parcels(
                    parcelTrips, tripMatrixParcels,
                    prevVan, prevVanHybr,
                    distArray, stadArray, buitenwegArray, snelwegArray, ZEZarray,
                    etInvDict['CO2'], emissionFacStad, emissionFacBuitenweg, emissionFacSnelweg,
                    numLinks, numET, linkDict,
                    indices,
                    doHybrRoutes)

                # Intensiteiten
                numTrips = parcelResults[0]

                # Number of trips for LS8 (=parcel deliveries)
                linkTripsArray[:, ls] += numTrips

                # De parcel trips per voertuigtype
                linkTripsArray[:, numLS + 2 + numVT + vt - 8] += numTrips
                linkTripsArray[:, numLS + 2 + vt] += numTrips

                # Total number of trips
                linkTripsArray[:, -1] += numTrips

                # Emissies
                linkEmissionsArray[ls] += parcelResults[1]

                # CO2 per trip
                for trip in parcelResults[2].keys():
                    parcelTripsCO2[int(trip)] = parcelResults[2][trip]

                del parcelResults

        logger.debug('\t\tBouw & Service segmenten...')

        # Van trips for service and construction purposes
        try:
            vanTripsService = read_mtx(settings.get_path("RittenBestelService"))
            vanTripsConstruction = read_mtx(settings.get_path("RittenBestelBouw"))
            vanTripsFound = True

        except FileNotFoundError:
            logger.warning(
                'Kon "RittenBestelService" en/of "RittenBestelBouw" niet vinden. ' +
                'Daarom worden geen service- of bouwritten toegedeeld aan het netwerk.')
            vanTripsFound = False

        if vanTripsFound:
            # Select half of ODs per multiroute iteration
            if numMultiRoute > 1:
                vanTripsService[r::numMultiRoute] = 0
                vanTripsConstruction[r::numMultiRoute] = 0

            # Reshape to square array
            vanTripsService = vanTripsService.reshape(numZones, numZones)
            vanTripsConstruction = vanTripsConstruction.reshape(numZones, numZones)

            indicesPerCPU = []
            for cpu in range(numCPU):
                indicesPerCPU.append([])

                indicesPerCPU[cpu].append(indices[cpu::numCPU])
                indicesPerCPU[cpu].append(vanTripsService[indices[cpu::numCPU], :])
                indicesPerCPU[cpu].append(vanTripsConstruction[indices[cpu::numCPU], :])
                indicesPerCPU[cpu].append(prevVan[indices[cpu::numCPU], :])
                if doHybrRoutes:
                    indicesPerCPU[cpu].append(prevVanHybr[indices[cpu::numCPU], :])

            # Make some space available on the RAM
            del vanTripsService, vanTripsConstruction, prevVan
            if doHybrRoutes:
                del prevVanHybr

            if numCPU > 1:
                # Initialize a pool object that spreads tasks over different CPUs
                p = mp.Pool(numCPU)

                # Calculate intensities for service / construction vans
                vanResultPerCPU = p.map(functools.partial(
                    calc_intensities_vans,
                    doHybrRoutes, probsCombustionVans, zoneIsZEZ, ZEZarray,
                    numLinks, linkDict), indicesPerCPU)

                # Wait for completion of processes
                p.close()
                p.join()

            else:
                # Calculate intensities for service / construction vans
                vanResultPerCPU = [calc_intensities_vans(
                    doHybrRoutes, probsCombustionVans, zoneIsZEZ, ZEZarray,
                    numLinks, linkDict, indicesPerCPU[0])]

            del indicesPerCPU

            # Combine the results from the different CPUs
            for cpu in range(numCPU):
                linkVanTripsArray += vanResultPerCPU[cpu][0]
                if doHybrRoutes:
                    linkCleanVanTripsArray += vanResultPerCPU[cpu][1]

            del vanResultPerCPU

    # Write the intensities and emissions into the links-DataFrame
    for field in intensityFields:
        links[field] = 0.0
    links.loc[:, volCols] += linkTripsArray.astype(int)

    # Total emissions and per logistic segment
    for ls in range(numLS):
        for et in range(numET):
            links[etDict[et].lower()] += linkEmissionsArray[ls][:, et]
            links[etDict[et].lower() + '_ls' + str(ls)] += linkEmissionsArray[ls][:, et]

    del linkTripsArray, linkEmissionsArray

    if vanTripsFound:
        links = write_emissions_vans_into_links(
            links, linkVanTripsArray, linkCleanVanTripsArray,
            distArray, stadArray, buitenwegArray, snelwegArray,
            emissionFacs, etDict, numET)

        del linkVanTripsArray

    logger.debug('\tEmissies schrijven in "Tours" en "PakketRondritten"...')

    # Zet emissies in de tours
    tours, parcelTours = put_emissions_into_tours(
        settings, tripsCO2, tripsCO2_NL, tripsDist, tripsDist_NL, parcelTripsCO2, logger)

    # Wegschrijven verrijkte vracht tours
    tours.to_csv(settings.get_path("Tours"), sep='\t', index=False)

    # Wegschrijven verrijkte pakket tours
    parcelTours.to_csv(settings.get_path("PakketRondritten"), sep='\t', index=False)

    logger.debug('\tEmissies schrijven in "ZendingenNaTourformatie"...')

    shipments = put_emissions_into_shipments(settings, tours, zones, logger)

    # Wegschrijven verrijkte zendingen
    shipments.to_csv(
        settings.get_path("ZendingenNaTourformatie"), sep='\t', index=False)

    logger.debug("\tGeladen netwerk naar .txt exporteren...")

    links.to_csv(settings.get_path("GeladenLinks"), sep='\t', index=False)

    # Wegschrijven shapefile geladen netwerk
    if settings.module.write_shape_traf:
        logger.debug("\tGeladen netwerk naar .shp exporteren...")

        write_links_to_shape(
            settings.get_path("GeladenLinksShape"),
            links, nodes, nodeDict, invNodeDict, intensityFields)


def get_truck_capacities(settings: Settings, numVT: int) -> np.ndarray:
    """
    Returns truck capacities (in kg).
    """
    carryingCapacity = -1 * np.ones(numVT)
    for row in pd.read_csv(settings.get_path("LaadvermogensLWM"), sep='\t').to_dict('records'):
        vt = int(row['vehicle_type_lwm'])
        weightTon = float(row['weight__kgram'])
        carryingCapacity[vt] = weightTon

    if np.any(carryingCapacity == -1):
        raise Exception('Er zijn missende voertuigtypes in "LaadvermogensLWM".')

    return carryingCapacity


def get_cost_freight(
    settings: Settings,
    ggSharesMS: pd.DataFrame,
    ggSharesCK: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Berekent de uurkosten en kilometerkosten voor vracht gemiddeld
    over alle goederengroepen, voertuigtypen en verschijningsvormen.

    Args:
        settings (Settings): _description_
        ggSharesMS (pd.DataFrame): _description_
        ggSharesCK (pd.DataFrame): _description_

    Returns:
        tuple: costPerHrFreight en costPerKmFreight
    """
    # Cost parameters freight
    costFreightMS = np.array(pd.read_csv(
        settings.get_path("KostenKentallenMS", mode='weg'),
        sep='\t',
        usecols=['commodity', 'distance_costs__eur_per_kmeter', 'time_costs__eur_per_hour']))
    costFreightCK = np.array(pd.read_csv(
        settings.get_path("KostenKentallenCK", mode='road'),
        sep='\t',
        usecols=['commodity', 'distance_costs__eur_per_kmeter', 'time_costs__eur_per_hour']))

    costPerKmFreight = 0.0
    costPerHrFreight = 0.0

    for gg in costFreightMS[:, 0]:
        if int(gg) in ggSharesMS.index:
            row = np.where(costFreightMS[:, 0] == gg)[0][0]
            costPerKmFreight += costFreightMS[row, 1] * ggSharesMS.at[int(gg), 'weight__ton']
            costPerHrFreight += costFreightMS[row, 2] * ggSharesMS.at[int(gg), 'weight__ton']

    for gg in costFreightCK[:, 0]:
        if int(gg) in ggSharesCK.index:
            row = np.where(costFreightCK[:, 0] == gg)[0][0]
            costPerKmFreight += costFreightCK[row, 1] * ggSharesCK.at[int(gg), 'weight__ton']
            costPerHrFreight += costFreightCK[row, 2] * ggSharesCK.at[int(gg), 'weight__ton']

    sumWeight = (np.sum(ggSharesMS) + np.sum(ggSharesCK))

    costPerKmFreight /= sumWeight
    costPerHrFreight /= sumWeight
    costPerKmFreight = float(costPerKmFreight)
    costPerHrFreight = float(costPerHrFreight)

    return costPerHrFreight, costPerKmFreight


def get_cost_van(settings: Settings) -> Tuple[float, float]:
    """
    Berekent de uurkosten en kilometerkosten voor bestelwagens gemiddeld
    over alle goederengroepen.

    Args:
        settings (Settings): _description_

    Returns:
        tuple: costPerHrVan en costPerKmVan
    """
    costVan = np.array(pd.read_csv(
        settings.get_path("KostenKentallenVoertuigType", vehicle_type="Bestel"),
        sep='\t',
        usecols=['commodity', 'distance_costs__eur_per_kmeter', 'time_costs__eur_per_hour']))
    costPerKmVan = np.average(costVan[:, 1])
    costPerHrVan = np.average(costVan[:, 2])

    return costPerHrVan, costPerKmVan


def get_emission_facs(
    settings: Settings,
    etInvDict: Dict[int, str],
    dimRoadType: List[str],
    dimVT: List[str]
) -> Dict[str, np.ndarray]:
    """
    Haal de emissiefactoren op.

    Args:
        settings (Settings): _description_
        etInvDict (Dict[str, int]): _description_
        dimRoadType (List[str]): _description_
        dimVT (List[str]): _description_

    Returns:
        dict: Per combinatie van wegtype en leeg/vol de emissiefactoren.
    """
    numET = len(etInvDict)
    numVT = len(dimVT)

    emissionFacs = dict(
        (f'{roadType.lower()}_{loadType}', np.zeros((numVT, numET)))
        for roadType in dimRoadType for loadType in ['leeg', 'vol'])

    for row in pd.read_csv(settings.get_path("Emissiefactoren"), sep='\t').to_dict('records'):
        key = f"{row['road_type']}_{row['loaded']}"
        et = etInvDict[row['emission_type']]
        vt = row['vehicle_type_lwm']
        value = row['factor__gram_per_kmeter']
        emissionFacs[key][vt, et] = value

    return emissionFacs


def get_link_dict(
    links: pd.DataFrame,
    maxNumConnections: int = 8
) -> np.ndarray:
    """
    Maak een array o.b.v. efficient een linknummer kan worden afgeleid o.b.v. de herkomst-
    en -bestemmingsknoop.

    Args:
        links (pd.DataFrame): _description_
        maxNumConnections (int, optional): _description_. Defaults to 8.

    Returns:
        np.ndarray: _description_
    """
    linkDict = -1 * np.ones((max(links['node_a']) + 1, 2 * maxNumConnections), dtype=np.int32)
    for i in links.index:
        aNode = links['node_a'][i]
        bNode = links['node_b'][i]

        for col in range(maxNumConnections):
            if linkDict[aNode][col] == -1:
                linkDict[aNode][col] = bNode
                linkDict[aNode][col + maxNumConnections] = i
                break

    return linkDict


def get_trips(
    settings: Settings,
    carryingCapacity: np.ndarray
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Inladen tekstbestanden met rondritten voor vracht en pakketbezorging.

    Args:
        settings (Settings): _description_
        carryingCapacity (np.ndarray): _description_

    Returns:
        tuple: Met daarin:
            - pd.DataFrame: de rondritten voor vracht
            - pd.DataFrame: de rondritten voor pakketbezorging
    """

    # Import trips csv
    allTrips = pd.read_csv(settings.get_path("Tours"), sep='\t')
    allTrips.loc[allTrips['trip_deptime__hour'] >= 24, 'trip_deptime__hour'] -= 24
    allTrips.loc[allTrips['trip_deptime__hour'] >= 24, 'trip_deptime__hour'] -= 24
    capUt = (
        (allTrips['trip_weight__ton'] * 1000) /
        carryingCapacity[np.array(allTrips['vehicle_type_lwm'], dtype=int)])
    allTrips['capacity_utilization'] = capUt
    allTrips['index'] = allTrips.index
    allTrips = allTrips[[
        'carrier_id', 'orig_nrm', 'dest_nrm',
        'vehicle_type_lwm', 'capacity_utilization',
        'logistic_segment', 'combustion_type', 'index']]

    # Import parcel schedule csv
    allParcelTrips = pd.read_csv(settings.get_path("PakketRondritten"), sep='\t')

    # Assume on average 50% of loading capacity used for parcel deliveries
    allParcelTrips['capacity_utilization'] = 0.5

    # Recode vehicle type from string to number
    allParcelTrips['vehicle_type_lwm'] = [
        {'Van': 8, 'LEVV': 9}[vt] for vt in allParcelTrips['vehicle_type_lwm']]

    # Fuel as basic combustion type
    allParcelTrips['combustion_type'] = 0

    # Trips coming from UCC to ZEZ use electric
    allParcelTrips.loc[allParcelTrips['orig_type'] == 'UCC', 'combustion_type'] = 1
    allParcelTrips['logistic_segment'] = 6
    allParcelTrips['index'] = allParcelTrips.index
    allParcelTrips = allParcelTrips[[
        'depot_id', 'orig_nrm', 'dest_nrm',
        'vehicle_type_lwm', 'capacity_utilization',
        'logistic_segment', 'combustion_type', 'index']]

    return allTrips, allParcelTrips


def get_probs_combustion_vans(
    settings: Settings,
    numCombType: int
) -> np.ndarray:
    """
    Haal de overgangskansen per energiebron op voor bestelauto's.

    Args:
        settings (Settings): _description_
        numCombType (int): _description_

    Returns:
        np.ndarray: _description_
    """
    # Vehicle/combustion shares (for ZEZ scenario)
    scenarioZEZ = pd.read_csv(settings.get_path("KansenVoertuigtypeZEZ"), sep='\t')

    probsCombustionVans = np.zeros(numCombType, dtype=float)

    for i in scenarioZEZ[scenarioZEZ['consolidated'] == 0].index:
        vt = int(scenarioZEZ.at[i, 'vehicle_type_lwm'])

        if vt == 8:
            ct = int(scenarioZEZ.at[i, 'combustion_type'])
            prob = scenarioZEZ.at[i, 'probability']
            probsCombustionVans[ct] += prob

    if np.sum(probsCombustionVans) == 0:
        probsCombustionVans = np.ones(numCombType, dtype=float) / numCombType

    if np.sum(probsCombustionVans) != 1:
        probsCombustionVans /= np.sum(probsCombustionVans)

    return probsCombustionVans


def get_trip_matrices(
    settings: Settings,
    columnOrder: list
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Haalt de HB-matrices met aantallen ritten voor vracht en
    pakketbezorging op.

    Args:
        settings (Settings): _description_
        columnOrder (list): _description_

    Returns:
        tuple: Met daarin:
            - numpy.ndarray: tripMatrix
            - numpy.ndarray: tripMatrixParcels
    """
    # Vracht
    tripMatrix = pd.read_csv(settings.get_path("RittenMatrixNRM"), sep='\t')
    tripMatrix['n_trips_ls8'] = 0
    tripMatrix['n_trips_van_service'] = 0
    tripMatrix['n_trips_van_construction'] = 0
    tripMatrix['n_trips_ls8_vt8'] = 0
    tripMatrix['n_trips_ls8_vt9'] = 0
    tripMatrix = tripMatrix[columnOrder]
    tripMatrix = np.array(tripMatrix)

    # Pakketbezorging
    tripMatrixParcels = pd.read_csv(settings.get_path("RittenMatrixPakketNRM"), sep='\t')
    tripMatrixParcels = np.array(tripMatrixParcels)

    return tripMatrix, tripMatrixParcels


def get_charges(
    settings: Settings
) -> Tuple[Dict[str, Dict[int, float]], Dict[str, Dict[int, bool]]]:
    """
    Haal de betekenis van de verschillende heffingssoorten uit het NRM-netwerk op.

    De sleutels zijn:
        - 'freight'
            - 0 t/m 9
        - 'van'
            - 0 t/m 9

    Args:
        settings (Settings): _description_

    Returns:
        tuple: Met daarin:
            - dict: De hoogte van de heffing.
            - dict: Of de heffing afstandsgebaseerd is of niet.
    """
    chargeValues = {'freight': {}, 'van': {}}
    chargeIsDistanceBased = {'freight': {}, 'van': {}}

    for row in pd.read_csv(settings.get_path("Heffingen"), sep='\t').to_dict('records'):
        chargeType = int(row['charge_type'])
        isFreight = bool(int(row['freight']))
        isDistanceBased = bool(int(row['distance_based']))
        value = float(row['value'])

        if isFreight:
            chargeValues['freight'][chargeType] = value
            chargeIsDistanceBased['freight'][chargeType] = isDistanceBased
        else:
            chargeValues['van'][chargeType] = value
            chargeIsDistanceBased['van'][chargeType] = isDistanceBased

    return chargeValues, chargeIsDistanceBased


def calc_cost_charge(
    links: pd.DataFrame,
    chargeValues: Dict[str, Dict[int, float]],
    chargeIsDistanceBased: Dict[str, Dict[int, bool]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bereken de heffing (in euro's) die geldt op elke link voor vracht en voor bestel.

    Args:
        links (pd.DataFrame): _description_
        chargeValues (Dict[str, Dict[int, float]]): _description_
        chargeIsDistanceBased (Dict[str, Dict[int, bool]]): _description_

    Returns:
        tuple: Met daarin:
            - np.ndarray: De heffing voor vracht.
            - np.ndarray: De heffing voor bestel.
    """
    numLinks = links.shape[0]

    linkDist = links['distance__meter'].values / 1000
    linkChargeType = np.array(links['charge_type'], dtype=int)

    linkChargeFreight = np.zeros(numLinks, dtype=float)
    linkChargeVan = np.zeros(numLinks, dtype=float)

    for i in range(numLinks):
        tmpFreightValue = chargeValues['freight'][linkChargeType[i]]
        tmpFreightIsDistanceBased = chargeIsDistanceBased['freight'][linkChargeType[i]]

        if tmpFreightIsDistanceBased:
            linkChargeFreight[i] = linkDist[i] * tmpFreightValue
        else:
            linkChargeFreight[i] = tmpFreightValue

        tmpVanValue = chargeValues['van'][linkChargeType[i]]
        tmpVanIsDistanceBased = chargeIsDistanceBased['van'][linkChargeType[i]]

        if tmpVanIsDistanceBased:
            linkChargeVan[i] = linkDist[i] * tmpVanValue
        else:
            linkChargeVan[i] = tmpVanValue

    return linkChargeFreight, linkChargeVan


def get_prev(
    csgraph: lil_matrix,
    indices: list
) -> np.ndarray:
    '''
    For each origin zone and destination node, determine the
    previously visited node on the shortest path.

    Args:
        csgraph (scipy.sparse.lil_matrix): _description_
        indices (list): _description_

    Returns:
        np.ndarray: 'prev' object van de Dijkstra-routine
    '''
    whichCPU = indices[1]
    indices = indices[0]
    numOrigSelection = len(indices)

    prev = [None for i in range(numOrigSelection)]

    for i in range(numOrigSelection):
        prev[i] = np.array(scipy.sparse.csgraph.dijkstra(
            csgraph, indices=indices[i], return_predecessors=True)[1], dtype=np.int32)

        if whichCPU == 0:
            if i % int(round(numOrigSelection / 100)) == 0:
                print(f'\t{round((i / numOrigSelection) * 100, 1)}%', end='\r')

    del csgraph

    prev = np.array(prev, dtype=np.int32)

    return prev


def get_all_prev(
    links: np.ndarray, linksRandArray: np.ndarray,
    doHybrRoutes: bool,
    cost: np.ndarray, costHybr: np.ndarray,
    numNodes: int, numZones: int,
    numMultiRoute: int, multiRoute: int,
    numCPU: int, origSelectionPerCPU: list,
    indices: list, indicesPerCPU: list,
    logger: logging.Logger
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Roept 'get_prev' aan vanuit alle herkomstzones, eventueel apart voor hybride en niet-hybride,
    met spreiding van taken over verschillende cores.

    Args:
        links (np.ndarray): _description_
        linksRandArray (np.ndarray): _description_
        doHybrRoutes (bool): _description_
        cost (np.ndarray): _description_
        costHybr (np.ndarray): _description_
        numNodes (int): _description_
        numZones (int): _description_
        numMultiRoute (int): _description_
        multiRoute (int): _description_
        numCPU (int): _description_
        origSelectionPerCPU (list): _description_
        indices (list): _description_
        indicesPerCPU (list): _description_
        logger (logging.Logger): _description_

    Returns:
        tuple: Met daarin:
            - np.ndarray: 'prev' object van de Dijkstra-routine voor standaardvoertuigen
            - np.ndarray: 'prev' object van de Dijkstra-routine voor hybridevoertuigen
    """
    if numCPU > 1:

        # The network with costs between nodes
        csgraph = lil_matrix((numNodes, numNodes))
        if numMultiRoute > 1:
            csgraph[links[:, 0], links[:, 1]] = (
                cost * 0.95 + 0.1 * linksRandArray[multiRoute])
        else:
            csgraph[links[:, 0], links[:, 1]] = cost

        # Initialize a pool object that spreads tasks over different CPUs
        p = mp.Pool(numCPU)

        # Execute the Dijkstra route search
        prevPerCPU = p.map(functools.partial(get_prev, csgraph), indicesPerCPU)

        # Wait for completion of processes
        p.close()
        p.join()

        # Combine the results from the different CPUs
        # Matrix with for each node the previous node on the shortest path
        prev = [None for i in range(numZones)]
        for cpu in range(numCPU - 1, -1, -1):
            for i in range(len(indicesPerCPU[cpu][0])):
                prev[origSelectionPerCPU[cpu][i]] = prevPerCPU[cpu][i, :]
            del prevPerCPU[cpu]
        prev = np.array(prev, dtype=np.int32)

        # Make some space available on the RAM
        del prevPerCPU

    else:
        # The network with costs between nodes
        csgraph = lil_matrix((numNodes, numNodes))
        if numMultiRoute > 1:
            csgraph[links[:, 0], links[:, 1]] = (
                cost * 0.95 + 0.1 * linksRandArray[multiRoute])
        else:
            csgraph[links[:, 0], links[:, 1]] = cost

        # Execute the Dijkstra route search
        prev = get_prev(csgraph, [indices, 0])

    # Make some space available on the RAM
    del csgraph

    if not doHybrRoutes:
        return prev, None

    logger.debug(
        f"\t\t(with hybrid combustion, part {multiRoute + 1})...")

    if numCPU > 1:

        # The network with costs between nodes
        csgraphHybr = lil_matrix((numNodes, numNodes))
        if numMultiRoute > 1:
            csgraphHybr[links[:, 0], links[:, 1]] = (
                costHybr * 0.95 + 0.1 * linksRandArray[multiRoute])
        else:
            csgraphHybr[links[:, 0], links[:, 1]] = costHybr

        # Initialize a pool object that spreads tasks over
        # different CPUs
        p = mp.Pool(numCPU)

        # Execute the Dijkstra route search
        prevHybrPerCPU = p.map(functools.partial(get_prev, csgraphHybr), indicesPerCPU)

        # Wait for completion of processes
        p.close()
        p.join()

        # Combine the results from the different CPUs.
        # Matrix with for each node the previous node on the shortest path.
        prevHybr = [None for i in range(numZones)]
        for cpu in range(numCPU - 1, -1, -1):
            for i in range(len(indicesPerCPU[cpu][0])):
                prevHybr[origSelectionPerCPU[cpu][i]] = prevHybrPerCPU[cpu][i, :]
            del prevHybrPerCPU[cpu]
        prevHybr = np.array(prevHybr, dtype=np.int32)

        # Make some space available on the RAM
        del prevHybrPerCPU

    else:

        # The network with costs between nodes
        csgraphHybr = lil_matrix((numNodes, numNodes))
        if numMultiRoute > 1:
            csgraphHybr[links[:, 0], links[:, 1]] = (
                costHybr * 0.95 + 0.1 * linksRandArray[multiRoute])
        else:
            csgraphHybr[links[:, 0], links[:, 1]] = costHybr

        # Execute the Dijkstra route search
        prevHybr = get_prev(csgraphHybr, [indices, 0])

    # Make some space available on the RAM
    del csgraphHybr

    return prev, prevHybr


@njit
def get_route(
    orig: int,
    dest: int,
    prev: np.ndarray,
    linkDict: np.ndarray,
    maxNumConnections: int = 8
) -> np.ndarray:
    """
    Deduce the path from the prev object for one OD.

    Args:
        orig (int): _description_
        dest (int): _description_
        prev (np.ndarray): _description_
        linkDict (np.ndarray): _description_
        maxNumConnections (int, optional): _description_. Defaults to 8.

    Returns:
        np.ndarray: Path sequence in terms of link IDs.
    """

    route = []

    # Deduce sequence of nodes on network
    sequenceNodes = []
    destNode = dest
    if prev[orig][destNode] >= 0:
        while prev[orig][destNode] >= 0:
            sequenceNodes.insert(0, destNode)
            destNode = prev[orig][destNode]
        else:
            sequenceNodes.insert(0, destNode)

    # Deduce sequence of links on network
    if len(sequenceNodes) > 1:

        for i in range(len(sequenceNodes) - 1):
            aNode = sequenceNodes[i]
            bNode = sequenceNodes[i + 1]

            tmp = linkDict[aNode]
            for col in range(maxNumConnections):
                if tmp[col] == bNode:
                    route.append(tmp[col + maxNumConnections])
                    break

    return np.array(route, dtype=int32)


def calc_emissions_freight(
    tripMatrix: np.ndarray, iterTrips: np.ndarray, tripMatrixOrigins: set,
    prevFreight: np.ndarray, prevFreightHybr: np.ndarray, linkDict: np.ndarray,
    stadArray: np.ndarray, buitenwegArray: np.ndarray, snelwegArray: np.ndarray,
    ZEZarray: np.ndarray, NLarray: np.ndarray, distArray: np.ndarray,
    emissionFacs: Dict[str, np.ndarray],
    numLinks: int, numET: int, numZones: int, numLS: int, numVolCols: int,
    doHybrRoutes: bool
) -> Tuple[
    np.ndarray, np.ndarray,
    Dict[int, float], Dict[int, float], Dict[int, float], Dict[int, float]
]:
    """_summary_

    Args:
        tripMatrix (np.ndarray): _description_
        iterTrips (np.ndarray): _description_
        tripMatrixOrigins (set): _description_
        prevFreight (np.ndarray): _description_
        prevFreightHybr (np.ndarray): _description_
        linkDict (np.ndarray): _description_
        stadArray (np.ndarray): _description_
        buitenwegArray (np.ndarray): _description_
        snelwegArray (np.ndarray): _description_
        ZEZarray (np.ndarray): _description_
        NLarray (np.ndarray): _description_
        distArray (np.ndarray): _description_
        emissionsFacs (Dict[str, np.ndarray]): _description_
        numLinks (int): _description_
        numET (int): _description_
        numZones (int): _description_
        numLS (int): _description_
        numVolCols (int): _description_
        doHybrRoutes (bool): _description_

    Returns:
        _type_: _description_
    """
    # Initialize arrays for intensities and emissions
    linkTripsArray = np.zeros((numLinks, numVolCols))
    linkEmissionsArray = [np.zeros((numLinks, numET)) for ls in range(numLS)]

    # Initialiseer dictionaries voor bijhouden emissies en afstand per rit
    tripsCO2, tripsCO2_NL = {}, {}
    tripsDist, tripsDist_NL = {}, {}

    whereODL = {}
    for i in range(len(tripMatrix)):
        orig = tripMatrix[i, 0]
        dest = tripMatrix[i, 1]
        for ls in range(numLS):
            whereODL[(orig, dest, ls)] = []
    for i in range(len(iterTrips)):
        orig = iterTrips[i, 1]
        dest = iterTrips[i, 2]
        ls = iterTrips[i, 5]
        whereODL[(orig, dest, ls)].append(i)

    for i in range(numZones):
        origZone = i + 1

        print('\tOrigin ' + str(origZone), end='\r')

        if origZone not in tripMatrixOrigins:
            continue

        destZoneIndex = np.where(tripMatrix[:, 0] == origZone)[0]
        destZones = tripMatrix[destZoneIndex, 1]

        # Regular routes
        routes = [get_route(i, destZone - 1, prevFreight, linkDict) for destZone in destZones]

        # Routes for hybrid vehicles
        # (no need for penalty on ZEZ-links here)
        if doHybrRoutes:
            hybrRoutes = [
                get_route(i, destZone - 1, prevFreightHybr, linkDict) for destZone in destZones]

        # Schrijf de volumes op de links
        for j in range(len(destZones)):
            destZone = destZones[j]
            route = routes[j]

            # Get route and part of route that is
            # stad/buitenweg/snelweg and ZEZ/non-ZEZ
            routeStad = route[stadArray[route]]
            routeBuitenweg = route[buitenwegArray[route]]
            routeSnelweg = route[snelwegArray[route]]

            # Het gedeelte van de route in NL
            routeNL = route[NLarray[route]]
            isRouteStadNL = NLarray[routeStad]
            isRouteBuitenwegNL = NLarray[routeBuitenweg]
            isRouteSnelwegNL = NLarray[routeSnelweg]

            distStad = distArray[routeStad]
            distBuitenweg = distArray[routeBuitenweg]
            distSnelweg = distArray[routeSnelweg]
            distTotal = np.sum(distArray[route])
            distTotalNL = np.sum(distArray[routeNL])

            if doHybrRoutes:
                hybrRoute = hybrRoutes[j]

                # Get route and part of route that is
                # stad/buitenweg/snelweg and ZEZ/non-ZEZ
                hybrRouteStad = hybrRoute[stadArray[hybrRoute]]
                hybrRouteBuitenweg = hybrRoute[buitenwegArray[hybrRoute]]
                hybrRouteSnelweg = hybrRoute[snelwegArray[hybrRoute]]
                hybrZEZstad = ZEZarray[hybrRouteStad]
                hybrZEZbuitenweg = ZEZarray[hybrRouteBuitenweg]
                hybrZEZsnelweg = ZEZarray[hybrRouteSnelweg]

                # Het gedeelte van de route in NL
                hybrRouteNL = hybrRoute[NLarray[hybrRoute]]
                hybrIsRouteStadNL = NLarray[hybrRouteStad]
                hybrIsRouteBuitenwegNL = NLarray[hybrRouteBuitenweg]
                hybrIsRouteSnelwegNL = NLarray[hybrRouteSnelweg]

                hybrDistStad = distArray[hybrRouteStad]
                hybrDistBuitenweg = distArray[hybrRouteBuitenweg]
                hybrDistSnelweg = distArray[hybrRouteSnelweg]
                hybrDistTotal = np.sum(distArray[hybrRoute])
                hybrDistTotalNL = np.sum(distArray[hybrRouteNL])

            # Bereken en schrijf de intensiteiten/emissies
            # op de links
            for ls in range(numLS):
                # Welke trips worden allemaal gemaakt op de HB
                # van de huidige iteratie van de ij-loop
                currentTrips = iterTrips[whereODL[(origZone, destZone, ls)], :]
                numCurrentTrips = len(currentTrips)

                if numCurrentTrips == 0:
                    continue

                capUt = currentTrips[:, 4]
                vt = [int(x) for x in currentTrips[:, 3]]

                emissionFacStad = [None for et in range(numET)]
                emissionFacBuitenweg = [None for et in range(numET)]
                emissionFacSnelweg = [None for et in range(numET)]

                for et in range(numET):
                    lowerFacStad = emissionFacs['stad_leeg'][vt, et]
                    upperFacStad = emissionFacs['stad_vol'][vt, et]
                    emissionFacStad[et] = lowerFacStad + capUt * (upperFacStad - lowerFacStad)

                    lowerFacBuitenweg = emissionFacs['buitenweg_leeg'][vt, et]
                    upperFacBuitenweg = emissionFacs['buitenweg_vol'][vt, et]
                    emissionFacBuitenweg[et] = lowerFacBuitenweg + capUt * (
                        upperFacBuitenweg - lowerFacBuitenweg)

                    lowerFacSnelweg = emissionFacs['snelweg_leeg'][vt, et]
                    upperFacSnelweg = emissionFacs['snelweg_vol'][vt, et]
                    emissionFacSnelweg[et] = lowerFacSnelweg + capUt * (
                        upperFacSnelweg - lowerFacSnelweg)

                for trip in range(numCurrentTrips):
                    vt = int(currentTrips[trip, 3])
                    ct = int(currentTrips[trip, 6])

                    # If combustion type is fuel or bio-fuel
                    if ct in [0, 4]:
                        stadEmissions = np.zeros((len(routeStad), numET))
                        buitenwegEmissions = np.zeros((len(routeBuitenweg), numET))
                        snelwegEmissions = np.zeros((len(routeSnelweg), numET))

                        for et in range(numET):
                            stadEmissions[:, et] = distStad * emissionFacStad[et][trip]
                            buitenwegEmissions[:, et] = (
                                distBuitenweg * emissionFacBuitenweg[et][trip])
                            snelwegEmissions[:, et] = distSnelweg * emissionFacSnelweg[et][trip]

                        linkEmissionsArray[ls][routeStad, :] += stadEmissions
                        linkEmissionsArray[ls][routeBuitenweg, :] += buitenwegEmissions
                        linkEmissionsArray[ls][routeSnelweg, :] += snelwegEmissions

                        linkTripsArray[route, numLS + 2 + vt] += 1
                        linkTripsArray[route, ls] += 1
                        linkTripsArray[route, -1] += 1

                    # If combustion type is hybrid
                    elif ct == 3:
                        stadEmissions = np.zeros((len(hybrRouteStad), numET))
                        buitenwegEmissions = np.zeros((len(hybrRouteBuitenweg), numET))
                        snelwegEmissions = np.zeros((len(hybrRouteSnelweg), numET))

                        for et in range(numET):
                            stadEmissions[:, et] = hybrDistStad * emissionFacStad[et][trip]
                            buitenwegEmissions[:, et] = (
                                hybrDistBuitenweg * emissionFacBuitenweg[et][trip])
                            snelwegEmissions[:, et] = (
                                hybrDistSnelweg * emissionFacSnelweg[et][trip])

                        # No emissions in ZEZ part of route
                        stadEmissions[hybrZEZstad, :] = 0
                        buitenwegEmissions[hybrZEZbuitenweg, :] = 0
                        snelwegEmissions[hybrZEZsnelweg, :] = 0

                        linkEmissionsArray[ls][hybrRouteStad, :] += stadEmissions
                        linkEmissionsArray[ls][hybrRouteBuitenweg, :] += buitenwegEmissions
                        linkEmissionsArray[ls][hybrRouteSnelweg, :] += snelwegEmissions

                        linkTripsArray[hybrRoute, numLS + 2 + vt] += 1
                        linkTripsArray[hybrRoute, ls] += 1
                        linkTripsArray[hybrRoute, -1] += 1

                    # Clean combustion types (no emissions)
                    else:
                        linkTripsArray[route, numLS + 2 + vt] += 1
                        linkTripsArray[route, ls] += 1
                        linkTripsArray[route, -1] += 1

                    tripIndex = currentTrips[trip, -1]

                    # Total CO2 for each trip
                    if ct in [0, 4]:
                        tripsCO2[tripIndex] = (
                            np.sum(stadEmissions[:, 0]) +
                            np.sum(buitenwegEmissions[:, 0]) +
                            np.sum(snelwegEmissions[:, 0]))
                        tripsCO2_NL[tripIndex] = (
                            np.sum(stadEmissions[isRouteStadNL, 0]) +
                            np.sum(buitenwegEmissions[isRouteBuitenwegNL, 0]) +
                            np.sum(snelwegEmissions[isRouteSnelwegNL, 0]))

                    elif ct == 3:
                        tripsCO2[tripIndex] = (
                            np.sum(stadEmissions[:, 0]) +
                            np.sum(buitenwegEmissions[:, 0]) +
                            np.sum(snelwegEmissions[:, 0]))
                        tripsCO2_NL[tripIndex] = (
                            np.sum(stadEmissions[hybrIsRouteStadNL, 0]) +
                            np.sum(buitenwegEmissions[hybrIsRouteBuitenwegNL, 0]) +
                            np.sum(snelwegEmissions[hybrIsRouteSnelwegNL, 0]))
                    else:
                        tripsCO2[tripIndex] = 0
                        tripsCO2_NL[tripIndex] = 0

                    # Total distance for each trip
                    if ct == 3:
                        tripsDist[tripIndex] = hybrDistTotal
                        tripsDist_NL[tripIndex] = hybrDistTotalNL
                    else:
                        tripsDist[tripIndex] = distTotal
                        tripsDist_NL[tripIndex] = distTotalNL

    for ls in range(numLS):
        linkEmissionsArray[ls] /= 1000

    return (
        linkTripsArray, linkEmissionsArray,
        tripsCO2, tripsCO2_NL, tripsDist, tripsDist_NL)


def calc_emissions_parcels(
    trips: np.ndarray,
    tripMatrixParcels: np.ndarray,
    prevVan: np.ndarray, prevVanHybr: np.ndarray,
    distArray: np.ndarray,
    stadArray: np.ndarray, buitenwegArray: np.ndarray, snelwegArray: np.ndarray,
    ZEZarray: np.ndarray,
    indexCO2: int,
    emissionFacStad: np.ndarray, emissionFacBuitenweg: np.ndarray, emissionFacSnelweg: np.ndarray,
    nLinks: int, nET: int,
    linkDict: np.ndarray,
    indices: np.ndarray,
    doHybrRoutes: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bereken de intensiteiten en emissies van bestelautoritten
    voor pakketbezorging.

    Args:
        trips (np.ndarray): _description_
        tripMatrixParcels (np.ndarray): _description_
        prevVan (np.ndarray): _description_
        prevVanHybr (np.ndarray): _description_
        distArray (np.ndarray): _description_
        stadArray (np.ndarray): _description_
        buitenwegArray (np.ndarray): _description_
        snelwegArray (np.ndarray): _description_
        ZEZarray (np.ndarray): _description_
        indexCO2 (int): _description_
        emissionFacStad (np.ndarray): _description_
        emissionFacBuitenweg (np.ndarray): _description_
        emissionFacSnelweg (np.ndarray): _description_
        nLinks (int): _description_
        nET (int): _description_
        linkDict (np.ndarray): _description_
        indices (np.ndarray): _description_
        doHybrRoutes (bool): _description_

    Returns:
        tuple: Met daarin:
            - np.ndarray: De intensiteiten op elke link
            - np.ndarray: De emissies op elke link
            - np.ndarray: De CO2-emissies van elke rit
    """
    linkTripsArray = np.zeros(nLinks)
    linkEmissionsArray = np.zeros((nLinks, nET))
    parcelTripsCO2 = {}

    for i in range(len(indices)):
        print('\t\tOrigin ' + '{:>5}'.format(indices[i] + 1), end='\r')

        origZone = indices[i]
        destZoneIndex = np.where(
            tripMatrixParcels[:, 0] == (origZone + 1))[0]

        if len(destZoneIndex) == 0:
            continue

        # Schrijf de volumes op de links
        for j in destZoneIndex:
            destZone = tripMatrixParcels[j, 1] - 1

            route = get_route(origZone, destZone, prevVan, linkDict)

            if doHybrRoutes:
                hybrRoute = get_route(origZone, destZone, prevVanHybr, linkDict)

            # Welke trips worden allemaal gemaakt op de HB van de
            # huidige iteratie van de ij-loop
            currentTrips = trips[
                (trips[:, 1] == (origZone + 1)) &
                (trips[:, 2] == (destZone + 1)), :]

            # Route per wegtype
            routeStad = route[stadArray[route]]
            routeBuitenweg = route[buitenwegArray[route]]
            routeSnelweg = route[snelwegArray[route]]

            # Emissies voor een enkele trip
            stadEmissions = np.zeros((len(routeStad), nET))
            buitenwegEmissions = np.zeros((len(routeBuitenweg), nET))
            snelwegEmissions = np.zeros((len(routeSnelweg), nET))
            for et in range(nET):
                stadEmissions[:, et] = distArray[routeStad] * emissionFacStad[et]
                buitenwegEmissions[:, et] = distArray[routeBuitenweg] * emissionFacBuitenweg[et]
                snelwegEmissions[:, et] = distArray[routeSnelweg] * emissionFacSnelweg[et]

            if doHybrRoutes:
                # Route per wegtype
                hybrRouteStad = hybrRoute[stadArray[hybrRoute]]
                hybrRouteBuitenweg = hybrRoute[buitenwegArray[hybrRoute]]
                hybrRouteSnelweg = hybrRoute[snelwegArray[hybrRoute]]

                # Welke links liggen in het ZE-gebied
                hybrZEZstad = ZEZarray[hybrRouteStad]
                hybrZEZbuitenweg = ZEZarray[hybrRouteBuitenweg]
                hybrZEZsnelweg = ZEZarray[hybrRouteSnelweg]

                # Emissies voor een enkele trip
                hybrStadEmissions = np.zeros((len(hybrRouteStad), nET))
                hybrBuitenwegEmissions = np.zeros((len(hybrRouteBuitenweg), nET))
                hybrSnelwegEmissions = np.zeros((len(hybrRouteSnelweg), nET))
                for et in range(nET):
                    hybrStadEmissions[:, et] = (
                        distArray[hybrRouteStad] * emissionFacStad[et])
                    hybrBuitenwegEmissions[:, et] = (
                        distArray[hybrRouteBuitenweg] * emissionFacBuitenweg[et])
                    hybrSnelwegEmissions[:, et] = (
                        distArray[hybrRouteSnelweg] * emissionFacSnelweg[et])

            # Bereken en schrijf de emissies op de links
            for trip in range(len(currentTrips)):
                ct = int(currentTrips[trip, 6])

                if ct == 3:
                    linkTripsArray[hybrRoute] += 1
                else:
                    linkTripsArray[route] += 1

                # If combustion type is fuel or bio-fuel
                if ct in [0, 4]:
                    stadEmissionsTrip = stadEmissions
                    buitenwegEmissionsTrip = buitenwegEmissions
                    snelwegEmissionsTrip = snelwegEmissions

                    for et in range(nET):
                        linkEmissionsArray[routeStad, et] += (stadEmissionsTrip[:, et])
                        linkEmissionsArray[routeBuitenweg, et] += (buitenwegEmissionsTrip[:, et])
                        linkEmissionsArray[routeSnelweg, et] += (snelwegEmissionsTrip[:, et])

                    # CO2-emissions for the current trip
                    parcelTripsCO2[currentTrips[trip, -1]] = (
                        np.sum(stadEmissionsTrip[:, indexCO2]) +
                        np.sum(buitenwegEmissionsTrip[:, indexCO2]) +
                        np.sum(snelwegEmissionsTrip[:, indexCO2]))

                # If hybrid combustion
                elif ct == 3:
                    stadEmissionsTrip = hybrStadEmissions
                    buitenwegEmissionsTrip = hybrBuitenwegEmissions
                    snelwegEmissionsTrip = hybrSnelwegEmissions

                    stadEmissionsTrip[hybrZEZstad] = 0
                    buitenwegEmissionsTrip[hybrZEZbuitenweg] = 0
                    snelwegEmissionsTrip[hybrZEZsnelweg] = 0

                    for et in range(nET):
                        linkEmissionsArray[hybrRouteStad, et] += (stadEmissionsTrip[:, et])
                        linkEmissionsArray[hybrRouteBuitenweg, et] += (
                            buitenwegEmissionsTrip[:, et])
                        linkEmissionsArray[hybrRouteSnelweg, et] += (snelwegEmissionsTrip[:, et])

                    # CO2-emissions for the current trip
                    parcelTripsCO2[currentTrips[trip, 1]] = (
                        np.sum(stadEmissionsTrip[:, indexCO2]) +
                        np.sum(buitenwegEmissionsTrip[:, indexCO2]) +
                        np.sum(snelwegEmissionsTrip[:, indexCO2]))

                else:
                    parcelTripsCO2[currentTrips[trip, -1]] = 0

    del prevVan

    linkEmissionsArray /= 1000

    return linkTripsArray, linkEmissionsArray, parcelTripsCO2


def calc_intensities_vans(
    doHybrRoutes: bool,
    probsCombustionVans: np.ndarray,
    zoneIsZEZ: np.ndarray, ZEZarray: np.ndarray,
    numLinks: int, linkDict: np.ndarray,
    indices: list
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bereken de intensiteiten en emissies van bestelautoritten voor
    bouw en service.

    Args:
        doHybrRoutes (bool): _description_
        probsCombustionVans (np.ndarray): _description_
        zoneIsZEZ (np.ndarray): Per zone of het een ZE-zone is.
        ZEZarray (np.ndarray): Per link of deze in een ZE-zone ligt.
        numLinks (int): _description_
        linkDict (np.ndarray): _description_
        indices (list): _description_

    Returns:
        tuple: Met daarin:
            - np.ndarray: De intensiteiten op elke link (gereden met conventionele aandrijving)
            - np.ndarray: De intensiteiten op elke link (gereden met schone aandrijving)
    """
    vanTripsService = indices[1]
    vanTripsConstruction = indices[2]

    prevVan = indices[3]
    if doHybrRoutes:
        prevVanHybr = indices[4]

    indices = indices[0]

    linkVanTripsArray = np.zeros((numLinks, 2))
    if doHybrRoutes:
        linkCleanVanTripsArray = np.zeros((numLinks, 2))
        probFuel = np.sum(probsCombustionVans[[0, 4]])
        probHybr = probsCombustionVans[3]
        probClean = np.sum(probsCombustionVans[[1, 2]])

    for i in range(len(indices)):
        print('\t\tOrigin ' + '{:>5}'.format(indices[i] + 1), end='\r')

        # Voor welke bestemmingen routes zoeken
        destZones = np.where(
            (vanTripsService[i, :] > 0) | (vanTripsConstruction[i, :] > 0))[0]

        # Dijkstra routezoekalgoritme toepassen
        routes = [get_route(i, destZone, prevVan, linkDict) for destZone in destZones]

        for j, destZone in enumerate(destZones):
            nTripsService = vanTripsService[i, destZone]
            nTripsConstruction = vanTripsConstruction[i, destZone]

            route = routes[j]

            if doHybrRoutes:

                if zoneIsZEZ[indices[i]] or zoneIsZEZ[destZone]:
                    hybrRoute = get_route(i, destZone, prevVanHybr, linkDict)

                    # Conventionele brandstof rijdt gewone route (met penalty in ZEZ) en stoot
                    # overal emissies uit
                    if probFuel > 0:
                        linkVanTripsArray[route, 0] += nTripsService * probFuel
                        linkVanTripsArray[route, 1] += nTripsConstruction * probFuel

                    # Hybride rijdt andere route (zonder penalty in ZEZ) en stoot alleen buiten
                    # de ZEZ emissies uit
                    hybrRouteZEZ = hybrRoute[ZEZarray[hybrRoute]]
                    hybrRouteNotZEZ = hybrRoute[~ZEZarray[hybrRoute]]
                    linkCleanVanTripsArray[hybrRouteZEZ, 0] += nTripsService * probHybr
                    linkCleanVanTripsArray[hybrRouteZEZ, 1] += nTripsConstruction * probHybr
                    linkVanTripsArray[hybrRouteNotZEZ, 0] += nTripsService * probHybr
                    linkVanTripsArray[hybrRouteNotZEZ, 1] += nTripsConstruction * probHybr

                    # Schoon voertuig rijdt andere (zonder penalty in ZEZ) en stoot niks uit
                    linkCleanVanTripsArray[hybrRoute, 0] += nTripsService * probClean
                    linkCleanVanTripsArray[hybrRoute, 1] += nTripsConstruction * probClean

                    continue

            # Het aantal ritten gemaakt op elke link
            if nTripsService > 0:
                linkVanTripsArray[route, 0] += nTripsService
            if nTripsConstruction > 0:
                linkVanTripsArray[route, 1] += nTripsConstruction

    del prevVan, vanTripsService, vanTripsConstruction

    if doHybrRoutes:
        del prevVanHybr
        return linkVanTripsArray, linkCleanVanTripsArray

    return linkVanTripsArray, None


def write_emissions_vans_into_links(
    links: pd.DataFrame,
    linkVanTripsArray: np.ndarray, linkCleanVanTripsArray: np.ndarray,
    distArray: np.ndarray,
    stadArray: np.ndarray, buitenwegArray: np.ndarray, snelwegArray: np.ndarray,
    emissionFacs: Dict[str, np.ndarray],
    etDict: dict, numET: int
) -> pd.DataFrame:
    """
    Zet de emissies en intensiteiten voor service/bouw in de links DataFrame.

    Args:
        links (pd.DataFrame): _description_
        linkVanTripsArray (np.ndarray): _description_
        linkCleanVanTripsArray (np.ndarray): _description_
        distArray (np.ndarray): _description_
        stadArray (np.ndarray): _description_
        buitenwegArray (np.ndarray): _description_
        snelwegArray (np.ndarray): _description_
        emissionFacs (Dict[str, np.ndarray]): _description_
        etDict (dict): _description_
        numET (int): _description_

    Returns:
        pd.DataFrame: De links met nu ook de intensiteiten en emissies voor bouw/service.
    """
    # Intensiteiten
    for col in ['n_trips_van_service', 'n_trips_vt8', 'n_trips']:
        links.loc[:, col] += linkVanTripsArray[:, 0]
        links.loc[:, col] += linkCleanVanTripsArray[:, 0]

    for col in ['n_trips_van_construction', 'n_trips_vt8', 'n_trips']:
        links.loc[:, col] += linkVanTripsArray[:, 1]
        links.loc[:, col] += linkCleanVanTripsArray[:, 1]

    vt = 8
    capUt = 0.5  # Assume half of loading capacity used

    # Generieke emissiefactoren die toepasbaar zijn voor alle bouw/service bestelauto's
    emissionFacStad = np.array([
        emissionFacs['stad_leeg'][vt, et] + capUt * (
            emissionFacs['stad_vol'][vt, et] - emissionFacs['stad_leeg'][vt, et])
        for et in range(numET)])
    emissionFacBuitenweg = np.array([
        emissionFacs['buitenweg_leeg'][vt, et] + capUt * (
            emissionFacs['buitenweg_vol'][vt, et] - emissionFacs['buitenweg_leeg'][vt, et])
        for et in range(numET)])
    emissionFacSnelweg = np.array([
        emissionFacs['snelweg_leeg'][vt, et] + capUt * (
            emissionFacs['snelweg_vol'][vt, et] - emissionFacs['snelweg_leeg'][vt, et])
        for et in range(numET)])

    # Bereken de emissies o.b.v. de intensiteiten en emissiefactoren
    for et in range(numET):
        links.loc[stadArray, etDict[et].lower() + '_' + 'van_service'] = (
            linkVanTripsArray[stadArray, 0] * distArray[stadArray] *
            emissionFacStad[et] / 1000)
        links.loc[buitenwegArray, etDict[et].lower() + '_' + 'van_service'] = (
            linkVanTripsArray[buitenwegArray, 0] * distArray[buitenwegArray] *
            emissionFacBuitenweg[et] / 1000)
        links.loc[snelwegArray, etDict[et].lower() + '_' + 'van_service'] = (
            linkVanTripsArray[snelwegArray, 0] * distArray[snelwegArray] *
            emissionFacSnelweg[et] / 1000)

        links.loc[stadArray, etDict[et].lower() + '_' + 'van_construction'] = (
            linkVanTripsArray[stadArray, 1] * distArray[stadArray] *
            emissionFacStad[et] / 1000)
        links.loc[buitenwegArray, etDict[et].lower() + '_' + 'van_construction'] = (
            linkVanTripsArray[buitenwegArray, 1] * distArray[buitenwegArray] *
            emissionFacBuitenweg[et] / 1000)
        links.loc[snelwegArray, etDict[et].lower() + '_' + 'van_construction'] = (
            linkVanTripsArray[snelwegArray, 1] * distArray[snelwegArray] *
            emissionFacSnelweg[et] / 1000)

        links[etDict[et].lower()] += links[etDict[et].lower() + '_' + 'van_service']
        links[etDict[et].lower()] += links[etDict[et].lower() + '_' + 'van_construction']

    return links


def put_emissions_into_tours(
    settings: Settings,
    tripsCO2: dict,
    tripsCO2_NL: dict,
    tripsDist: dict,
    tripsDist_NL: dict,
    parcelTripsCO2: dict,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Zet emissies (en afstanden) in 'tours' en 'parcelTours'.

    Args:
        settings (Settings): _description_
        tripsCO2 (dict): _description_
        tripsCO2_NL (dict): _description_
        tripsDist (dict): _description_
        tripsDist_NL (dict): _description_
        parcelTripsCO2 (dict): _description_
        logger (logging.Logger): _description_

    Returns:
        tuple: Met daarin:
            - pd.DataFrame: tours verrijkt met CO2 en afstand
            - pd.DataFrame: parcelTours verrijkt met CO2
    """
    # Inlezen vracht tours
    tours = pd.read_csv(settings.get_path("Tours"), sep='\t')

    # Koppelen CO2 en kilometers vracht
    toursCO2 = np.zeros(tours.shape[0], dtype=float)
    toursCO2_NL = np.zeros(tours.shape[0], dtype=float)
    toursDist = np.zeros(tours.shape[0], dtype=float)
    toursDist_NL = np.zeros(tours.shape[0], dtype=float)

    keyErrorRows = []

    for i in tours.index:
        try:
            toursCO2[i] = np.round(tripsCO2[i] / 1000, 3)
            toursCO2_NL[i] = np.round(tripsCO2_NL[i] / 1000, 3)
            toursDist[i] = tripsDist[i]
            toursDist_NL[i] = tripsDist_NL[i]
        except KeyError:
            keyErrorRows.append(i)

    if keyErrorRows:
        logger.warning(
            f"Geen emissies of afstand bepaald voor de volgende rijen in Tours: {keyErrorRows}.")

    tours['co2__gram'] = toursCO2
    tours['co2_nl__gram'] = toursCO2_NL
    tours['distance__kmeter'] = toursDist
    tours['distance_nl__kmeter'] = toursDist_NL

    # Inlezen pakket tours
    parcelTours = pd.read_csv(settings.get_path("PakketRondritten"), sep='\t')

    # Koppelen CO2 en kilometers pakket
    parcelToursCO2 = np.zeros(parcelTours.shape[0], dtype=float)

    keyErrorRows = []

    for i in parcelTours.index:
        try:
            parcelToursCO2[i] = np.round(parcelTripsCO2[i] / 1000, 3)
        except KeyError:
            keyErrorRows.append(i)

    if keyErrorRows:
        logger.warning(
            "Geen emissies of afstand bepaald voor de volgende rijen in ParcelTours: " +
            f"{keyErrorRows}.")

    parcelTours['co2__gram'] = parcelToursCO2

    return tours, parcelTours


def put_emissions_into_shipments(
    settings: Settings,
    tours: pd.DataFrame,
    zones: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Zet emissies in 'shipments'.

    Args:
        settings (Settings): _description_
        tours (pd.DataFrame): _description_
        zones (pd.DataFrame): _description_
        logger (logging.Logger): _description_

    Returns:
        pd.DataFrame: De zendingen met daaraan de CO2-waarde gekoppeld.
    """
    # Calculate emissions at the tour level instead of trip level
    tours['tour_id'] = [
        str(tours.at[i, 'carrier_id']) + '_' + str(tours.at[i, 'tour_id'])
        for i in tours.index]
    toursCO2 = pd.pivot_table(
        tours, values=['co2__gram'], index=['tour_id'], aggfunc=np.sum)
    tourIDDict = dict(np.transpose(np.vstack((toursCO2.index, np.arange(len(toursCO2))))))
    toursCO2 = np.array(toursCO2['co2__gram'])

    # Read the shipments
    shipments = pd.read_csv(settings.get_path("ZendingenNaTourformatie"), sep='\t')
    shipments = shipments.sort_values('tour_id')
    shipments.index = np.arange(len(shipments))

    # For each tour, which shipments belong to it
    tourIDs = [tourIDDict[x] for x in shipments['tour_id']]
    shipIDs = []
    currentShipIDs = [0]
    for i in range(1, len(shipments)):
        if tourIDs[i - 1] == tourIDs[i]:
            currentShipIDs.append(i)
        else:
            shipIDs.append(currentShipIDs.copy())
            currentShipIDs = [i]
    shipIDs.append(currentShipIDs.copy())

    # Network distance of each shipment
    skimDistanceFreight = read_mtx(settings.get_path("SkimVrachtAfstandNRM"))
    skimDistanceVan = read_mtx(settings.get_path("SkimBestelAfstandNRM"))

    # Dit stukje is om te voorkomen dat grote skimbestanden moeten worden opgenomen in Git
    # om tests met de rekenmodules te kunnen uitvoeren
    if len(skimDistanceFreight) == 0 or len(skimDistanceVan) == 0:
        logger.warning(
            "De skimbestanden bevatten geen data. " +
            "Er worden nu skims berekend o.b.v. hemelsbrede afstand.")
        skimDistanceFreight = get_euclidean_skim(
            zones['x_rd__meter'].values, zones['y_rd__meter'].values)
        skimDistanceVan = skimDistanceFreight.copy()

    numZones = int(len(skimDistanceFreight) ** 0.5)
    origNRM = np.array(shipments['orig_nrm'], dtype=int)
    destNRM = np.array(shipments['dest_nrm'], dtype=int)

    shipDist = []
    for i in range(len(shipments)):
        if shipments.at[i, 'vehicle_type_lwm'] >= 8:
            shipDist.append(skimDistanceVan[(origNRM[i] - 1) * numZones + (destNRM[i] - 1)])
        else:
            shipDist.append(skimDistanceFreight[(origNRM[i] - 1) * numZones + (destNRM[i] - 1)])
    shipDist = np.array(shipDist)

    # Divide CO2 of each tour over its shipments based on distance
    shipCO2 = np.zeros(len(shipments))
    for tourID in np.unique(tourIDs):
        currentDists = shipDist[shipIDs[tourID]]
        currentCO2 = toursCO2[tourID]
        if np.sum(currentDists) == 0:
            shipCO2[shipIDs[tourID]] = 0.0
        else:
            shipCO2[shipIDs[tourID]] = currentDists / np.sum(currentDists) * currentCO2
    shipments['co2__gram'] = shipCO2

    # Sorteren op shipment ID
    shipments = shipments.sort_values('shipment_id')
    shipments.index = np.arange(len(shipments))

    return shipments


def write_links_to_shape(
    path: str,
    links: pd.DataFrame,
    nodes: pd.DataFrame,
    nodeDict: dict,
    invNodeDict: dict,
    intensityFields: list
):
    """
    Schrijf het geladen netwerk naar een shapefile in de uitvoerfolder.

    Args:
        path (str): _description_
        links (pd.DataFrame): _description_
        nodes (pd.DataFrame): _description_
        nodeDict (dict): _description_
        invNodeDict (dict): _description_
        intensityFields (list): _description_
    """
    # Set travel times of connectors at 0 for in
    # the output network shape
    links.loc[links['link_type'] == 99, 'time_freight__s'] = 0.0
    links.loc[links['link_type'] == 99, 'time_van__s'] = 0.0

    links['node_a'] = [nodeDict[x] for x in links['node_a']]
    links['node_b'] = [nodeDict[x] for x in links['node_b']]

    # Initialize shapefile fields
    w = shp.Writer(path)
    w.field('link',   'N', size=7, decimal=0)
    w.field('node_a', 'N', size=7, decimal=0)
    w.field('node_b', 'N', size=7, decimal=0)
    w.field('speed_ff', 'N', size=3, decimal=0)
    w.field('speed',    'N', size=3, decimal=0)
    w.field('capacity',    'N', size=5, decimal=0)
    w.field('link_type',   'N', size=2, decimal=0)
    w.field('distance',    'N', size=8, decimal=0)
    w.field('user_type',   'N', size=1, decimal=0)
    w.field('charge_type', 'N', size=1, decimal=0)
    w.field('nrm',         'N', size=1, decimal=0)

    for col in ['PA_ETM', 'MVR_ETM', 'ZVR_ETM', 'APA_ETM', 'AMVR_ETM', 'AZVR_ETM']:
        if col in links.columns:
            size = len(str(links[col].astype(int).max()))
            w.field(col, 'N', size=size, decimal=0)

    w.field('zez', 'N', size=1, decimal=0)
    w.field('nl',  'N', size=1, decimal=0)
    w.field('t_frg__s',   'N', size=7, decimal=1)
    w.field('t_van__s',   'N', size=7, decimal=1)
    w.field('c_frg__eur', 'N', size=8, decimal=3)
    w.field('c_van__eur', 'N', size=8, decimal=3)

    for field in intensityFields:
        decimal = 5 if 'van' in field or 'n_trips' not in field else 1
        size = len(str(links[field].astype(int).max())) + decimal
        w.field(field.replace('trips_', ''), 'N', size=size, decimal=decimal)

    nodesX = np.array(nodes['x_rd__meter'], dtype=int)
    nodesY = np.array(nodes['y_rd__meter'], dtype=int)
    linksA = [invNodeDict[x] for x in links['node_a']]
    linksB = [invNodeDict[x] for x in links['node_b']]

    dbfData = np.array(links, dtype=object)
    numLinks = dbfData.shape[0]

    for i in range(numLinks):
        # Add geometry
        line = []
        line.append([nodesX[linksA[i]], nodesY[linksA[i]]])
        line.append([nodesX[linksB[i]], nodesY[linksB[i]]])
        w.line([line])

        # Add data fields
        w.record(*dbfData[i, :])

        if i % int(round(numLinks / 100)) == 0:
            print('\t' + str(round(i / numLinks * 100, 1)) + '%', end='\r')

    print('\t100.0%', end='\r')

    w.close()
