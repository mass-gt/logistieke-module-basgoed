import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import shapefile as shp
from calculation.common.data import numpy_pivot_table
from calculation.ship.types import (ContainerStatistics, CostFigure,
                                    Probabilities, Routes)
from numba import float64, njit
from settings import Settings


def get_truck_capacities(settings: Settings, numVT: int) -> np.ndarray:
    """
    Returns truck capacities (in tonnes).
    """
    carryingCapacity = -1 * np.ones(numVT)
    for row in pd.read_csv(settings.get_path("LaadvermogensLWM"), sep='\t').to_dict('records'):
        vt = int(row['vehicle_type_lwm'])
        weightTon = float(row['weight__kgram']) / 1000
        carryingCapacity[vt] = weightTon

    if np.any(carryingCapacity == -1):
        raise Exception('Er zijn missende voertuigtypes in "LaadvermogensLWM".')

    return carryingCapacity


def get_employment(settings: Settings) -> Dict[str, int]:
    """
    Returns employment sectors.
    """
    return {
        name: i for i, name in enumerate(
            'empl_' + empl.lower() for empl in settings.dimensions.employment_category)}


def get_make_use(
    settings: Settings,
    employmentDict: Dict[str, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Haal de maak-gebruik tabellen op en maak er een kansverdeling van.
    """
    # Conversietabel voor goederengroepen in make-use tabel
    convGG = np.array(
        pd.read_csv(settings.get_path("Goederengroep"), sep='\t'), dtype=int)
    convGG = dict((convGG[i, :]) for i in range(len(convGG)))

    # Conversietabel voor sectoren in make-use tabel
    convSector = np.array(pd.read_csv(settings.get_path("KoppeltabelSectoren"), sep='\t'))
    convSector[:, 1] = [
        employmentDict['empl_' + sector.lower()] for sector in convSector[:, 1]]
    convSector = np.array(convSector, dtype=int)
    convSector = dict(tuple(convSector[i, :]) for i in range(len(convSector)))

    # Import make-use table BasGoed
    makeUseTable = np.array(pd.read_csv(settings.get_path("MakeUseEerstJr"), sep='\t'))

    # Convert the goods types and sectors
    for i in range(len(makeUseTable)):
        makeUseTable[i, 0] = convGG.get(makeUseTable[i, 0], -1)
        makeUseTable[i, 1] = convSector.get(makeUseTable[i, 1], -1)

    # Remove rows with no known conversion for the goods type or sector
    makeUseTable = makeUseTable[
        (makeUseTable[:, 0] >= 0) & (makeUseTable[:, 1] >= 0), :]

    # Create make and use distribution tables (by commodity and NRM-employment-sector)
    numEmployment = int(max(makeUseTable[:, 1])) + 1
    numGG = int(max(makeUseTable[:, 0]))

    makeDistribution = np.zeros((numGG, numEmployment))
    useDistribution = np.zeros((numGG, numEmployment))

    for i in range(len(makeUseTable)):
        gg = int(makeUseTable[i, 0])
        empl = int(makeUseTable[i, 1])
        makeWeight = makeUseTable[i, 2]
        useWeight = makeUseTable[i, 3]

        makeDistribution[gg - 1, empl] += makeWeight
        useDistribution[gg - 1, empl] += useWeight

    for gg in range(numGG):
        sumMake = np.sum(makeDistribution[gg, :])
        sumUse = np.sum(useDistribution[gg, :])

        if sumMake > 0:
            makeDistribution[gg, :] /= sumMake
        else:
            makeDistribution[gg, :] = (np.ones(numEmployment) / numEmployment)

        if sumUse > 0:
            useDistribution[gg, :] /= sumUse
        else:
            useDistribution[gg, :] = (np.ones(numEmployment) / numEmployment)

    return makeDistribution, useDistribution


def get_gg_to_nstr(settings: Settings) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bepaal de cumulatieve kansverdeling per combinatie van
    BasGoed-goederengroep en NSTR-goederengroep voor container en voor niet-container.

    Args:
        settings (Settings): _description_

    Returns:
        tuple: Met daarin:
            - np.ndarray: Cumulatieve kansen GG naar NSTR (niet-container)
            - np.ndarray: Cumulatieve kansen GG naar NSTR (container)
    """
    data = pd.read_csv(
        settings.get_path('Koppeltabel_Commodity_NSTR'), sep='\t').to_dict('records')

    numGG = max([row['commodity'] for row in data])
    numNSTR = max([row['nstr'] for row in data]) + 1

    cumProbsCommodityToNSTR = (
        np.zeros((numGG, numNSTR), dtype=float),
        np.zeros((numGG, numNSTR), dtype=float))

    for row in data:
        gg = row['commodity']
        nstr = row['nstr']
        container = row['container']
        weight = row['weight__ton']

        cumProbsCommodityToNSTR[container][gg - 1, nstr] = weight

    for gg in range(numGG):
        for container in [0, 1]:
            if np.sum(cumProbsCommodityToNSTR[container][gg, :]) == 0:
                cumProbsCommodityToNSTR[container][gg, :] = np.ones(numNSTR)

            cumProbsCommodityToNSTR[container][gg, :] = np.cumsum(
                cumProbsCommodityToNSTR[container][gg, :])

            cumProbsCommodityToNSTR[container][gg, :] /= cumProbsCommodityToNSTR[container][gg, -1]

    return cumProbsCommodityToNSTR


def get_gg_to_ls(settings: Settings) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bepaal de  kansverdeling (niet cumulatief) per combinatie van
    BasGoed-goederengroep en logistiek segment voor container en voor
    niet-container.

    Args:
        settings (Settings): _description_

    Returns:
        tuple: Met daarin:
            - np.ndarray: Cumulatieve kansen GG naar LS (niet-container)
            - np.ndarray: Cumulatieve kansen GG naar LS (container)
    """
    data = pd.read_csv(
        settings.get_path('Koppeltabel_Commodity_LogisticSegment'), sep='\t').to_dict('records')

    numGG = max([row['commodity'] for row in data])
    numLS = max([row['logistic_segment'] for row in data]) + 1

    probsCommodityToLS = (
        np.zeros((numGG, numLS), dtype=float),
        np.zeros((numGG, numLS), dtype=float))

    for row in data:
        gg = row['commodity']
        ls = row['logistic_segment']
        container = row['container']
        weight = row['weight__ton']

        probsCommodityToLS[container][gg - 1, ls] = weight

    for gg in range(numGG):
        for container in [0, 1]:
            if np.sum(probsCommodityToLS[container][gg, :]) > 0:
                probsCommodityToLS[container][gg, :] /= np.sum(
                    probsCommodityToLS[container][gg, :])
            else:
                probsCommodityToLS[container][gg, :] = (np.ones(numLS) / numLS)

    return probsCommodityToLS


def get_nrm_zones(settings: Settings) -> pd.DataFrame:
    """
    Returns the NRM-mother-zones
    """
    zones = pd.read_csv(settings.get_path("ZonesNRM"), sep='\t')
    zones.index = zones['zone_nrm']
    return zones


def get_dict_to_nrm(
    settings: Settings,
    zones: pd.DataFrame
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Maak een dictionary van BG-zone naar NRM-zone en van CK-zone naar NRM-zone.

    Args:
        settings (Settings): _description_
        zones (pd.DataFrame): _description_
        BGwithOverlapNRM (set): _description_

    Returns:
        tuple: Met daarin:
            - dict: Koppeling BG-zone naar NRM-zone.
            - dict: Koppeling CK-zone naar NRM-zone.
    """
    BGwithOverlapNRM = np.unique(zones['zone_bg'])

    zonesBG = pd.read_csv(settings.get_path("Koppeltabel_ZONEBG"), sep='\t')
    zonesBG = zonesBG.astype(int)
    zonesBG.index = zonesBG['zone_bg']

    dictBGtoNRM = {}
    dictCKtoNRM = {}

    for zoneCK in np.unique(zones['zone_ck']):
        dictCKtoNRM[zoneCK] = np.array(zones.loc[zones['zone_ck'] == zoneCK, 'zone_nrm'])

    for i in zonesBG.index:
        zoneBG = i
        zoneCK = zonesBG.at[i, 'zone_ck']

        if i in BGwithOverlapNRM:
            dictBGtoNRM[zoneBG] = np.array(zones.loc[zones['zone_bg'] == zoneBG, 'zone_nrm'])
        else:
            dictBGtoNRM[zoneBG] = np.array([zonesBG.at[i, 'zone_nrm']])

        if dictCKtoNRM.get(zoneCK, None) is None:
            dictCKtoNRM[zoneCK] = np.array([zonesBG.at[i, 'zone_nrm']])

    return dictBGtoNRM, dictCKtoNRM


def get_dict_to_bg(
    settings: Settings, zones: pd.DataFrame
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Maak een dictionary van CK-zone naar BG-zone en van NRM-zone naar BG-zone.

    Args:
        settings (Settings): _description_
        zones (pd.DataFrame): _description_

    Returns:
        tuple: Met daarin:
            - dict: Koppeling CK-zone naar BG-zone.
            - dict: Koppeling NRM-zone naar BG-zone.
    """
    zonesBG = pd.read_csv(settings.get_path("Koppeltabel_ZONEBG"), sep='\t')
    zonesBG = zonesBG.astype(int)

    maxDutchZoneBG = settings.zones.maximum_dutch

    dictCKtoBG = {}
    dictNRMtoBG = {}

    for i in zones.index:
        zoneCK = zones.at[i, 'zone_ck']
        zoneNRM = zones.at[i, 'zone_nrm']
        zoneBG = zones.at[i, 'zone_bg']
        dictCKtoBG[zoneCK] = zoneBG
        dictNRMtoBG[zoneNRM] = zoneBG

    for i in zonesBG.index:
        zoneCK = zonesBG.at[i, 'zone_ck']
        zoneBG = zonesBG.at[i, 'zone_bg']

        if zoneBG > maxDutchZoneBG:
            dictCKtoBG[zoneCK] = zoneBG

    return dictCKtoBG, dictNRMtoBG


def get_distribution_centers(settings: Settings, zones: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a dataframe with distribution centers,
    based on zones that have a surface greater than 0.
    """
    distrCenters = zones[['zone_bg', 'zone_ck', 'zone_nrm', 'surface_distr__meter2']]
    distrCenters = distrCenters[distrCenters['surface_distr__meter2'] > 0]
    distrCenters['employment'] = (
        distrCenters['surface_distr__meter2'] * settings.module.empl_per_m2_dc)
    distrCenters.index = np.arange(len(distrCenters))
    return distrCenters


# def get_terminals(settings: Settings, dictCKtoBG: Dict[int, int]) -> Terminals:
#     terminals_ck = get_terminals_ck()

#     return Terminals()


def get_terminals_ck(settings: Settings, dictCKtoBG: Dict[int, int]) -> pd.DataFrame:
    """
    Returns ck-terminals from file.
    Appends BasGoed zone to each terminal.

    Args:
        settings (Settings): _description_
        dictCKtoBG (Dict[int, int]): _description_

    Returns:
        DataFrame: _description_
    """
    terminals = pd.read_csv(settings.get_path("Terminals"), sep='\t')
    terminals['zone_bg'] = [dictCKtoBG.get(x, None) for x in terminals['zone_ck']]
    return terminals


def get_terminals_ms(settings: Settings) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Maak DataFrames voor spoor- en binnenvaarterminals o.b.v. de NEMO- en BIVAS-knopen.

    Args:
        settings (Settings): _description_

    Returns:
        tuple: Met daarin:
            - pd.DataFrame: De spoorterminals.
            - pd.DataFrame: De binnenvaartterminals.
    """
    # Importeer de spoorterminals voor niet-container
    terminalsRail = pd.read_csv(settings.get_path("KnopenSpoor"), sep='\t').fillna(0)
    terminalsRail = terminalsRail[
        (terminalsRail['zone_nrm'] != 0) &
        (terminalsRail['zone_bg'] != 0)]
    terminalsRail = terminalsRail[['terminal', 'zone_bg', 'zone_nrm']]
    terminalsRail.index = np.arange(len(terminalsRail))

    # Importeer de binnenvaartterminals voor niet-container
    terminalsWater = pd.read_csv(settings.get_path("KnopenBinnenvaart"), sep='\t').fillna(0)
    terminalsWater = terminalsWater[
        (terminalsWater['zone_nrm'] != 0) &
        (terminalsWater['zone_bg'] != 0)]
    terminalsWater = terminalsWater[['terminal', 'zone_bg', 'zone_nrm']]
    terminalsWater.index = np.arange(len(terminalsWater))

    return terminalsRail, terminalsWater


def update_zones_by_terminals_and_distribution_centers(
    zones: pd.DataFrame, distribution_centers: pd.DataFrame, terminals_ck: pd.DataFrame
):
    """
    Prevents double counting of flows to/from TTs.
    Prevents double counting of flows to/from DCs.
    """
    zones.loc[
        [x for x in terminals_ck['zone_nrm'].values if not pd.isna(x)], 'empl_industrie'
    ] = 0.0

    # Prevent double counting of flows to/from DCs
    for i in range(len(distribution_centers)):
        zones.loc[
            distribution_centers.at[i, 'zone_nrm'], 'empl_industrie'
        ] -= distribution_centers.at[i, 'employment']
    zones['empl_industrie'] = [max(0.0, x) for x in zones['empl_industrie'].values]


def get_fractions(settings: Settings) -> pd.DataFrame:
    """
    Returns load fractions that travel via, from or two a distribution center.

    Args:
        settings (Settings): _description_

    Returns:
        pd.DataFrame: _description_
    """
    fractionsDC = pd.read_csv(settings.get_path("FractiesDC"), sep='\t')
    fractionsDC.index = fractionsDC['commodity']
    return fractionsDC


def get_tonnes_ms(
    settings: Settings,
    probsCommodityToLS: Tuple[np.ndarray],
    dimGG: List[int], dimLS: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Haal de tonnen voor wegvervoer uit de Modal Split module op en koppel
    er een logistiek segment aan.

    Args:
        settings (Settings): _description_
        probsCommodityToLS (Tuple[np.ndarray]): _description_
        year (str): _description_
        dimGG (List[int]): _description_
        dimLS (List[str]): _description_

    Returns:
        tuple: Met daarin:
            - pd.DataFrame: De wegtonnages per H-B-GG.
            - pd.DataFrame: De spoortonnages per H-B-GG.
            - pd.DataFrame: De binnenvaarttonnages per H-B-GG.
    """
    dimCols = ['origin_bg', 'destination_bg', 'commodity', 'logistic_segment']
    weightCols = [f'{mode}_weight__ton' for mode in ['road', 'rail', 'water']]

    # Tonnen van de goederengroepen onder elkaar plakken
    tonnes = pd.read_csv(
        settings.get_path("NietContainerTonnen", commodity=str(dimGG[0])), sep='\t')
    tonnes['commodity'] = dimGG[0]

    if len(dimGG) > 1:
        for gg in dimGG[1:]:
            tonnesGG = pd.read_csv(
                settings.get_path("NietContainerTonnen", commodity=str(gg)), sep='\t')
            tonnesGG['commodity'] = gg
            tonnes = pd.concat([tonnes, tonnesGG.copy()])

    tonnes = tonnes.sort_values(by=['origin_bg', 'destination_bg', 'commodity'])
    tonnes.index = np.arange(len(tonnes))

    # Terugbrengen van jaar- naar dagtonnage
    tonnes[weightCols] /= settings.module.year_factor_goods

    # Spreid tonnen niet-container over logistieke segmenten
    numLS = len(dimLS) - 1
    tonnesLS = np.zeros((numLS * len(tonnes), 7), dtype=float)

    row = 0
    for i in tonnes.index:
        orig = tonnes.at[i, 'origin_bg']
        dest = tonnes.at[i, 'destination_bg']
        gg = tonnes.at[i, 'commodity']
        weights = [tonnes.at[i, col] for col in weightCols]

        for ls in range(numLS):
            fracLS = probsCommodityToLS[0][gg - 1, ls]

            if fracLS > 0:
                tonnesLS[row, 0] = orig
                tonnesLS[row, 1] = dest
                tonnesLS[row, 2] = gg
                tonnesLS[row, 3] = ls
                tonnesLS[row, 4:] = [weight * fracLS for weight in weights]

            row += 1

    # Verwijder rijen zonder tonnage
    tonnesLS = tonnesLS[np.sum(tonnesLS[:, 4:], axis=1) > 0]

    # Zet terug in DataFrame met juiste datatypes
    tonnes = pd.DataFrame(
        tonnesLS,
        columns=dimCols + weightCols)
    tonnes[dimCols] = tonnes[dimCols].astype(int)

    # Aparte DataFrame per modaliteit, zonder nulcellen
    tonnesByMode = [tonnes[dimCols + [weightCol]] for weightCol in weightCols]
    for i, weightCol in enumerate(weightCols):
        tonnesByMode[i] = tonnesByMode[i][tonnesByMode[i][weightCol] > 0]
        tonnesByMode[i].index = np.arange(len(tonnesByMode[i]))

    return tuple(tonnesByMode)


def get_tonnes_ck(
    settings: Settings,
    probsCommodityToLS: Tuple[np.ndarray],
    dimGG: list, dimLS: list
) -> pd.DataFrame:
    """
    Haal de tonnen voor wegvervoer uit de Container Keten module op en koppel
    er een logistiek segment aan.

    Args:
        settings (Settings): _description_
        probsCommodityToLS (Tuple[np.ndarray]): _description_
        forecast (bool): _description_
        dimGG (list): _description_
        dimLS (list): _description_

    Returns:
        pd.DataFrame: De tonnages per H-B-GG.
    """

    # Tonnen van de goederengroepen onder elkaar plakken
    tonnesCK = pd.read_csv(
        settings.get_path("ContainerTonnen", commodity=str(dimGG[0])), sep='\t')[
            ['origin_ck', 'destination_ck', 'weight__ton']]
    tonnesCK['commodity'] = dimGG[0]

    if len(dimGG) > 1:
        for gg in dimGG[1:]:
            tonnesGG = pd.read_csv(
                settings.get_path("ContainerTonnen", commodity=str(gg)), sep='\t')[
                    ['origin_ck', 'destination_ck', 'weight__ton']]
            tonnesGG['commodity'] = gg
            tonnesCK = pd.concat([tonnesCK, tonnesGG.copy()])

    tonnesCK = tonnesCK.sort_values(by=['origin_ck', 'destination_ck', 'commodity'])
    tonnesCK.index = np.arange(len(tonnesCK))

    # Terugbrengen van jaar- naar dagtonnage
    tonnesCK['weight__ton'] /= settings.module.year_factor_goods

    # Spreid tonnen container over logistieke segmenten
    numLS = len(dimLS) - 1
    tonnesLS = np.zeros((numLS * len(tonnesCK), 5), dtype=float)
    row = 0
    for i in tonnesCK.index:
        orig = tonnesCK.at[i, 'origin_ck']
        dest = tonnesCK.at[i, 'destination_ck']
        gg = tonnesCK.at[i, 'commodity']
        weight = tonnesCK.at[i, 'weight__ton']

        for ls in range(numLS):
            fracLS = probsCommodityToLS[1][gg - 1, ls]

            if fracLS > 0:
                tonnesLS[row, :] = (orig, dest, gg, ls, weight * fracLS)

            row += 1

    # Verwijder rijen zonder tonnage
    tonnesLS = tonnesLS[tonnesLS[:, 4] > 0]

    # Zet terug in DataFrame met juiste datatypes
    tonnesCK = pd.DataFrame(
        tonnesLS,
        columns=['origin_ck', 'destination_ck', 'commodity', 'logistic_segment', 'weight__ton'])
    intCols = ['origin_ck', 'destination_ck', 'commodity', 'logistic_segment']
    tonnesCK[intCols] = tonnesCK[intCols].astype(int)

    return tonnesCK


def get_routes_ck(
    settings: Settings, dimGG: List[int], terminals: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Haal de routes uit de Container Keten module op.

    Args:
        settings (Settings): _description_
        dimGG (List[int]): _description_
        terminals (pandas.DataFrame): _description_

    Returns:
        tuple: Met daarin:
            - pd.DataFrame: De directe routes.
            - pd.DataFrame: De routes naar een terminal toe.
            - pd.DataFrame: De routes vanaf een terminal.
    """
    routes = pd.read_csv(settings.get_path("RoutesCompleet", commodity=str(dimGG[0])), sep='\t')
    routes['commodity'] = 1

    if len(dimGG) > 1:
        for gg in dimGG[1:]:
            routesGG = pd.read_csv(
                settings.get_path("RoutesCompleet", commodity=str(gg)), sep='\t')
            routesGG['commodity'] = gg
            routes = pd.concat([routes, routesGG])

    routes.index = np.arange(len(routes))

    # Maak arrays met tonnages van alle routes zonder overslag, naar terminal en vanuit terminal
    routesDirect = []
    routesToTT = []
    routesFromTT = []

    origs = routes['origin_ck'].values
    dests = routes['destination_ck'].values
    weights = routes['weight__ton'].values
    origTTs = routes['terminal_1'].values
    destTTs = routes['terminal_2'].values
    origModes = routes['mode_1'].values
    destModes = routes['mode_2'].values
    ggs = routes['commodity'].values

    for i in routes.index:
        orig, dest = origs[i], dests[i]
        weight = weights[i]
        origTT, destTT = origTTs[i], destTTs[i]
        origMode, destMode = origModes[i], destModes[i]
        gg = ggs[i]

        if pd.isna(origTT) and pd.isna(destTT):
            routesDirect.append([orig, dest, gg, weight])
        if not pd.isna(origTT) and not pd.isna(destTT):
            routesToTT.append([orig, origTT, gg, weight])
            routesFromTT.append([destTT, dest, gg, weight])
        if not pd.isna(origTT) and pd.isna(destTT):
            if origMode == 1:
                routesToTT.append([orig, origTT, gg, weight])
            elif destMode == 1:
                routesFromTT.append([origTT, dest, gg, weight])

    routesDirect = pd.DataFrame(
        np.array(routesDirect),
        columns=['origin_ck', 'destination_ck', 'commodity', 'weight__ton'])
    routesToTT = pd.DataFrame(
        np.array(routesToTT),
        columns=['origin_ck', 'destination_terminal', 'commodity', 'weight__ton'])
    routesFromTT = pd.DataFrame(
        np.array(routesFromTT),
        columns=['origin_terminal', 'destination_ck', 'commodity', 'weight__ton'])

    # Koppel de CK-zones o.b.v. de terminal
    terminalToZone = dict(
        (terminals.at[i, 'terminal'], terminals.at[i, 'zone_ck']) for i in terminals.index)

    routesToTT['destination_ck'] = [
        terminalToZone[tt] for tt in routesToTT['destination_terminal'].values]
    routesFromTT['origin_ck'] = [
        terminalToZone[tt] for tt in routesFromTT['origin_terminal'].values]

    # Volgorde kolommen
    routesToTT = routesToTT[
        ['origin_ck', 'destination_ck', 'destination_terminal', 'commodity', 'weight__ton']]
    routesFromTT = routesFromTT[
        ['origin_ck', 'destination_ck', 'origin_terminal', 'commodity', 'weight__ton']]

    # Integers waar nodig
    intCols = ['origin_ck', 'destination_ck', 'commodity']
    routesDirect[intCols] = routesDirect[intCols].astype(int)
    routesToTT[intCols] = routesToTT[intCols].astype(int)
    routesFromTT[intCols] = routesFromTT[intCols].astype(int)
    routesToTT['destination_terminal'] = routesToTT['destination_terminal'].astype(int)
    routesFromTT['origin_terminal'] = routesFromTT['origin_terminal'].astype(int)

    return (routesDirect, routesToTT, routesFromTT)


def get_cost_freight_vt(settings: Settings) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lees de uur- en kilometerkosten voor vracht per voertuigtype in.

    Args:
        settings (Settings): _description_
        dimVT (list): _description_
        dimGG (list): _description_

    Returns:
        tuple: Met daarin:
            - np.ndarray: Uurkosten per goederengroep en voertuigtype
            - np.ndarray: Kilometerkosten per goederengroep en voertuigtype
    """
    dimVT = settings.dimensions.vehicle_type_lwm
    dimGG = settings.commodities

    numVT = len(dimVT) - 2

    costFreight = [None for vt in range(numVT)]

    rowVehicleType = {
        0: 'vrachtwagen',
        3: 'aanhanger',
        5: 'oplegger',
        6: 'speciaal',
        7: 'lzv',
        8: 'bestel'}

    for row, vehicle_type in rowVehicleType.items():
        costFreight[row] = np.array(pd.read_csv(
            settings.get_path("KostenKentallenVoertuigType", vehicle_type=vehicle_type),
            sep='\t',
            usecols=['commodity', 'distance_costs__eur_per_kmeter', 'time_costs__eur_per_hour']))

    costFreight[1] = costFreight[0]
    costFreight[2] = costFreight[0]
    costFreight[4] = costFreight[3]

    # Vul missende goederensoorten in
    numGG = max(dimGG)
    for vt in range(numVT):
        newArray = np.zeros((numGG, 3), dtype=float)

        for gg in range(numGG):
            if (gg + 1) not in costFreight[vt][:, 0]:
                newArray[gg, :] = [gg + 1, 0, 0]
            else:
                row = np.where(costFreight[vt][:, 0] == (gg + 1))[0]
                newArray[gg, :] = costFreight[vt][row, :]

        costFreight[vt] = newArray

    # Herstructureer als numpy array
    costFreightKm = np.zeros((numGG, numVT), dtype=float)
    costFreightHr = np.zeros((numGG, numVT), dtype=float)
    for gg in range(numGG):
        for vt in range(numVT):
            costFreightKm[gg, vt] = costFreight[vt][gg, 1]
            costFreightHr[gg, vt] = costFreight[vt][gg, 2]

    return costFreightHr, costFreightKm


def get_cost_freight_ck(settings: Settings) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lees de CK uur- en kilometerkosten voor vracht in (zonder voertuigtype als dimensie).

    Args:
        settings (Settings): _description_
        dimGG (list): _description_

    Returns:
        tuple: _description_
    """
    dimGG = settings.commodities
    numGG = max(dimGG)

    costFreightKm = np.zeros(numGG)
    costFreightHr = np.zeros(numGG)

    costRoadCK = pd.read_csv(settings.get_path("KostenKentallenCK", mode='road'), sep='\t')

    for row in costRoadCK.to_dict('records'):
        gg = int(row['commodity'])
        costFreightKm[gg - 1] = row['distance_costs__eur_per_kmeter']
        costFreightHr[gg - 1] = row['time_costs__eur_per_hour']

    return (costFreightHr, costFreightKm)


def get_cost_freight_ms(settings: Settings) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lees de MS uur- en kilometerkosten voor vracht in (zonder voertuigtype als dimensie).

    Args:
        settings (Settings): _description_
        dimGG (list): _description_

    Returns:
        tuple: _description_
    """
    dimGG = settings.commodities
    numGG = max(dimGG)

    costFreightKm = np.zeros(numGG)
    costFreightHr = np.zeros(numGG)

    costRoadMS = pd.read_csv(settings.get_path("KostenKentallenMS", mode='weg'), sep='\t')

    for row in costRoadMS.to_dict('records'):
        gg = int(row['commodity'])
        costFreightKm[gg - 1] = row['distance_costs__eur_per_kmeter']
        costFreightHr[gg - 1] = row['time_costs__eur_per_hour']

    return (costFreightHr, costFreightKm)


def get_coeffs_ship(
    settings: Settings,
    numGG: int, numSS: int, numVT: int,
) -> Dict[str, np.ndarray]:
    """
    Lees de coefficienten voor het keuzemodel voor zendingsgrootte en voertuigtype in.

    De sleutels zijn:
        - transport_costs
        - inventory_costs
        - from_dc
        - to_dc
        - const_shipment_size
        - const_vehicle_type_lwm_short_haul
        - const_vehicle_type_lwm_long_haul

    Args:
        settings (Settings): _description_
        numGG (int): Het aantal BasGoed-goederengroepen.
        numSS (int): Het aantal zendingsgrootte-klassen.
        numVT (int): Het aantal voertuigtypen.

    Returns:
        dict: De coefficienten.
    """
    coeffs = {}

    coeffs1 = pd.read_csv(
        settings.get_path("CoefficientenZendingen_Commodity"), sep='\t')
    coeffs2 = pd.read_csv(
        settings.get_path("CoefficientenZendingen_Commodity_ShipmentSize"), sep='\t')
    coeffs3 = pd.read_csv(
        settings.get_path("CoefficientenZendingen_Commodity_VehicleTypeLWM_LongHaul"), sep='\t')

    # Herformateer coefficienten met dimensie (commodity)
    for parameter in ['transport_costs', 'inventory_costs', 'from_dc', 'to_dc']:
        coeffs[parameter] = coeffs1.loc[coeffs1['parameter'] == parameter, 'value'].values

    # Herformateer coefficienten met dimensie (commodity, shipment_size)
    coeffs['const_shipment_size'] = np.zeros((numGG, numSS), dtype=float)
    for row in coeffs2.to_dict('records'):
        gg = row['commodity']
        ss = row['shipment_size']
        value = row['value']
        coeffs['const_shipment_size'][gg - 1, ss] = value

    # Herformateer coefficienten met dimensie (commodity, vehicle_type_lwm, long_haul)
    coeffs['const_vehicle_type_lwm_short_haul'] = np.zeros((numGG, numVT), dtype=float)
    coeffs['const_vehicle_type_lwm_long_haul'] = np.zeros((numGG, numVT), dtype=float)
    for row in coeffs3.to_dict('records'):
        gg = row['commodity']
        vt = row['vehicle_type_lwm']
        longHaul = row['long_haul']
        value = row['value']
        if longHaul:
            coeffs['const_vehicle_type_lwm_long_haul'][gg - 1, vt] = value
        else:
            coeffs['const_vehicle_type_lwm_short_haul'][gg - 1, vt] = value

    return coeffs


def get_coeffs_dist_decay(settings: Settings):
    coeffs = pd.read_csv(
        settings.get_path("CoefficientenAfstandsverval"), sep='\t', index_col=0
    ).to_dict()['value']
    return coeffs


def calculate_container_stats(settings: Settings) -> ContainerStatistics:
    """
    Calculates and returns container statistics
    """
    containerStats = pd.read_csv(
        settings.get_path("ContainerStatistieken"), sep='\t', index_col='commodity')

    return ContainerStatistics(
        np.array(containerStats['shipment_size_average__ton'])[:-1],
        containerStats.iloc[-1, 1] / np.sum(containerStats.iloc[:-1, -1]),
        containerStats.at[-1, 'shipment_size_average__ton']
    )


def calculate_container_probabilities(
    zones: pd.DataFrame,
    makeDistribution: np.ndarray,
    useDistribution: np.ndarray,
    dictCKtoNRM: dict,
    employmentNames: list,
    dimGG: list
) -> Probabilities:
    """
    Bepaal per CK-zone de kans van een van de NRM-zones erbinnen de producent of consument
    is van een zending.

    Args:
        zones (pd.DataFrame): _description_
        makeDistribution (np.ndarray): _description_
        useDistribution (np.ndarray): _description_
        dictCKtoNRM (dict): _description_
        employmentNames (list): _description_
        dimGG (list): _description_

    Returns:
        tuple: Met daarin:
            - dict: Kansen voor producent.
            - dict: Kansen voor consument.
    """
    numGG = max(dimGG)

    # Bepaal de kansen van NRM-zones als producent van een zending
    producerProbs = {}

    for zoneCK in dictCKtoNRM.keys():

        zonesNRM = dictCKtoNRM[zoneCK]
        jobsNRM = np.array(zones.loc[zonesNRM, employmentNames])

        producerProbs[zoneCK] = {}
        for gg in range(numGG):
            producerProbs[zoneCK][gg] = np.sum(makeDistribution[gg, :] * jobsNRM, axis=1)
            if np.sum(producerProbs[zoneCK][gg]) == 0:
                producerProbs[zoneCK][gg] = np.ones(len(producerProbs[zoneCK][gg]))

    # Bepaal de cumulatieve kansen van NRM-zones als consument van een zending
    consumerCumProbs = {}

    for zoneCK in dictCKtoNRM.keys():

        zonesNRM = dictCKtoNRM[zoneCK]
        jobsNRM = np.array(zones.loc[zonesNRM, employmentNames])

        consumerCumProbs[zoneCK] = {}
        for gg in range(numGG):
            consumerCumProbs[zoneCK][gg] = np.sum(useDistribution[gg, :] * jobsNRM, axis=1)
            if np.sum(consumerCumProbs[zoneCK][gg]) == 0:
                consumerCumProbs[zoneCK][gg] = np.ones(len(consumerCumProbs[zoneCK][gg]))
            consumerCumProbs[zoneCK][gg] = np.cumsum(consumerCumProbs[zoneCK][gg])
            consumerCumProbs[zoneCK][gg] /= consumerCumProbs[zoneCK][gg][-1]

    return Probabilities(producerProbs, consumerCumProbs)


def calculate_non_container_probabilities(
    zones: pd.DataFrame,
    makeDistribution: np.ndarray,
    useDistribution: np.ndarray,
    dictBGtoNRM: dict,
    employmentNames: list,
    dimGG: list
) -> Probabilities:
    """
    Bepaal per BG-zone de kans van een van de NRM-zones erbinnen de producent of consument
    is van een zending.

    Args:
        zones (pd.DataFrame): _description_
        makeDistribution (np.ndarray): _description_
        useDistribution (np.ndarray): _description_
        dictBGtoNRM (dict): _description_
        employmentNames (list): _description_
        dimGG (list): _description_

    Returns:
        tuple: Met daarin:
            - dict: Kansen voor producent.
            - dict: Kansen voor consument.
    """
    numGG = max(dimGG)

    # Bepaal de kansen van NRM-zones als producent van een zending
    producerProbs = {}

    for zoneBG in dictBGtoNRM.keys():

        zonesNRM = dictBGtoNRM[zoneBG]
        jobsNRM = np.array(zones.loc[zonesNRM, employmentNames])

        producerProbs[zoneBG] = {}
        for gg in range(numGG):
            producerProbs[zoneBG][gg] = np.sum(makeDistribution[gg, :] * jobsNRM, axis=1)
            if np.sum(producerProbs[zoneBG][gg]) == 0:
                producerProbs[zoneBG][gg] = np.ones(len(producerProbs[zoneBG][gg]))

    # Bepaal de kansen van NRM-zones als consument van een zending
    consumerCumProbs = {}

    for zoneBG in dictBGtoNRM.keys():

        zonesNRM = dictBGtoNRM[zoneBG]
        jobsNRM = np.array(zones.loc[zonesNRM, employmentNames])

        consumerCumProbs[zoneBG] = {}
        for gg in range(numGG):
            consumerCumProbs[zoneBG][gg] = np.sum(useDistribution[gg, :] * jobsNRM, axis=1)
            if np.sum(consumerCumProbs[zoneBG][gg]) == 0:
                consumerCumProbs[zoneBG][gg] = np.ones(len(consumerCumProbs[zoneBG][gg]))
            consumerCumProbs[zoneBG][gg] = np.cumsum(consumerCumProbs[zoneBG][gg])
            consumerCumProbs[zoneBG][gg] /= consumerCumProbs[zoneBG][gg][-1]

    return Probabilities(producerProbs, consumerCumProbs)


def get_where_routes(
    routesDirect: pd.DataFrame,
    routesToTT: pd.DataFrame,
    routesFromTT: pd.DataFrame,
) -> Tuple[
    Dict[Tuple[int, int, int], List[int]],
    Dict[Tuple[int, int, int], List[int]],
    Dict[Tuple[int, int, int], List[int]]
]:
    """
    Bepaal op welke rijen elke combo van H-B-GG te vinden is in de DataFrames met routes.

    Args:
        routesDirect (pd.DataFrame): _description_
        routesToTT (pd.DataFrame): _description_
        routesFromTT (pd.DataFrame): _description_

    Returns:
        tuple: Met daarin:
            - dict: De rijen binnen routeDirect.
            - dict: De rijen binnen routesToTT.
            - dict: De rijen binnen routeFromTT.
    """
    whereRoutesDirect = {}
    for i, row in enumerate(routesDirect.to_dict('records')):
        key = tuple(int(row[col]) for col in ['origin_ck', 'destination_ck', 'commodity'])
        whereRoutesDirect[key] = [i]

    whereRoutesToTT = {}
    for i, row in enumerate(routesToTT.to_dict('records')):
        key = tuple(int(row[col]) for col in ['origin_ck', 'destination_ck', 'commodity'])
        try:
            whereRoutesToTT[key].append(i)
        except KeyError:
            whereRoutesToTT[key] = [i]

    whereRoutesFromTT = {}
    for i, row in enumerate(routesFromTT.to_dict('records')):
        key = tuple(int(row[col]) for col in ['origin_ck', 'destination_ck', 'commodity'])
        try:
            whereRoutesFromTT[key].append(i)
        except KeyError:
            whereRoutesFromTT[key] = [i]

    return (whereRoutesDirect, whereRoutesToTT, whereRoutesFromTT)


def calc_distance_decay(
    origZones: np.ndarray, destZone: int,
    costHr: float, costKm: float,
    skim: np.ndarray,
    alpha: float, beta: float
) -> np.ndarray:
    """
    Bereken de waardes van de afstandsvervalfunctie, gegeven de bestemmingszone en de toe te
    passen kostenparameters.

    Args:
        origZones (np.ndarray): _description_
        destZone (int): _description_
        costHr (float): _description_
        costKm (float): _description_
        skim (np.ndarray): _description_
        alpha (float): _description_
        beta (float): _description_

    Returns:
        np.ndarray: _description_
    """
    numZones = int(skim.shape[0] ** 0.5)

    travelCost = (
        costHr / 3600 * (skim[(destZone - 1)::numZones, 0][origZones - 1]) +
        costKm / 1000 * (skim[(destZone - 1)::numZones, 1][origZones - 1]) +
        skim[(destZone - 1)::numZones, 2][origZones - 1])

    distanceDecay = (1.0 / (1.0 + np.exp(alpha + beta * np.log(travelCost))))

    distanceDecay /= np.sum(distanceDecay)

    return distanceDecay


def get_probs_vt_zez(
    settings: Settings,
    numLS: int, numVT: int,
) -> np.ndarray:
    """
    Haal de kansen per voertuigtype voor het ZEZ-scenario op.

    Args:
        settings (Settings): _description_
        numLS (int): _description_
        numVT (int): _description_

    Returns:
        np.ndarray: _description_
    """
    # Vehicle/combustion shares (for ZEZ scenario)
    scenarioZEZ = pd.read_csv(settings.get_path("KansenVoertuigtypeZEZ"), sep='\t')

    # Only vehicle shares
    cumProbsVehicleTypesZEZ = np.zeros((numLS, numVT))
    for i in scenarioZEZ[scenarioZEZ['consolidated'] == 1].index:
        ls = int(scenarioZEZ.at[i, 'logistic_segment'])
        vt = int(scenarioZEZ.at[i, 'vehicle_type_lwm'])
        prob = scenarioZEZ.at[i, 'probability']
        cumProbsVehicleTypesZEZ[ls, vt] += prob

    for ls in range(numLS):
        tmpCumSum = np.cumsum(cumProbsVehicleTypesZEZ[ls, :])
        tmpSum = tmpCumSum[-1]
        if tmpSum > 0:
            cumProbsVehicleTypesZEZ[ls, :] = tmpCumSum / tmpSum
        else:
            cumProbsVehicleTypesZEZ[ls, :] = np.arange(1, numVT + 1) / numVT

    return cumProbsVehicleTypesZEZ


def create_container_shipments(
    tonnes: pd.DataFrame, routes: Routes, probabilities: Probabilities,
    distrCenters: pd.DataFrame, terminals: pd.DataFrame,
    fractionsDC: pd.DataFrame, coeffsDistDecay: dict, statistics: ContainerStatistics,
    costFigures: CostFigure, skim: np.ndarray,
    dictToNRM: dict, dictToBG: dict, maxDutchZone: int,
    cumProbsCommodityToNSTR: Tuple[np.ndarray], seed: int, discrFac: float = 1.0,
    logger: logging.Logger = None
) -> pd.DataFrame:
    """
        Voer de procedure voor het maken van discrete containerzendingen uit.

    Args:
        tonnes (pd.DataFrame): _description_
        routes (Routes): _description_
        probabilities (Probabilities): _description_
        distrCenters (pd.DataFrame): _description_
        terminals (pd.DataFrame): _description_
        fractionsDC (pd.DataFrame): _description_
        coeffsDistDecay (dict): _description_
        statistics (ContainerStatistics): _description_
        costFigures (CostFigure): _description_
        skim (np.ndarray): _description_
        dictToNRM (dict): _description_
        dictToBG (dict): _description_
        maxDutchZone (int): _description_
        cumProbsCommodityToNSTR (Tuple[np.ndarray]): _description_
        seed (int): _description_
        discrFac (float, optional): _description_. Defaults to 1.0.
        logger (logging.Logger, optional): _description_. Defaults to None.

    Returns:
        pd.DataFrame: De aangemaakte zendingen. Elke rij is 1 zending.
    """
    # Counters voor bijhouden voortgang
    totalWeight = np.sum(tonnes['weight__ton']) * discrFac
    progressStep = int(totalWeight / 500)
    allocatedWeight = 0.0

    alpha, beta = coeffsDistDecay['alpha'], coeffsDistDecay['beta']

    # Op welke rijen is elke combo van H-B-GG te vinden in de DataFrames met routes
    whereRoutesDirect, whereRoutesToTT, whereRoutesFromTT = get_where_routes(
        routes.direct, routes.toTerminal, routes.fromTerminal)

    # DC-attributen in arrays voor sneller accessen
    distrSurface = np.array(distrCenters['surface_distr__meter2'])
    distrZoneCK = np.array(distrCenters['zone_ck'], dtype=int)
    distrZoneNRM = np.array(distrCenters['zone_nrm'], dtype=int)

    terminalZoneNRM = np.array(terminals['zone_nrm'].fillna(-1), dtype=int)

    # Tonnen en routes in arrays voor sneller accessen
    tonnesArray = np.array(tonnes)
    routesDirectArray = np.array(routes.direct)
    routesToTTArray = np.array(routes.toTerminal)
    routesFromTTArray = np.array(routes.fromTerminal)

    # Initialiseer lijst met zendingen
    shipmentsCK = []
    count = 0

    # Voor alle HB's in de tonnenmatrix zendingen maken
    for i in range(len(tonnesArray)):

        # CK-origin, -destination and -goods type of the current cell
        origCK = int(tonnesArray[i, 0])
        destCK = int(tonnesArray[i, 1])
        gg = int(tonnesArray[i, 2])
        ls = int(tonnesArray[i, 3])

        # Seed vastzetten per cel in de tonnenmatrix
        np.random.seed(seed + int(f"{origCK}{destCK}") + gg * (ls + 1))

        # Voertuigtype is altijd 5 (trekker-oplegger)
        vt = 5

        # Welke NRM-zones vallen onder de huidige CK-herkomst en -bestemming
        origZonesNRM = dictToNRM[origCK]
        destZonesNRM = dictToNRM[destCK]

        # Aantal toe te wijzen tonnen voor huidige combinatie van
        # CK-herkomst, CK-bestemming en BasGoed-goederengroep
        totalWeightCurrent = tonnesArray[i, 4] * discrFac

        # Verdeel gewicht over beladen en lege containers
        totalWeightCurrentLoaded = (1 - statistics.emptyShareCK) * totalWeightCurrent
        totalWeightCurrentEmpty = statistics.emptyShareCK * totalWeightCurrent

        # Bepaal het aantal zendingen
        numShipsCurrentLoaded = max(
            1, int(np.round(totalWeightCurrentLoaded / statistics.avgShipSizeCK[gg - 1])))
        numShipsCurrentEmpty = int(
            np.round(totalWeightCurrentEmpty / statistics.avgShipSizeEmptyCK))
        numShipsCurrent = numShipsCurrentLoaded + numShipsCurrentEmpty

        # Bepaal de zendingsgrootte
        if numShipsCurrentLoaded > 0:
            shipSizeCurrentLoaded = totalWeightCurrentLoaded / numShipsCurrentLoaded
        else:
            shipSizeCurrentLoaded = statistics.avgShipSizeCK[gg - 1]

        if numShipsCurrentEmpty > 0:
            shipSizeCurrentEmpty = totalWeightCurrentEmpty / numShipsCurrentEmpty
        else:
            shipSizeCurrentEmpty = statistics.avgShipSizeEmptyCK

        # Bepaal de kans van en naar DC
        probFromDC = fractionsDC.at[str(gg), 'ck_fraction_from_dc']
        probToDC = fractionsDC.at[str(gg), 'ck_fraction_to_dc']

        # Bepaal de kansen voor producenten en herkomst-DC's
        recalcOrigProbs = False
        if i == 0:
            recalcOrigProbs = True
        elif origCK != tonnesArray[i - 1, 0]:
            recalcOrigProbs = True

        if recalcOrigProbs:
            whereDCinOrig = (
                [] if origCK > maxDutchZone else
                list(np.where(distrZoneCK == origCK)[0]))
            if whereDCinOrig:
                origProbsDC = distrSurface[whereDCinOrig]

        # Bepaal de kansen voor consumenten en bestemmings-DC's
        recalcDestProbs = False
        if i == 0:
            recalcDestProbs = True
        elif destCK != tonnesArray[i - 1, 1]:
            recalcDestProbs = True

        if recalcDestProbs:
            whereDCinDest = (
                [] if destCK > maxDutchZone else
                list(np.where(distrZoneCK == destCK)[0]))
            if whereDCinDest:
                destProbsDC = distrSurface[whereDCinDest]
                destCumProbsDC = np.cumsum(destProbsDC)
                destCumProbsDC = destCumProbsDC / destCumProbsDC[-1]
            destCumProbsNRM = probabilities.consumerCumulative[destCK]

        # Bepaal de kans van en naar TT
        currentRoutesDirect = routesDirectArray[
            whereRoutesDirect.get((origCK, destCK, gg), []), :]
        currentRoutesToTT = routesToTTArray[
            whereRoutesToTT.get((origCK, destCK, gg), []), :]
        currentRoutesFromTT = routesFromTTArray[
            whereRoutesFromTT.get((origCK, destCK, gg), []), :]

        synthDirect = (
            np.sum(currentRoutesDirect[:, -1]) if len(currentRoutesDirect) > 0 else 0.0)
        synthToTT = (
            np.sum(currentRoutesToTT[:, -1]) if len(currentRoutesToTT) > 0 else 0.0)
        synthFromTT = (
            np.sum(currentRoutesFromTT[:, -1]) if len(currentRoutesFromTT) > 0 else 0.0)

        probTT = np.array([synthDirect, synthToTT, synthFromTT])
        if np.sum(probTT) > 0:
            cumProbTT = np.cumsum(probTT)
            cumProbTT = cumProbTT / cumProbTT[-1]
        else:
            cumProbTT = np.array([1.0, 1.0, 1.0])

        # Bepaal de kansen voor herkomstterminals
        if synthFromTT > 0:
            whereTTinOrig = [int(x - 1) for x in np.unique(currentRoutesFromTT[:, 2])]
            origProbsTT = numpy_pivot_table(currentRoutesFromTT, index=2, column=4, aggfunc=sum)
            origProbsTT = [origProbsTT[tt + 1] for tt in whereTTinOrig]

        # Bepaal de kansen voor bestemmingsterminals
        if synthToTT > 0:
            whereTTinDest = [int(x - 1) for x in np.unique(currentRoutesToTT[:, 2])]
            destProbsTT = numpy_pivot_table(currentRoutesToTT, index=2, column=4, aggfunc=sum)
            destProbsTT = [destProbsTT[tt + 1] for tt in whereTTinDest]
            destCumProbsTT = np.cumsum(destProbsTT)
            destCumProbsTT = destCumProbsTT / destCumProbsTT[-1]

        # Maak nu de discrete zendingen aan
        for ship in range(numShipsCurrent):

            # Bepaal NSTR-goederensoort
            nstr = draw_choice(cumProbsCommodityToNSTR[1][gg - 1, :])

            # Van of naar een terminal
            choiceTT = draw_choice(cumProbTT)
            if choiceTT == 0:
                toTT, fromTT = False, False
            elif choiceTT == 1:
                toTT, fromTT = True, False
            else:
                toTT, fromTT = False, True

            # Van of naar een DC
            toDC, fromDC = False, False
            if (not fromTT) and whereDCinOrig:
                fromDC = (np.random.rand() < probFromDC)
            if (not toTT) and whereDCinDest:
                toDC = (np.random.rand() < probToDC)

            # Welke terminal als bestemming
            if toTT:
                if terminalZoneNRM[whereTTinDest[0]] < 0:
                    # Niet toewijzen aan terminal in gebied buiten bereik NRM
                    toTT = False
                    destTT = -9999
                else:
                    destTT = whereTTinDest[draw_choice(destCumProbsTT)] + 1
                    destNRM = terminalZoneNRM[destTT - 1]
            else:
                destTT = -9999

            # Welke DC als bestemming
            if toDC:
                destDC = whereDCinDest[draw_choice(destCumProbsDC)] + 1
                destNRM = distrZoneNRM[destDC - 1]
            else:
                destDC = -9999

            # Welke NRM-zone als bestemming (consument)
            if not toTT and not toDC:
                destNRM = destZonesNRM[draw_choice(destCumProbsNRM[gg - 1])]

            # Welke terminal als herkomst
            if fromTT:
                if terminalZoneNRM[whereTTinOrig[0]] < 0:
                    # Niet toewijzen aan terminal in gebied buiten bereik NRM
                    fromTT = False
                    origTT = -9999
                else:
                    distanceDecay = calc_distance_decay(
                        terminalZoneNRM[whereTTinOrig], destNRM,
                        costFigures.eurPerHour[gg - 1], costFigures.eurPerKilometer[gg - 1],
                        skim, alpha, beta)
                    origCumProbsTT = np.cumsum(origProbsTT * distanceDecay)
                    origCumProbsTT = origCumProbsTT / origCumProbsTT[-1]
                    origTT = whereTTinOrig[draw_choice(origCumProbsTT)] + 1
                    origNRM = terminalZoneNRM[origTT - 1]
            else:
                origTT = -9999

            # Welke DC als herkomst
            if fromDC:
                distanceDecay = calc_distance_decay(
                    distrZoneNRM[whereDCinOrig], destNRM,
                    costFigures.eurPerHour[gg - 1], costFigures.eurPerKilometer[gg - 1],
                    skim, alpha, beta)
                origCumProbsDC = np.cumsum(origProbsDC * distanceDecay)
                origCumProbsDC = origCumProbsDC / origCumProbsDC[-1]
                origDC = whereDCinOrig[draw_choice(origCumProbsDC)] + 1
                origNRM = distrZoneNRM[origDC - 1]
            else:
                origDC = -9999

            # Welke NRM-zone als herkomst (producent)
            if not fromTT and not fromDC:
                distanceDecay = calc_distance_decay(
                    origZonesNRM, destNRM,
                    costFigures.eurPerHour[gg - 1], costFigures.eurPerKilometer[gg - 1],
                    skim, alpha, beta)
                origCumProbsNRM = np.cumsum(probabilities.producer[origCK][gg - 1] * distanceDecay)
                origCumProbsNRM = origCumProbsNRM / origCumProbsNRM[-1]
                origNRM = origZonesNRM[draw_choice(origCumProbsNRM)]

            # Zendingsgrootte en BasGoed-goederengroep
            if ship < numShipsCurrentLoaded:
                weight = shipSizeCurrentLoaded
                tmpGG = gg
            else:
                weight = shipSizeCurrentEmpty
                tmpGG = -1
                nstr = -1

            # Om het terug te brengen naar dagbasis, bewaar random een deel van de zendingen
            if np.random.rand() < (1 / discrFac):
                shipmentsCK.append([
                    origNRM, destNRM, dictToBG[origCK], dictToBG[destCK],
                    tmpGG, ls, nstr,
                    vt,
                    weight,
                    origDC, destDC, origTT, destTT])

            allocatedWeight += weight
            count += 1

            if count % progressStep == 0:
                progress = round(100 * allocatedWeight / totalWeight, 1)
                if logger is None:
                    print(f'\t\t{progress}%', end='\r')
                else:
                    logger.debug(f"\t\t{progress}%")

    if logger is None:
        print('\t\t100.0%', end='\r')
    else:
        logger.debug("\t\t100.0%")

    # Shipments als DataFrame met de juiste headers teruggeven
    shipmentsCK = pd.DataFrame(
        np.array(shipmentsCK),
        columns=[
            'orig_nrm', 'dest_nrm', 'orig_bg', 'dest_bg',
            'commodity', 'logistic_segment', 'nstr',
            'vehicle_type_lwm', 'weight__ton',
            'orig_dc', 'dest_dc', 'orig_tt', 'dest_tt'])

    return shipmentsCK


def create_non_container_shipments_direct(
    tonnes: pd.DataFrame, probabilities: Probabilities, distrCenters: pd.DataFrame,
    fractionsDC: pd.DataFrame, coeffsShip: dict, coeffsDistDecay: dict, skim: np.ndarray,
    costFiguresVehicleTypes: CostFigure, costFigures: CostFigure,
    truckCapacities: np.ndarray, dictToNRM: dict, maxDutchZone: int,
    cumProbsCommodityToNSTR: Tuple[np.ndarray],
    vehicleTypeIds: List[str], shipmentSizeIds: List[str], seed: int, discrFac: float = 1.0,
    logger: logging.Logger = None
) -> pd.DataFrame:
    """
    Voer de procedure voor het maken van discrete niet-containerzendingen uit,
    voor goederenstromen niet van/naar een terminal.

    Args:
        tonnes (pd.DataFrame): _description_
        probabilities (Probabilities): _description_
        distrCenters (pd.DataFrame): _description_
        fractionsDC (pd.DataFrame): _description_
        coeffsShip (dict): _description_
        coeffsDistDecay (dict): _description_
        skim (np.ndarray): _description_
        costFiguresVehicleTypes (CostFigure): _description_
        costFigures (CostFigure): _description_
        truckCapacities (np.ndarray): _description_
        dictToNRM (dict): _description_
        maxDutchZone (int): _description_
        cumProbsCommodityToNSTR (Tuple[np.ndarray]): _description_
        vehicleTypeIds (List[str]): _description_
        shipmentSizeIds (List[str]): _description_
        seed (int): _description_
        discrFac (float, optional): _description_. Defaults to 1.0.
        logger (logging.Logger, optional): _description_. Defaults to None.

    Returns:
        pd.DataFrame: De aangemaakte zendingen. Elke rij is 1 zending.
    """
    numVT = len(vehicleTypeIds) - 3
    numSS = len(shipmentSizeIds)
    numZones = int(skim.shape[0] ** 0.5)
    absoluteSS = np.array([row['median'] for row in shipmentSizeIds])

    # Counters voor bijhouden voortgang
    totalWeight = np.sum(tonnes['road_weight__ton']) * discrFac
    progressStep = int(totalWeight / 500)
    allocatedWeight = 0.0

    alpha, beta = coeffsDistDecay['alpha'], coeffsDistDecay['beta']

    # DC-attributen in arrays voor sneller accessen
    distrSurface = np.array(distrCenters['surface_distr__meter2'])
    distrZoneBG = np.array(distrCenters['zone_bg'], dtype=int)
    distrZoneNRM = np.array(distrCenters['zone_nrm'], dtype=int)

    # Tonnen en routes in arrays voor sneller accessen
    tonnesArray = np.array(tonnes)

    # Initialiseer lijst met zendingen
    shipments = []
    count = 0

    # Voor alle HB's in de tonnenmatrix zendingen maken
    for i in range(len(tonnesArray)):

        # CK-origin, -destination and -goods type of the current cell
        origBG = int(tonnesArray[i, 0])
        destBG = int(tonnesArray[i, 1])
        gg = int(tonnesArray[i, 2])
        ls = int(tonnesArray[i, 3])

        # Seed vastzetten per cel in de tonnenmatrix
        np.random.seed(seed + int(f"{origBG}{destBG}") + gg * (ls + 1))

        # Welke NRM-zones vallen onder de huidige BG-herkomst en -bestemming
        origZonesNRM = dictToNRM[origBG]
        destZonesNRM = dictToNRM[destBG]

        # Aantal toe te wijzen tonnen voor huidige combinatie van
        # BG-herkomst, BG-bestemming en BasGoed-goederengroep
        totalWeightCurrent = tonnesArray[i, 4] * discrFac
        allocatedWeightCurrent = 0.0

        # Bepaal de kans van en naar DC
        probFromDC = fractionsDC.at[str(gg), 'ms_fraction_from_dc']
        probToDC = fractionsDC.at[str(gg), 'ms_fraction_to_dc']

        # Bepaal de kansen voor producenten en herkomst-DC's
        recalcOrigProbs = False
        if i == 0:
            recalcOrigProbs = True
        elif origBG != tonnesArray[i - 1, 0]:
            recalcOrigProbs = True

        if recalcOrigProbs:
            whereDCinOrig = (
                [] if origBG > maxDutchZone else
                list(np.where(distrZoneBG == origBG)[0]))
            if whereDCinOrig:
                origProbsDC = distrSurface[whereDCinOrig]

        # Bepaal de kansen voor consumenten en bestemmings-DC's
        recalcDestProbs = False
        if i == 0:
            recalcDestProbs = True
        elif destBG != tonnesArray[i - 1, 1]:
            recalcDestProbs = True

        if recalcDestProbs:
            whereDCinDest = (
                [] if destBG > maxDutchZone else
                list(np.where(distrZoneBG == destBG)[0]))
            if whereDCinDest:
                destProbsDC = distrSurface[whereDCinDest]
                destCumProbsDC = np.cumsum(destProbsDC)
                destCumProbsDC = destCumProbsDC / destCumProbsDC[-1]
            destCumProbsNRM = probabilities.consumerCumulative[destBG]

        # De coefficienten van het keuzemodel binnen de huidige goederengroep
        betaTransportCosts = coeffsShip['transport_costs'][gg - 1]
        betaInventoryCosts = coeffsShip['inventory_costs'][gg - 1]
        betaFromDC = coeffsShip['from_dc'][gg - 1]
        betaToDC = coeffsShip['to_dc'][gg - 1]
        constsLongHaulVT = coeffsShip['const_vehicle_type_lwm_long_haul'][gg - 1, :]
        constsShortHaulVT = coeffsShip['const_vehicle_type_lwm_short_haul'][gg - 1, :]
        constsSS = coeffsShip['const_shipment_size'][gg - 1, :]

        while allocatedWeightCurrent < totalWeightCurrent:

            # Bepaal NSTR-goederensoort
            nstr = draw_choice(cumProbsCommodityToNSTR[0][gg - 1, :])

            # Van of naar een DC
            fromDC = (np.random.rand() < probFromDC) if whereDCinOrig else False
            toDC = (np.random.rand() < probToDC) if whereDCinDest else False

            # Welke DC als bestemming
            if toDC:
                destDC = whereDCinDest[draw_choice(destCumProbsDC)] + 1
                destNRM = distrZoneNRM[destDC - 1]
            else:
                destDC = -9999

            # Welke NRM-zone als bestemming (consument)
            if not toDC:
                destNRM = destZonesNRM[draw_choice(destCumProbsNRM[gg - 1])]

            # Welke DC als herkomst
            if fromDC:
                distanceDecay = calc_distance_decay(
                    distrZoneNRM[whereDCinOrig], destNRM,
                    costFigures.eurPerHour[gg - 1],
                    costFigures.eurPerKilometer[gg - 1],
                    skim, alpha, beta)
                origCumProbsDC = np.cumsum(origProbsDC * distanceDecay)
                origCumProbsDC = origCumProbsDC / origCumProbsDC[-1]
                origDC = whereDCinOrig[draw_choice(origCumProbsDC)] + 1
                origNRM = distrZoneNRM[origDC - 1]
            else:
                origDC = -9999

            # Welke NRM-zone als herkomst (producent)
            if not fromDC:
                distanceDecay = calc_distance_decay(
                    origZonesNRM, destNRM,
                    costFigures.eurPerHour[gg - 1],
                    costFigures.eurPerKilometer[gg - 1],
                    skim, alpha, beta)
                origCumProbsNRM = np.cumsum(probabilities.producer[origBG][gg - 1] * distanceDecay)
                origCumProbsNRM /= origCumProbsNRM[-1]
                origNRM = origZonesNRM[draw_choice(origCumProbsNRM)]

            # Trek een simultane keuze voor zendingsgrootte en voertuigtype
            travTime = skim[(origNRM - 1) * numZones + (destNRM - 1), 0] / 3600
            distance = skim[(origNRM - 1) * numZones + (destNRM - 1), 1] / 1000
            choiceSSVT = choice_model_ssvt(
                costFiguresVehicleTypes.eurPerKilometer,
                costFiguresVehicleTypes.eurPerHour,
                gg, fromDC, toDC,
                travTime, distance,
                absoluteSS, truckCapacities,
                betaTransportCosts, betaInventoryCosts, betaFromDC, betaToDC,
                constsShortHaulVT, constsLongHaulVT, constsSS,
                numVT, numSS)

            # Herleid naar zendingsgrootte en voertuigtype
            shipmentSizeCat = int(np.floor(choiceSSVT / numVT))
            weight = min(
                absoluteSS[shipmentSizeCat],
                totalWeightCurrent - allocatedWeightCurrent)

            # The chosen vehicle type
            vt = choiceSSVT - shipmentSizeCat * numVT

            # Om het terug te brengen naar dagbasis, bewaar random een deel van de zendingen
            if np.random.rand() < (1 / discrFac):
                shipments.append([
                    origNRM, destNRM, origBG, destBG,
                    gg, ls, nstr,
                    vt,
                    weight,
                    origDC, destDC, -9999, -9999])

            allocatedWeightCurrent += weight
            allocatedWeight += weight
            count += 1

            if count % progressStep == 0:
                progress = round(100 * allocatedWeight / totalWeight, 1)
                if logger is None:
                    print(f'\t\t{progress}%', end='\r')
                else:
                    logger.debug(f"\t\t{progress}%")

    if logger is None:
        print('\t\t100.0%', end='\r')
    else:
        logger.debug("\t\t100.0%")

    # Shipments als DataFrame met de juiste headers teruggeven
    shipments = pd.DataFrame(
        np.array(shipments),
        columns=[
            'orig_nrm', 'dest_nrm', 'orig_bg', 'dest_bg',
            'commodity', 'logistic_segment', 'nstr',
            'vehicle_type_lwm', 'weight__ton',
            'orig_dc', 'dest_dc', 'orig_tt', 'dest_tt'])

    return shipments


def create_non_container_shipments_non_direct(
    tonnes: pd.DataFrame, terminals: pd.DataFrame, probabilities: Probabilities,
    distrCenters: pd.DataFrame, fractionsDC: pd.DataFrame,
    coeffsShip: dict, coeffsDistDecay: dict, skim: np.ndarray,
    costFiguresVehicleTypes: CostFigure, costFigures: CostFigure,
    truckCapacities: np.ndarray, dictToNRM: dict, maxDutchZone: int,
    cumProbsCommodityToNSTR: Tuple[np.ndarray],
    vehicleTypeIds: List[str], shipmentSizeIds: List[str],
    dimAccessEgressGG: List[int], dimAccessGG: List[int], seed: int, discrFac: float = 1.0
) -> pd.DataFrame:
    """
    Voer de procedure voor het maken van discrete niet-containerzendingen uit, voor
    goederenstromen van/naar een terminal.

    Args:
        tonnes (pd.DataFrame): _description_
        terminals (pd.DataFrame): _description_
        probabilities (Probabilities): _description_
        distrCenters (pd.DataFrame): _description_
        fractionsDC (pd.DataFrame): _description_
        coeffsShip (dict): _description_
        coeffsDistDecay (dict): _description_
        skim (np.ndarray): _description_
        costFiguresVehicleTypes (CostFigure): _description_
        costFigures (CostFigure): _description_
        truckCapacities (np.ndarray): _description_
        dictToNRM (dict): _description_
        maxDutchZone (int): _description_
        cumProbsCommodityToNSTR (Tuple[np.ndarray]): _description_
        vehicleTypeIds (List[str]): _description_
        shipmentSizeIds (List[str]): _description_
        dimAccessEgressGG (List[int]): _description_
        dimAccessGG (List[int]): _description_
        seed (int): _description_
        discrFac (float, optional): _description_. Defaults to 1.0.

    Raises:
        Exception: _description_
        Exception: _description_

    Returns:
        pd.DataFrame: _description_
    """
    numVT = len(vehicleTypeIds) - 3
    numSS = len(shipmentSizeIds)
    numZones = int(skim.shape[0] ** 0.5)
    absoluteSS = np.array([row['median'] for row in shipmentSizeIds])

    alpha, beta = coeffsDistDecay['alpha'], coeffsDistDecay['beta']

    # Tonnen en routes in arrays voor sneller accessen
    tonnesArray = np.array(tonnes)

    # Alleen de relevante goederengroepen erin laten
    setGG = set([int(gg) for gg in dimAccessEgressGG + dimAccessGG])
    tonnesArray = tonnesArray[
        [i for i in range(len(tonnesArray)) if tonnesArray[i, 2] in setGG], :]

    # DC-attributen in arrays voor sneller accessen
    distrSurface = np.array(distrCenters['surface_distr__meter2'])
    distrZoneBG = np.array(distrCenters['zone_bg'], dtype=int)
    distrZoneNRM = np.array(distrCenters['zone_nrm'], dtype=int)

    # Terminalattributen in arrays voor sneller accessen
    terminalZoneBG = np.array(terminals['zone_bg'], dtype=int)
    terminalZoneNRM = np.array(terminals['zone_nrm'], dtype=int)
    terminalID = np.array(terminals['terminal'], dtype=int)

    # Initialiseer lijst met zendingen
    shipments = []

    # Voor de HB's in de tonnenmatrix zendingen maken
    for i in range(len(tonnesArray)):

        # BG-origin, -destination and -goods type of the current cell
        origBG = int(tonnesArray[i, 0])
        destBG = int(tonnesArray[i, 1])
        gg = int(tonnesArray[i, 2])
        ls = int(tonnesArray[i, 3])

        # Seed vastzetten per cel in de tonnenmatrix
        np.random.seed(seed + int(f"{origBG}{destBG}") + gg * (ls + 1))

        # Welke NRM-zones vallen onder de huidige BG-herkomst en -bestemming
        origZonesNRM = dictToNRM[origBG]
        destZonesNRM = dictToNRM[destBG]

        # Aantal toe te wijzen tonnen voor huidige combinatie van
        # BG-herkomst, BG-bestemming en BasGoed-goederengroep
        totalWeightAccessCurrent = (
            discrFac * tonnesArray[i, 4] if gg in dimAccessGG else 0.0)
        totalWeightEgressCurrent = (
            discrFac * tonnesArray[i, 4] if gg in dimAccessEgressGG else 0.0)
        allocatedWeightAccessCurrent = 0.0
        allocatedWeightEgressCurrent = 0.0

        # Bepaal de kans van en naar DC
        probFromDC = fractionsDC.at[str(gg), 'ms_fraction_from_dc']
        probToDC = fractionsDC.at[str(gg), 'ms_fraction_to_dc']

        # Bepaal de kansen voor producenten en herkomst-DC's
        recalcOrigProbs = False
        if i == 0:
            recalcOrigProbs = True
        elif origBG != tonnesArray[i - 1, 0]:
            recalcOrigProbs = True

        if recalcOrigProbs:
            whereDCinOrig = (
                [] if origBG > maxDutchZone else
                list(np.where(distrZoneBG == origBG)[0]))
            if whereDCinOrig:
                origProbsDC = distrSurface[whereDCinOrig]

            whereTTinOrig = (
                [] if origBG > maxDutchZone else
                list(np.where(terminalZoneBG == origBG)[0]))
            numTTinOrig = len(whereTTinOrig)

        if origBG > maxDutchZone:
            totalWeightAccessCurrent = 0.0

        if numTTinOrig == 0 and totalWeightAccessCurrent > 0.0:
            raise Exception(
                f'Geen terminals in BasGoed-zone {origBG}.')

        # Bepaal de kansen voor consumenten en bestemmings-DC's
        recalcDestProbs = False
        if i == 0:
            recalcDestProbs = True
        elif destBG != tonnesArray[i - 1, 1]:
            recalcDestProbs = True

        if recalcDestProbs:
            whereDCinDest = (
                [] if destBG > maxDutchZone else
                list(np.where(distrZoneBG == destBG)[0]))
            if whereDCinDest:
                destProbsDC = distrSurface[whereDCinDest]
                destCumProbsDC = np.cumsum(destProbsDC)
                destCumProbsDC = destCumProbsDC / destCumProbsDC[-1]

            destCumProbsNRM = probabilities.consumerCumulative[destBG]

            whereTTinDest = (
                [] if destBG > maxDutchZone else
                list(np.where(terminalZoneBG == destBG)[0]))
            numTTinDest = len(whereTTinDest)

        if gg in dimAccessEgressGG and destBG > maxDutchZone:
            totalWeightEgressCurrent = 0.0

        if numTTinDest == 0 and totalWeightEgressCurrent > 0.0:
            raise Exception(
                f'Geen terminals in BasGoed-zone {destBG}.')

        # De coefficienten van het keuzemodel binnen de huidige goederengroep
        betaTransportCosts = coeffsShip['transport_costs'][gg - 1]
        betaInventoryCosts = coeffsShip['inventory_costs'][gg - 1]
        betaFromDC = coeffsShip['from_dc'][gg - 1]
        betaToDC = coeffsShip['to_dc'][gg - 1]
        constsLongHaulVT = coeffsShip['const_vehicle_type_lwm_long_haul'][gg - 1, :]
        constsShortHaulVT = coeffsShip['const_vehicle_type_lwm_short_haul'][gg - 1, :]
        constsSS = coeffsShip['const_shipment_size'][gg - 1, :]

        # Aanmaken zendingen naar terminal in herkomstzone
        while allocatedWeightAccessCurrent < totalWeightAccessCurrent:

            # Bepaal NSTR-goederensoort
            nstr = draw_choice(cumProbsCommodityToNSTR[0][gg - 1, :])

            # Flow naar terminal, dus geen herkomstterminal en geen bestemmings-DC
            origTT = -9999
            destDC = -9999

            # Welke TT als bestemming
            destTT = whereTTinOrig[np.random.randint(numTTinOrig)]
            destNRM = terminalZoneNRM[destTT]
            destTT = terminalID[destTT]

            # Vanuit een DC of niet
            fromDC = (np.random.rand() < probFromDC) if whereDCinOrig else False

            # Welke DC als herkomst
            if fromDC:
                distanceDecay = calc_distance_decay(
                    distrZoneNRM[whereDCinOrig], destNRM,
                    costFigures.eurPerHour[gg - 1], costFigures.eurPerKilometer[gg - 1],
                    skim, alpha, beta)
                origCumProbsDC = np.cumsum(origProbsDC * distanceDecay)
                origCumProbsDC = origCumProbsDC / origCumProbsDC[-1]
                origDC = whereDCinOrig[draw_choice(origCumProbsDC)] + 1
                origNRM = distrZoneNRM[origDC - 1]
            else:
                origDC = -9999

            # Welke NRM-zone als herkomst (producent)
            if not fromDC:
                distanceDecay = calc_distance_decay(
                    origZonesNRM, destNRM,
                    costFigures.eurPerHour[gg - 1], costFigures.eurPerKilometer[gg - 1],
                    skim, alpha, beta)
                origCumProbsNRM = np.cumsum(probabilities.producer[origBG][gg - 1] * distanceDecay)
                origCumProbsNRM /= origCumProbsNRM[-1]
                origNRM = origZonesNRM[draw_choice(origCumProbsNRM)]

            # Trek een simultane keuze voor zendingsgrootte en voertuigtype
            travTime = skim[(origNRM - 1) * numZones + (destNRM - 1), 0] / 3600
            distance = skim[(origNRM - 1) * numZones + (destNRM - 1), 1] / 1000
            choiceSSVT = choice_model_ssvt(
                costFiguresVehicleTypes.eurPerKilometer, costFiguresVehicleTypes.eurPerHour,
                gg, fromDC, False,
                travTime, distance,
                absoluteSS, truckCapacities,
                betaTransportCosts, betaInventoryCosts, betaFromDC, betaToDC,
                constsShortHaulVT, constsLongHaulVT, constsSS,
                numVT, numSS)

            # Herleid naar zendingsgrootte en voertuigtype
            shipmentSizeCat = int(np.floor(choiceSSVT / numVT))
            weight = min(
                absoluteSS[shipmentSizeCat],
                totalWeightAccessCurrent - allocatedWeightAccessCurrent)

            # The chosen vehicle type
            vt = choiceSSVT - shipmentSizeCat * numVT

            shipments.append([
                origNRM, destNRM, origBG, origBG,
                gg, ls, nstr,
                vt,
                weight,
                origDC, destDC, origTT, destTT])

            allocatedWeightAccessCurrent += weight

        # Aanmaken zendingen vanuit terminal in bestemmingszone
        while allocatedWeightEgressCurrent < totalWeightEgressCurrent:

            # Bepaal NSTR-goederensoort
            nstr = draw_choice(cumProbsCommodityToNSTR[0][gg - 1, :])

            # Flow vanuit terminal, dus geen bestemmingsterminal en geen herkomst-DC
            destTT = -9999
            origDC = -9999

            # Naar een DC of niet
            toDC = (np.random.rand() < probToDC) if whereDCinDest else False

            # Welke DC als bestemming
            if toDC:
                destDC = whereDCinDest[draw_choice(destCumProbsDC)] + 1
                destNRM = distrZoneNRM[destDC - 1]
            else:
                destDC = -9999

            # Welke NRM-zone als bestemming (consument)
            if not toDC:
                destNRM = destZonesNRM[draw_choice(destCumProbsNRM[gg - 1])]

            # Welke TT als herkomst
            distanceDecay = calc_distance_decay(
                terminalZoneNRM[whereTTinDest], destNRM,
                costFigures.eurPerHour[gg - 1], costFigures.eurPerKilometer[gg - 1],
                skim, alpha, beta)
            origCumProbsTT = np.cumsum(distanceDecay)
            origCumProbsTT /= origCumProbsTT[-1]
            origTT = whereTTinDest[draw_choice(origCumProbsTT)]
            origNRM = terminalZoneNRM[origTT]
            origTT = terminalID[origTT]

            # Trek een simultane keuze voor zendingsgrootte en voertuigtype
            travTime = skim[(origNRM - 1) * numZones + (destNRM - 1), 0] / 3600
            distance = skim[(origNRM - 1) * numZones + (destNRM - 1), 1] / 1000
            choiceSSVT = choice_model_ssvt(
                costFiguresVehicleTypes.eurPerKilometer, costFiguresVehicleTypes.eurPerHour,
                gg, False, toDC,
                travTime, distance,
                absoluteSS, truckCapacities,
                betaTransportCosts, betaInventoryCosts, betaFromDC, betaToDC,
                constsShortHaulVT, constsLongHaulVT, constsSS,
                numVT, numSS)

            # Herleid naar zendingsgrootte en voertuigtype
            shipmentSizeCat = int(np.floor(choiceSSVT / numVT))
            weight = min(
                absoluteSS[shipmentSizeCat],
                totalWeightEgressCurrent - allocatedWeightEgressCurrent)

            # The chosen vehicle type
            vt = choiceSSVT - shipmentSizeCat * numVT

            shipments.append([
                origNRM, destNRM, destBG, destBG,
                gg, ls, nstr,
                vt,
                weight,
                origDC, destDC, origTT, destTT])

            allocatedWeightEgressCurrent += weight

    # Om het terug te brengen naar dagbasis, selecteer nu random een deel van de zendingen
    numShipsInit = len(shipments)
    numShipsDay = int(round(numShipsInit / discrFac))

    shipments = [
        shipments[i] for i in np.random.choice(
            range(numShipsInit), size=numShipsDay, replace=False)]

    # Shipments als DataFrame met de juiste headers teruggeven
    shipments = pd.DataFrame(
        np.array(shipments),
        columns=[
            'orig_nrm', 'dest_nrm', 'orig_bg', 'dest_bg',
            'commodity', 'logistic_segment', 'nstr',
            'vehicle_type_lwm', 'weight__ton',
            'orig_dc', 'dest_dc', 'orig_tt', 'dest_tt'])

    return shipments


def combine_shipments(
    shipmentsCK: pd.DataFrame,
    shipmentsDirectMS: pd.DataFrame, shipmentsRailMS: pd.DataFrame, shipmentsWaterMS: pd.DataFrame
) -> pd.DataFrame:
    """
    Combineer de vier soorten zendingen in DataFrame met alle zendingen.

    Args:
        shipmentsCK (pd.DataFrame): _description_
        shipmentsDirectMS (pd.DataFrame): _description_
        shipmentsRailMS (pd.DataFrame): _description_
        shipmentsWaterMS (pd.DataFrame): _description_

    Returns:
        _type_: _description_
    """
    # Extra kolom voor verschijningsvorm
    shipmentsCK['container'] = 1
    shipmentsDirectMS['container'] = 0
    shipmentsRailMS['container'] = 0
    shipmentsWaterMS['container'] = 0

    shipmentsCK['ms_type'] = 0
    shipmentsDirectMS['ms_type'] = 1
    shipmentsRailMS['ms_type'] = 2
    shipmentsWaterMS['ms_type'] = 3

    # Zendingen achter elkaar plakken en een identifier geven
    shipments = pd.concat([shipmentsCK, shipmentsDirectMS, shipmentsRailMS, shipmentsWaterMS])
    shipments['shipment_id'] = np.arange(len(shipments))
    shipments = shipments[['shipment_id'] + list(shipments.columns)[:-1]]
    shipments.index = np.arange(len(shipments))

    shipments['from_ucc'] = 0
    shipments['to_ucc'] = 0

    return shipments


def apply_zez_zones(
    settings: Settings, logger: logging.Logger,
    shipments: pd.DataFrame, zones: pd.DataFrame, dictNRMtoBG: dict
) -> pd.DataFrame:
    """_summary_

    Args:
        settings (Settings): Configuration.
        shipments (pd.DataFrame): _description_
        zones (pd.DataFrame): _description_
        dictNRMtoBG (dict): _description_

    Raises:
        Exception: _description_

    Returns:
        pd.DataFrame: _description_
    """
    if not settings.module.apply_zez:
        return shipments

    logger.debug("\tZendingen herrouteren via consolidatiecentra...")

    dimLS = settings.dimensions.logistic_segment
    dimVT = settings.dimensions.vehicle_type_lwm
    maxDutchZone = settings.zones.maximum_dutch

    zonesZEZ = pd.read_csv(settings.get_path("ZonesZEZ"), sep='\t')
    zezToUCC = dict(
        (zonesZEZ.at[i, 'zone_nrm'], zonesZEZ.at[i, 'zone_nrm_ucc']) for i in zonesZEZ.index)
    isZEZ = np.zeros(len(zones), dtype=int)
    isZEZ[zonesZEZ['zone_nrm'].values - 1] = 1

    probsVehicleTypesZEZ = get_probs_vt_zez(
        settings, len(dimLS), len(dimVT))

    probsConsolidationZEZ = pd.read_csv(
        settings.get_path("KansenConsolidatieZEZ"), sep='\t')
    probsConsolidationZEZ = {
        probsConsolidationZEZ.at[i, 'logistic_segment']: probsConsolidationZEZ.at[i, 'probability']
        for i in probsConsolidationZEZ.index}

    # Bepaal welke zendingen van/naar een ZE-zone gaan
    whereOrigZEZ = np.array([
        i for i in shipments[shipments['orig_bg'] <= maxDutchZone].index
        if isZEZ[int(shipments.at[i, 'orig_nrm'] - 1)]], dtype=int)
    whereDestZEZ = np.array([
        i for i in shipments[shipments['dest_bg'] <= maxDutchZone].index
        if isZEZ[int(shipments.at[i, 'dest_nrm'] - 1)]], dtype=int)
    setWhereOrigZEZ = set(whereOrigZEZ)
    setWhereDestZEZ = set(whereDestZEZ)

    whereBothZEZ = [
        i for i in shipments.index if i in setWhereOrigZEZ and i in setWhereDestZEZ]

    shipmentsArray = np.array(shipments, dtype=object)

    # Initialiseer
    newShipmentsArray = np.zeros(shipments.shape, dtype=object)

    j = 0
    for col, dtype in {
        'shipment_id': int, 'orig_nrm': int, 'dest_nrm': int, 'orig_bg': int, 'dest_bg': int,
        'commodity': int, 'logistic_segment': int, 'nstr': int,
        'vehicle_type_lwm': int, 'weight__ton': float,
        'orig_dc': int, 'dest_dc': int, 'orig_tt': int, 'dest_tt': int,
        'container': int, 'ms_type': int,
        'from_ucc': int, 'to_ucc': int
    }.items():
        try:
            shipments[col] = shipments[col].astype(dtype)
            shipmentsArray[:, j] = np.array(shipmentsArray[:, j], dtype=dtype)
            newShipmentsArray[:, j] = np.array(newShipmentsArray[:, j], dtype=dtype)
            j += 1
        except KeyError:
            raise Exception(f"Interne variabele 'shipments' bevat geen kolom genaamd '{col}'.")

    count = 0

    for i in whereOrigZEZ:

        if i not in setWhereDestZEZ:
            ls = shipmentsArray[i, 6]

            if probsConsolidationZEZ[ls] > np.random.rand():
                trueOrigin = shipmentsArray[i, 1]
                newOrigin = zezToUCC[trueOrigin]

                # Redirect to UCC
                shipmentsArray[i, 1] = newOrigin
                shipmentsArray[i, -2] = 1

                # Add shipment from ZEZ to UCC
                newShipmentsArray[count, :] = shipmentsArray[i, :].copy()
                newShipmentsArray[count, [1, 2, -2, -1, 8]] = (
                    trueOrigin, newOrigin, 0, 1,
                    draw_choice(probsVehicleTypesZEZ[ls, :]))
                newShipmentsArray[count, 14] = 0

                count += 1

    for i in whereDestZEZ:

        if i not in setWhereOrigZEZ:
            ls = shipmentsArray[i, 6]

            if probsConsolidationZEZ[ls] > np.random.rand():
                trueDest = shipmentsArray[i, 2]
                newDest = zezToUCC[trueDest]

                # Redirect to UCC
                shipmentsArray[i, 2] = newDest
                shipmentsArray[i, -1] = 1

                # Add shipment to ZEZ from UCC
                newShipmentsArray[count, :] = shipmentsArray[i, :].copy()
                newShipmentsArray[count, [1, 2, -2, -1, 8]] = [
                    newDest, trueDest, 1, 0, draw_choice(probsVehicleTypesZEZ[ls, :])]
                newShipmentsArray[count, 14] = 0

                count += 1

    # Also change vehicle type and rerouting for shipments
    # that go from a ZEZ area to a ZEZ area
    for i in whereBothZEZ:
        ls = shipmentsArray[i, 6]
        trueOrigin = shipmentsArray[i, 1]
        trueDest = shipmentsArray[i, 2]

        # Als het binnen dezelfde gemeente (i.e. dezelfde ZEZ) blijft,
        # dan hoeven we alleen maar het voertuigtype aan te passen
        # Assume dangerous goods keep the same vehicle type
        if (zones.at[trueOrigin, 'name_gemeente'] == zones.at[trueDest, 'name_gemeente']):
            if ls != 7:
                shipmentsArray[i, 10] = draw_choice(probsVehicleTypesZEZ[ls, :])

        # Als het van de ene ZEZ naar de andere ZEZ gaat,
        # maken we 3 legs: ZEZ1--> UCC1, UCC1-->UCC2, UCC2-->ZEZ2
        else:
            if probsConsolidationZEZ[ls] > np.random.rand():
                newOrigin = zezToUCC[trueOrigin]
                newDest = zezToUCC[trueDest]

                # Redirect origin to UCC
                shipmentsArray[i, 1] = newOrigin
                shipmentsArray[i, -2] = 1

                # Add shipment from ZEZ1 to UCC1
                newShipmentsArray[count, :] = shipmentsArray[i, :].copy()
                newShipmentsArray[count, [1, 2, -1, -1, 8]] = [
                    trueOrigin, newOrigin, 0, 1,
                    draw_choice(probsVehicleTypesZEZ[ls, :])]
                newShipmentsArray[count, 14] = 0

                count += 1

                # Redirect destination to UCC
                shipmentsArray[i, 2] = newDest
                shipmentsArray[i, -1] = 1

                # Add shipment from UCC2 to ZEZ2
                newShipmentsArray[count, :] = shipmentsArray[i, :].copy()
                newShipmentsArray[count, [1, 2, -1, -1, 8]] = [
                    newDest, trueDest, 1, 0, draw_choice(probsVehicleTypesZEZ[ls, :])]
                newShipmentsArray[count, 14] = 0

                count += 1

    newShipmentsArray = newShipmentsArray[np.arange(count), :]

    newShipmentsArray[:, 3] = [dictNRMtoBG[x]for x in newShipmentsArray[:, 1]]
    newShipmentsArray[:, 4] = [dictNRMtoBG[x]for x in newShipmentsArray[:, 2]]

    shipments = pd.DataFrame(
        np.r_[shipmentsArray, newShipmentsArray], columns=shipments.columns)
    shipments['shipment_id'] = np.arange(len(shipments))
    shipments.index = np.arange(len(shipments))

    return shipments


def write_shipments_to_file(
    settings: Settings, logger: logging.Logger, shipments: pd.DataFrame, sep: str = '\t'
):
    """
    Schrijf de aangemaakte zendingen weg naar een tekstbestand.

    Args:
        path (str): _description_
        shipments (pd.DataFrame): _description_
        sep (str, optional): _description_. Defaults to '\t'.

    Raises:
        Exception: Header niet gevonden in lokale variabele 'shipments'.
        Exception: Kolom in lokale variabele 'shipments' kan niet naar juiste datatype
            geconverteerd worden.
    """
    logger.debug("\tWegschrijven zendingen naar tekstbestand...")

    # Datatypes goed zetten
    dtypes = {
        'orig_nrm': int, 'dest_nrm': int, 'orig_bg': int, 'dest_bg': int,
        'commodity': int, 'logistic_segment': int, 'nstr': int, 'container': int, 'ms_type': int,
        'vehicle_type_lwm': int,
        'weight__ton': float,
        'orig_dc': int, 'dest_dc': int, 'orig_tt': int, 'dest_tt': int,
        'from_ucc': int, 'to_ucc': int}

    for col, dtype in dtypes.items():
        try:
            shipments[col] = shipments[col].astype(dtype)
        except KeyError:
            raise Exception(
                f"Interne variabele 'shipments' bevat geen kolom genaamd '{col}'.")
        except ValueError:
            raise Exception(
                f"Kon niet alle waardes behorend bij kolom '{col}' in interne variabele " +
                f"'shipments' niet converteren naar datatype {dtype}.")

    shipments['weight__ton'] = shipments['weight__ton'].round(5)

    # Wegschrijven
    shipments.to_csv(settings.get_path("Zendingen"), sep=sep, index=False)


def write_shipments_to_shp(
    settings: Settings, logger: logging.Logger, shipments: pd.DataFrame, zones: pd.DataFrame
):
    """_summary_

    Args:
        path (str): _description_
        shipments (pd.DataFrame): _description_
        zones (pd.DataFrame): _description_
        applyZEZ (bool): _description_
    """
    if not settings.module.write_shape_ship:
        return

    logger.debug("\tWegschrijven zendingen naar shapefile...")

    zonesX = np.array(zones['x_rd__meter'])
    zonesY = np.array(zones['y_rd__meter'])

    Ax = zonesX[shipments['orig_nrm'].values - 1]
    Ay = zonesY[shipments['orig_nrm'].values - 1]
    Bx = zonesX[shipments['dest_nrm'].values - 1]
    By = zonesY[shipments['dest_nrm'].values - 1]

    # Initialize shapefile fields
    w = shp.Writer(settings.get_path("ZendingenShape"))
    w.field('shipment_id', 'N', size=6, decimal=0)
    w.field('orig_nrm', 'N', size=4, decimal=0)
    w.field('dest_nrm', 'N', size=4, decimal=0)
    w.field('orig_bg', 'N', size=3, decimal=0)
    w.field('dest_bg', 'N', size=3, decimal=0)
    w.field('commodity', 'N', size=2, decimal=0)
    w.field('logistic_segment', 'N', size=2, decimal=0)
    w.field('nstr', 'N', size=2, decimal=0)
    w.field('vehicle_type_lwm', 'N', size=2, decimal=0)
    w.field('weight__ton', 'N', size=4, decimal=2)
    w.field('orig_dc', 'N', size=5, decimal=0)
    w.field('dest_dc', 'N', size=5, decimal=0)
    w.field('orig_tt', 'N', size=5, decimal=0)
    w.field('dest_tt', 'N', size=5, decimal=0)
    w.field('container', 'N', size=2, decimal=0)
    w.field('ms_type', 'N', size=2, decimal=0)
    w.field('from_ucc', 'N', size=2, decimal=0)
    w.field('to_ucc', 'N', size=2, decimal=0)

    dbfData = np.array(shipments, dtype=object)
    numShips = dbfData.shape[0]
    for i in range(numShips):
        # Add geometry
        w.line([[[Ax[i], Ay[i]], [Bx[i], By[i]]]])

        # Add data fields
        w.record(*dbfData[i, :])

    w.close()


@njit
def choice_model_ssvt(
    costFreightKm: np.ndarray, costFreightHr: np.ndarray,
    gg: int,
    fromDC: int, toDC: int,
    travTime: float, distance: float,
    absoluteShipmentSizes: np.ndarray,
    truckCapacities: np.ndarray,
    betaTransportCosts: float, betaInventoryCosts: float,
    betaFromDC: float, betaToDC: float,
    constsShortHaulVT: np.ndarray, constsLongHaulVT: np.ndarray, constsSS: np.ndarray,
    numVT: int, numSS: int,
    container: bool = False
) -> int:
    """
    Bereken de utilities en trek een keuze voor een voertuigtype en zendingsgrootte.

    Args:
        costFreightKm (np.ndarray): _description_
        costFreightHr (np.ndarray): _description_
        gg (int): _description_
        fromDC (int): _description_
        toDC (int): _description_
        travTime (float): _description_
        distance (float): _description_
        absoluteShipmentSizes (np.ndarray): _description_
        truckCapacities (np.ndarray): _description_
        betaTransportCosts (float): _description_
        betaInventoryCosts (float): _description_
        betaFromDC (float): _description_
        betaToDC (float): _description_
        constantsShortHaulVT (np.ndarray): _description_
        constantsLongHaulVT (np.ndarray): _description_
        constantsSS (np.ndarray): _description_
        numVT (int): _description_
        numSS (int): _description_
        container (bool, optional): _description_. Defaults to False.

    Returns:
        int: De index van het getrokken alternatief.
    """
    # Determine the utility and probability for each alternative
    utilities = np.zeros(numVT * numSS, dtype=float64)

    inventoryCosts = absoluteShipmentSizes
    longHaul = int(distance > 100)
    shortHaul = int(not longHaul)

    for ss in range(numSS):
        for vt in range(numVT):
            index = ss * numVT + vt
            costPerHr = costFreightHr[gg - 1, vt]
            costPerKm = costFreightKm[gg - 1, vt]

            # If vehicle type is available for the current goods type
            if costPerHr > 0:
                transportCosts = costPerHr * travTime + costPerKm * distance

                # Multiply transport costs by number of required vehicles
                transportCosts *= np.ceil(absoluteShipmentSizes[ss] / truckCapacities[vt])

                # Utility function
                if container and vt != 5:
                    utilities[index] = -100
                else:
                    utilities[index] = (
                        betaTransportCosts * transportCosts +
                        betaInventoryCosts * inventoryCosts[ss] +
                        betaFromDC * fromDC * (vt == 0) +
                        betaToDC * toDC * (vt in [3, 4, 5]) +
                        constsShortHaulVT[vt] * shortHaul +
                        constsLongHaulVT[vt] * longHaul +
                        constsSS[ss])
            else:
                utilities[index] = -100

    probs = np.exp(utilities) / np.sum(np.exp(utilities))
    cumProbs = np.cumsum(probs)

    # Sample one choice based on the cumulative probability distribution
    ssvt = draw_choice(cumProbs)

    return ssvt


@njit
def draw_choice(cumProbs: np.ndarray) -> int:
    '''
    Trek een keuze uit een array met cumulatieve kansen.

    Args:
        cumProbs (numpy.ndarray): _description_

    Returns:
        int: De index van het getrokken alternatief.
    '''
    nAlt = len(cumProbs)

    rand = np.random.rand()
    for alt in range(nAlt):
        if cumProbs[alt] >= rand:
            return alt

    raise Exception(
        '\nError in function "draw_choice", random draw was ' +
        'outside range of cumulative probability distribution.')
