import logging

import numpy as np
import pandas as pd
import shapefile as shp

from calculation.common.data import get_euclidean_skim
from calculation.common.io import get_skims
from calculation.common.params import set_seed

from settings import Settings


def run_module(settings: Settings, logger: logging.Logger):

    set_seed(settings.module.seed_parcel_schd, logger, 'seed-parcel-schd')

    logger.debug("\tInladen en prepareren invoerdata...")

    # Importeer de NRM-moederzones
    zones = pd.read_csv(settings.get_path("ZonesNRM"), sep='\t')
    zones = zones[zones['landsdeel'] <= 4]
    zones.index = zones['zone_nrm']
    numZones = len(zones)

    # De pakketvraag uit de vorige module
    parcels = get_parcels(settings, zones)

    # Pakketsorteercentra
    parcelNodes = pd.read_csv(settings.get_path("PakketSorteercentra"), sep='\t')
    parcelNodes.index = parcelNodes['depot'].astype(int)
    parcelNodes = parcelNodes.sort_index()

    parcelNodesCEP = dict(
        (parcelNodes.at[i, 'depot'], parcelNodes.at[i, 'courier'])
        for i in parcelNodes.index)

    # Vraagparameters
    parcelParams = pd.read_csv(settings.get_path("PakketParameters"), sep='\t')
    parcelParams = dict(
        (parcelParams.at[i, 'parameter'], parcelParams.at[i, 'value'])
        for i in parcelParams.index)
    maxLoadVan = int(parcelParams['max_num_parcels_van'])

    if settings.module.apply_zez:
        capacities = pd.read_csv(settings.get_path("LaadvermogensLWM"), sep='\t')
        capacities = dict(
            (capacities.at[i, 'vehicle_type_lwm'], capacities.at[i, 'weight__kgram'])
            for i in capacities.index)
        maxLoadLEVV = int(parcelParams['max_num_parcels_levv'])

    # Haal skim met reistijden en afstanden op
    skimTravTime, skimDistance = get_skims(
        settings, logger, zones, freight=True, changeZeroValues=True)

    numZonesSkim = int(len(skimTravTime)**0.5)

    # Alleen de zones in Nederland doen ertoe voor de pakkettenmodule
    skimTravTime = skimTravTime.reshape(numZonesSkim, numZonesSkim)
    skimTravTime = skimTravTime[:numZones, :]
    skimTravTime = skimTravTime[:, :numZones]
    skimDistance = skimDistance.reshape(numZonesSkim, numZonesSkim)
    skimDistance = skimDistance[:numZones, :]
    skimDistance = skimDistance[:, :numZones]

    # Intrazonal impedances
    for i in range(numZones):
        skimTravTime[i, i] = 0.7 * np.min(skimTravTime[i, skimTravTime[i, :] > 0])
    for i in range(numZones):
        skimDistance[i, i] = 0.7 * np.min(skimDistance[i, skimDistance[i, :] > 0])

    # Weer plat slaan naar lange array
    skimTravTime = skimTravTime.flatten()
    skimDistance = skimDistance.flatten()

    logger.debug('\tRuimtelijke clusters vormen...')

    # A measure of euclidean distance based on the coordinates
    skimEuclidean = get_euclidean_skim(zones['x_rd__meter'].values, zones['y_rd__meter'].values)
    skimEuclidean /= np.sum(skimEuclidean)

    # To prevent instability related to possible mistakes in skim,
    # use average of skim and euclidean distance (both normalized to a sum of 1)
    skimClustering = skimDistance.copy()
    skimClustering /= np.sum(skimClustering)
    skimClustering += skimEuclidean

    del skimEuclidean

    if settings.module.apply_zez:

        # Divide parcels into the 4 tour types, namely:
        # 0: Depots to households
        # 1: Depots to UCCs
        # 2: From UCCs, by van
        # 3: From UCCs, by LEVV
        numTourType = 4

        parcelsUCC = {}
        parcelsUCC[0] = pd.DataFrame(parcels[
            (parcels['from_ucc'] == 0) & (parcels['to_ucc'] == 0)])
        parcelsUCC[1] = pd.DataFrame(parcels[
            (parcels['from_ucc'] == 0) & (parcels['to_ucc'] == 1)])
        parcelsUCC[2] = pd.DataFrame(parcels[
            (parcels['from_ucc'] == 1) & (parcels['vehicle_type_lwm'] == 8)])
        parcelsUCC[3] = pd.DataFrame(parcels[
            (parcels['from_ucc'] == 1) & (parcels['vehicle_type_lwm'] >= 9)])

        # Cluster parcels based on proximity and
        # constrained by vehicle capacity
        for i in range(numTourType - 1):
            logger.debug('\t\tTour type ' + str(i + 1) + '...')

            parcelsUCC[i] = cluster_parcels(parcelsUCC[i], maxLoadVan, skimClustering)

        # LEVV have smaller capacity
        logger.debug(f'\t\tTour type {numTourType}...')

        parcelsUCC[numTourType - 1] = cluster_parcels(
            parcelsUCC[numTourType - 1], maxLoadLEVV, skimClustering)

        # Aggregate parcels based on depot, cluster and destination
        for i in range(numTourType):

            if i <= 1:
                parcelsUCC[i] = pd.pivot_table(
                    parcelsUCC[i],
                    values=['parcel_id'],
                    index=['depot_id', 'cluster', 'orig_nrm', 'dest_nrm'],
                    aggfunc={'parcel_id': 'count'})
                parcelsUCC[i] = parcelsUCC[i].rename(columns={'parcel_id': 'n_parcels'})
                parcelsUCC[i]['Depot'] = [x[0] for x in parcelsUCC[i].index]
                parcelsUCC[i]['cluster'] = [x[1] for x in parcelsUCC[i].index]
                parcelsUCC[i]['Orig'] = [x[2] for x in parcelsUCC[i].index]
                parcelsUCC[i]['Dest'] = [x[3] for x in parcelsUCC[i].index]

            else:
                parcelsUCC[i] = pd.pivot_table(
                    parcelsUCC[i],
                    values=['parcel_id'],
                    index=['orig_nrm', 'cluster', 'dest_nrm'],
                    aggfunc={'parcel_id': 'count'})
                parcelsUCC[i] = parcelsUCC[i].rename(columns={'parcel_id': 'n_parcels'})
                parcelsUCC[i]['Depot'] = [x[0] for x in parcelsUCC[i].index]
                parcelsUCC[i]['cluster'] = [x[1] for x in parcelsUCC[i].index]
                parcelsUCC[i]['Orig'] = [x[0] for x in parcelsUCC[i].index]
                parcelsUCC[i]['Dest'] = [x[2] for x in parcelsUCC[i].index]

            parcelsUCC[i].index = np.arange(len(parcelsUCC[i]))

    if not settings.module.apply_zez:

        # Cluster parcels based on proximity and constrained by vehicle capacity
        parcels = cluster_parcels(parcels, maxLoadVan, skimClustering)

        # Aggregate parcels based on depot, cluster and destination
        parcels = pd.pivot_table(
            parcels,
            values=['parcel_id'],
            index=['depot_id', 'cluster', 'orig_nrm', 'dest_nrm'],
            aggfunc={'parcel_id': 'count'})
        parcels = parcels.rename(columns={'parcel_id': 'n_parcels'})
        parcels['Depot'] = [x[0] for x in parcels.index]
        parcels['cluster'] = [x[1] for x in parcels.index]
        parcels['Orig'] = [x[2] for x in parcels.index]
        parcels['Dest'] = [x[3] for x in parcels.index]
        parcels.index = np.arange(len(parcels))

    del skimClustering

    logger.debug('\tRondritten vormen...')

    if settings.module.apply_zez:
        schedules = [
            create_schedules(
                parcelsUCC[tourType], skimTravTime, skimDistance, parcelNodesCEP, tourType + 1)
            for tourType in range(numTourType)]
        schedules = pd.concat(schedules)
        schedules.index = np.arange(len(schedules))

    else:
        tourType = 0
        schedules = create_schedules(
            parcels, skimTravTime, skimDistance, parcelNodesCEP, tourType + 1)

    logger.debug('\tPakketrondritten wegschrijven naar tekstbestand.')

    schedules.to_csv(settings.get_path("PakketRondritten"), index=False, sep='\t')

    if settings.module.write_shape_parcel_schd:

        logger.debug('\tPakketrondritten wegschrijven naar shapefile.')

        write_parcel_schedules_to_shp(settings, schedules, parcelNodes, zones)

    logger.debug('\tRitmatrices opstellen en wegschrijven naar tekstbestand.')

    tripMatrix = get_trip_matrix(schedules)

    tripMatrix.to_csv(
        settings.get_path("RittenMatrixPakketNRM"), index=False, sep='\t')


def get_parcels(
    settings: Settings,
    zones: pd.DataFrame
) -> pd.DataFrame:
    """
    Lees het tekstbestand met de pakketvraag in en herstructureer deze zodat
    iedere rij 1 pakket is.

    Args:
        settings (Settings): _description_
        zones (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: Alle pakketten die vervoerd moeten worden.
    """
    # Inlezen CSV met pakketvraag
    parcelsAggr = pd.read_csv(settings.get_path("PakketVraag"), sep='\t')
    numParcelsTotal = np.sum(parcelsAggr['n_parcels'])

    # Parcels herstructureren zodat iedere rij 1 pakket is
    parcelColumns = [
        'parcel_id',
        'orig_nrm', 'dest_nrm',
        'depot_id', 'courier',
        'vehicle_type_lwm']
    if settings.module.apply_zez:
        parcelColumns.append('from_ucc')
        parcelColumns.append('to_ucc')

    parcels = np.zeros((numParcelsTotal, len(parcelColumns)), dtype=object)

    count = 0
    for i in range(len(parcelsAggr)):
        orig = parcelsAggr.at[i, 'orig_nrm']
        dest = parcelsAggr.at[i, 'dest_nrm']
        numParcels = parcelsAggr.at[i, 'n_parcels']
        depotNo = parcelsAggr.at[i, 'depot_id']
        cep = parcelsAggr.at[i, 'courier']
        vehType = parcelsAggr.at[i, 'vehicle_type_lwm']

        parcels[count:(count + numParcels), 1] = orig
        parcels[count:(count + numParcels), 2] = dest
        parcels[count:(count + numParcels), 3] = depotNo
        parcels[count:(count + numParcels), 4] = cep
        parcels[count:(count + numParcels), 5] = vehType

        if settings.module.apply_zez:
            fromUCC = parcelsAggr.at[i, 'from_ucc']
            toUCC = parcelsAggr.at[i, 'to_ucc']

            parcels[count:(count + numParcels), 6] = fromUCC
            parcels[count:(count + numParcels), 7] = toUCC

        count += numParcels

    # Stop in een DataFrame met de juiste kolomnamen en data types
    parcels = pd.DataFrame(parcels, columns=parcelColumns)
    parcels['parcel_id'] = np.arange(numParcelsTotal)
    parcels = parcels.astype({
        'parcel_id': int,
        'orig_nrm': int, 'dest_nrm': int,
        'depot_id': int, 'courier': str,
        'vehicle_type_lwm': int})

    # Coordinaten bestemming
    zonesX = zones['x_rd__meter'].values
    zonesY = zones['y_rd__meter'].values
    parcels['X'] = [zonesX[x - 1] for x in parcels['dest_nrm'].values]
    parcels['Y'] = [zonesY[x - 1] for x in parcels['dest_nrm'].values]

    return parcels


def create_schedules(
    parcelsAggr: pd.DataFrame,
    skimTravTime: np.ndarray, skimDistance: np.ndarray,
    parcelNodesCEP: dict,
    tourType: int
) -> pd.DataFrame:
    """
    Vorm de rondritten en bewaar de informatie in een DataFrame.

    Args:
        parcelsAggr (pd.DataFrame): _description_
        skimTravTime (np.ndarray): _description_
        skimDistance (np.ndarray): _description_
        parcelNodesCEP (dict): _description_
        tourType (int): _description_

    Returns:
        pd.DataFrame: De rondritten.
    """

    numZones = int(len(skimTravTime)**0.5)

    tours = {}
    parcelsDelivered = {}
    numTripsTotal = 0

    for depot in np.unique(parcelsAggr['Depot']):
        depotParcels = parcelsAggr[parcelsAggr['Depot'] == depot]

        tours[depot] = {}
        parcelsDelivered[depot] = {}

        for cluster in np.unique(depotParcels['cluster']):
            tour = []

            clusterParcels = depotParcels[depotParcels['cluster'] == cluster]
            depotZone = list(clusterParcels['Orig'])[0]
            destZones = list(clusterParcels['Dest'])
            nParcelsPerZone = dict(zip(destZones, clusterParcels['n_parcels']))

            # Nearest neighbor
            tour.append(depotZone)
            for i in range(len(destZones)):
                distances = [
                    skimDistance[(tour[i] - 1) * numZones + (dest - 1)]
                    for dest in destZones]
                nextIndex = np.argmin(distances)
                tour.append(destZones[nextIndex])
                destZones.pop(nextIndex)
            tour.append(depotZone)

            # Shuffle the order of tour locations and accept the shuffle
            # if it reduces the tour distance
            numStops = len(tour)
            tour = np.array(tour, dtype=int)
            tourDist = np.sum(skimDistance[(tour[:-1] - 1) * numZones + (tour[1:] - 1)])
            if numStops > 4:
                for shiftLocA in range(1, numStops - 1):
                    for shiftLocB in range(1, numStops - 1):
                        if shiftLocA != shiftLocB:
                            swappedTour = tour.copy()
                            swappedTour[shiftLocA] = tour[shiftLocB]
                            swappedTour[shiftLocB] = tour[shiftLocA]
                            swappedTourDist = np.sum(
                                skimDistance[
                                    (swappedTour[:-1] - 1) * numZones + (swappedTour[1:] - 1)])

                            if swappedTourDist < tourDist:
                                tour = swappedTour.copy()
                                tourDist = swappedTourDist

            # Add current tour to dictionary with all formed tours
            tours[depot][cluster] = list(tour.copy())

            # Store the number of parcels delivered at each
            # location in the tour
            numParcelsPerStop = []
            for i in range(1, numStops - 1):
                numParcelsPerStop.append(nParcelsPerZone[tour[i]])
            numParcelsPerStop.append(0)
            parcelsDelivered[depot][cluster] = list(numParcelsPerStop.copy())

            numTripsTotal += (numStops - 1)

    schedulesCols = [
        'tour_type', 'courier',
        'depot_id', 'tour_id', 'trip_id', 'unique_id',
        'orig_nrm', 'dest_nrm',
        'n_parcels']
    schedules = np.zeros((numTripsTotal, len(schedulesCols)), dtype=object)

    tripcount = 0
    for depot in tours.keys():
        for tour in tours[depot].keys():
            for trip in range(len(tours[depot][tour]) - 1):
                orig = tours[depot][tour][trip]
                dest = tours[depot][tour][trip + 1]

                # Depot to HH (0) or UCC (1), UCC to HH by van (2)/LEVV (3)
                schedules[tripcount, 0] = tourType

                # Name of the couriers
                if tourType <= 1:
                    schedules[tripcount, 1] = parcelNodesCEP[depot]
                else:
                    schedules[tripcount, 1] = 'ConsolidatedUCC'

                # Depot_ID, Tour_ID, Trip_ID,
                # Unique ID under consideration of tour type
                schedules[tripcount, 2] = depot
                schedules[tripcount, 3] = f'{depot}_{tour}'
                schedules[tripcount, 4] = f'{depot}_{tour}_{trip}'
                schedules[tripcount, 5] = f'{depot}_{tour}_{trip}_{tourType}'

                # Origin and destination zone
                schedules[tripcount, 6] = orig
                schedules[tripcount, 7] = dest

                # Number of parcels
                schedules[tripcount, 8] = parcelsDelivered[depot][tour][trip]

                tripcount += 1

    # Place in DataFrame with the right data type per column
    schedules = pd.DataFrame(schedules, columns=schedulesCols)
    dtypes = {
        'tour_type': int, 'courier': str,
        'depot_id': int, 'tour_id': str,
        'trip_id': str, 'unique_id': str,
        'orig_nrm': int, 'dest_nrm': int,
        'n_parcels': float}
    for col in schedulesCols:
        schedules[col] = schedules[col].astype(dtypes[col])

    vehTypes = ['Van', 'Van', 'Van', 'LEVV']
    origTypes = ['Depot', 'Depot', 'UCC', 'UCC']
    destTypes = ['HH', 'UCC', 'HH', 'HH']

    schedules['vehicle_type_lwm'] = vehTypes[tourType - 1]
    schedules['orig_type'] = origTypes[tourType - 1]
    schedules['dest_type'] = destTypes[tourType - 1]

    return schedules


def cluster_parcels(
    parcels: pd.DataFrame,
    maxVehicleLoad: int,
    skimDistance: np.ndarray
) -> pd.DataFrame:
    """
    Assign parcels to clusters based on spatial proximity with cluster size constraints.
    The cluster variable is added as extra column to the DataFrame.

    Args:
        parcels (pd.DataFrame): _description_
        maxVehicleLoad (int): _description_
        skimDistance (np.ndarray): _description_

    Returns:
        pd.DataFrame: De pakketten met een cluster toegekend als laatste kolom.
    """
    depotNumbers = np.unique(parcels['depot_id'])
    numParcelsTotal = len(parcels)
    firstClusterID = 0
    numZones = int(len(skimDistance)**0.5)

    parcels.index = np.arange(numParcelsTotal)

    parcelsCluster = - np.ones(numParcelsTotal)

    # First check for depot/destination combination with more than 'maxVehicleLoad' parcels.
    # For these we don't need to use the clustering algorithm.
    counts = pd.pivot_table(
        parcels, values=['vehicle_type_lwm'], index=['depot_id', 'dest_nrm'], aggfunc=len)
    whereLargeCluster = list(counts.index[np.where(counts >= maxVehicleLoad)[0]])

    whereDepotDest = {}
    parcelsDepotID = np.array(parcels['depot_id'], dtype=int)
    parcelsDestZone = np.array(parcels['dest_nrm'], dtype=int)
    for i in range(numParcelsTotal):
        try:
            whereDepotDest[(parcelsDepotID[i], parcelsDestZone[i])].append(i)
        except KeyError:
            whereDepotDest[(parcelsDepotID[i], parcelsDestZone[i])] = [i]

    for x in whereLargeCluster:
        depotNumber = x[0]
        destZone = x[1]

        indices = whereDepotDest[(depotNumber, destZone)]

        for i in range(int(np.floor(len(indices) / maxVehicleLoad))):
            parcelsCluster[indices[:maxVehicleLoad]] = firstClusterID
            indices = indices[maxVehicleLoad:]

            firstClusterID += 1

    parcels['cluster'] = parcelsCluster

    # For each depot, cluster remaining parcels into batches of
    # 'maxVehicleLoad' parcels
    for depotNumber in depotNumbers:
        # Select parcels of the depot that are not assigned a cluster yet
        parcelsToFit = parcels[
            (parcels['depot_id'] == depotNumber) &
            (parcels['cluster'] == -1)].copy()

        # Sort parcels descending based on distance to depot
        # so that at the end of the loop the remaining parcels
        # are all nearby the depot and form a somewhat reasonable
        # parcels cluster
        parcelsToFit['Distance'] = skimDistance[
            (parcelsToFit['orig_nrm'] - 1) * numZones + (parcelsToFit['dest_nrm'] - 1)]
        parcelsToFit = parcelsToFit.sort_values('Distance', ascending=False)
        parcelsToFitIndex = list(parcelsToFit.index)
        parcelsToFit.index = np.arange(len(parcelsToFit))
        dests = np.array(parcelsToFit['dest_nrm'])

        # How many tours are needed to deliver these parcels
        numToursNeeded = int(np.ceil(len(parcelsToFit) / maxVehicleLoad))

        # In the case of 1 tour it's simple, all parcels belong to the
        # same cluster
        if numToursNeeded == 1:
            parcels.loc[parcelsToFitIndex, 'cluster'] = firstClusterID
            firstClusterID += 1

        # When there are multiple tours needed, the heuristic is
        # a little bit more complex
        else:
            clusters = np.ones(len(parcelsToFit), dtype=int) * -1

            for tour in range(numToursNeeded):
                # Select the first parcel for the new cluster that
                # is now initialized
                yetAssigned = (clusters != -1)
                notYetAssigned = np.where(~yetAssigned)[0]
                firstParcelIndex = notYetAssigned[0]
                clusters[firstParcelIndex] = firstClusterID

                # Find the nearest {maxVehicleLoad-1} parcels to
                # this first parcel that are not in a cluster yet
                distances = skimDistance[(dests[firstParcelIndex] - 1) * numZones + (dests - 1)]
                distances[notYetAssigned[0]] = 99999
                distances[yetAssigned] = 99999
                whereFirstClusterID = (np.argsort(distances)[:(maxVehicleLoad - 1)])
                clusters[whereFirstClusterID] = firstClusterID

                firstClusterID += 1

            # Group together remaining parcels, these are all nearby the depot
            yetAssigned = (clusters != -1)
            notYetAssigned = np.where(~yetAssigned)[0]
            clusters[notYetAssigned] = firstClusterID
            firstClusterID += 1

            parcels.loc[parcelsToFitIndex, 'cluster'] = clusters

    parcels['cluster'] = parcels['cluster'].astype(int)

    return parcels


def write_parcel_schedules_to_shp(
    settings: Settings,
    schedules: pd.DataFrame,
    parcelNodes: pd.DataFrame,
    zones: pd.DataFrame
):
    """_summary_

    Args:
        settings (Settings): _description_
        schedules (pd.DataFrame): _description_
        parcelNodes (pd.DataFrame): _description_
        zones (pd.DataFrame): _description_

    Raises:
        Exception: _description_
        Exception: _description_
    """
    zonesX = zones['x_rd__meter'].values
    zonesY = zones['y_rd__meter'].values

    # Initialize arrays with coordinates
    numRecords = schedules.shape[0]
    Ax = np.zeros(numRecords, dtype=float)
    Ay = np.zeros(numRecords, dtype=float)
    Bx = np.zeros(numRecords, dtype=float)
    By = np.zeros(numRecords, dtype=float)

    # Determine coordinates of LineString for each trip
    tripIDs = [x.split('_')[-1] for x in schedules['trip_id']]
    tourTypes = np.array(schedules['tour_type'], dtype=int)
    depotIDs = np.array(schedules['depot_id'])

    for i in schedules.index:

        # First trip of tour
        if tripIDs[i] == '0' and tourTypes[i] <= 1:
            Ax[i] = parcelNodes.at[depotIDs[i], 'x_rd__meter']
            Ay[i] = parcelNodes.at[depotIDs[i], 'y_rd__meter']
            Bx[i] = zonesX[schedules['dest_nrm'][i] - 1]
            By[i] = zonesY[schedules['dest_nrm'][i] - 1]

        # Last trip of tour
        if i == (numRecords - 1) and tourTypes[i] <= 1:
            Ax[i] = zonesX[schedules['orig_nrm'][i] - 1]
            Ay[i] = zonesY[schedules['orig_nrm'][i] - 1]
            Bx[i] = parcelNodes.at[depotIDs[i], 'x_rd__meter']
            By[i] = parcelNodes.at[depotIDs[i], 'y_coord']
        elif tripIDs[i + 1] == '0' and tourTypes[i] <= 1:
            Ax[i] = zonesX[schedules['orig_nrm'][i] - 1]
            Ay[i] = zonesY[schedules['orig_nrm'][i] - 1]
            Bx[i] = parcelNodes.at[depotIDs[i], 'x_rd__meter']
            By[i] = parcelNodes.at[depotIDs[i], 'y_rd__meter']

        # Intermediate trips of tour
        else:
            Ax[i] = zonesX[schedules['orig_nrm'][i] - 1]
            Ay[i] = zonesY[schedules['orig_nrm'][i] - 1]
            Bx[i] = zonesX[schedules['dest_nrm'][i] - 1]
            By[i] = zonesY[schedules['dest_nrm'][i] - 1]

    # Bestandsnaam
    path = settings.get_path("PakketRondrittenShape")
    if path is None:
        raise Exception('Kan geen bestandsnaam ophalen voor "PakketRondrittenShape".')
    if type(path) == str:
        extension = path.split('.')[-1]
        if extension != 'shp':
            raise Exception(
                'Vul een bestandsnaam met extentie "shp" in voor "PakketRondrittenShape".')

    # Initialize shapefile fields
    w = shp.Writer(path)
    w.field('tour_type', 'N', size=2, decimal=0)
    w.field('courier',   'C', size=max([len(x) for x in schedules['courier'].values]))
    w.field('depot_id',  'N', size=4, decimal=0)
    w.field('tour_id',   'C', size=max([len(x) for x in schedules['tour_id'].values]))
    w.field('trip_id',   'C', size=max([len(x) for x in schedules['trip_id'].values]))
    w.field('unique_id', 'C', size=max([len(x) for x in schedules['unique_id'].values]))
    w.field('orig_nrm',  'N', size=4, decimal=0)
    w.field('dest_nrm',  'N', size=4, decimal=0)
    w.field('n_parcels', 'N', size=8, decimal=0)
    w.field('orig_type', 'C', size=max([len(x) for x in schedules['orig_type'].values]))
    w.field('dest_type', 'C', size=max([len(x) for x in schedules['dest_type'].values]))

    dbfData = np.array(schedules, dtype=object)

    for i in range(numRecords):
        # Add geometry
        w.line([[
            [Ax[i], Ay[i]],
            [Bx[i], By[i]]]])

        # Add data fields
        w.record(*dbfData[i, :])

    w.close()


def get_trip_matrix(deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Maak een HB-matrix met aantallen ritten op NRM-niveau.

    Args:
        deliveries (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: De HB-matrix.
    """
    cols = ['orig_nrm', 'dest_nrm', 'n_trips']
    deliveries['n_trips'] = 1

    # Gebruik N_TOT om het aantal ritten per HB te bepalen,
    # voor elk logistiek segment, voertuigtype en totaal
    tripMatrix = pd.pivot_table(
        deliveries, values=['n_trips'], index=['orig_nrm', 'dest_nrm'], aggfunc=np.sum)
    tripMatrix['orig_nrm'] = [x[0] for x in tripMatrix.index]
    tripMatrix['dest_nrm'] = [x[1] for x in tripMatrix.index]
    tripMatrix = tripMatrix[cols]

    # Assume one intrazonal trip for each zone with
    # multiple deliveries visited in a tour
    intrazonalTrips = {}
    for i in deliveries[deliveries['n_parcels'] > 1].index:
        zone = deliveries.at[i, 'dest_nrm']
        if zone in intrazonalTrips.keys():
            intrazonalTrips[zone] += 1
        else:
            intrazonalTrips[zone] = 1

    intrazonalKeys = list(intrazonalTrips.keys())
    for zone in intrazonalKeys:
        if (zone, zone) in tripMatrix.index:
            tripMatrix.at[(zone, zone), 'n_trips'] += intrazonalTrips[zone]
            del intrazonalTrips[zone]

    intrazonalTripsDF = pd.DataFrame(np.zeros((len(intrazonalTrips), 3)), columns=cols)
    intrazonalTripsDF['orig_nrm'] = intrazonalTrips.keys()
    intrazonalTripsDF['dest_nrm'] = intrazonalTrips.keys()
    intrazonalTripsDF['n_trips'] = intrazonalTrips.values()
    tripMatrix = pd.concat([tripMatrix, intrazonalTripsDF])
    tripMatrix = tripMatrix.sort_values(['orig_nrm', 'dest_nrm'])
    tripMatrix.index = np.arange(len(tripMatrix))

    return tripMatrix
