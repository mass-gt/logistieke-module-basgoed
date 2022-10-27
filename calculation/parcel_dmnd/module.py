import logging

import numpy as np
import pandas as pd
import shapefile as shp

from calculation.common.io import get_skims
from calculation.common.params import set_seed

from settings import Settings


def run_module(settings: Settings, logger: logging.Logger):

    set_seed(settings.module.seed_parcel_dmnd, logger, 'seed-parcel-dmnd')

    logger.debug("\tInladen en prepareren invoerdata...")

    # Dimensies
    dimLS = settings.dimensions.logistic_segment
    dimVT = settings.dimensions.vehicle_type_lwm
    numLS = len(dimLS)
    numVT = len(dimVT)

    # Importeer de NRM-moederzones
    zones = pd.read_csv(settings.get_path("ZonesNRM"), sep='\t')
    zones = zones[zones['landsdeel'] <= 4]
    zones.index = zones['zone_nrm']
    numZones = len(zones)

    # Pakketsorteercentra
    parcelNodes = pd.read_csv(settings.get_path("PakketSorteercentra"), sep='\t')
    parcelNodes.index = parcelNodes['depot'].astype(int)
    parcelNodes = parcelNodes.sort_index()
    numParcelNodes = len(parcelNodes)

    # Vraagparameters
    parcelParams = pd.read_csv(settings.get_path("PakketParameters"), sep='\t')
    parcelParams = dict(
        (parcelParams.at[i, 'parameter'], parcelParams.at[i, 'value'])
        for i in parcelParams.index)

    # Marktaandelen van de postkoeriers
    cepShares = pd.read_csv(settings.get_path("PakketKoerierAandelen"), sep='\t')
    cepShares.index = cepShares['courier']
    cepList = np.unique(parcelNodes['courier'])
    cepNodes = [np.where(parcelNodes['courier'] == str(cep)) for cep in cepList]
    cepNodes = [x[0] if x else [] for x in cepNodes]
    cepNodeDict = dict((cepList[cepNr], cepNodes[cepNr]) for cepNr in range(len(cepList)))

    # Haal skim met reistijden en afstanden op
    skimDistance = get_skims(settings, logger, zones, freight=True, changeZeroValues=True)[1]

    # Skim with travel times between parcel nodes and all other zones
    parcelSkim = np.zeros((numZones, numParcelNodes))
    for i, orig in enumerate(parcelNodes['zone_nrm'].values):
        dests = 1 + np.arange(numZones)
        parcelSkim[:, i] = np.round(skimDistance[(orig - 1) * numZones + (dests - 1)] / 1000, 4)

    logger.debug("\tPakketvraag genereren...")

    # Calculate number of parcels per zone based on number of households and jobs
    numParcelsPerZone = (
        (zones['households'] * (
            parcelParams['parcels_per_household'] / parcelParams['success_rate_b2c'])) +
        (zones['empl_totaal'] * (
            parcelParams['parcels_per_job'] / parcelParams['success_rate_b2b'])))
    numParcelsPerZone = np.array(np.round(numParcelsPerZone), dtype=int)

    # Spread over couriers based on market shares
    zones['parcels'] = 0
    for cep in cepList:
        zones['parcels_' + str(cep)] = np.round(cepShares.at[cep, 'share'] * numParcelsPerZone)
        zones['parcels_' + str(cep)] = zones['parcels_' + str(cep)].astype(int)
        zones['parcels'] += zones['parcels_' + str(cep)]

    # Total number of parcels per courier
    numParcelsTotal = int(zones['parcels'].sum())

    # Maak een DataFrame met parcels per HB en depot
    parcels = generate_parcels(
        numParcelsTotal, zones, cepNodeDict, cepList, parcelSkim, parcelNodes)

    if settings.module.apply_zez:

        logger.debug("\tPakketvraag wegschrijven naar tekstbestand...")

        # Logistic segment is 6: Parcels
        ls = [i for i in range(numLS) if 'Parcel (deliveries)' in dimLS[i]]
        if ls:
            ls = ls[0]
        else:
            raise Exception(
                'Logistiek segment "Parcel (deliveries) niet teruggevonden ' +
                'in dimensie "logistic-segment"')

        # Consolidation potential per logistic segment (for UCC scenario)
        probConsolidation = pd.read_csv(settings.get_path("KansenConsolidatieZEZ"), sep='\t')
        probConsolidation = dict(
            (probConsolidation.at[i, 'logistic_segment'], probConsolidation.at[i, 'probability'])
            for i in probConsolidation.index)
        probConsolidation = probConsolidation.get(ls, 0.0)

        # Vehicle/combustion shares (for UCC scenario)
        scenarioZEZ = pd.read_csv(settings.get_path("KansenVoertuigtypeZEZ"), sep='\t')
        scenarioZEZ = scenarioZEZ[scenarioZEZ['logistic_segment'] == ls]

        if scenarioZEZ.shape[0] == 0:
            raise Exception(
                f'Geen overgangskansen gevonden voor logistiek segment {ls} in ' +
                'invoertabel "KansenVoertuigtypeZEZ".')

        # Only vehicle shares (summed up combustion types)
        sharesVehInZEZ = pd.pivot_table(
            scenarioZEZ, values='probability', index='vehicle_type_lwm', aggfunc=sum)
        sharesVehInZEZ = np.array([
            sharesVehInZEZ.at[vt, 'probability'] if vt in sharesVehInZEZ.index else 0.0
            for vt in range(numVT)])
        sharesVehInZEZ = np.cumsum(sharesVehInZEZ)
        sharesVehInZEZ /= sharesVehInZEZ[-1]

        # Welke zones zijn zero-emissie en welke UCC hoort erbij
        zezZones = pd.read_csv(settings.get_path("ZonesZEZ"), sep='\t')
        zezArray = np.zeros(numZones, dtype=int)
        uccArray = np.zeros(numZones, dtype=int)
        zezArray[np.array(zezZones['zone_nrm'], dtype=int) - 1] = 1
        uccArray[np.array(zezZones['zone_nrm'], dtype=int) - 1] = zezZones['zone_nrm_ucc']

        logger.debug("\tPakketten herrouteren via UCC's...")

        parcels = reroute_parcels_through_uccs(
            parcels, probConsolidation, sharesVehInZEZ, zezArray, uccArray)

        numParcelsTotal = len(parcels)

    parcelsAggr = aggregate_parcels(settings, parcels)

    logger.debug("\tPakketvraag wegschrijven naar tekstbestand...")

    parcelsAggr.to_csv(settings.get_path("PakketVraag"), index=False, sep='\t')

    if settings.module.write_shape_parcel_dmnd:

        logger.debug("\tPakketvraag wegschrijven naar shapefile...")

        write_parcels_to_shp(settings, parcelsAggr, parcelNodes, zones)


def generate_parcels(
    numParcelsTotal: int,
    zones: pd.DataFrame,
    cepNodeDict: dict,
    cepList: list,
    parcelSkim: np.ndarray,
    parcelNodes: pd.DataFrame
) -> pd.DataFrame:
    """
    Maak een DataFrame met een pakket op elke rij.
    Kolommen zijn parcel_id, orig_nrm, dest_nrm en depot_id.

    Args:
        numParcelsTotal (int): _description_
        zones (pd.DataFrame): _description_
        cepNodeDict (dict): _description_
        cepList (list): _description_
        parcelSkim (np.ndarray): _description_
        parcelNodes (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: De pakketten.
    """
    # Put parcel demand in Numpy array (faster indexing)
    cols = ['parcel_id', 'orig_nrm', 'dest_nrm', 'depot_id']
    parcels = np.zeros((numParcelsTotal, len(cols)), dtype=int)
    parcelsCep = np.array(['' for i in range(numParcelsTotal)], dtype=object)

    # Now determine for each zone and courier from which
    # depot the parcels are delivered
    count = 0
    for zoneID in zones['zone_nrm']:

        if zones.at[zoneID, 'parcels'] > 0:

            for cep in cepList:
                # Select DC of current CEP based shortest distance in skim
                parcelNodeIndex = [
                    cepNodeDict[cep][
                        parcelSkim[zoneID - 1, cepNodeDict[cep]].argmin()]]

                # Fill the df parcels with parcels, zone after zone.
                # Parcels consist of ID, D and O zone and parcel node
                # number in ongoing df from index count-1 the next
                # x=no. of parcels rows, fill the cell in the
                # column Parcel_ID with a number
                n = zones.loc[zoneID, 'parcels_' + str(cep)]

                # Parcel_ID
                parcels[count:(count + n), 0] = np.arange(
                    count + 1, count + 1 + n,
                    dtype=int)

                # ORIG_NRM and DEST_NRM
                parcels[count:(count + n), 1] = (
                    parcelNodes['zone_nrm'][parcelNodeIndex[0] + 1])
                parcels[count:(count + n), 2] = zoneID

                # DEPOT_ID
                if len(parcelNodeIndex) > 1:
                    start = count
                    halfway = int(count + np.floor(n / 2))
                    end = count + n
                    parcels[start:halfway, 3] = parcelNodeIndex[0] + 1
                    parcels[halfway:end, 3] = parcelNodeIndex[1] + 1
                else:
                    parcels[count:(count + n), 3] = parcelNodeIndex[0] + 1

                # CEP
                parcelsCep[count:(count + n)] = cep

                count += zones['parcels_' + str(cep)][zoneID]

    # Put the parcel demand data back in a DataFrame
    parcels = pd.DataFrame(parcels, columns=cols)
    parcels['courier'] = parcelsCep

    # Default vehicle type for parcel deliveries: vans
    parcels['vehicle_type_lwm'] = 8

    return parcels


def reroute_parcels_through_uccs(
    parcels: pd.DataFrame,
    probConsolidation: float, sharesVehInZEZ: np.ndarray,
    zezArray: np.ndarray, uccArray: np.ndarray
) -> pd.DataFrame:
    """
    Laat een deel van de pakketten van/naar ZE-zones via een UCC gaan.

    Args:
        parcels (pd.DataFrame): _description_
        probConsolidation (float): _description_
        sharesVehInZEZ (np.ndarray): _description_
        zezArray (np.ndarray): _description_
        uccArray (np.ndarray): _description_

    Returns:
        pd.DataFrame: De pakketten, waarvan een deel gereroute door de UCCs.
    """
    parcels['from_ucc'] = 0
    parcels['to_ucc'] = 0

    parcelsDestNRM = np.array(parcels['dest_nrm'].astype(int))
    parcelsDepotID = np.array(parcels['depot_id'].astype(int))
    parcelsCEP = [str(x) for x in parcels['courier'].values]

    isDestZEZ = (
        (zezArray[parcelsDestNRM - 1] == 1) &
        (probConsolidation > np.random.rand(len(parcels))))
    whereDestZEZ = np.where(isDestZEZ)[0]

    newParcels = np.zeros(parcels.shape, dtype=object)

    count = 0

    for i in whereDestZEZ:
        trueDest = parcelsDestNRM[i]

        # Redirect to UCC
        parcels.at[i, 'dest_nrm'] = uccArray[trueDest - 1]
        parcels.at[i, 'to_ucc'] = 1

        # Add parcel set to ZEZ from UCC
        newParcels[count, 1] = uccArray[trueDest - 1]  # Origin
        newParcels[count, 2] = trueDest                # Destination
        newParcels[count, 3] = parcelsDepotID[i]       # Depot ID
        newParcels[count, 4] = parcelsCEP[i]           # Courier name
        newParcels[count, 6] = 1  # From UCC
        newParcels[count, 7] = 0  # To UCC

        # Vehicle type
        newParcels[count, 5] = np.where(sharesVehInZEZ > np.random.rand())[0][0]

        count += 1

    newParcels = pd.DataFrame(newParcels)
    newParcels.columns = parcels.columns
    newParcels = newParcels.iloc[np.arange(count), :]

    dtypes = {
        'parcel_id': int,
        'orig_nrm': int, 'dest_nrm': int,
        'depot_id': int, 'courier': str,
        'vehicle_type_lwm': int,
        'from_ucc': int, 'to_ucc': int}

    for col in dtypes.keys():
        newParcels[col] = newParcels[col].astype(dtypes[col])

    parcels = pd.concat([parcels, newParcels])
    parcels.index = np.arange(len(parcels))
    parcels['parcel_id'] = np.arange(1, len(parcels) + 1)

    return parcels


def aggregate_parcels(
    settings: Settings,
    parcels: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregeer naar aantallen pakketten i.p.v. elke rij 1 pakket. De exacte dimensies van de
    aggregatie hangen af van of 'apply-zez' aanstaat of niet.

    Args:
        settings (Settings): _description_
        parcels (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: Geaggregeerde pakketten.
    """
    if settings.module.apply_zez:
        parcelsAggr = pd.pivot_table(
            parcels,
            values=['parcel_id'],
            index=[
                'depot_id', 'courier', 'dest_nrm', 'orig_nrm',
                'vehicle_type_lwm', 'from_ucc', 'to_ucc'
            ],
            aggfunc={
                'depot_id': np.mean, 'courier': 'first',
                'orig_nrm': np.mean, 'dest_nrm': np.mean,
                'parcel_id': 'count', 'vehicle_type_lwm': np.mean,
                'from_ucc': np.mean, 'to_ucc': np.mean})
        parcelsAggr = parcelsAggr.rename(columns={'parcel_id': 'n_parcels'})
        parcelsAggr = parcelsAggr.set_index(np.arange(len(parcelsAggr)))
        parcelsAggr = parcelsAggr.reindex(
            columns=[
                'orig_nrm', 'dest_nrm',
                'n_parcels', 'depot_id',
                'courier', 'vehicle_type_lwm',
                'from_ucc', 'to_ucc'])
        parcelsAggr = parcelsAggr.astype({
            'depot_id': int,
            'orig_nrm': int, 'dest_nrm': int,
            'n_parcels': int, 'vehicle_type_lwm': int,
            'from_ucc': int, 'to_ucc': int})

    else:
        parcelsAggr = pd.pivot_table(
            parcels,
            values=['parcel_id'],
            index=['depot_id', 'courier', 'dest_nrm', 'orig_nrm', 'vehicle_type_lwm'],
            aggfunc={
                'depot_id': np.mean, 'courier': 'first',
                'orig_nrm': np.mean, 'dest_nrm': np.mean,
                'vehicle_type_lwm': np.mean, 'parcel_id': 'count'})
        parcelsAggr = parcelsAggr.rename(columns={'parcel_id': 'n_parcels'})
        parcelsAggr = parcelsAggr.set_index(np.arange(len(parcelsAggr)))
        parcelsAggr = parcelsAggr.reindex(
            columns=[
                'orig_nrm', 'dest_nrm',
                'n_parcels', 'depot_id',
                'courier', 'vehicle_type_lwm'])
        parcelsAggr = parcelsAggr.astype({
            'depot_id': int,
            'orig_nrm': int, 'dest_nrm': int,
            'n_parcels': int, 'vehicle_type_lwm': int})

    return parcelsAggr


def write_parcels_to_shp(
    settings: Settings,
    parcelsAggr: pd.DataFrame,
    parcelNodes: pd.DataFrame,
    zones: pd.DataFrame
):
    """
    Schrijf de pakketvraag naar een shapefile.

    Args:
        settings (Settings): _description_
        parcelsAggr (pd.DataFrame): _description_
        parcelNodes (pd.DataFrame): _description_
        zones (pd.DataFrame): _description_

    Raises:
        Exception: _description_
        Exception: _description_
    """
    zonesX = np.array(zones['x_rd__meter'])
    zonesY = np.array(zones['y_rd__meter'])

    # Initialize arrays with coordinates
    numRecords = parcelsAggr.shape[0]
    Ax = np.zeros(numRecords, dtype=float)
    Ay = np.zeros(numRecords, dtype=float)
    Bx = np.zeros(numRecords, dtype=float)
    By = np.zeros(numRecords, dtype=float)

    # Determine coordinates of LineString for each trip
    depotIDs = np.array(parcelsAggr['depot_id'])
    for i in parcelsAggr.index:
        if settings.module.apply_zez:
            if parcelsAggr.at[i, 'from_ucc'] == 1:
                Ax[i] = zonesX[parcelsAggr.at[i, 'orig_nrm'] - 1]
                Ay[i] = zonesY[parcelsAggr.at[i, 'orig_nrm'] - 1]
                Bx[i] = zonesX[parcelsAggr.at[i, 'dest_nrm'] - 1]
                By[i] = zonesY[parcelsAggr.at[i, 'dest_nrm'] - 1]
        else:
            Ax[i] = parcelNodes.at[depotIDs[i], 'x_rd__meter']
            Ay[i] = parcelNodes.at[depotIDs[i], 'y_rd__meter']
            Bx[i] = zonesX[parcelsAggr.at[i, 'dest_nrm'] - 1]
            By[i] = zonesY[parcelsAggr.at[i, 'dest_nrm'] - 1]

    # Bestandsnaam
    path = settings.get_path("PakketVraagShape")
    if path is None:
        raise Exception('Kan geen bestandsnaam ophalen voor "PakketVraagShape".')
    if type(path) == str:
        extension = path.split('.')[-1]
        if extension != 'shp':
            raise Exception(
                'Vul een bestandsnaam met extentie "shp" in voor "PakketVraagShape".')

    # Initialize shapefile fields
    w = shp.Writer(path)
    w.field('orig_nrm',  'N', size=4, decimal=0)
    w.field('dest_nrm',  'N', size=4, decimal=0)
    w.field('n_parcels', 'N', size=6, decimal=0)
    w.field('depot_id',  'N', size=4, decimal=0)
    w.field('courier',   'C', size=max([len(x) for x in parcelsAggr['courier'].values]))
    w.field('vehicle_type_lwm', 'N', size=2, decimal=0)
    if settings.module.apply_zez:
        w.field('from_ucc', 'N', size=2, decimal=0)
        w.field('to_ucc',   'N', size=2, decimal=0)

    dbfData = np.array(parcelsAggr, dtype=object)
    for i in range(numRecords):
        # Add geometry
        w.line([[
            [Ax[i], Ay[i]],
            [Bx[i], By[i]]]])

        # Add data fields
        w.record(*dbfData[i, :])

    w.close()
