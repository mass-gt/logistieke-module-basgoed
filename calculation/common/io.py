import array
import logging
import numpy as np
import os.path
import pandas as pd
import shapefile as shp
from typing import Tuple
from calculation.common.data import get_euclidean_skim

from settings import Settings


def read_mtx(filepath: str) -> np.ndarray:
    """
    Lees in bestand in binair formaat (.mtx).

    Args:
        filepath (str): _description_

    Returns:
        np.ndarray: _description_
    """
    mtxData = array.array('f')  # f for float
    mtxData.fromfile(
        open(filepath, 'rb'),
        os.path.getsize(filepath) // mtxData.itemsize)

    # The number of zones is in the first byte
    if len(mtxData) > 0:
        mtxData = np.array(mtxData, dtype=float)[1:]

    return mtxData


def write_mtx(
    filepath: str,
    mat: np.ndarray,
    numZones: int
):
    """
    Schrijf een array naar binair formaat (.mtx).

    Args:
        filepath (str): _description_
        mat (np.ndarray): _description_
        numZones (int): _description_
    """
    mat = np.append(numZones, mat)
    matBin = array.array('f')
    matBin.fromlist(list(mat))
    matBin.tofile(open(filepath, 'wb'))


def read_shape(
    filepath: str,
    encoding: str = 'latin1',
    returnGeometry: bool = False
):
    '''
    Read a shapefile.
    '''
    # Load the shape
    sf = shp.Reader(filepath, encoding=encoding)
    records = sf.records()
    if returnGeometry:
        geometry = sf.__geo_interface__
        geometry = geometry['features']
        geometry = [geometry[i]['geometry'] for i in range(len(geometry))]
    fields = sf.fields
    sf.close()

    # Get information on the fields in the DBF
    columns = [x[0] for x in fields[1:]]
    colTypes = [x[1:] for x in fields[1:]]
    nRecords = len(records)

    # Check for headers that appear twice
    for col in range(len(columns)):
        name = columns[col]
        whereName = [i for i in range(len(columns)) if columns[i] == name]
        if len(whereName) > 1:
            for i in range(1, len(whereName)):
                columns[whereName[i]] = (
                    str(columns[whereName[i]]) + '_' + str(i))

    # Put all the data records into a NumPy array
    #  (much faster than Pandas DataFrame)
    shape = np.zeros((nRecords, len(columns)), dtype=object)
    for i in range(nRecords):
        shape[i, :] = records[i][0:]

    # Then put this into a Pandas DataFrame with the right headers
    # and data types
    shape = pd.DataFrame(shape, columns=columns)
    for col in range(len(columns)):
        if colTypes[col][0] == 'C':
            shape[columns[col]] = shape[columns[col]].astype(str)
        else:
            shape.loc[pd.isna(shape[columns[col]]), columns[col]] = -99999
            if colTypes[col][-1] > 0:
                shape[columns[col]] = shape[columns[col]].astype(float)
            else:
                shape[columns[col]] = shape[columns[col]].astype(int)

    if returnGeometry:
        return (shape, geometry)
    else:
        return shape


def read_nodes(filepath: str):
    '''
    Read NRM nodes.
    '''
    fieldWidths = [6, 6, 10, 10]
    colSpecs = [(0, fieldWidths[0])]
    for i in range(len(fieldWidths) - 1):
        colSpecs.append((colSpecs[i][1], colSpecs[i][1] + fieldWidths[i + 1]))

    with open(filepath, 'r') as f:
        data = f.readlines()

    nRows = len(data)
    nCols = len(fieldWidths)
    nodes = np.zeros((nRows, nCols), dtype=object)

    for row in range(nRows):
        for col in range(nCols):
            nodes[row, col] = data[row][colSpecs[col][0]:colSpecs[col][1]]

    nodes = pd.DataFrame(nodes, columns=['NODENR', 'DEL', 'X', 'Y'])
    nodes = nodes.drop(columns=['DEL'])
    nodes = nodes.astype(int)

    return nodes


def read_links(filepath: str):
    '''
    Read NRM links.
    '''

    # Check op dataformaat (eerste 2 regels)
    with open(filepath, 'r') as f:
        firstRow = f.readline()
        secondRow = f.readline()

    if len(secondRow) == 61:
        loaded = False
    elif len(firstRow) == 225:
        loaded = True
    else:
        raise BaseException(
            "\nUnexpected file format for road network links (" +
            str(filepath) + ")\n" +
            "Expected either a file with a width of 225 characters (loaded network) or " +
            "a file with a width of 61 characters (unloaded network).")

    # Lees het hele bestand nu in
    with open(filepath, 'r') as f:
        data = f.readlines()

    # Geladen netwerk
    if loaded:
        fieldWidths = [
            6, 6,
            7, 7, 7, 7, 7,
            7, 7, 7, 7, 7,
            2, 3, 2,
            6, 4, 7, 2, 4, 4,
            10, 10, 10, 10,
            3, 3,
            8, 8, 8, 8, 8,
            4, 3, 2, 3, 10]
        colSpecs = [(0, fieldWidths[0])]
        for i in range(len(fieldWidths)-1):
            colSpecs.append((colSpecs[i][1], colSpecs[i][1]+fieldWidths[i + 1]))

        nRows = len(data)
        nCols = len(fieldWidths)
        links = np.zeros((nRows, nCols), dtype=object)

        for row in range(nRows):
            for col in range(nCols):
                links[row, col] = data[row][colSpecs[col][0]:colSpecs[col][1]]

        columns = [
            'Anode', 'Bnode',
            'Flow', 'Q_bb', 'Fcap', 'Q_in', 'N_bb',
            'FFcost', 'Wcost', 'Dcost', 'Cost', 'Dist',
            'Lane', 'HWN', 'HOV',
            'FBlk', 'WvType', 'FlCpW', 'Ltype', 'IvF', 'Hef',
            'Nbb_in', 'Nek of Nprim', 'Nbb_rest', 'Buffer',
            'Leidend', 'Leidlink',
            'WvDist', 'Fileduur', 'VVU', 'VrachtTijdFF', 'VrachtTijd',
            'WetSnel', 'NRM-type', 'Geb-C',
            'DEL1',  'DEL2']

        links = pd.DataFrame(links, columns=columns)
        links['SNEL_MOD'] = (
            3.6 * links['Dist'].astype(float) / links['VrachtTijd'].astype(float))
        links = links[
            ['Anode', 'Bnode', 'WetSnel', 'SNEL_MOD',
             'Fcap', 'NRM-type', 'Dist', 'HOV', 'Geb-C']]

        links.columns = [
            'KNOOP_A',  'KNOOP_B', 'SNEL_FF', 'SNEL',
            'CAP', 'LINKTYPE', 'DISTANCE', 'DOELSTROOK', 'GEB_CODE']

        for col in [
            'KNOOP_A', 'KNOOP_B', 'SNEL_FF', 'CAP',
            'LINKTYPE', 'DISTANCE', 'DOELSTROOK', 'GEB_CODE'
        ]:
            links[col] = links[col].astype(float).astype(int)

        links['SNEL'] = links['SNEL'].astype(float)
        links = links.fillna(-99999)

        return (links, loaded)

    # Ongeladen netwerk
    else:
        fieldWidths = [
            6, 6, 5, 3,
            5, 2, 3, 7,
            2, 2, 2, 2,
            2, 2, 2, 2, 7]
        colSpecs = [(0, fieldWidths[0])]
        for i in range(len(fieldWidths)-1):
            colSpecs.append((colSpecs[i][1], colSpecs[i][1]+fieldWidths[i + 1]))

        data = data[1:]

        nRows = len(data)
        nCols = len(fieldWidths)
        links = np.zeros((nRows, nCols), dtype=object)

        for row in range(nRows):
            for col in range(nCols):
                links[row, col] = data[row][colSpecs[col][0]:colSpecs[col][1]]

        links = pd.DataFrame(
            links, columns=[
                'KNOOP_A', 'KNOOP_B', 'SNEL_FF', 'SNEL',
                'CAP', 'LINKTYPE', '1', 'DISTANCE',
                '2', '3', '4', 'DOELSTROOK',
                '5', '6', '7', '8', '9'])
        links = links.drop(columns=[str(x + 1) for x in range(9)])
        links['SNEL'] = links['SNEL_FF']
        links['GEB_CODE'] = 0
        links = links.fillna(-99999)
        links = links.astype(int)

        return (links, loaded)


def get_skims(
    settings: Settings,
    logger: logging.Logger,
    zones: pd.DataFrame,
    freight: bool = True,
    changeZeroValues: bool = False,
    returnSkimCostCharge: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Lees de skim met reistijden en de skim met afstanden in.

    Args:
        settings (Settings): _description_
        logger (logging.Logger): _description_
        zones (pd.DataFrame): _description_
        freight (bool, optional): _description_. Defaults to True.
        changeZeroValues (bool, optional): _description_. Defaults to False.
        returnSkimCostCharge (bool, optional): _description_. Defaults to False.

    Returns:
        tuple: Met daarin:
            - numpy.ndarray: Skim met reistijden (in seconden)
            - numpy.ndarray: Skim met afstanden (in meters)
            - numpy.ndarray: Skim met tol (in euro's)
    """
    if freight:
        skimTravTime = read_mtx(settings.get_path("SkimVrachtTijdNRM"))
        skimDistance = read_mtx(settings.get_path("SkimVrachtAfstandNRM"))
        if returnSkimCostCharge:
            skimCostCharge = np.array(
                read_mtx(settings.get_path("SkimVrachtTolNRM")), dtype=float)
    else:
        skimTravTime = read_mtx(settings.get_path("SkimBestelTijdNRM"))
        skimDistance = read_mtx(settings.get_path("SkimBestelAfstandNRM"))
        if returnSkimCostCharge:
            skimCostCharge = np.array(
                read_mtx(settings.get_path("SkimBestelTolNRM")), dtype=float)

    numZones = int(len(skimTravTime)**0.5)

    if numZones == 0:
        # Dit stukje is om te voorkomen dat grote skimbestanden moeten worden opgenomen in Git
        # om tests met de rekenmodules te kunnen uitvoeren
        logger.warning(
            "De skimbestanden bevatten geen data. " +
            "Er worden nu skims berekend o.b.v. hemelsbrede afstand.")
        skimDistance = get_euclidean_skim(
            zones['x_rd__meter'].values, zones['y_rd__meter'].values)
        skimTravTime = skimDistance / 1000 / 60

        if returnSkimCostCharge:
            skimCostCharge = np.zeros(len(zones) ** 2, dtype=float)

    skimTravTime[skimTravTime < 0] = 0
    skimDistance[skimDistance < 0] = 0

    if returnSkimCostCharge:
        skimCostCharge[skimCostCharge < 0] = 0

    # For zero times and distances assume half the value to the nearest (non-zero) zone
    # (otherwise we get problem in the distance decay function)
    if changeZeroValues:
        for orig in range(numZones):

            whereZero = np.where(
                skimTravTime[orig * numZones + np.arange(numZones)] == 0)[0]
            whereNonZero = np.where(
                skimTravTime[orig * numZones + np.arange(numZones)] != 0)[0]
            if len(whereZero) > 0:
                skimTravTime[orig * numZones + whereZero] = (
                    0.5 * np.min(skimTravTime[orig * numZones + whereNonZero]))

            whereZero = np.where(
                skimDistance[orig * numZones + np.arange(numZones)] == 0)[0]
            whereNonZero = np.where(
                skimDistance[orig * numZones + np.arange(numZones)] != 0)[0]
            if len(whereZero) > 0:
                skimDistance[orig * numZones + whereZero] = (
                    0.5 * np.min(skimDistance[orig * numZones + whereNonZero]))

    if returnSkimCostCharge:
        return skimTravTime, skimDistance, skimCostCharge
    else:
        return skimTravTime, skimDistance
