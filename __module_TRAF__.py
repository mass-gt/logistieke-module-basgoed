import numpy as np
import pandas as pd
import shapefile as shp
import time
import datetime
from numba import njit, int32
import scipy.sparse.csgraph
from scipy.sparse import lil_matrix
import multiprocessing as mp
import functools
from __functions__ import read_mtx, read_shape, determine_num_cpu
import psutil

# Modules nodig voor de user interface
import tkinter as tk
from tkinter.ttk import Progressbar
import zlib
import base64
import tempfile
from threading import Thread


def main(varDict):
    '''
    Start the GUI object which runs the module
    '''
    root = Root(varDict)

    return root.returnInfo


class Root:

    def __init__(self, args):
        '''
        Initialize a GUI object
        '''
        # Set graphics parameters
        self.width = 500
        self.height = 60
        self.bg = 'black'
        self.fg = 'white'
        self.font = 'Verdana'

        # Create a GUI window
        self.root = tk.Tk()
        self.root.title("Progress Emission Calculation")
        self.root.geometry(f'{self.width}x{self.height}+0+200')
        self.root.resizable(False, False)
        self.canvas = tk.Canvas(
            self.root,
            width=self.width,
            height=self.height,
            bg=self.bg)
        self.canvas.place(
            x=0,
            y=0)
        self.statusBar = tk.Label(
            self.root,
            text="",
            anchor='w',
            borderwidth=0,
            fg='black')
        self.statusBar.place(
            x=2,
            y=self.height - 22,
            width=self.width,
            height=22)

        # Remove the default tkinter icon from the window
        icon = zlib.decompress(base64.b64decode(
            'eJxjYGAEQgEBBiDJwZDBy' +
            'sAgxsDAoAHEQCEGBQaIOAg4sDIgACMUj4JRMApGwQgF/ykEAFXxQRc='))
        _, self.iconPath = tempfile.mkstemp()
        with open(self.iconPath, 'wb') as iconFile:
            iconFile.write(icon)
        self.root.iconbitmap(bitmap=self.iconPath)

        # Create a progress bar
        self.progressBar = Progressbar(self.root, length=self.width - 20)
        self.progressBar.place(x=10, y=10)

        self.returnInfo = ""

        if __name__ == '__main__':
            self.args = [[self, args]]
        else:
            self.args = [args]

        self.run_module()

        # Keep GUI active until closed
        self.root.mainloop()

    def update_statusbar(self, text):
        self.statusBar.configure(text=text)

    def error_screen(self, text='', event=None,
                     size=[800, 50], title='Error message'):
        '''
        Pop up a window with an error message
        '''
        windowError = tk.Toplevel(self.root)
        windowError.title(title)
        windowError.geometry(f'{size[0]}x{size[1]}+0+{200+50+self.height}')
        windowError.minsize(width=size[0], height=size[1])
        windowError.iconbitmap(default=self.iconPath)
        labelError = tk.Label(
            windowError,
            text=text,
            anchor='w',
            justify='left')
        labelError.place(x=10, y=10)

    def run_module(self, event=None):
        Thread(target=actually_run_module, args=self.args, daemon=True).start()


def actually_run_module(args):
    '''
    Emission Calculation: Main body of the script where all
    calculations are performed.
    '''
    try:

        start_time = time.time()

        root = args[0]
        varDict = args[1]

        if root != '':
            root.progressBar['value'] = 0

        exportShp = True

        log_file = open(
            varDict['OUTPUTFOLDER'] + "Logfile_EmissionCalculation.log", 'w')
        log_file.write(
            "Start simulation at: " +
            datetime.datetime.now().strftime("%y-%m-%d %H:%M") + "\n")

        if varDict['SEED'] != '':
            np.random.seed(varDict['SEED'])

        dimLS = pd.read_csv(
            varDict['DIMFOLDER'] + 'logistic_segment.txt',
            sep='\t')
        dimET = pd.read_csv(
            varDict['DIMFOLDER'] + 'emission_type.txt',
            sep='\t')
        dimVT = pd.read_csv(
            varDict['DIMFOLDER'] + 'vehicle_type.txt',
            sep='\t')

        nLS = len(dimLS)
        nET = len(dimET)
        nVT = len(dimVT)

        # To convert emissions to kilograms
        emissionDivFac = [1000, 1000000, 1000000, 1000]
        etDict = dict((i, dimET.at[i, 'Comment']) for i in range(nET))
        etInvDict = dict((v, k) for k, v in etDict.items())

        # Which vehicle type can be used in the parcel module
        vehTypesParcels = np.where(dimVT['IsAvailableInParcelModule'] == 1)[0]

        # Carrying capacity in kg
        carryingCapacity = np.array(pd.read_csv(
            varDict['VEHCAPACITY'],
            index_col='Vehicle Type'))

        # Number of CPUs over which the processes are parallelized
        ramSize = psutil.virtual_memory().total
        if ramSize < 10000000000:
            maxCPU = 3
        elif ramSize < 20000000000:
            maxCPU = 6
        elif ramSize < 40000000000:
            maxCPU = 8
        else:
            maxCPU = 12

        nCPU = determine_num_cpu(varDict, maxCPU=maxCPU)

        # Aantal routes waarover gespreid wordt per HB
        # TODO: nMultiRoute instelbaar maken of verwijderen
        nMultiRoute = 1

        if root != '':
            root.progressBar['value'] = 0.5

        print("Importing and preprocessing network...")
        log_file.write("Importing and preprocessing network...\n")

        print('\tReading zones...')
        zones = read_shape(varDict['ZONES'])
        zones = zones[['SEGNR_2018', 'N', 'O', 'W', 'Z', 'XCOORD', 'YCOORD']]
        nZones = len(zones)

        if root != '':
            root.progressBar['value'] = 1.5

        print('\tReading nodes...')
        nodes = pd.read_csv(varDict['NODES'], sep=',')
        nNodes = len(nodes)

        if root != '':
            root.progressBar['value'] = 2.0

        print('\tReading links...')
        links = read_shape(varDict['LINKS'])
        nLinks = len(links)

        # Hercoderen nodes
        nodeDict = dict((i, nodes.iloc[i, 0]) for i in range(nNodes))
        invNodeDict = dict((nodes.iloc[i, 0], i) for i in range(nNodes))

        nodes['NODENR'] = [invNodeDict[x] for x in nodes['NODENR'].values]
        nodes.index = np.arange(nNodes)

        links['KNOOP_A'] = [invNodeDict[x] for x in links['KNOOP_A'].values]
        links['KNOOP_B'] = [invNodeDict[x] for x in links['KNOOP_B'].values]

        # Van meter naar kilometer
        links['DISTANCE'] /= 1000

        # Van seconde naar uur
        links['T0_FREIGHT'] /= 3600
        links['T0_VAN'] /= 3600

        if root != '':
            root.progressBar['value'] = 3.0

        print('\tReading other input files...')

        shipments = pd.read_csv(
            varDict['OUTPUTFOLDER'] + 'Shipments_' + varDict['LABEL'] + '.csv',
            sep=',')
        ggSharesMS = pd.pivot_table(
            shipments[shipments['CONTAINER'] == 0],
            values='WEIGHT',
            index='BG_GG',
            aggfunc=sum)
        ggSharesCKM = pd.pivot_table(
            shipments[shipments['CONTAINER'] == 1],
            values='WEIGHT',
            index='BG_GG',
            aggfunc=sum)

        ggSharesMS.index = [int(x) for x in ggSharesMS.index]
        ggSharesCKM.index = [int(x) for x in ggSharesCKM.index]

        del shipments

        # Calculate average cost weighted by weight per goods type
        # and containerized/non-containerized
        costPerHrFreight, costPerKmFreight = get_cost_freight(
            varDict, ggSharesMS, ggSharesCKM)

        # Cost parameters van
        costPerHrVan, costPerKmVan = get_cost_van(varDict)

        # Variabele om de linkID te verkrijgen o.b.v. A- en B-knoop
        maxNumConnections = 8
        linkDict = -1 * np.ones(
            (max(links['KNOOP_A']) + 1, 2 * maxNumConnections), dtype=int)
        for i in links.index:
            aNode = links['KNOOP_A'][i]
            bNode = links['KNOOP_B'][i]

            for col in range(maxNumConnections):
                if linkDict[aNode][col] == -1:
                    linkDict[aNode][col] = bNode
                    linkDict[aNode][col + maxNumConnections] = i
                    break

        # Travel times and travel costs
        links['COST_FREIGHT'] = (
            costPerKmFreight * links['DISTANCE'] +
            costPerHrFreight * links['T0_FREIGHT'])
        links['COST_VAN'] = (
            costPerKmVan * links['DISTANCE'] +
            costPerHrVan * links['T0_VAN'])

        # Set connector travel costs high so these are not chosen other
        # than for entering/leaving network
        links.loc[links['LINKTYPE'] == 99, 'COST_FREIGHT'] = 10000
        links.loc[links['LINKTYPE'] == 99, 'COST_VAN'] = 10000

        # Set travel costs for forbidden-for-freight-links high so these
        # are not chosen for freight
        links.loc[links['DOELSTROOK'] == 2, 'COST_FREIGHT'] = 10000

        # Set travel costs for only-freight-links high so these are
        # not chosen for vans
        links.loc[links['DOELSTROOK'] == 3, 'COST_VAN'] = 10000

        # Set travel times on links in ZEZ Rotterdam high so these are
        # only used to go to UCC and not for through traffic
        costFreightHybr = links['COST_FREIGHT'].copy()
        costVanHybr = links['COST_VAN'].copy()
        if varDict['LABEL'] == 'UCC':
            links.loc[links['ZEZ'] == 1, 'COST_FREIGHT'] += 10000
            links.loc[links['ZEZ'] == 1, 'COST_VAN'] += 10000

        # Initialize empty fields with emissions and traffic intensity per link
        # (also save list with all field names)
        volCols = [
            'N_LS0', 'N_LS1', 'N_LS2', 'N_LS3', 'N_LS4',
            'N_LS5', 'N_LS6', 'N_LS7', 'N_LS8',
            'N_VAN_S', 'N_VAN_C',
            'N_VEH0', 'N_VEH1', 'N_VEH2', 'N_VEH3',
            'N_VEH4', 'N_VEH5', 'N_VEH6', 'N_VEH7',
            'N_VEH8', 'N_VEH9', 'N_VEH10',
            'N_LS8_VEH8', 'N_LS8_VEH9',
            'N_TOT']
        intensityFields = []
        for et in etDict.values():
            intensityFields.append(et)
        for volCol in volCols:
            intensityFields.append(volCol)
        for ls in range(nLS):
            for et in etDict.values():
                intensityFields.append(et + '_LS' + str(ls))
        for ls in ['VAN_S', 'VAN_C']:
            for et in etDict.values():
                intensityFields.append(et + '_' + str(ls))

        # Make some space available on the RAM
        del zones

        if root != '':
            root.progressBar['value'] = 4.0

        # Lees de emissiefactoren in
        (emissionsBuitenwegLeeg, emissionsBuitenwegVol,
         emissionsSnelwegLeeg, emissionsSnelwegVol,
         emissionsStadLeeg, emissionsStadVol) = get_emission_facs(varDict)

        # To which vehicle type in the emission factors (value)
        # does each of our vehicle types (key) belong
        vtDict = {
            0: 2,
            1: 3,
            2: 5,
            3: 4,
            4: 6,
            5: 7,
            6: 9,
            7: 9,
            8: 0,
            9: 0,
            10: 0}

        print('\tReading freight and parcel trips...')

        # Inladen tourbestanden
        allTrips, allParcelTrips = get_trips(varDict, carryingCapacity)

        # Determine linktypes (urban/rural/highway)
        stadLinkTypes = [6, 7]
        buitenwegLinkTypes = [3, 4, 5]
        snelwegLinkTypes = [1, 2]

        whereStad = [
            links['LINKTYPE'][i] in stadLinkTypes for i in links.index]
        whereBuitenweg = [
            links['LINKTYPE'][i] in buitenwegLinkTypes for i in links.index]
        whereSnelweg = [
            links['LINKTYPE'][i] in snelwegLinkTypes for i in links.index]

        roadtypeArray = np.zeros((len(links)))

        roadtypeArray[whereStad] = 1
        roadtypeArray[whereBuitenweg] = 2
        roadtypeArray[whereSnelweg] = 3

        stadArray = (roadtypeArray == 1)
        buitenwegArray = (roadtypeArray == 2)
        snelwegArray = (roadtypeArray == 3)

        distArray = np.array(links['DISTANCE'])
        ZEZarray = np.array(links['ZEZ'] == 1, dtype=int)
        NLarray = np.array(links['NL'] == 1, dtype=int)

        # Bring ORIG and DEST to the front of the list of column names
        newColOrder = volCols.copy()
        newColOrder.insert(0, 'DEST_NRM')
        newColOrder.insert(0, 'ORIG_NRM')

        # HB-matrices met aantallen ritten ophalen
        tripMatrix, tripMatrixParcels = get_trip_matrices(varDict, newColOrder)

        # For which origin zones do we need to find the routes
        origSelection = np.arange(nZones)
        nOrigSelection = len(origSelection)

        # Initialize arrays for intensities and emissions
        linkTripsArray = np.zeros((nLinks, len(volCols)))
        linkVanTripsArray = np.zeros((nLinks, 2))
        linkEmissionsArray = [
            np.zeros((nLinks, nET)) for ls in range(nLS)]
        linkVanEmissionsArray = [
            np.zeros((nLinks, nET)) for ls in ['VAN_S', 'VAN_C']]

        if root != '':
            root.progressBar['value'] = 5.0

        tripMatrixOrigins = set(tripMatrix[:, 0])
        trips = np.array(allTrips[allTrips['VEHTYPE'] != 8])

        if nMultiRoute >= 2:
            np.random.seed(100)
            whereTripsByIter = [[] for r in range(nMultiRoute)]
            for i in range(len(trips)):
                whereTripsByIter[np.random.randint(nMultiRoute)].append(i)
        elif nMultiRoute == 1:
            whereTripsByIter = [np.arange(len(trips))]
        else:
            raise Exception(
                'nMultiRoute should be >= 1, now it is ' +
                str(nMultiRoute))

        # Vaste seed voor spreiding reiskosten op links
        if nMultiRoute >= 2:
            linksRandArray = [[] for r in range(nMultiRoute)]
            linksA = np.array(links['KNOOP_A'])
            linksB = np.array(links['KNOOP_B'])
            for i in range(len(links)):
                np.random.seed(linksA[i] + linksB[i])
                for r in range(nMultiRoute):
                    linksRandArray[r].append(np.random.rand())

        if varDict['SEED'] != '':
            np.random.seed(varDict['SEED'])

        if root != '':
            root.progressBar['value'] = 6.0

        tripsCO2 = {}
        tripsCO2_NL = {}
        tripsDist = {}
        tripsDist_NL = {}
        parcelTripsCO2 = {}

        # From which nodes do we need to perform the shortest path algoritm
        indices = origSelection

        # From which nodes does every CPU perform the shortest path algorithm
        indicesPerCPU = [
            [indices[cpu::nCPU], cpu] for cpu in range(nCPU)]
        origSelectionPerCPU = [
            np.arange(nOrigSelection)[cpu::nCPU] for cpu in range(nCPU)]

        # Whether a separate route search needs to be done
        # or hybrid vehicles or not
        doHybrRoutes = np.any(trips[:, 6] == 3)

        for r in range(nMultiRoute):

            print(f"Route search (freight part {r + 1})...")
            log_file.write(f"Route search (freight part {r + 1})...\n")

            # Route search freight
            if nCPU > 1:

                # The network with costs between nodes (freight)
                csgraphFreight = lil_matrix((nNodes, nNodes))
                tmp = np.array(links['COST_FREIGHT'])
                if nMultiRoute > 1:
                    tmp *= 0.95 + 0.1 * linksRandArray[r]
                csgraphFreight[
                    np.array(links['KNOOP_A']),
                    np.array(links['KNOOP_B'])] = tmp

                # Initialize a pool object that spreads tasks
                # over different CPUs
                p = mp.Pool(nCPU)

                # Execute the Dijkstra route search
                prevFreightPerCPU = p.map(
                    functools.partial(get_prev, csgraphFreight, nNodes),
                    indicesPerCPU)

                # Wait for completion of processes
                p.close()
                p.join()

                # Combine the results from the different CPUs
                # Matrix with for each node the previous node on
                # the shortest path
                prevFreight = np.zeros((nOrigSelection, nNodes), dtype=int)
                for cpu in range(nCPU):
                    for i in range(len(indicesPerCPU[cpu][0])):
                        prevFreight[origSelectionPerCPU[cpu][i], :] = (
                            prevFreightPerCPU[cpu][i, :])

                # Make some space available on the RAM
                del prevFreightPerCPU

            else:

                # The network with costs between nodes (freight)
                csgraphFreight = lil_matrix((nNodes, nNodes))
                tmp = np.array(links['COST_FREIGHT'])
                if nMultiRoute > 1:
                    tmp *= 0.95 + 0.1 * linksRandArray[r]
                csgraphFreight[
                    np.array(links['KNOOP_A']),
                    np.array(links['KNOOP_B'])] = tmp

                # Execute the Dijkstra route search
                prevFreight = get_prev(csgraphFreight, nNodes, [indices, 0])

            if root != '':
                root.progressBar['value'] = 8.0 + (r + 1) * 10.0 + r * 10.0

            # Make some space available on the RAM
            del csgraphFreight

            # Route search freight (for clean vehicles in the UCC scenario)
            if doHybrRoutes:
                message = (
                    "Route search " +
                    f"(freight with hybrid combustion part {r + 1})...")
                print(message), log_file.write(message + "\n")

                # Route search freight
                if nCPU > 1:

                    # The network with costs between nodes (freight)
                    csgraphFreightHybr = lil_matrix((nNodes, nNodes))
                    tmp = costFreightHybr
                    if nMultiRoute > 1:
                        tmp *= 0.95 + 0.1 * linksRandArray[r]
                    csgraphFreightHybr[
                        np.array(links['KNOOP_A']),
                        np.array(links['KNOOP_B'])] = tmp

                    # Initialize a pool object that spreads tasks over
                    # different CPUs
                    p = mp.Pool(nCPU)

                    # Execute the Dijkstra route search
                    prevFreightHybrPerCPU = p.map(
                        functools.partial(
                            get_prev, csgraphFreightHybr, nNodes),
                        indicesPerCPU)

                    # Wait for completion of processes
                    p.close()
                    p.join()

                    # Combine the results from the different CPUs.
                    # Matrix with for each node the previous node on
                    # the shortest path.
                    prevFreightHybr = np.zeros(
                        (nOrigSelection, nNodes), dtype=int)
                    for cpu in range(nCPU):
                        for i in range(len(indicesPerCPU[cpu][0])):
                            prevFreightHybr[origSelectionPerCPU[cpu][i], :] = (
                                prevFreightHybrPerCPU[cpu][i, :])

                    # Make some space available on the RAM
                    del prevFreightHybrPerCPU

                else:

                    # The network with costs between nodes (freight)
                    csgraphFreightHybr = lil_matrix((nNodes, nNodes))
                    tmp = costFreightHybr
                    if nMultiRoute > 1:
                        tmp *= 0.95 + 0.1 * linksRandArray[r]
                    csgraphFreightHybr[
                        np.array(links['KNOOP_A']),
                        np.array(links['KNOOP_B'])] = tmp

                    # Execute the Dijkstra route search
                    prevFreightHybr = get_prev(
                        csgraphFreightHybr, nNodes, [indices, 0])

                if root != '':
                    root.progressBar['value'] = 8.0 + (r + 1) * 10.0 + r * 10.0

                # Make some space available on the RAM
                del csgraphFreightHybr

            message = (
                "Calculating emissions and traffic intensities " +
                f"(freight part {r + 1})")
            print(message), log_file.write(message + "\n")

            iterTrips = trips[whereTripsByIter[r], :]

            whereODL = {}
            for i in range(len(tripMatrix)):
                orig = tripMatrix[i, 0]
                dest = tripMatrix[i, 1]
                for ls in range(nLS):
                    whereODL[(orig, dest, ls)] = []
            for i in range(len(iterTrips)):
                orig = iterTrips[i, 1]
                dest = iterTrips[i, 2]
                ls = iterTrips[i, 5]
                whereODL[(orig, dest, ls)].append(i)

            for i in range(nOrigSelection):
                origZone = i + 1

                print('\tOrigin ' + str(origZone), end='\r')

                if origZone not in tripMatrixOrigins:
                    continue

                destZoneIndex = np.where(tripMatrix[:, 0] == origZone)[0]
                destZones = tripMatrix[destZoneIndex, 1]

                # Regular routes
                routes = [
                    get_route(i, destZone - 1, prevFreight, linkDict)
                    for destZone in destZones]

                # Routes for hybrid vehicles
                # (no need for penalty on ZEZ-links here)
                if doHybrRoutes:
                    hybrRoutes = [
                        get_route(
                            i, destZone - 1,
                            prevFreightHybr,
                            linkDict)
                        for destZone in destZones]

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
                    routeStadNL = np.where(
                        NLarray[routeStad] == 1)[0]
                    routeBuitenwegNL = np.where(
                        NLarray[routeBuitenweg] == 1)[0]
                    routeSnelwegNL = np.where(
                        NLarray[routeSnelweg] == 1)[0]

                    distStad = distArray[routeStad]
                    distBuitenweg = distArray[routeBuitenweg]
                    distSnelweg = distArray[routeSnelweg]
                    distTotal = np.sum(distArray[route])
                    distTotalNL = np.sum(
                        distArray[np.where(NLarray[route] == 1)[0]])

                    if doHybrRoutes:
                        hybrRoute = hybrRoutes[j]

                        # Get route and part of route that is
                        # stad/buitenweg/snelweg and ZEZ/non-ZEZ
                        hybrRouteStad = hybrRoute[
                            stadArray[hybrRoute]]
                        hybrRouteBuitenweg = hybrRoute[
                            buitenwegArray[hybrRoute]]
                        hybrRouteSnelweg = hybrRoute[
                            snelwegArray[hybrRoute]]
                        hybrZEZstad = (ZEZarray[hybrRouteStad] == 1)
                        hybrZEZbuitenweg = (ZEZarray[hybrRouteBuitenweg] == 1)
                        hybrZEZsnelweg = (ZEZarray[hybrRouteSnelweg] == 1)

                        # Het gedeelte van de route in NL
                        hybrRouteStadNL = np.where(
                            NLarray[hybrRouteStad] == 1)[0]
                        hybrRouteBuitenwegNL = np.where(
                            NLarray[hybrRouteBuitenweg] == 1)[0]
                        hybrRouteSnelwegNL = np.where(
                            NLarray[hybrRouteSnelweg] == 1)[0]

                        hybrDistStad = distArray[hybrRouteStad]
                        hybrDistBuitenweg = distArray[hybrRouteBuitenweg]
                        hybrDistSnelweg = distArray[hybrRouteSnelweg]
                        hybrDistTotal = np.sum(distArray[hybrRoute])
                        hybrDistTotalNL = np.sum(
                            distArray[np.where(NLarray[hybrRoute] == 1)[0]])

                    # Bereken en schrijf de intensiteiten/emissies
                    # op de links
                    for ls in range(nLS):
                        # Welke trips worden allemaal gemaakt op de HB
                        # van de huidige iteratie van de ij-loop
                        currentTrips = iterTrips[
                            whereODL[(origZone, destZone, ls)], :]
                        nCurrentTrips = len(currentTrips)

                        if nCurrentTrips == 0:
                            continue

                        capUt = currentTrips[:, 4]
                        vt = [vtDict[x] for x in currentTrips[:, 3]]

                        emissionFacStad = [None for et in range(nET)]
                        emissionFacBuitenweg = [None for et in range(nET)]
                        emissionFacSnelweg = [None for et in range(nET)]

                        for et in range(nET):
                            lowerFacStad = emissionsStadLeeg[vt, et]
                            upperFacStad = emissionsStadVol[vt, et]
                            emissionFacStad[et] = (
                                lowerFacStad +
                                capUt * (upperFacStad - lowerFacStad))

                            lowerFacBuitenweg = emissionsBuitenwegLeeg[vt, et]
                            upperFacBuitenweg = emissionsBuitenwegVol[vt, et]
                            emissionFacBuitenweg[et] = (
                                lowerFacBuitenweg +
                                capUt * (
                                    upperFacBuitenweg - lowerFacBuitenweg))

                            lowerFacSnelweg = emissionsSnelwegLeeg[vt, et]
                            upperFacSnelweg = emissionsSnelwegVol[vt, et]
                            emissionFacSnelweg[et] = (
                                lowerFacSnelweg +
                                capUt * (upperFacSnelweg - lowerFacSnelweg))

                        for trip in range(nCurrentTrips):
                            vt = int(currentTrips[trip, 3])
                            ct = int(currentTrips[trip, 6])

                            # If combustion type is fuel or bio-fuel
                            if ct in [0, 4]:
                                stadEmissions = np.zeros(
                                    (len(routeStad), nET))
                                buitenwegEmissions = np.zeros(
                                    (len(routeBuitenweg), nET))
                                snelwegEmissions = np.zeros(
                                    (len(routeSnelweg), nET))

                                for et in range(nET):
                                    stadEmissions[:, et] = (
                                        distStad *
                                        emissionFacStad[et][trip])
                                    buitenwegEmissions[:, et] = (
                                        distBuitenweg *
                                        emissionFacBuitenweg[et][trip])
                                    snelwegEmissions[:, et] = (
                                        distSnelweg *
                                        emissionFacSnelweg[et][trip])

                                linkEmissionsArray[ls][routeStad, :] += (
                                    stadEmissions)
                                linkEmissionsArray[ls][routeBuitenweg, :] += (
                                    buitenwegEmissions)
                                linkEmissionsArray[ls][routeSnelweg, :] += (
                                    snelwegEmissions)

                                linkTripsArray[route, nLS + 2 + vt] += 1
                                linkTripsArray[route, ls] += 1
                                linkTripsArray[route, -1] += 1

                            # If combustion type is hybrid
                            elif ct == 3:
                                stadEmissions = np.zeros(
                                    (len(hybrRouteStad), nET))
                                buitenwegEmissions = np.zeros(
                                    (len(hybrRouteBuitenweg), nET))
                                snelwegEmissions = np.zeros(
                                    (len(hybrRouteSnelweg), nET))

                                for et in range(nET):
                                    stadEmissions[:, et] = (
                                        hybrDistStad *
                                        emissionFacStad[et][trip])
                                    buitenwegEmissions[:, et] = (
                                        hybrDistBuitenweg *
                                        emissionFacBuitenweg[et][trip])
                                    snelwegEmissions[:, et] = (
                                        hybrDistSnelweg *
                                        emissionFacSnelweg[et][trip])

                                # No emissions in ZEZ part of route
                                stadEmissions[hybrZEZstad, :] = 0
                                buitenwegEmissions[hybrZEZbuitenweg, :] = 0
                                snelwegEmissions[hybrZEZsnelweg, :] = 0

                                linkEmissionsArray[ls][
                                    hybrRouteStad, :] += stadEmissions
                                linkEmissionsArray[ls][
                                    hybrRouteBuitenweg, :] += (
                                        buitenwegEmissions)
                                linkEmissionsArray[ls][
                                    hybrRouteSnelweg, :] += snelwegEmissions

                                linkTripsArray[hybrRoute, nLS + 2 + vt] += 1
                                linkTripsArray[hybrRoute, ls] += 1
                                linkTripsArray[hybrRoute, -1] += 1

                            # Clean combustion types (no emissions)
                            else:
                                linkTripsArray[route, nLS + 2 + vt] += 1
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
                                    np.sum(stadEmissions[
                                        routeStadNL, 0]) +
                                    np.sum(buitenwegEmissions[
                                        routeBuitenwegNL, 0]) +
                                    np.sum(snelwegEmissions[
                                        routeSnelwegNL, 0]))

                            elif ct == 3:
                                tripsCO2[tripIndex] = (
                                    np.sum(stadEmissions[:, 0]) +
                                    np.sum(buitenwegEmissions[:, 0]) +
                                    np.sum(snelwegEmissions[:, 0]))
                                tripsCO2_NL[tripIndex] = (
                                    np.sum(stadEmissions[
                                        hybrRouteStadNL, 0]) +
                                    np.sum(buitenwegEmissions[
                                        hybrRouteBuitenwegNL, 0]) +
                                    np.sum(snelwegEmissions[
                                        hybrRouteSnelwegNL, 0]))
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

                if root != '':
                    root.progressBar['value'] = (
                        8.0 + (r + 1) * 10.0 + 10.0 * i / nOrigSelection)

            del prevFreight

            if doHybrRoutes:
                del prevFreightHybr

        # Select van trips
        trips = np.array(allTrips[allTrips['VEHTYPE'] == 8])

        if nMultiRoute >= 2:
            whereTripsByIter = [[] for r in range(nMultiRoute)]
            for i in range(len(trips)):
                whereTripsByIter[np.random.randint(nMultiRoute)].append(i)
        elif nMultiRoute == 1:
            whereTripsByIter = [np.arange(len(trips))]
        else:
            raise Exception(
                'nMultiRoute should be >= 1, now it is ' +
                str(nMultiRoute))

        if nMultiRoute >= 2:
            whereParcelTripsByIter = [[] for r in range(nMultiRoute)]
            for i in range(len(allParcelTrips)):
                whereParcelTripsByIter[
                    np.random.randint(nMultiRoute)].append(i)
        elif nMultiRoute == 1:
            whereParcelTripsByIter = [np.arange(len(allParcelTrips))]

        for r in range(nMultiRoute):

            print(f"Route search (vans part {r + 1})...")
            log_file.write(f"Route search (vans part {r + 1})...\n")

            if nCPU > 1:
                # From which nodes does every CPU perform
                # the shortest path algorithm
                indicesPerCPU = [
                    [indices[cpu::nCPU], cpu]
                    for cpu in range(nCPU)]
                origSelectionPerCPU = [
                    np.arange(nOrigSelection)[cpu::nCPU]
                    for cpu in range(nCPU)]

                # The network with costs between nodes (freight)
                csgraphVan = lil_matrix((nNodes, nNodes))
                tmp = np.array(links['COST_VAN'])
                if nMultiRoute > 1:
                    tmp *= 0.95 + 0.1 * linksRandArray[r]
                csgraphVan[
                    np.array(links['KNOOP_A']),
                    np.array(links['KNOOP_B'])] = tmp

                # Initialize a pool object that spreads
                # tasks over different CPUs
                p = mp.Pool(nCPU)

                # Execute the Dijkstra route search
                prevVanPerCPU = p.map(
                    functools.partial(get_prev, csgraphVan, nNodes),
                    indicesPerCPU)

                # Wait for completion of processes
                p.close()
                p.join()

                # Combine the results from the different CPUs.
                # Matrix with for each node the previous node on
                # the shortest path.
                prevVan = np.zeros(
                    (nOrigSelection, nNodes), dtype=int)
                for cpu in range(nCPU):
                    for i in range(len(indicesPerCPU[cpu][0])):
                        prevVan[origSelectionPerCPU[cpu][i], :] = (
                            prevVanPerCPU[cpu][i, :])

                # Make some space available on the RAM
                del prevVanPerCPU

            else:

                # The network with costs between nodes (freight)
                csgraphVan = lil_matrix((nNodes, nNodes))
                tmp = np.array(links['COST_VAN'])
                if nMultiRoute > 1:
                    tmp *= 0.95 + 0.1 * linksRandArray[r]
                csgraphVan[
                    np.array(links['KNOOP_A']),
                    np.array(links['KNOOP_B'])] = tmp

                # Execute the Dijkstra route search
                prevVan = get_prev(csgraphVan, nNodes, [indices, 0])

            if root != '':
                root.progressBar['value'] = 40.0 + (r + 1) * 10.0 + r * 10.0

            # Make some space available on the RAM
            del csgraphVan

            # Route search freight (for clean vehicles in the UCC scenario)
            if doHybrRoutes:
                message = (
                    "Route search " +
                    f"(vans with hybrid combustion part {r + 1})...")
                print(message), log_file.write(message + "\n")

                # Route search van
                if nCPU > 1:

                    # The network with costs between nodes (van)
                    csgraphVanHybr = lil_matrix((nNodes, nNodes))
                    tmp = costVanHybr
                    if nMultiRoute > 1:
                        tmp *= 0.95 + 0.1 * linksRandArray[r]
                    csgraphVanHybr[
                        np.array(links['KNOOP_A']),
                        np.array(links['KNOOP_B'])] = tmp

                    # Initialize a pool object that spreads tasks
                    # over different CPUs
                    p = mp.Pool(nCPU)

                    # Execute the Dijkstra route search
                    prevVanHybrPerCPU = p.map(
                        functools.partial(get_prev, csgraphVanHybr, nNodes),
                        indicesPerCPU)

                    # Wait for completion of processes
                    p.close()
                    p.join()

                    # Combine the results from the different CPUs
                    # Matrix with for each node the previous node
                    # on the shortest path
                    prevVanHybr = np.zeros(
                        (nOrigSelection, nNodes), dtype=int)
                    for cpu in range(nCPU):
                        for i in range(len(indicesPerCPU[cpu][0])):
                            prevVanHybr[origSelectionPerCPU[cpu][i], :] = (
                                prevVanHybrPerCPU[cpu][i, :])

                    # Make some space available on the RAM
                    del prevVanHybrPerCPU

                else:

                    # The network with costs between nodes (van)
                    csgraphVanHybr = lil_matrix((nNodes, nNodes))
                    tmp = costVanHybr
                    if nMultiRoute > 1:
                        tmp *= 0.95 + 0.1 * linksRandArray[r]
                    csgraphVanHybr[
                        np.array(links['KNOOP_A']),
                        np.array(links['KNOOP_B'])] = tmp

                    # Execute the Dijkstra route search
                    prevVanHybr = get_prev(
                        csgraphVanHybr, nNodes, [indices, 0])

                if root != '':
                    root.progressBar['value'] = 8.0 + (r + 1) * 10.0 + r * 10.0

                # Make some space available on the RAM
                del csgraphVanHybr

            message = (
                "Calculating emissions and traffic intensities " +
                f"(vans part {r + 1})")
            print(message), log_file.write(message + "\n")

            print('\tGoods vans...')
            log_file.write('\tGoods vans...\n')

            iterTrips = trips[whereTripsByIter[r], :]

            whereODL = {}
            for i in range(len(tripMatrix)):
                orig = tripMatrix[i, 0]
                dest = tripMatrix[i, 1]
                for ls in range(nLS):
                    whereODL[(orig, dest, ls)] = []
            for i in range(len(iterTrips)):
                orig = iterTrips[i, 1]
                dest = iterTrips[i, 2]
                ls = iterTrips[i, 5]
                whereODL[(orig, dest, ls)].append(i)

            for i in range(nOrigSelection):
                origZone = i + 1

                print('\t\tOrigin ' + str(origZone), end='\r')

                if origZone not in tripMatrixOrigins:
                    continue

                destZoneIndex = np.where(tripMatrix[:, 0] == origZone)[0]
                destZones = tripMatrix[destZoneIndex, 1]

                routes = [
                    get_route(i, destZone - 1, prevVan, linkDict)
                    for destZone in destZones]

                # Routes for hybrid vehicles
                # (no need for penalty on ZEZ-links here)
                if doHybrRoutes:
                    hybrRoutes = [
                        get_route(i, destZone - 1, prevVanHybr, linkDict)
                        for destZone in destZones]

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
                    routeStadNL = np.where(
                        NLarray[routeStad] == 1)[0]
                    routeBuitenwegNL = np.where(
                        NLarray[routeBuitenweg] == 1)[0]
                    routeSnelwegNL = np.where(
                        NLarray[routeSnelweg] == 1)[0]

                    distStad = distArray[routeStad]
                    distBuitenweg = distArray[routeBuitenweg]
                    distSnelweg = distArray[routeSnelweg]
                    distTotal = np.sum(distArray[route])
                    distTotalNL = np.sum(distArray[
                        np.where(NLarray[route] == 1)[0]])

                    if doHybrRoutes:
                        hybrRoute = hybrRoutes[j]

                        # Get route and part of route that is
                        # stad/buitenweg/snelweg and ZEZ/non-ZEZ
                        hybrRouteStad = hybrRoute[
                            stadArray[hybrRoute]]
                        hybrRouteBuitenweg = hybrRoute[
                            buitenwegArray[hybrRoute]]
                        hybrRouteSnelweg = hybrRoute[
                            snelwegArray[hybrRoute]]
                        hybrZEZstad = (ZEZarray[hybrRouteStad] == 1)
                        hybrZEZbuitenweg = (ZEZarray[hybrRouteBuitenweg] == 1)
                        hybrZEZsnelweg = (ZEZarray[hybrRouteSnelweg] == 1)

                        # Het gedeelte van de route in NL
                        hybrRouteStadNL = np.where(
                            NLarray[hybrRouteStad] == 1)[0]
                        hybrRouteBuitenwegNL = np.where(
                            NLarray[hybrRouteBuitenweg] == 1)[0]
                        hybrRouteSnelwegNL = np.where(
                            NLarray[hybrRouteSnelweg] == 1)[0]

                        hybrDistStad = distArray[hybrRouteStad]
                        hybrDistBuitenweg = distArray[hybrRouteBuitenweg]
                        hybrDistSnelweg = distArray[hybrRouteSnelweg]
                        hybrDistTotal = np.sum(distArray[hybrRoute])
                        hybrDistTotalNL = np.sum(distArray[
                            np.where(NLarray[hybrRoute] == 1)[0]])

                    # Bereken en schrijf de intensiteiten/emissies op de links
                    for ls in range(nLS):
                        # Welke trips worden allemaal gemaakt op de HB
                        # van de huidige iteratie van de ij-loop
                        currentTrips = iterTrips[
                            whereODL[(origZone, destZone, ls)], :]
                        nCurrentTrips = len(currentTrips)

                        if nCurrentTrips == 0:
                            continue

                        capUt = currentTrips[:, 4]
                        vt = vtDict[8]

                        emissionFacStad = [
                            None for et in range(nET)]
                        emissionFacBuitenweg = [
                            None for et in range(nET)]
                        emissionFacSnelweg = [
                            None for et in range(nET)]

                        for et in range(nET):
                            lowerFacStad = emissionsStadLeeg[vt, et]
                            upperFacStad = emissionsStadVol[vt, et]
                            emissionFacStad[et] = (
                                lowerFacStad +
                                capUt * (upperFacStad - lowerFacStad))

                            lowerFacBuitenweg = emissionsBuitenwegLeeg[vt, et]
                            upperFacBuitenweg = emissionsBuitenwegVol[vt, et]
                            emissionFacBuitenweg[et] = (
                                lowerFacBuitenweg +
                                capUt * (
                                    upperFacBuitenweg - lowerFacBuitenweg))

                            lowerFacSnelweg = emissionsSnelwegLeeg[vt, et]
                            upperFacSnelweg = emissionsSnelwegVol[vt, et]
                            emissionFacSnelweg[et] = (
                                lowerFacSnelweg +
                                capUt * (upperFacSnelweg - lowerFacSnelweg))

                        for trip in range(nCurrentTrips):
                            vt = int(currentTrips[trip, 3])
                            ct = int(currentTrips[trip, 6])

                            # If combustion type is fuel or bio-fuel
                            if ct in [0, 4]:
                                stadEmissions = np.zeros(
                                    (len(routeStad), nET))
                                buitenwegEmissions = np.zeros(
                                    (len(routeBuitenweg), nET))
                                snelwegEmissions = np.zeros(
                                    (len(routeSnelweg), nET))

                                for et in range(nET):
                                    stadEmissions[:, et] = (
                                        distStad *
                                        emissionFacStad[et][trip])
                                    buitenwegEmissions[:, et] = (
                                        distBuitenweg *
                                        emissionFacBuitenweg[et][trip])
                                    snelwegEmissions[:, et] = (
                                        distSnelweg *
                                        emissionFacSnelweg[et][trip])

                                linkEmissionsArray[ls][
                                    routeStad, :] += stadEmissions
                                linkEmissionsArray[ls][
                                    routeBuitenweg, :] += buitenwegEmissions
                                linkEmissionsArray[ls][
                                    routeSnelweg, :] += snelwegEmissions

                                linkTripsArray[route, nLS + 2 + vt] += 1
                                linkTripsArray[route, ls] += 1
                                linkTripsArray[route, -1] += 1

                            # If combustion type is hybrid
                            elif ct == 3:
                                stadEmissions = np.zeros(
                                    (len(hybrRouteStad), nET))
                                buitenwegEmissions = np.zeros(
                                    (len(hybrRouteBuitenweg), nET))
                                snelwegEmissions = np.zeros(
                                    (len(hybrRouteSnelweg), nET))

                                for et in range(nET):
                                    stadEmissions[:, et] = (
                                        hybrDistStad *
                                        emissionFacStad[et][trip])
                                    buitenwegEmissions[:, et] = (
                                        hybrDistBuitenweg *
                                        emissionFacBuitenweg[et][trip])
                                    snelwegEmissions[:, et] = (
                                        hybrDistSnelweg *
                                        emissionFacSnelweg[et][trip])

                                # No emissions in ZEZ part of route
                                stadEmissions[hybrZEZstad, :] = 0
                                buitenwegEmissions[hybrZEZbuitenweg, :] = 0
                                snelwegEmissions[hybrZEZsnelweg, :] = 0

                                linkEmissionsArray[ls][
                                    hybrRouteStad, :] += stadEmissions
                                linkEmissionsArray[ls][
                                    hybrRouteBuitenweg, :] += (
                                        buitenwegEmissions)
                                linkEmissionsArray[ls][
                                    hybrRouteSnelweg, :] += snelwegEmissions

                                linkTripsArray[hybrRoute, nLS + 2 + vt] += 1
                                linkTripsArray[hybrRoute, ls] += 1
                                linkTripsArray[hybrRoute, -1] += 1

                            # Clean combustion types (no emissions)
                            else:
                                linkTripsArray[route, nLS + 2 + vt] += 1
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
                                    np.sum(stadEmissions[
                                        routeStadNL, 0]) +
                                    np.sum(buitenwegEmissions[
                                        routeBuitenwegNL, 0]) +
                                    np.sum(snelwegEmissions[
                                        routeSnelwegNL, 0]))
                            elif ct == 3:
                                tripsCO2[tripIndex] = (
                                    np.sum(stadEmissions[:, 0]) +
                                    np.sum(buitenwegEmissions[:, 0]) +
                                    np.sum(snelwegEmissions[:, 0]))
                                tripsCO2_NL[tripIndex] = (
                                    np.sum(stadEmissions[
                                        hybrRouteStadNL, 0]) +
                                    np.sum(buitenwegEmissions[
                                        hybrRouteBuitenwegNL, 0]) +
                                    np.sum(snelwegEmissions[
                                        hybrRouteSnelwegNL, 0]))
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

                if root != '':
                    root.progressBar['value'] = (
                        40.0 +
                        (r + 1) * 15.0 + 4.0 * i / nOrigSelection)

            del whereODL

            # ------------- Emissions and intensities (parcel vans) -----------

            print('\tParcel vans...')
            log_file.write('\tParcels vans...\n')

            # Logistic segment: parcel deliveries
            ls = 8

            # Assume half of vehicle capacity (in terms of weight) used
            capUt = 0.5

            for vt in vehTypesParcels:
                parcelTrips = allParcelTrips.loc[
                    whereParcelTripsByIter[r], :]
                parcelTrips = parcelTrips.loc[
                    (parcelTrips['VEHTYPE'] == vt), :]
                parcelTrips = np.array(parcelTrips)

                emissionFacStad = np.array([
                    emissionsStadLeeg[vtDict[vt], et] +
                    capUt * (
                        emissionsStadVol[vtDict[vt], et] -
                        emissionsStadLeeg[vtDict[vt], et])
                    for et in range(nET)])
                emissionFacBuitenweg = np.array([
                    emissionsBuitenwegLeeg[vtDict[vt], et] +
                    capUt * (
                        emissionsBuitenwegVol[vtDict[vt], et] -
                        emissionsBuitenwegLeeg[vtDict[vt], et])
                    for et in range(nET)])
                emissionFacSnelweg = np.array([
                    emissionsSnelwegLeeg[vtDict[vt], et] +
                    capUt * (
                        emissionsSnelwegVol[vtDict[vt], et] -
                        emissionsSnelwegLeeg[vtDict[vt], et])
                    for et in range(nET)])

                if len(parcelTrips) > 0:
                    parcelResults = emissions_parcels(
                        parcelTrips,
                        tripMatrixParcels,
                        prevVan, prevVanHybr if doHybrRoutes else None,
                        distArray,
                        stadArray, buitenwegArray, snelwegArray,
                        ZEZarray,
                        etInvDict['CO2'],
                        emissionFacStad, emissionFacBuitenweg,
                        emissionFacSnelweg,
                        nLinks, nET, linkDict,
                        indices,
                        doHybrRoutes)

                    # Intensiteiten
                    nTrips = parcelResults[0]

                    # Number of trips for LS8 (=parcel deliveries)
                    linkTripsArray[:, ls] += nTrips

                    # De parcel trips per voertuigtype
                    linkTripsArray[:, nLS + 2 + nVT + vt - 8] += nTrips

                    # Number of trips for vehicle type
                    linkTripsArray[:, nLS + 2 + vt] += nTrips

                    # Total number of trips
                    linkTripsArray[:, -1] += nTrips

                    # Emissies
                    linkEmissionsArray[ls] += parcelResults[1]

                    # CO2 per trip
                    for trip in parcelResults[2].keys():
                        parcelTripsCO2[int(trip)] = parcelResults[2][trip]

                    del parcelResults

            if root != '':
                root.progressBar['value'] = 40.0 + (r + 1) * 15.0 + 5.0

            # ---------- Emissions and intensities (serv/constr vans) ---------

            print('\tService & construction vans...')
            log_file.write('\tService & construction vans...\n')

            # Van trips for service and construction purposes
            try:
                vanTripsService = read_mtx(
                    varDict['OUTPUTFOLDER'] + 'TripsVanService.mtx')
                vanTripsConstruction = read_mtx(
                    varDict['OUTPUTFOLDER'] + 'TripsVanConstruction.mtx')
                vanTripsFound = True

            except FileNotFoundError:
                message = (
                    'Could not find TripsVanService.mtx and/or ' +
                    'TripsVanConstruction.mtx in specified outputfolder. ' +
                    'Hence no service/construction vans are ' +
                    'assigned to the network.')
                print(message), log_file.write(message + '\n')
                vanTripsFound = False

            if vanTripsFound:
                # Select half of ODs per multiroute iteration
                if nMultiRoute > 1:
                    vanTripsService[r::nMultiRoute] = 0
                    vanTripsConstruction[r::nMultiRoute] = 0

                # Reshape to square array
                vanTripsService = vanTripsService.reshape(
                    nZones, nZones)
                vanTripsConstruction = vanTripsConstruction.reshape(
                    nZones, nZones)

                vt = 8       # Vehicle type: Van
                capUt = 0.5  # Assume half of loading capacity used

                emissionFacStad = np.array([
                    emissionsStadLeeg[vtDict[vt], et] +
                    capUt * (
                        emissionsStadVol[vtDict[vt], et] -
                        emissionsStadLeeg[vtDict[vt], et])
                    for et in range(nET)])
                emissionFacBuitenweg = np.array([
                    emissionsBuitenwegLeeg[vtDict[vt], et] +
                    capUt * (
                        emissionsBuitenwegVol[vtDict[vt], et] -
                        emissionsBuitenwegLeeg[vtDict[vt], et])
                    for et in range(nET)])
                emissionFacSnelweg = np.array([
                    emissionsSnelwegLeeg[vtDict[vt], et] +
                    capUt * (
                        emissionsSnelwegVol[vtDict[vt], et] -
                        emissionsSnelwegLeeg[vtDict[vt], et])
                    for et in range(nET)])

                indicesPerCPU = []
                for cpu in range(nCPU):
                    indicesPerCPU.append([])

                    indicesPerCPU[cpu].append(
                        indices[cpu::nCPU])
                    indicesPerCPU[cpu].append(
                        vanTripsService[indices[cpu::nCPU], :])
                    indicesPerCPU[cpu].append(
                        vanTripsConstruction[indices[cpu::nCPU], :])
                    if doHybrRoutes:
                        indicesPerCPU[cpu].append(
                            prevVanHybr[indices[cpu::nCPU], :])
                    else:
                        indicesPerCPU[cpu].append(
                            prevVan[indices[cpu::nCPU], :])

                # Make some space available on the RAM
                del vanTripsService, vanTripsConstruction, prevVan
                if doHybrRoutes:
                    del prevVanHybr

                if nCPU > 1:
                    # Initialize a pool object that spreads tasks
                    # over different CPUs
                    p = mp.Pool(nCPU)

                    # Calculate emissions for service / construction vans
                    vanResultPerCPU = p.map(
                        functools.partial(
                            emissions_vans,
                            distArray,
                            stadArray, buitenwegArray, snelwegArray,
                            emissionFacStad, emissionFacBuitenweg,
                            emissionFacSnelweg,
                            nLinks, nET, linkDict),
                        indicesPerCPU)

                    # Wait for completion of processes
                    p.close()
                    p.join()

                else:
                    # Calculate emissions for service / construction vans
                    vanResultPerCPU = [
                        emissions_vans(
                            distArray,
                            stadArray, buitenwegArray, snelwegArray,
                            emissionFacStad, emissionFacBuitenweg,
                            emissionFacSnelweg,
                            nLinks, nET, linkDict,
                            indicesPerCPU[0])]

                if root != '':
                    root.progressBar['value'] = (
                        40.0 +
                        (r + 1) * 15.0 + 15.0 * i / nOrigSelection)

                del indicesPerCPU

                # Combine the results from the different CPUs
                for cpu in range(nCPU):
                    linkVanTripsArray += vanResultPerCPU[cpu][0]
                    linkVanEmissionsArray[0] += vanResultPerCPU[cpu][1][0]
                    linkVanEmissionsArray[1] += vanResultPerCPU[cpu][1][1]

                del vanResultPerCPU

        # Write the intensities and emissions into the links-DataFrame
        for field in intensityFields:
            links[field] = 0.0
        links.loc[:, volCols] += linkTripsArray.astype(int)

        # Total emissions and per logistic segment
        for ls in range(nLS):
            for et in range(nET):
                links[etDict[et]] += (
                    linkEmissionsArray[ls][:, et] / emissionDivFac[et])
                links[etDict[et] + '_LS' + str(ls)] += (
                    linkEmissionsArray[ls][:, et] / emissionDivFac[et])

        del linkTripsArray, linkEmissionsArray

        if root != '':
            root.progressBar['value'] = 90.0

        if vanTripsFound:
            # Number of van trips
            linkVanTripsArray = np.round(linkVanTripsArray, 3)
            links.loc[:, 'N_VAN_S'] = linkVanTripsArray[:, 0]
            links.loc[:, 'N_VAN_C'] = linkVanTripsArray[:, 1]
            links.loc[:, 'N_VEH7'] += linkVanTripsArray[:, 0]
            links.loc[:, 'N_VEH7'] += linkVanTripsArray[:, 1]
            links.loc[:, 'N_TOT'] += linkVanTripsArray[:, 0]
            links.loc[:, 'N_TOT'] += linkVanTripsArray[:, 1]

            # Emissions from van trips
            for et in range(nET):
                links[etDict[et] + '_' + 'VAN_S'] = (
                    linkVanEmissionsArray[0][:, et] / emissionDivFac[et])
                links[etDict[et] + '_' + 'VAN_C'] = (
                    linkVanEmissionsArray[1][:, et] / emissionDivFac[et])
                links[etDict[et]] += (
                    linkVanEmissionsArray[0][:, et] / emissionDivFac[et])
                links[etDict[et]] += (
                    linkVanEmissionsArray[1][:, et] / emissionDivFac[et])

            del linkVanTripsArray, linkVanEmissionsArray

        if root != '':
            root.progressBar['value'] = 91.0

        # ------------------- Enriching tours and shipments -------------------

        try:
            print("Writing emissions into Tours and ParcelSchedule...")
            log_file.write(
                "Writing emissions into Tours and ParcelSchedule...\n")

            # Zet emissies in de tours
            tours, parcelTours = put_emissions_into_tours(
                varDict,
                tripsCO2, tripsCO2_NL,
                tripsDist, tripsDist_NL,
                parcelTripsCO2,
                emissionDivFac)

            # Wegschrijven verrijkte vracht tours
            tours.to_csv(
                varDict['OUTPUTFOLDER'] + 'Tours_' + varDict['LABEL'] + '.csv',
                index=False)

            # Wegschrijven verrijkte pakket tours
            parcelTours.to_csv(
                (
                    varDict['OUTPUTFOLDER'] +
                    'ParcelSchedule_' +
                    varDict['LABEL'] +
                    '.csv'
                ),
                index=False)

            if root != '':
                root.progressBar['value'] = 94.0

            print("Writing emissions into Shipments...")
            log_file.write("Writing emissions into Shipments...\n")

            shipments = put_emissions_into_shipments(
                varDict, tours)

            # Export enriched shipments with CO2 field
            shipments.to_csv(
                (
                    varDict['OUTPUTFOLDER'] +
                    'Shipments_AfterScheduling_' +
                    varDict['LABEL'] +
                    '.csv'
                ),
                index=False)

        except Exception:
            message = (
                "Writing emissions into " +
                "Tours/ParcelSchedule/Shipments failed!")
            print(message)
            log_file.write(message + '\n')

            try:
                import sys
                print(sys.exc_info()[0]),
                log_file.write(str(sys.exc_info()[0])),
                log_file.write("\n")
                import traceback
                print(traceback.format_exc()),
                log_file.write(str(traceback.format_exc())),
                log_file.write("\n")
            except Exception:
                pass

        if root != '':
            root.progressBar['value'] = 95.0

        # Export loaded network to shapefile
        if exportShp:
            print("Exporting network to .shp...")
            log_file.write("Exporting network to .shp...\n")

            write_links_to_shape(
                varDict,
                links, nodes,
                nodeDict, invNodeDict,
                intensityFields,
                root, 93.0, 100.0)

        print('\t100%', end='\r')

        totaltime = round(time.time() - start_time, 2)
        log_file.write("Total runtime: %s seconds\n" % (totaltime))
        log_file.write(
            "End simulation at: " +
            datetime.datetime.now().strftime("%y-%m-%d %H:%M") + "\n")
        log_file.close()

        if root != '':
            root.update_statusbar("Emission Calculation: Done")
            root.progressBar['value'] = 100

            # 0 means no errors in execution
            root.returnInfo = [0, [0, 0]]

            return root.returnInfo

        else:
            return [0, [0, 0]]

    except BaseException:
        import sys
        log_file.write(str(sys.exc_info()[0])), log_file.write("\n")
        import traceback
        log_file.write(str(traceback.format_exc())), log_file.write("\n")
        log_file.write("Execution failed!")
        log_file.close()
        print(sys.exc_info()[0])
        print(traceback.format_exc())
        print("Execution failed!")

        if root != '':
            # Use this information to display as error message in GUI
            root.returnInfo = [1, [sys.exc_info()[0], traceback.format_exc()]]

            if __name__ == '__main__':
                root.update_statusbar(
                    "Emission Calculation: Execution failed!")
                errorMessage = (
                    'Execution failed!\n\n' +
                    str(root.returnInfo[1][0]) +
                    '\n\n' +
                    str(root.returnInfo[1][1]))
                root.error_screen(text=errorMessage, size=[900, 350])

            else:
                return root.returnInfo

        else:
            return [1, [sys.exc_info()[0], traceback.format_exc()]]


def get_prev(
    csgraph: lil_matrix,
    nNodes: int,
    indices: list
):
    '''
    For each origin zone and destination node, determine the
    previously visited node on the shortest path.

    Args:
        csgraph (scipy.sparse.lil_matrix): _description_
        nNodes (int): _description_
        indices (list): _description_

    Returns:
        numpy.ndarray: 'prev' object van de Dijkstra-routine
    '''
    whichCPU = indices[1]
    indices = indices[0]
    nOrigSelection = len(indices)

    prev = np.zeros((nOrigSelection, nNodes), dtype=int)
    for i in range(nOrigSelection):
        prev[i, :] = scipy.sparse.csgraph.dijkstra(
            csgraph,
            indices=indices[i],
            return_predecessors=True)[1]

        if whichCPU == 0:
            if i % int(round(nOrigSelection / 20)) == 0:
                print(
                    '\t' + str(int(round((i / nOrigSelection) * 100))) + '%',
                    end='\r')

    del csgraph

    return prev


@njit
def get_route(
    orig: int,
    dest: int,
    prev: np.ndarray,
    linkDict: np.ndarray,
    maxNumConnections: int = 8
):
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

    if orig != dest:
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


def emissions_parcels(
    trips: np.ndarray,
    tripMatrixParcels: np.ndarray,
    prevVan: np.ndarray, prevVanHybr: np.ndarray,
    distArray: np.ndarray,
    stadArray: np.ndarray,
    buitenwegArray: np.ndarray,
    snelwegArray: np.ndarray,
    ZEZarray: np.ndarray,
    indexCO2: int,
    emissionFacStad: np.ndarray,
    emissionFacBuitenweg: np.ndarray,
    emissionFacSnelweg: np.ndarray,
    nLinks: int, nET: int,
    linkDict: np.ndarray,
    indices: np.ndarray,
    doHybrRoutes: bool
):
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

            if doHybrRoutes:
                route = get_route(
                    origZone, destZone, prevVan, linkDict)
                hybrRoute = get_route(
                    origZone, destZone, prevVanHybr, linkDict)
            else:
                route = get_route(origZone, destZone, prevVan, linkDict)

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
                stadEmissions[:, et] = (
                    distArray[routeStad] * emissionFacStad[et])
                buitenwegEmissions[:, et] = (
                    distArray[routeBuitenweg] * emissionFacBuitenweg[et])
                snelwegEmissions[:, et] = (
                    distArray[routeSnelweg] * emissionFacSnelweg[et])

            if doHybrRoutes:
                # Route per wegtype
                hybrRouteStad = hybrRoute[stadArray[hybrRoute]]
                hybrRouteBuitenweg = hybrRoute[buitenwegArray[hybrRoute]]
                hybrRouteSnelweg = hybrRoute[snelwegArray[hybrRoute]]

                # Welke links liggen in het ZE-gebied
                hybrZEZstad = (ZEZarray[hybrRouteStad] == 1)
                hybrZEZbuitenweg = (ZEZarray[hybrRouteBuitenweg] == 1)
                hybrZEZsnelweg = (ZEZarray[hybrRouteSnelweg] == 1)

                # Emissies voor een enkele trip
                hybrStadEmissions = np.zeros(
                    (len(hybrRouteStad), nET))
                hybrBuitenwegEmissions = np.zeros(
                    (len(hybrRouteBuitenweg), nET))
                hybrSnelwegEmissions = np.zeros(
                    (len(hybrRouteSnelweg), nET))
                for et in range(nET):
                    hybrStadEmissions[:, et] = (
                        distArray[hybrRouteStad] *
                        emissionFacStad[et])
                    hybrBuitenwegEmissions[:, et] = (
                        distArray[hybrRouteBuitenweg] *
                        emissionFacBuitenweg[et])
                    hybrSnelwegEmissions[:, et] = (
                        distArray[hybrRouteSnelweg] *
                        emissionFacSnelweg[et])

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
                        linkEmissionsArray[routeStad, et] += (
                            stadEmissionsTrip[:, et])
                        linkEmissionsArray[routeBuitenweg, et] += (
                            buitenwegEmissionsTrip[:, et])
                        linkEmissionsArray[routeSnelweg, et] += (
                            snelwegEmissionsTrip[:, et])

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
                        linkEmissionsArray[hybrRouteStad, et] += (
                            stadEmissionsTrip[:, et])
                        linkEmissionsArray[hybrRouteBuitenweg, et] += (
                            buitenwegEmissionsTrip[:, et])
                        linkEmissionsArray[hybrRouteSnelweg, et] += (
                            snelwegEmissionsTrip[:, et])

                    # CO2-emissions for the current trip
                    parcelTripsCO2[currentTrips[trip, 1]] = (
                        np.sum(stadEmissionsTrip[:, indexCO2]) +
                        np.sum(buitenwegEmissionsTrip[:, indexCO2]) +
                        np.sum(snelwegEmissionsTrip[:, indexCO2]))

                else:
                    parcelTripsCO2[currentTrips[trip, -1]] = 0

    del prevVan

    return linkTripsArray, linkEmissionsArray, parcelTripsCO2


def emissions_vans(
    distArray: np.ndarray,
    stadArray: np.ndarray,
    buitenwegArray: np.ndarray,
    snelwegArray: np.ndarray,
    emissionFacStad: np.ndarray,
    emissionFacBuitenweg: np.ndarray,
    emissionFacSnelweg: np.ndarray,
    nLinks: int, nET: int,
    linkDict: np.ndarray,
    indices: list
):
    """
    Bereken de intensiteiten en emissies van bestelautoritten voor
    bouw en service.

    Args:
        distArray (np.ndarray): _description_
        stadArray (np.ndarray): _description_
        buitenwegArray (np.ndarray): _description_
        snelwegArray (np.ndarray): _description_
        emissionFacStad (np.ndarray): _description_
        emissionFacBuitenweg (np.ndarray): _description_
        emissionFacSnelweg (np.ndarray): _description_
        nLinks (int): _description_
        nET (int): _description_
        linkDict (np.ndarray): _description_
        indices (list): _description_

    Returns:
        tuple: Met daarin:
            - np.ndarray: De intensiteiten op elke link
            - np.ndarray: De emissies op elke link
    """
    vanTripsService = indices[1]
    vanTripsConstruction = indices[2]
    prevVan = indices[3]
    indices = indices[0]

    linkVanTripsArray = np.zeros((nLinks, 2))
    linkVanEmissionsArray = [
        np.zeros((nLinks, nET)) for i in ['VAN_S', 'VAN_C']]

    for i in range(len(indices)):
        print('\t\tOrigin ' + '{:>5}'.format(indices[i] + 1), end='\r')

        # For which destination zones to search routes
        destZones = np.where(
            (vanTripsService[i, :] > 0) |
            (vanTripsConstruction[i, :] > 0))[0]

        # Dijkstra route search
        routes = [get_route(i, j, prevVan, linkDict) for j in destZones]

        j = 0
        for destZone in destZones:
            nTripsService = vanTripsService[i, destZone]
            nTripsConstruction = vanTripsConstruction[i, destZone]

            route = np.array(list(routes[j]), dtype=int)

            # Route per wegtype
            routeStad = route[stadArray[route]]
            routeBuitenweg = route[buitenwegArray[route]]
            routeSnelweg = route[snelwegArray[route]]

            # Afstanden van de links op de route
            distStad = distArray[routeStad]
            distBuitenweg = distArray[routeBuitenweg]
            distSnelweg = distArray[routeSnelweg]

            # Emissies voor een enkele trip
            stadEmissions = np.zeros((len(routeStad), nET))
            buitenwegEmissions = np.zeros((len(routeBuitenweg), nET))
            snelwegEmissions = np.zeros((len(routeSnelweg), nET))
            for et in range(nET):
                stadEmissions[:, et] = (
                    distStad * emissionFacStad[et])
                buitenwegEmissions[:, et] = (
                    distBuitenweg * emissionFacBuitenweg[et])
                snelwegEmissions[:, et] = (
                    distSnelweg * emissionFacSnelweg[et])

            # Van: Service segment
            if nTripsService > 0:
                # Number of trips made on each link on the route
                linkVanTripsArray[route, 0] += nTripsService

                # Emissions on each link on the route
                stadEmissionsService = stadEmissions * nTripsService
                buitenwegEmissionsService = buitenwegEmissions * nTripsService
                snelwegEmissionsService = snelwegEmissions * nTripsService

                for et in range(nET):
                    linkVanEmissionsArray[0][routeStad, et] += (
                        stadEmissionsService[:, et])
                    linkVanEmissionsArray[0][routeBuitenweg, et] += (
                        buitenwegEmissionsService[:, et])
                    linkVanEmissionsArray[0][routeSnelweg, et] += (
                        snelwegEmissionsService[:, et])

            # Van: Construction segment
            if nTripsConstruction > 0:
                # Number of trips made on each link on the route
                linkVanTripsArray[route, 1] += nTripsConstruction

                # Emissions on each link on the route
                stadEmissionsConstruction = (
                    stadEmissions * nTripsConstruction)
                buitenwegEmissionsConstruction = (
                    buitenwegEmissions * nTripsConstruction)
                snelwegEmissionsConstruction = (
                    snelwegEmissions * nTripsConstruction)

                for et in range(nET):
                    linkVanEmissionsArray[1][routeStad, et] += (
                        stadEmissionsConstruction[:, et])
                    linkVanEmissionsArray[1][routeBuitenweg, et] += (
                        buitenwegEmissionsConstruction[:, et])
                    linkVanEmissionsArray[1][routeSnelweg, et] += (
                        snelwegEmissionsConstruction[:, et])

            j += 1

    del prevVan, vanTripsService, vanTripsConstruction

    return linkVanTripsArray, linkVanEmissionsArray


def get_cost_freight(
    varDict: dict,
    ggSharesMS: pd.DataFrame,
    ggSharesCKM: pd.DataFrame
):
    """
    Berekent de uurkosten en kilometerkosten voor vracht gemiddeld
    over alle goederengroepen, voertuigtypen en verschijningsvormen.

    Args:
        varDict (dict): _description_
        ggSharesMS (pd.DataFrame): _description_
        ggSharesCKM (pd.DataFrame): _description_

    Returns:
        tuple: costPerHrFreight en costPerKmFreight
    """
    # Cost parameters freight
    costFreightMS = np.array(pd.read_csv(
        varDict['COST_ROAD_MS'],
        sep='\t'))[:, [0, 2, 3]]
    costFreightCKM = np.array(pd.read_csv(
        varDict['COST_ROAD_CKM'],
        sep='\t'))[:, [0, 1, 2]]

    costPerKmFreight = 0.0
    costPerHrFreight = 0.0

    for gg in costFreightMS[:, 0]:
        if int(gg) in ggSharesMS.index:
            row = np.where(costFreightMS[:, 0] == gg)[0][0]
            costPerKmFreight += (
                costFreightMS[row, 1] * ggSharesMS.at[int(gg), 'WEIGHT'])
            costPerHrFreight += (
                costFreightMS[row, 2] * ggSharesMS.at[int(gg), 'WEIGHT'])

    for gg in costFreightCKM[:, 0]:
        if int(gg) in ggSharesCKM.index:
            row = np.where(costFreightCKM[:, 0] == gg)[0][0]
            costPerKmFreight += (
                costFreightCKM[row, 1] * ggSharesCKM.at[int(gg), 'WEIGHT'])
            costPerHrFreight += (
                costFreightCKM[row, 2] * ggSharesCKM.at[int(gg), 'WEIGHT'])

    sumWeight = (np.sum(ggSharesMS) + np.sum(ggSharesCKM))

    costPerKmFreight /= sumWeight
    costPerHrFreight /= sumWeight
    costPerKmFreight = float(costPerKmFreight)
    costPerHrFreight = float(costPerHrFreight)

    return costPerHrFreight, costPerKmFreight


def get_cost_van(
    varDict: dict
):
    """
    Berekent de uurkosten en kilometerkosten voor bestelwagens gemiddeld
    over alle goederengroepen.

    Args:
        varDict (dict): _description_

    Returns:
        tuple: costPerHrVan en costPerKmVan
    """
    costVan = np.array(pd.read_csv(
        varDict['COST_BESTEL'],
        sep='\t'))[:, [0, 1, 2]]
    costPerKmVan = np.average(costVan[:, 1])
    costPerHrVan = np.average(costVan[:, 2])

    return costPerHrVan, costPerKmVan


def get_emission_facs(
    varDict: dict
):
    """
    Haal de emissiefactoren op.

    Args:
        varDict (dict): _description_

    Returns:
        tuple: Met daarin:
            - numpy.ndarray: emissiefactoren buitenweg leeg
            - numpy.ndarray: emissiefactoren buitenweg vol
            - numpy.ndarray: emissiefactoren snelweg leeg
            - numpy.ndarray: emissiefactoren snelweg vol
            - numpy.ndarray: emissiefactoren stad leeg
            - numpy.ndarray: emissiefactoren stad vol
    """
    # Lees emissiefactoren in (kolom 0=CO2, 1=SO2, 2=PM, 3=NOX)
    emissionsBuitenwegLeeg = np.array(pd.read_csv(
        varDict['EM_BUITENWEG_LEEG'], index_col='Voertuigtype'))
    emissionsBuitenwegVol = np.array(pd.read_csv(
        varDict['EM_BUITENWEG_VOL'], index_col='Voertuigtype'))
    emissionsSnelwegLeeg = np.array(pd.read_csv(
        varDict['EM_SNELWEG_LEEG'], index_col='Voertuigtype'))
    emissionsSnelwegVol = np.array(pd.read_csv(
        varDict['EM_SNELWEG_VOL'], index_col='Voertuigtype'))
    emissionsStadLeeg = np.array(pd.read_csv(
        varDict['EM_STAD_LEEG'], index_col='Voertuigtype'))
    emissionsStadVol = np.array(pd.read_csv(
        varDict['EM_STAD_VOL'], index_col='Voertuigtype'))

    # Gemiddelde van small en large tractor+trailer
    emissionsBuitenwegLeeg[7, :] = (
        (emissionsBuitenwegLeeg[7, :] + emissionsBuitenwegLeeg[8, :]) / 2)
    emissionsBuitenwegVol[7, :] = (
        (emissionsBuitenwegVol[7, :] + emissionsBuitenwegVol[8, :]) / 2)
    emissionsSnelwegLeeg[7, :] = (
        (emissionsSnelwegLeeg[7, :] + emissionsSnelwegLeeg[8, :]) / 2)
    emissionsSnelwegVol[7, :] = (
        (emissionsSnelwegVol[7, :] + emissionsSnelwegVol[8, :]) / 2)
    emissionsStadLeeg[7, :] = (
        (emissionsStadLeeg[7, :] + emissionsStadLeeg[8, :]) / 2)
    emissionsStadVol[7, :] = (
        (emissionsStadVol[7, :] + emissionsStadVol[8, :]) / 2)

    # Gemiddelde van small en large van
    emissionsBuitenwegLeeg[0, :] = (
        (emissionsBuitenwegLeeg[0, :] + emissionsBuitenwegLeeg[1, :]) / 2)
    emissionsBuitenwegVol[0, :] = (
        (emissionsBuitenwegVol[0, :] + emissionsBuitenwegVol[1, :]) / 2)
    emissionsSnelwegLeeg[0, :] = (
        (emissionsSnelwegLeeg[0, :] + emissionsSnelwegLeeg[1, :]) / 2)
    emissionsSnelwegVol[0, :] = (
        (emissionsSnelwegVol[0, :] + emissionsSnelwegVol[1, :]) / 2)
    emissionsStadLeeg[0, :] = (
        (emissionsStadLeeg[0, :] + emissionsStadLeeg[1, :]) / 2)
    emissionsStadVol[0, :] = (
        (emissionsStadVol[0, :] + emissionsStadVol[1, :]) / 2)

    return (
        emissionsBuitenwegLeeg, emissionsBuitenwegVol,
        emissionsSnelwegLeeg, emissionsSnelwegVol,
        emissionsStadLeeg, emissionsStadVol)


def get_trips(
    varDict: dict,
    carryingCapacity: np.ndarray
):
    """
    Inladen CSV-bestanden met rondritten voor vracht en pakketbezorging.

    Args:
        varDict (dict): _description_
        carryingCapacity (np.ndarray): _description_

    Returns:
        tuple: Met daarin:
            - pd.DataFrame: de rondritten voor vracht
            - pd.DataFrame: de rondritten voor pakketbezorging
    """

    # Import trips csv
    allTrips = pd.read_csv(
        varDict['OUTPUTFOLDER'] + "Tours_" + varDict['LABEL'] + ".csv",
        sep=',')
    allTrips.loc[allTrips['TRIP_DEPTIME'] >= 24, 'TRIP_DEPTIME'] -= 24
    allTrips.loc[allTrips['TRIP_DEPTIME'] >= 24, 'TRIP_DEPTIME'] -= 24
    capUt = (
        (allTrips['TRIP_WEIGHT'] * 1000) /
        carryingCapacity[np.array(allTrips['VEHTYPE'], dtype=int)][:, 0])
    allTrips['CAP_UT'] = capUt
    allTrips['INDEX'] = allTrips.index
    allTrips = allTrips[
        ['CARRIER_ID', 'ORIG_NRM', 'DEST_NRM',
         'VEHTYPE', 'CAP_UT', 'LOG_SEG', 'COMBTYPE', 'INDEX']]

    # Import parcel schedule csv
    allParcelTrips = pd.read_csv(
        (
            varDict['OUTPUTFOLDER'] +
            "ParcelSchedule_" +
            varDict['LABEL'] +
            ".csv"
        ), sep=',')
    allParcelTrips = allParcelTrips.rename(
        columns={'ORIG_NRM': 'ORIG',
                 'DEST_NRM': 'DEST'})
    allParcelTrips.loc[
        allParcelTrips['TRIP_DEPTIME'] >= 24, 'TRIP_DEPTIME'] -= 24
    allParcelTrips.loc[
        allParcelTrips['TRIP_DEPTIME'] >= 24, 'TRIP_DEPTIME'] -= 24

    # Assume on average 50% of loading capacity used for parcel deliveries
    allParcelTrips['CAP_UT'] = 0.5

    # Recode vehicle type from string to number
    allParcelTrips['VEHTYPE'] = [
        {'Van': 8, 'LEVV': 9}[vt] for vt in allParcelTrips['VEHTYPE']]

    # Fuel as basic combustion type
    allParcelTrips['COMBTYPE'] = 0

    # Trips coming from UCC to ZEZ use electric
    allParcelTrips.loc[allParcelTrips['ORIGTYPE'] == 'UCC', 'COMBTYPE'] = 1
    allParcelTrips['LOG_SEG'] = 6
    allParcelTrips['INDEX'] = allParcelTrips.index
    allParcelTrips = allParcelTrips[
        ['DEPOT_ID', 'ORIG', 'DEST',
         'VEHTYPE', 'CAP_UT', 'LOG_SEG',
         'COMBTYPE', 'INDEX']]

    return allTrips, allParcelTrips


def get_trip_matrices(
    varDict: dict,
    columnOrder: list
):
    """
    Haalt de HB-matrices met aantallen ritten voor vracht en
    pakketbezorging op.

    Args:
        varDict (dict): _description_

    Returns:
        tuple: Met daarin:
            - numpy.ndarray: tripMatrix
            - numpy.ndarray: tripMatrixParcels
    """
    # Vracht
    tripMatrix = pd.read_csv(
        (
            varDict['OUTPUTFOLDER'] +
            'TripMatrix_Freight_NRM_' +
            varDict['LABEL'] +
            '.csv'
        ),
        sep=',')
    tripMatrix['N_LS8'] = 0
    tripMatrix['N_VAN_S'] = 0
    tripMatrix['N_VAN_C'] = 0
    tripMatrix['N_LS8_VEH8'] = 0
    tripMatrix['N_LS8_VEH9'] = 0
    tripMatrix = tripMatrix[columnOrder]
    tripMatrix = np.array(tripMatrix)

    # Pakketbezorging
    tripMatrixParcels = pd.read_csv(
        (
            varDict['OUTPUTFOLDER'] +
            'TripMatrix_ParcelVans_NRM_' +
            varDict['LABEL'] +
            '.csv'
        ),
        sep=',')
    tripMatrixParcels = np.array(tripMatrixParcels)

    return tripMatrix, tripMatrixParcels


def put_emissions_into_tours(
    varDict: dict,
    tripsCO2: dict,
    tripsCO2_NL: dict,
    tripsDist: dict,
    tripsDist_NL: dict,
    parcelTripsCO2: dict,
    emissionDivFac: list
):
    """
    Zet emissies (en afstanden) in 'tours' en 'parcelTours'.

    Args:
        varDict (dict): _description_
        tripsCO2 (dict): _description_
        tripsCO2_NL (dict): _description_
        tripsDist (dict): _description_
        tripsDist_NL (dict): _description_
        parcelTripsCO2 (dict): _description_
        emissionDivFac (list): _description_

    Returns:
        tuple: Met daarin:
            - pd.DataFrame: tours verrijkt met CO2 en afstand
            - pd.DataFrame: parcelTours verrijkt met CO2
    """
    # Inlezen vracht tours
    tours = pd.read_csv(
        varDict['OUTPUTFOLDER'] + 'Tours_' + varDict['LABEL'] + '.csv',
        sep=',')

    # Koppelen CO2 en kilometers vracht
    tours['CO2'] = [tripsCO2[i] for i in tours.index]
    tours['CO2'] = np.round(tours['CO2'] / emissionDivFac[0], 3)
    tours['CO2_NL'] = [tripsCO2_NL[i] for i in tours.index]
    tours['CO2_NL'] = np.round(tours['CO2_NL'] / emissionDivFac[0], 3)
    tours['AFSTAND'] = [tripsDist[i] for i in tours.index]
    tours['AFSTAND_NL'] = [tripsDist_NL[i] for i in tours.index]

    # Inlezen pakket tours
    parcelTours = pd.read_csv(
        (
            varDict['OUTPUTFOLDER'] +
            'ParcelSchedule_' +
            varDict['LABEL'] +
            '.csv'
        ),
        sep=',')

    # Koppelen CO2 en kilometers pakket
    parcelTours['CO2'] = [
        parcelTripsCO2[i] for i in parcelTours.index]
    parcelTours['CO2'] = np.round(
        parcelTours['CO2'] / emissionDivFac[0], 3)

    return tours, parcelTours


def put_emissions_into_shipments(
    varDict: dict,
    tours: pd.DataFrame
):
    """
    Zet emissies in 'shipments'.

    Args:
        varDict (dict): _description_
        tours (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: De zendingen met daaraan de CO2-waarde gekoppeld.
    """
    # Calculate emissions at the tour level instead of trip level
    tours['TOUR_ID'] = [
        (
            str(tours.at[i, 'CARRIER_ID']) +
            '_' +
            str(tours.at[i, 'TOUR_ID'])
        )
        for i in tours.index]
    toursCO2 = pd.pivot_table(
        tours,
        values=['CO2'],
        index=['TOUR_ID'],
        aggfunc=np.sum)
    tourIDDict = dict(np.transpose(np.vstack(
        (toursCO2.index, np.arange(len(toursCO2))))))
    toursCO2 = np.array(toursCO2['CO2'])

    # Read the shipments
    shipments = pd.read_csv(
        (
            varDict['OUTPUTFOLDER'] +
            'Shipments_AfterScheduling_' +
            varDict['LABEL'] +
            '.csv'
        ),
        sep=',')
    shipments = shipments.sort_values('TOUR_ID')
    shipments.index = np.arange(len(shipments))

    # For each tour, which shipments belong to it
    tourIDs = [tourIDDict[x] for x in shipments['TOUR_ID']]
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
    skimDistance = read_mtx(varDict['SKIMDISTANCE'])
    nZones = int(len(skimDistance) ** 0.5)
    origNRM = np.array(shipments['ORIG_NRM'], dtype=int)
    destNRM = np.array(shipments['DEST_NRM'], dtype=int)
    shipDist = skimDistance[(origNRM - 1) * nZones + (destNRM - 1)]

    # Divide CO2 of each tour over its shipments based on distance
    shipCO2 = np.zeros(len(shipments))
    for tourID in np.unique(tourIDs):
        currentDists = shipDist[shipIDs[tourID]]
        currentCO2 = toursCO2[tourID]
        if np.sum(currentDists) == 0:
            shipCO2[shipIDs[tourID]] = 0
        else:
            shipCO2[shipIDs[tourID]] = (
                currentDists / np.sum(currentDists) * currentCO2)
    shipments['CO2'] = shipCO2

    # Sorteren op shipment ID
    shipments = shipments.sort_values('SHIP_ID')
    shipments.index = np.arange(len(shipments))

    return shipments


def write_links_to_shape(
    varDict: dict,
    links: pd.DataFrame,
    nodes: pd.DataFrame,
    nodeDict: dict,
    invNodeDict: dict,
    intensityFields: list,
    root='', startProgress=93.0, endProgress=100.0
):
    """
    Schrijf het geladen netwerk naar een shapefile in de uitvoerfolder.

    Args:
        varDict (dict): _description_
        links (pd.DataFrame): _description_
        nodes (pd.DataFrame): _description_
        nodeDict (dict): _description_
        invNodeDict (dict): _description_
        intensityFields (list): _description_
        root (str, optional): _description_. Defaults to ''.
        startProgress (float, optional): _description_. Defaults to 93.
        endProgress (float, optional): _description_. Defaults to 100.
    """
    # Set travel times of connectors at 0 for in
    # the output network shape
    links.loc[links['LINKTYPE'] == 99, 'T0_FREIGHT'] = 0.0
    links.loc[links['LINKTYPE'] == 99, 'T0_VAN'] = 0.0

    links['KNOOP_A'] = [nodeDict[x] for x in links['KNOOP_A']]
    links['KNOOP_B'] = [nodeDict[x] for x in links['KNOOP_B']]

    # Van kilometer terug naar meter
    links['DISTANCE'] *= 1000

    # Van uur terug naar secondes
    links['T0_FREIGHT'] *= 3600
    links['T0_VAN'] *= 3600

    # Initialize shapefile fields
    w = shp.Writer(
        varDict['OUTPUTFOLDER'] +
        'links_loaded_' +
        varDict['LABEL'] +
        '.shp')
    w.field('LINKNR',       'N', size=7, decimal=0)
    w.field('KNOOP_A',      'N', size=7, decimal=0)
    w.field('KNOOP_B',      'N', size=7, decimal=0)
    w.field('SNEL_FF',      'N', size=3, decimal=0)
    w.field('SNEL',         'N', size=3, decimal=0)
    w.field('CAP',          'N', size=5, decimal=0)
    w.field('LINKTYPE',     'N', size=2, decimal=0)
    w.field('DISTANCE',     'N', size=8, decimal=0)
    w.field('DOELSTROOK',   'N', size=1, decimal=0)
    w.field('NRM',          'N', size=1, decimal=0)
    w.field('ZEZ',          'N', size=1, decimal=0)
    w.field('NL',           'N', size=1, decimal=0)
    w.field('T0_FREIGHT',   'N', size=7, decimal=1)
    w.field('T0_VAN',       'N', size=7, decimal=1)
    w.field('COST_FREIGHT', 'N', size=8, decimal=3)
    w.field('COST_VAN',     'N', size=8, decimal=3)
    for field in intensityFields[:4]:
        w.field(field, 'N', size=13, decimal=3)
    for field in intensityFields[4:]:
        if field[:2] == 'N_':
            w.field(field, 'N', size=7, decimal=1)
        else:
            w.field(field, 'N', size=12, decimal=3)

    nodesX = np.array(nodes['X'], dtype=int)
    nodesY = np.array(nodes['Y'], dtype=int)
    linksA = [invNodeDict[x] for x in links['KNOOP_A']]
    linksB = [invNodeDict[x] for x in links['KNOOP_B']]

    dbfData = np.array(links, dtype=object)
    nLinks = dbfData.shape[0]
    for i in range(nLinks):
        # Add geometry
        line = []
        line.append([nodesX[linksA[i]], nodesY[linksA[i]]])
        line.append([nodesX[linksB[i]], nodesY[linksB[i]]])
        w.line([line])

        # Add data fields
        w.record(*dbfData[i, :])

        if i % 500 == 0:
            print(
                '\t' + str(round((i / nLinks) * 100, 1)) + '%',
                end='\r')

            if root != '':
                root.progressBar['value'] = (
                    startProgress + (endProgress - startProgress) * i / nLinks)

    w.close()
