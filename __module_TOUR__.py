import numpy as np
import pandas as pd
import time
import datetime
import multiprocessing as mp
import shapefile as shp
import functools
from __functions__ import read_shape, get_skims, determine_num_cpu

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
        self.root.title("Progress Tour Formation")
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

    try:

        start_time = time.time()

        root = args[0]
        varDict = args[1]

        if root != '':
            root.progressBar['value'] = 0

        log_file = open(
            varDict['OUTPUTFOLDER'] + "Logfile_TourFormation.log", "w")
        log_file.write(
            "Start simulation at: " +
            datetime.datetime.now().strftime("%y-%m-%d %H:%M") + "\n")

        if varDict['SEED'] != '':
            np.random.seed(varDict['SEED'])

        # Number of CPUs over which the tour formation is parallelized
        nCPU = determine_num_cpu(varDict)

        # Carrying capacity for each vehicle type
        if varDict['LABEL'] == 'REF':
            carryingCapacity = np.array(pd.read_csv(
                varDict['VEHCAPACITY'],
                index_col=0))[:-2, 0]
        elif varDict['LABEL'] == 'UCC':
            # In the UCC scenario we also need to define a capacity
            # for zero emission vehiclesn(last one in list, value=-1)
            carryingCapacity = np.array(pd.read_csv(
                varDict['VEHCAPACITY'],
                index_col=0))[:, 0]

        # The number of carriers that transport the shipments
        # not going to or from a DC
        nCarNonDC = 80

        # Maximum number of shipments in tour
        maxNumShips = 10

        dimLS = pd.read_csv(
            varDict['DIMFOLDER'] + 'logistic_segment.txt',
            sep='\t')
        dimNSTR = pd.read_csv(
            varDict['DIMFOLDER'] + 'nstr.txt',
            sep='\t')
        dimGG = pd.read_csv(
            varDict['DIMFOLDER'] + 'basgoed_goods_type.txt',
            sep='\t')
        dimVT = pd.read_csv(
            varDict['DIMFOLDER'] + 'vehicle_type.txt',
            sep='\t')
        dimCombType = pd.read_csv(
            varDict['DIMFOLDER'] + 'combustion_type.txt',
            sep='\t')

        nLS = len(dimLS) - 1
        nNSTR = len(dimNSTR) - 1
        nGG = len(dimGG)
        nVT = len(dimVT)
        nCombType = len(dimCombType)

        # Plek van de zonale attributen
        # Alle attributen komen 4 x voor
        # De volgorde is: 2030, 2040, 2050, 2018
        # Het basisjaar staat dus achteraan.
        if varDict['YEAR'] == 2018:
            yrAppendix = '_3'
        elif varDict['YEAR'] == 2030:
            yrAppendix = ''
        elif varDict['YEAR'] == 2040:
            yrAppendix = '_1'
        elif varDict['YEAR'] == 2050:
            yrAppendix = '_2'
        else:
            raise Exception(
                'YEAR parameter is not 2018, 2030, 2040 or 2050, but ' +
                str(varDict['YEAR']) + ' instead.')

        if root != '':
            root.progressBar['value'] = 0.1

        print("Importing shipments and zones...")
        log_file.write("Importing shipments and zones...\n")

        # Import zones data
        zones = read_shape(varDict['ZONES'])
        zones.index = zones['SEGNR_2018']
        zonesX = np.array(zones['XCOORD'])
        zonesY = np.array(zones['YCOORD'])
        nZones = len(zones)

        # Add level of urbanization based on segs (# houses / jobs)
        # and surface (square m)
        zonesSted = np.array(
            (100 * (zones['INWONERS' + yrAppendix] / zones['OPP'])) >= 2500,
            dtype=int)
        zonesLMS = np.array(zones['LMSVAM'], dtype=int)

        if root != '':
            root.progressBar['value'] = 0.5

        # Import logistic nodes data
        logNodes = pd.read_csv(varDict['DISTRIBUTIECENTRA'], sep=',')
        logNodes = logNodes[~pd.isna(logNodes['SEGNR_2018'])]
        nDC = len(logNodes)
        logNodes.index = np.arange(nDC)
        dcZones = np.array(logNodes['SEGNR_2018'])

        # Koppeltabel NRM naar BG
        NRMtoBG = pd.read_csv(varDict['NRM_TO_BG'], sep=',')
        dictNRMtoBG = dict(
            (NRMtoBG.at[i, 'SEGNR_2018'], NRMtoBG.at[i, 'ZONEBG2018'])
            for i in NRMtoBG.index)
        BGwithOverlapNRM = set(np.array(np.unique(
            NRMtoBG['ZONEBG2018']),
            dtype=int))

        # Import shipments data
        shipments, shipmentsCKM = get_shipments(varDict)

        # niet-container zendingen in kilogrammen
        shipments['WEIGHT'] *= 1000

        # Voeg kenmerken toe m.b.t. locatietypes aan niet-container zendingen
        # (header: LOGNODE_LOADING, LOGNODE_UNLOADING, URBAN, EXTERNAL)
        shipments = add_location_types_to_shipments(
            shipments, zonesSted, BGwithOverlapNRM)

        if root != '':
            root.progressBar['value'] = 1.0

        # Import ZEZ scenario input
        if varDict['LABEL'] == 'UCC':
            # Which zones are ZEZ and by which UCC-zone are they served
            zezZones = pd.read_csv(varDict['ZEZ_ZONES'], sep=',')
            zezToUCC = dict(
                (zezZones.at[i, 'SEGNR_2018'], zezZones.at[i, 'UCC'])
                for i in zezZones.index)
            uccZones = np.array(np.unique(zezZones['UCC']), dtype=int)
            zezZones = np.array(zezZones['SEGNR_2018'], dtype=int)

            # Cumulatieve kansen van de energiebronnen binnen elk
            # voertuigtype en logistiek segment
            cumProbCombustion = get_cum_prob_combustion(
                varDict, nCombType, nLS, nVT)

        print("Preparing shipments for tour formation...")
        log_file.write("Preparing shipments for tour formation...\n")

        # Add a column with a carrierID
        # - Shipments are first grouped by DC, one carrier per DC assumed
        #   [carrierID 0 to nDC].
        # - Shipments not loaded or unloaded at DC are randomly assigned
        #   to one of the other carriers
        #   [carrierID nDC to nDC+nCarNonDC].
        if varDict['LABEL'] == 'UCC':
            shipments = assign_carrier_to_shipments(
                shipments, varDict, nDC, nCarNonDC,
                zezZones, uccZones, zezToUCC)
        else:
            shipments = assign_carrier_to_shipments(
                shipments, varDict, nDC, nCarNonDC,
                [], [], {})

        # Place shipments in Numpy array for faster accessing of values
        shipments = shipments[
            ['SHIP_ID', 'ORIG_NRM', 'DEST_NRM', 'CARRIER',
             'VEHTYPE', 'NSTR', 'WEIGHT', 'LOGNODE_LOADING',
             'LOGNODE_UNLOADING', 'URBAN', 'LOGSEG', 'ORIG_BG',
             'DEST_BG', 'EXTERNAL', 'BG_GG', 'CONTAINER',
             'ORIG_LMS', 'DEST_LMS']]
        shipments = np.array(shipments, dtype=object)

        # Split shipments that are larger than the vehicle capacity
        # into multiple shipments
        shipments = split_shipments_with_capacity_exceedence(
            shipments, carryingCapacity)

        # Sort shipments by carrierID
        shipments = shipments[shipments[:, 3].argsort()]
        shipmentDict = dict(np.transpose(np.vstack(
            (np.arange(len(shipments)), shipments[:, 0].copy()))))

        # Give the shipments a new shipmentID after ordering the
        # array by carrierID
        shipments[:, 0] = np.arange(len(shipments))

        # Make sure SHIP_ID, ORIG_NRM and DEST_NRM are integers
        shipments[:, [0, 1, 2]] = np.array(shipments[:, [0, 1, 2]], dtype=int)

        if root != '':
            root.progressBar['value'] = 2.5

        print('Importing skims...')
        log_file.write('Importing skims...\n')

        # Import binary skim files
        skimTravTime, skimDistance = get_skims(varDict, changeZeroValues=True)
        nZones = int(len(skimTravTime) ** 0.5)

        # Concatenate time and distance arrays as a 2-column matrix
        skim = np.c_[skimTravTime, skimDistance]

        if root != '':
            root.progressBar['value'] = 4.0

        print('Importing logit parameters...')
        log_file.write('Importing logit parameters...\n')

        logitParams_ETfirst = np.array(pd.read_csv(
            varDict['PARAMS_ET_FIRST'], index_col=0))[:, 0]
        logitParams_ETlater = np.array(pd.read_csv(
            varDict['PARAMS_ET_LATER'], index_col=0))[:, 0]

        print('Obtaining other information required before tour formation...')
        log_file.write(
            'Obtaining other information required before tour formation...\n')

        # Total number of shipments and carriers
        nShipments = len(shipments)
        nCarriers = len(np.unique(shipments[:, 3]))

        # At which index is the first shipment of each carrier
        carMarkers = [0]
        for i in range(1, nShipments):
            if shipments[i, 3] != shipments[i - 1, 3]:
                carMarkers.append(i)
        carMarkers.append(nShipments)

        # How many shipments does each carrier have?
        nShipmentsPerCarrier = [None for car in range(nCarriers)]
        if nCarriers == 1:
            nShipmentsPerCarrier[0] = nShipments
        else:
            for i in range(nCarriers):
                nShipmentsPerCarrier[i] = carMarkers[i + 1] - carMarkers[i]

        if root != '':
            root.progressBar['value'] = 5.0

        # Bepaal het aantal CPUs dat we gebruiken en welke CPU
        # welke carriers doet
        chunks = [np.arange(nCarriers)[i::nCPU] for i in range(nCPU)]

        if nCPU > 1:

            print(f'Start tour formation (parallelized over {nCPU} cores)')
            log_file.write(
                f'Start tour formation (parallelized over {nCPU} cores)\n')

            # Initialiseer een pool object dat de taken verdeelt over de CPUs
            p = mp.Pool(nCPU)

            # Voer de tourformatie uit
            tourformationResult = p.map(
                functools.partial(
                    form_tours, carMarkers, shipments,
                    skim, nZones,
                    maxNumShips, carryingCapacity, dcZones,
                    nShipmentsPerCarrier, nCarriers,
                    logitParams_ETfirst, logitParams_ETlater,
                    nNSTR),
                chunks)

            # Wait for completion of parallellization processes
            p.close()
            p.join()

            if root != '':
                root.progressBar['value'] = 78.0

            # Initialization of lists with information regarding tours
            # tours:         here we store the shipmentIDs of each tour
            # tourSequences: here we store the order of (un)loading locations
            #                of each tour
            # nTours:        number of tours constructed for each carrier
            tours = [[] for car in range(nCarriers)]
            tourSequences = [[] for car in range(nCarriers)]
            nTours = np.zeros(nCarriers, dtype=int)
            tourIsExternal = [[] for car in range(nCarriers)]

            # Pak de tourformatieresultaten uit
            for cpu in range(nCPU):
                for car in chunks[cpu]:
                    tours[car] = tourformationResult[cpu][0][car]
                    tourSequences[car] = tourformationResult[cpu][1][car]
                    nTours[car] = tourformationResult[cpu][2][car]
                    tourIsExternal[car] = tourformationResult[cpu][3][car]

            del tourformationResult

        else:
            print('Start tour formation')
            log_file.write('Start tour formation\n')

            tours, tourSequences, nTours, tourIsExternal = form_tours(
                carMarkers, shipments,
                skim, nZones,
                maxNumShips, carryingCapacity, dcZones,
                nShipmentsPerCarrier, nCarriers,
                logitParams_ETfirst, logitParams_ETlater,
                nNSTR, chunks[0])

        print('\tTour formation completed for all carriers')
        log_file.write('\tTour formation completed for all carriers\n')

        if root != '':
            root.progressBar['value'] = 81.0

        print('Adding empty trips...')
        log_file.write('Adding empty trips...\n')

        # Add empty trips
        emptytripadded = [
            [None for tour in range(nTours[car])]
            for car in range(nCarriers)]
        for car in range(nCarriers):
            for tour in range(nTours[car]):
                # Only add empty trip if tour does not end at start location
                # and does not visit BG-zone outside of NRM-scope
                emptytripadded[car][tour] = (
                    (
                        tourSequences[car][tour][0] !=
                        tourSequences[car][tour][-1]
                    ) and
                    (tourIsExternal[car][tour] == 0))

                if emptytripadded[car][tour]:
                    tourSequences[car][tour].append(
                        tourSequences[car][tour][0])

        if root != '':
            root.progressBar['value'] = 82.0

        print('Obtaining number of shipments and trip weights...')
        log_file.write('Obtaining number of shipments and trip weights...\n')

        # Number of shipments and trips of each tour
        nShipmentsPerTour = [
            [len(tours[car][tour]) for tour in range(nTours[car])]
            for car in range(nCarriers)]
        nTripsPerTour = [
            [len(tourSequences[car][tour]) - 1 for tour in range(nTours[car])]
            for car in range(nCarriers)]
        nTripsTotal = np.sum(np.sum(nTripsPerTour))

        # Weight per trip
        tripWeights = [
            [np.zeros(nTripsPerTour[car][tour], dtype=int)
                for tour in range(nTours[car])]
            for car in range(nCarriers)]

        for car in range(nCarriers):
            for tour in range(nTours[car]):
                origs = [
                    tourSequences[car][tour][trip]
                    for trip in range(nTripsPerTour[car][tour])]
                shipmentLoaded = [
                    False for ship in range(nShipmentsPerTour[car][tour])]
                shipmentUnloaded = [
                    False for ship in range(nShipmentsPerTour[car][tour])]

                for trip in range(nTripsPerTour[car][tour]):
                    # Startzone of trip
                    orig = origs[trip]

                    # If it's the first trip of the tour, initialize a
                    # counter to add and subtract weight from
                    if trip == 0:
                        tmpTripWeight = 0

                    for i in range(len(tours[car][tour])):
                        ship = tours[car][tour][i]

                        # If loading location of the shipment was
                        # the startpoint of the trip and shipment has
                        # not been loaded/unloaded yet
                        if (
                            shipments[ship][1] == orig and
                            ~(shipmentLoaded[i] or shipmentUnloaded[i])
                        ):
                            # Add weight of shipment to counter
                            tmpTripWeight += shipments[ship][6]
                            shipmentLoaded[i] = True

                        # If unloading location of the shipment was
                        # the startpoint of the trip and shipment has
                        # not been unloaded yet
                        if (
                            shipments[ship][2] == orig and
                            shipmentLoaded[i] and not
                            shipmentUnloaded[i]
                        ):
                            # Remove weight of shipment from counter
                            tmpTripWeight -= shipments[ship][6]
                            shipmentUnloaded[i] = True

                    tripWeights[car][tour][trip] = tmpTripWeight

        if root != '':
            root.progressBar['value'] = 84.0

        print('Drawing departure times of tours ...')
        log_file.write('Drawing departure times of tours...\n')

        # Import probability distribution of departure time
        # (per logistic segment)
        probDepTime = pd.read_csv(varDict['DEPTIME_FREIGHT'], sep=',')
        cumProbDepTime = np.array(np.cumsum(probDepTime))

        depTimeTour = [
            [None for tour in range(nTours[car])]
            for car in range(nCarriers)]

        for car in range(nCarriers):
            for tour in range(nTours[car]):
                logSeg = shipments[tours[car][tour][0], 10]
                rand = np.random.rand()
                depTimeTour[car][tour] = np.where(
                    cumProbDepTime[:, logSeg] > rand)[0][0]

        if root != '':
            root.progressBar['value'] = 85.0

        print('Determining combustion type of tours...')
        log_file.write('Determining combustion type of tours...\n')

        # In the not-UCC scenario everything is assumed to be fuel
        combTypeTour = [
            [0 for tour in range(nTours[car])]
            for car in range(nCarriers)]

        if varDict['LABEL'] == 'UCC':
            zezZones = pd.read_csv(varDict['ZEZ_ZONES'], sep=',').astype(int)
            zezZones = np.array(zezZones['SEGNR_2018'], dtype=int)
            zezZones = set(list(zezZones))

            for car in range(nCarriers):
                for tour in range(nTours[car]):

                    # Combustion type for tours from/to UCCs
                    if car >= nDC + nCarNonDC:
                        ls = shipments[tours[car][tour][0], 10]
                        vt = shipments[tours[car][tour][0],  4]
                        rand = np.random.rand()
                        combTypeTour[car][tour] = np.where(
                            cumProbCombustion[vt][ls, :] > rand)[0][0]
                    else:
                        inZEZ = [
                            x in zezZones
                            for x in np.unique(tourSequences[car][tour])
                            if x != 0]

                        # Combustion type for tours within ZEZ
                        # (but not from/to UCCs)
                        if np.all(inZEZ):
                            rand = np.random.rand()
                            combTypeTour[car][tour] = np.where(
                                cumProbCombustion[vt][ls, :] > rand)[0][0]

                        # Hybrids for tours directly entering/leaving the ZEZ
                        elif np.any(inZEZ):
                            combTypeTour[car][tour] = 3

                        # Fuel for all other tours that do not go to the ZEZ
                        else:
                            combTypeTour[car][tour] = 0

        if root != '':
            root.progressBar['value'] = 86.0

        print('Writing tour data to CSV...')
        log_file.write('Writing tour data to CSV...\n')

        columns = [
            "CARRIER_ID", "TOUR_ID", "TRIP_ID",
            "ORIG_NRM", "DEST_NRM",
            "ORIG_BG", "DEST_BG",
            "ORIG_LMS", "DEST_LMS",
            "X_ORIG", "X_DEST",
            "Y_ORIG", "Y_DEST",
            "VEHTYPE",
            "NSTR",  "BG_GG", "LOG_SEG",
            "N_SHIP",
            "DC_ID",
            "TOUR_WEIGHT", "TRIP_WEIGHT",
            "TOUR_DEPTIME", "TRIP_DEPTIME", "TRIP_ARRTIME",
            "COMBTYPE",
            "EXT_BG",
            "CONTAINER"]
        dataTypes = [
            str, int,  int,
            int, int,
            int, int,
            int, int,
            float, float,
            float, float,
            int,
            int, int, int,
            int,
            int,
            float, float,
            float, float, float,
            int,
            int,
            int]

        outputTours = np.zeros((nTripsTotal, len(columns)), dtype=object)

        tripcount = 0
        for car in range(nCarriers):
            for tour in range(nTours[car]):
                for trip in range(nTripsPerTour[car][tour]):
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
                    outputTours[tripcount][7] = zonesLMS[origNRM - 1]
                    outputTours[tripcount][8] = zonesLMS[destNRM - 1]

                    # X/Y coordinates of origin and destination of trip
                    orig = tourSequences[car][tour][trip] - 1
                    dest = tourSequences[car][tour][trip + 1] - 1
                    outputTours[tripcount][9] = zonesX[orig]
                    outputTours[tripcount][10] = zonesX[dest]
                    outputTours[tripcount][11] = zonesY[orig]
                    outputTours[tripcount][12] = zonesY[dest]

                    # Vehicle type
                    outputTours[tripcount][13] = shipments[
                        tours[car][tour][0], 4]

                    # Dominant NSTR and BasGoed goods type (by weight)
                    outputTours[tripcount][14] = max_nstr(
                        tours[car][tour], shipments, nNSTR)
                    outputTours[tripcount][15] = max_bggg(
                        tours[car][tour], shipments, nGG)

                    # Logistic segment of tour
                    outputTours[tripcount][16] = shipments[
                        tours[car][tour][0], 10]

                    # Number of shipments transported in tour
                    outputTours[tripcount][17] = nShipmentsPerTour[car][tour]

                    # ID for DC zones
                    outputTours[tripcount][18] = (
                        dcZones[outputTours[tripcount][0]]
                        if car < nDC else -9999)

                    # Sum of the weight carried in the tour
                    outputTours[tripcount][19] = sum_weight(
                        tours[car][tour], shipments)

                    # Weight carried in the trip
                    outputTours[tripcount][20] = (
                        tripWeights[car][tour][trip] / 1000)

                    # Departure time of tour
                    outputTours[tripcount][21] = depTimeTour[car][tour]

                    # combustion type of the vehicle
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
                            skim[(origNRM - 1) * nZones + (destNRM - 1)][0] /
                            3600)
                        outputTours[tripcount][22] = depTime
                        outputTours[tripcount][23] = depTime + travTime
                    else:
                        dwellTime = 0.5 * np.random.rand()
                        depTime = outputTours[tripcount - 1][21] + dwellTime
                        origNRM = outputTours[tripcount][3]
                        destNRM = outputTours[tripcount][4]
                        travTime = (
                            skim[(origNRM - 1) * nZones + (destNRM - 1)][0] /
                            3600)
                        outputTours[tripcount][22] = depTime
                        outputTours[tripcount][23] = depTime + travTime

                    # BG-GG of empty trips is -1
                    if tripWeights[car][tour][trip] == 0:
                        outputTours[tripcount][15] = -1

                    tripcount += 1

                # BG-GG of empty trips is -1
                if emptytripadded[car][tour]:
                    outputTours[tripcount - 1][15] = -1

        nTripsMS = tripcount

        if root != '':
            root.progressBar['value'] = 87.0

        # Containerized trips
        nTripsCKM = len(shipmentsCKM)
        outputToursCKM = np.zeros((nTripsCKM, len(columns)), dtype=object)
        origsNRM = np.array(shipmentsCKM['ORIG_NRM'])
        destsNRM = np.array(shipmentsCKM['DEST_NRM'])
        origsBG = np.array(shipmentsCKM['ORIG_BG'])
        destsBG = np.array(shipmentsCKM['DEST_BG'])
        for i in range(nTripsCKM):
            outputToursCKM[i, 0] = -9999
            outputToursCKM[i, 1] = i
            outputToursCKM[i, 2] = 0

            origNRM = int(origsNRM[i])
            destNRM = int(destsNRM[i])
            origBG = int(origsBG[i])
            destBG = int(destsBG[i])
            origLMS = zonesLMS[origNRM - 1]
            destLMS = zonesLMS[destNRM - 1]

            nstr = shipmentsCKM.at[i, 'NSTR']
            bggg = shipmentsCKM.at[i, 'BG_GG']
            logseg = shipmentsCKM.at[i, 'LOGSEG']

            outputToursCKM[i, 3] = origNRM
            outputToursCKM[i, 4] = destNRM
            outputToursCKM[i, 5] = origBG
            outputToursCKM[i, 6] = destBG
            outputToursCKM[i, 7] = origLMS
            outputToursCKM[i, 8] = destLMS
            outputToursCKM[i, 9] = zonesX[origNRM - 1]
            outputToursCKM[i, 10] = zonesY[origNRM - 1]
            outputToursCKM[i, 11] = zonesX[destNRM - 1]
            outputToursCKM[i, 12] = zonesY[destNRM - 1]
            outputToursCKM[i, 13] = shipmentsCKM.at[i, 'VEHTYPE']
            outputToursCKM[i, 14] = nstr
            outputToursCKM[i, 15] = bggg
            outputToursCKM[i, 16] = logseg
            outputToursCKM[i, 17] = 1
            outputToursCKM[i, 18] = -9999
            outputToursCKM[i, 19] = shipmentsCKM.at[i, 'WEIGHT']
            outputToursCKM[i, 20] = shipmentsCKM.at[i, 'WEIGHT']

            depTime = np.where(
                cumProbDepTime[:, logSeg] > np.random.rand())[0][0]
            travTime = skim[(origNRM - 1) * nZones + (destNRM - 1), 0] / 3600

            outputToursCKM[i, 21] = depTime
            outputToursCKM[i, 22] = depTime
            outputToursCKM[i, 23] = depTime + travTime

            # Combustion type
            outputToursCKM[i, 24] = 0

            # If BG-zone outside of scope of NRM-zone system
            if origBG not in BGwithOverlapNRM:
                outputToursCKM[i, 25] = origBG
            elif destBG not in BGwithOverlapNRM:
                outputToursCKM[i, 25] = destBG
            else:
                outputToursCKM[i, 25] = -9999

            # Containerized
            outputToursCKM[i, 26] = 1

        outputTours = np.append(outputTours, outputToursCKM, axis=0)

        # Create DataFrame object for easy formatting and exporting to csv
        outputTours = pd.DataFrame(outputTours, columns=columns)
        for i in range(len(outputTours.columns)):
            outputTours.iloc[:, i] = (
                outputTours.iloc[:, i].astype(dataTypes[i]))

        outputTours.to_csv(
            varDict['OUTPUTFOLDER'] + 'Tours_' + varDict['LABEL'] + '.csv',
            index=None,
            header=True)

        message = (
            'Tour data written to ' +
            varDict['OUTPUTFOLDER'] + 'Tours_' + varDict['LABEL'] + '.csv')
        print(message), log_file.write(message + '\n')

        if root != '':
            root.progressBar['value'] = 89.0

        print('Enriching Shipments CSV...')
        log_file.write('Enriching Shipments CSV...\n')

        shipmentTourID = {}
        for car in range(nCarriers):
            for tour in range(nTours[car]):
                for ship in range(len(tours[car][tour])):
                    shipmentTourID[shipmentDict[tours[car][tour][ship]]] = (
                        f'{car}_{tour}')

        # Place shipments in DataFrame with the right headers
        shipments = pd.DataFrame(shipments)
        shipments.columns = [
            'SHIP_ID', 'ORIG_NRM', 'DEST_NRM', 'CARRIER',
            'VEHTYPE', 'NSTR', 'WEIGHT', 'LOGNODE_LOADING',
            'LOGNODE_UNLOADING', 'URBAN', 'LOGSEG', 'ORIG_BG',
            'DEST_BG', 'EXTERNAL', 'BG_GG', 'CONTAINER',
            'ORIG_LMS', 'DEST_LMS']

        shipments['WEIGHT'] /= 1000
        shipments['SHIP_ID'] = [
            shipmentDict[x] for x in shipments['SHIP_ID']]
        shipments['TOUR_ID'] = [
            shipmentTourID[x] for x in shipments['SHIP_ID']]
        nShipsMS = len(shipments)

        # Extra columns also for containerized shipments
        shipmentsCKM['CARRIER'] = -9999
        shipmentsCKM['LOGNODE_LOADING'] = -9999
        shipmentsCKM['LOGNODE_UNLOADING'] = -9999
        shipmentsCKM['URBAN'] = -9999
        shipmentsCKM['SHIP_ID'] = nShipsMS + np.arange(len(shipmentsCKM))
        shipmentsCKM['TOUR_ID'] = [
            f'-9999_{i}' for i in range(len(shipmentsCKM))]
        shipmentsCKM['EXTERNAL'] = 0
        for i in shipmentsCKM.index:
            origBG = shipments.at[i, 'ORIG_BG']

            if origBG not in BGwithOverlapNRM:
                shipmentsCKM.at[i, 'EXTERNAL'] = 1
            else:
                destBG = shipments.at[i, 'DEST_BG']
                if destBG not in BGwithOverlapNRM:
                    shipmentsCKM.at[i, 'EXTERNAL'] = 1

        # Append containerized shipments
        shipments = shipments.append(shipmentsCKM[list(shipments.columns)])
        shipments.index = np.arange(len(shipments))

        shipments = shipments.sort_values('SHIP_ID')

        # Change order of columns back to original
        shipments = shipments[
            ["SHIP_ID", "CARRIER", "TOUR_ID",
             "ORIG_BG", "DEST_BG",
             "ORIG_NRM", "DEST_NRM",
             "ORIG_LMS", "DEST_LMS",
             "BG_GG", "NSTR",
             "LOGSEG",
             "WEIGHT",
             "VEHTYPE", "CONTAINER"]]

        shipments.to_csv(
            (
                varDict['OUTPUTFOLDER'] +
                "Shipments_AfterScheduling_" +
                varDict['LABEL'] +
                '.csv'
            ),
            index=False)

        if root != '':
            root.progressBar['value'] = 90.0

        # Export as shapefile
        print("Writing Shapefile...")
        log_file.write("Writing Shapefile...\n")
        percStart = 90
        percEnd = 98
        if root != '':
            root.progressBar['value'] = percStart

        Ax = list(outputTours['X_ORIG'])
        Ay = list(outputTours['Y_ORIG'])
        Bx = list(outputTours['X_DEST'])
        By = list(outputTours['Y_DEST'])

        # Initialize shapefile fields
        w = shp.Writer(
            varDict['OUTPUTFOLDER'] + 'Tours_' + varDict['LABEL'] + '.shp')
        w.field('CARRIER_ID',   'N', size=5, decimal=0)
        w.field('TOUR_ID',      'N', size=5, decimal=0)
        w.field('TRIP_ID',      'N', size=3, decimal=0)
        w.field('ORIG_NRM',     'N', size=4, decimal=0)
        w.field('DEST_NRM',     'N', size=4, decimal=0)
        w.field('ORIG_BG',      'N', size=4, decimal=0)
        w.field('DEST_BG',      'N', size=4, decimal=0)
        w.field('ORIG_LMS',     'N', size=4, decimal=0)
        w.field('DEST_LMS',     'N', size=4, decimal=0)
        w.field('VEHTYPE',      'N', size=2, decimal=0)
        w.field('NSTR',         'N', size=2, decimal=0)
        w.field('BG_GG',        'N', size=2, decimal=0)
        w.field('LOGSEG',       'N', size=2, decimal=0)
        w.field('N_SHIP',       'N', size=3, decimal=0)
        w.field('DC_ID',        'N', size=4, decimal=0)
        w.field('TOUR_WEIGHT',  'N', size=5, decimal=2)
        w.field('TRIP_WEIGHT',  'N', size=5, decimal=2)
        w.field('TOUR_DEPTIME', 'N', size=4, decimal=2)
        w.field('TRIP_DEPTIME', 'N', size=5, decimal=2)
        w.field('TRIP_ARRTIME', 'N', size=5, decimal=2)
        w.field('COMBTYPE',     'N', size=2, decimal=0)
        w.field('EXT_BG',       'N', size=5, decimal=0)
        w.field('CONTAINER',    'N', size=2, decimal=0)

        coordCols = ['X_ORIG', 'Y_ORIG',
                     'X_DEST', 'Y_DEST']
        dbfData = np.array(outputTours.drop(columns=coordCols), dtype=object)

        nTrips = nTripsMS + nTripsCKM

        for i in range(nTrips):
            # Add geometry
            w.line([[[Ax[i], Ay[i]],
                     [Bx[i], By[i]]]])

            # Add data fields
            w.record(*dbfData[i, :])

            if i % 500 == 0:
                print('\t' + str(round((i / nTrips) * 100, 1)) + '%', end='\r')

                if root != '':
                    root.progressBar['value'] = (
                        percStart + (percEnd - percStart) * i / nTrips)

        w.close()

        message = (
            'Tours written to ' +
            varDict['OUTPUTFOLDER'] + 'Tours_' + varDict['LABEL'] + 'shp')
        print(message), log_file.write(message + '\n')

        print('Generating trip matrix...')
        log_file.write('Generating trip matrix...\n')

        # Maak dummies in tours variabele per logistiek segment,
        # voertuigtype en N_TOT (altijd 1 hier)
        for ls in range(nLS):
            outputTours['N_LS' + str(ls)] = (
                outputTours['LOG_SEG'] == ls).astype(int)
        for vt in range(nVT):
            outputTours['N_VEH' + str(vt)] = (
                outputTours['VEHTYPE'] == vt).astype(int)
        outputTours['N_TOT'] = 1

        # NRM-tripmatrix
        cols = [
            'ORIG_NRM', 'DEST_NRM',
            'N_LS0', 'N_LS1', 'N_LS2', 'N_LS3',
            'N_LS4', 'N_LS5', 'N_LS6', 'N_LS7',
            'N_VEH0', 'N_VEH1', 'N_VEH2', 'N_VEH3',
            'N_VEH4', 'N_VEH5', 'N_VEH6', 'N_VEH7',
            'N_VEH8', 'N_VEH9', 'N_VEH10',
            'N_TOT']

        # Gebruik deze dummies om het aantal ritten per HB te bepalen,
        # voor elk logistiek segment, voertuigtype en totaal
        pivotTable = pd.pivot_table(
            outputTours,
            values=cols[2:],
            index=['ORIG_NRM', 'DEST_NRM'],
            aggfunc=np.sum)
        pivotTable['ORIG_NRM'] = [x[0] for x in pivotTable.index]
        pivotTable['DEST_NRM'] = [x[1] for x in pivotTable.index]
        pivotTable = pivotTable[cols]

        pivotTable.to_csv(
            (
                varDict['OUTPUTFOLDER'] +
                "TripMatrix_Freight_NRM_" +
                varDict['LABEL'] +
                ".csv"
            ),
            index=False,
            sep=',')

        message = (
            'Trip matrix written to ' +
            varDict['OUTPUTFOLDER'] +
            "TripMatrix_Freight_NRM_" +
            varDict['LABEL'] +
            ".csv")
        print(message), log_file.write(message + '\n')

        if root != '':
            root.progressBar['value'] = 99.0

        # BG-tripmatrix
        cols = [
            'ORIG_BG', 'DEST_BG',
            'N_LS0', 'N_LS1', 'N_LS2', 'N_LS3',
            'N_LS4', 'N_LS5', 'N_LS6', 'N_LS7',
            'N_VEH0', 'N_VEH1', 'N_VEH2', 'N_VEH3',
            'N_VEH4', 'N_VEH5', 'N_VEH6', 'N_VEH7',
            'N_VEH8', 'N_VEH9', 'N_VEH10',
            'N_TOT']

        # Gebruik deze dummies om het aantal ritten per HB te bepalen,
        # voor elk logistiek segment, voertuigtype en totaal
        pivotTable = pd.pivot_table(
            outputTours,
            values=cols[2:],
            index=['ORIG_BG', 'DEST_BG'],
            aggfunc=np.sum)
        pivotTable['ORIG_BG'] = [x[0] for x in pivotTable.index]
        pivotTable['DEST_BG'] = [x[1] for x in pivotTable.index]
        pivotTable = pivotTable[cols]

        pivotTable.to_csv(
            (
                varDict['OUTPUTFOLDER'] +
                "TripMatrix_Freight_BG_" +
                varDict['LABEL'] +
                ".csv"
            ),
            index=False,
            sep=',')

        message = (
            'Trip matrix written to' +
            varDict['OUTPUTFOLDER'] +
            "TripMatrix_Freight_BG_" +
            varDict['LABEL'] +
            ".csv")
        print(message), log_file.write(message + '\n')

        totaltime = round(time.time() - start_time, 2)
        log_file.write("Total runtime: %s seconds\n" % (totaltime))
        log_file.write(
            "End simulation at: " +
            datetime.datetime.now().strftime("%y-%m-%d %H:%M") + "\n")
        log_file.close()

        if root != '':
            root.update_statusbar("Tour Formation: Done")
            root.progressBar['value'] = 100

            # 0 means no errors in execution
            root.returnInfo = [0, [0, 0]]

            return root.returnInfo

        else:
            return [0, [0, 0]]

    except Exception:
        import sys
        log_file.write(str(sys.exc_info()[0])), log_file.write("\n")
        import traceback
        log_file.write(str(traceback.format_exc())), log_file.write("\n")
        log_file.write("Execution failed!")
        log_file.close()

        if root != '':
            # Use this information to display as error message in GUI
            root.returnInfo = [1, [sys.exc_info()[0], traceback.format_exc()]]

            if __name__ == '__main__':
                root.update_statusbar("Tour Formation: Execution failed!")
                errorMessage = (
                    'Execution failed!\n\n' +
                    str(root.returnInfo[1][0]) + '\n\n' +
                    str(root.returnInfo[1][1]))
                root.error_screen(text=errorMessage, size=[900, 350])

            else:
                return root.returnInfo

        else:
            return [1, [sys.exc_info()[0], traceback.format_exc()]]


def get_traveltime(
    orig: int, dest: int,
    skim: np.ndarray,
    nZones: int
):
    '''
    Obtain the travel time [h] for orig to a destination zone.

    Args:
        orig (int): Zone number of the origin.
        dest (int): Zone number of the destination.
        skim (int): Long skim with travel times in column 0
            and distances in column 1.
        nZones (int): The number of zones.

    Returns:
        float: The travel time between 'orig' and 'dest' in hours.
    '''
    return skim[(orig - 1) * nZones + (dest - 1)][0] / 3600


def get_distance(
    orig: int, dest: int,
    skim: np.ndarray,
    nZones: int
):
    '''
    Obtain the distance [km] for an origin to a destination zone.

    Args:
        orig (int): Zone number of the origin.
        dest (int): Zone number of the destination.
        skim (int): Long skim with travel times in column 0
            and distances in column 1.
        nZones (int): The number of zones.

    Returns:
        float: The distance between 'orig' and 'dest' in kilometers.
    '''
    return skim[(orig - 1) * nZones + (dest - 1)][1] / 1000


def nearest_neighbor(
    tourLocs: np.ndarray,
    dc: int,
    skim: np.ndarray,
    nZones: int
):
    '''
    Creates a tour sequence to visit all loading and unloading locations.
    First a nearest-neighbor search is applied,then a 2-opt posterior
    improvement phase.

    Args:
        tourLocs (numpy.ndarray): Array with loading locations of shipments
            in column 0 and unloading locations in column 1
        dc (int): Zone ID of the DC in case the carrier is located in a DC zone

    Returns:
        numpy.ndarray: The order of visiting the zones of the tour.
    '''
    # ------------------------ Initialization -------------------------------
    # Arrays of unique loading and unloading locations to be visited
    loading = np.unique(tourLocs[:, 0])
    unloading = np.unique(tourLocs[:, 1])

    # Total number of loading and unloading locations to visit
    nLoad = len(loading)
    nUnload = len(unloading)

    # Here we will store the sequence of locations
    tourSequence = np.zeros(nLoad + nUnload, dtype=int)

    # ----------------------- Loading sequence -------------------------------
    # First visited location = first listed location
    tourSequence[0] = tourLocs[0, 0]

    # This location has already been visited,
    # remove from list of remaining loading locations
    loading = loading[loading != tourLocs[0, 0]]

    # For each shipment (except the last), decide the one that is
    # visited for loading
    for currentship in range(nLoad - 1):

        # (Re)initialize array with travel times from current to
        # each remaining shipment
        timeCurrentToRemaining = np.zeros(len(loading))

        # Go over all remaining loading locations
        for remainship in range(len(loading)):

            # Fill in travel time current location to
            # each remaining loading location
            timeCurrentToRemaining[remainship] = get_traveltime(
                tourSequence[currentship],
                loading[remainship],
                skim,
                nZones)

        # Index of the nearest unvisited loading location
        nearestShipment = np.argmin(timeCurrentToRemaining)

        # Fill in as the next location in tour sequence
        tourSequence[currentship + 1] = loading[nearestShipment]

        # Remove this shipment from list of remaining loading locations
        loading = np.delete(loading, nearestShipment)

    # ------------------ Unloading sequence ------------------------
    for currentship in range(nUnload):

        # (Re)initialize array with travel times from current to
        # each remaining shipment
        timeCurrentToRemaining = np.zeros(len(unloading))

        # Go over all remaining unloading locations
        for remainship in range(len(unloading)):

            # Fill in travel time current location to
            # each remaining loading location
            timeCurrentToRemaining[remainship] = get_traveltime(
                tourSequence[nLoad - 1 + currentship],
                unloading[remainship],
                skim,
                nZones)

        # Index of the nearest unvisited unloading location
        nearestShipment = np.argmin(timeCurrentToRemaining)

        # Fill in as the next location in tour sequence
        tourSequence[nLoad + currentship] = unloading[nearestShipment]

        # Remove this shipment from list of remaining unloading locations
        unloading = np.delete(unloading, nearestShipment)

    # ---------- Some restructuring of tourSequence variable ------------------
    tourSequence = list(tourSequence)

    # If the carrier is at a DC, start the tour here
    # (not always necessarily the first loading location)
    if dc is not None:
        if tourSequence[0] != dc:
            tourSequence.insert(0, int(dc))

    # Make sure the tour does not visit the homebase in between
    # (otherwise it's not 1 tour but 2 tours)
    nStartLoc = set(np.where(np.array(tourSequence) == tourSequence[0])[0][1:])
    if len(nStartLoc) > 1:
        tourSequence = [
            tourSequence[x]
            for x in range(len(tourSequence))
            if x not in nStartLoc]
        tourSequence.append(tourSequence[0])
    lenTourSequence = len(tourSequence)

    # ----------------- 2-opt posterior improvement ---------------------
    # Only do 2-opt if tour has more than 3 stops
    if lenTourSequence > 3:
        startLocations = np.array(tourSequence[:-1]) - 1
        endLocations = np.array(tourSequence[1:]) - 1
        tourDuration = np.sum(
            skim[startLocations * nZones + endLocations, 0]) / 3600

        for shiftLocA in range(1, lenTourSequence - 1):
            for shiftLocB in range(1, lenTourSequence - 1):
                if shiftLocA != shiftLocB:
                    swappedTourSequence = tourSequence.copy()
                    swappedTourSequence[shiftLocA] = tourSequence[shiftLocB]
                    swappedTourSequence[shiftLocB] = tourSequence[shiftLocA]

                    swappedStartLocations = np.array(
                        swappedTourSequence[:-1]) - 1
                    swappedEndLocations = np.array(
                        swappedTourSequence[1:]) - 1
                    swappedTourDuration = np.sum(
                        skim[swappedStartLocations * nZones +
                             swappedEndLocations, 0]) / 3600

                    # Only make the swap definitive
                    # if it reduces the tour duration
                    if swappedTourDuration < tourDuration:

                        # Check if the loading locations are visited
                        # before the unloading locations
                        precedence = [None for i in range(len(tourLocs))]
                        for i in range(len(tourLocs)):
                            load, unload = tourLocs[i, 0], tourLocs[i, 1]
                            precedence[i] = (
                                np.where(
                                    swappedTourSequence == unload
                                )[0][-1] >
                                np.where(
                                    swappedTourSequence == load
                                )[0][0])

                        if np.all(precedence):
                            tourSequence = swappedTourSequence.copy()
                            tourDuration = swappedTourDuration

    return np.array(tourSequence, dtype=int)


def get_tourdur(
    tourSequence: np.ndarray,
    skim: np.ndarray,
    nZones: int
):
    '''
    Calculates the tour duration of the tour (so far)

    tourSequence = array with ordered sequence of visited locations in tour
    '''
    tourSequence = np.array(tourSequence)
    tourSequence = tourSequence[tourSequence != 0]
    startLocations = tourSequence[: -1] - 1
    endLocations = tourSequence[1:] - 1
    tourDuration = np.sum(
        skim[startLocations * nZones + endLocations, 0]) / 3600

    return tourDuration


def cap_utilization(
    veh: int,
    weights: np.ndarray,
    carryingCapacity: np.ndarray
):
    '''
    Calculates the capacity utilzation of the tour (so far)
    Assume that vehicle type chosen for 1st shipment
    in tour defines the carrying capacity
    '''
    cap = carryingCapacity[veh]
    weight = sum(weights)
    return weight / cap


def proximity(
    tourlocs: np.ndarray,
    universal: np.ndarray,
    skim: np.ndarray,
    nZones: int
):
    '''
    Reports the proximity value [km] of each shipment in
    the universal choice set

    tourlocs = array with the locations visited in the tour so far
    universal = the universal choice set
    '''
    # Unique locations visited in the tour so far
    # (except for tour starting point)
    if tourlocs[0, 0] == tourlocs[0, 1]:
        tourlocs = [
            x for x in np.unique(tourlocs)
            if x != 0]
    else:
        tourlocs = [
            x for x in np.unique(tourlocs)
            if x != 0 and x != tourlocs[0, 0]]

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
):
    '''
    Returns a Boolean that states whether a logistical node
    is visited in the tour for loading
    '''
    if len(tour) == 1:
        return shipments[tour][0][7] == 2
    else:
        return np.any(shipments[tour][0][7] == 2)


def lognode_unloading(
    tour: np.ndarray,
    shipments: np.ndarray
):
    '''
    Returns a Boolean that states whether a logistical node
    is visited in the tour for unloading
    '''
    if len(tour) == 1:
        return shipments[tour][0][8] == 2
    else:
        return np.any(shipments[tour][0][8] == 2)


def transship(
    tour: np.ndarray,
    shipments: np.ndarray
):
    '''
    Returns a Boolean that states whether a transshipment zone
    is visited in the tour
    '''
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
):
    '''
    Returns a Boolean that states whether an urban zone is visited in the tour
    '''
    if len(tour) == 1:
        return shipments[tour][0][9]
    else:
        return np.any(shipments[tour][0][9] == 1)


def is_concrete():
    '''
    Returns a Boolean that states whether concrete is transported in the tour
    '''
    return False


def max_nstr(
    tour: np.ndarray,
    shipments: np.ndarray,
    nNSTR: int
):
    '''
    Returns the NSTR goods type (0-9) of which the highest weight is
    transported in the tour (so far)

    tour = array with the IDs of all shipments in the tour
    '''
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
):
    '''
    Returns the BasGoed goods type (1-13) of which the highest weight is
    transported in the tour (so far)

    tour = array with the IDs of all shipments in the tour
    '''
    bgggWeight = np.zeros(nGG)

    for i in range(len(tour)):
        shipBGGG = shipments[tour[i], 14]
        shipWeight = shipments[tour[i], 6]
        bgggWeight[shipBGGG - 1] += shipWeight

    return np.argmax(bgggWeight) + 1


def sum_weight(
    tour: np.ndarray,
    shipments: np.ndarray
):
    '''
    Returns the weight of all goods that are transported in the tour

    tour = array with the IDs of all shipments in the tour
    '''
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
    params: np.ndarray,
    shipments: np.ndarray,
    nNSTR: int
):
    '''
    Returns True if we decide to end the tour, False if we decide
    to add another shipment
    '''
    # Calculate explanatory variables
    vehicleTypeIs0 = (shipments[tour[0], 4] == 0) * 1
    vehicleTypeIs1 = (shipments[tour[0], 4] == 1) * 1
    maxNSTR = max_nstr(tour, shipments, nNSTR)
    nstrIs0 = (maxNSTR == 0) * 1
    nstrIs1 = (maxNSTR == 1) * 1
    nstrIs2to5 = (maxNSTR in [2, 3, 4, 5]) * 1
    nstrIs6 = (maxNSTR == 6) * 1
    nstrIs7 = (maxNSTR == 7) * 1
    nstrIs8 = (maxNSTR == 8) * 1

    # Calculate utility
    etUtility = (params[0] +
                 params[1] * tourDuration**0.5 +
                 params[2] * capUt**2 +
                 params[3] * transship(tour, shipments) +
                 params[4] * lognode_loading(tour, shipments) +
                 params[5] * lognode_unloading(tour, shipments) +
                 params[6] * urbanzone(tour, shipments) +
                 params[7] * vehicleTypeIs0 +
                 params[8] * vehicleTypeIs1 +
                 params[9] * nstrIs0 +
                 params[10] * nstrIs1 +
                 params[11] * nstrIs2to5 +
                 params[12] * nstrIs6 +
                 params[13] * nstrIs7 +
                 params[14] * nstrIs8)

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
    params: np.ndarray,
    shipments: np.ndarray,
    nNSTR: int
):
    '''
    Returns True if we decide to end the tour, False if we decide
    to add another shipment
    '''
    # Calculate explanatory variables
    tourDuration = get_tourdur(
        tourSequence,
        skim,
        nZones)
    prox = np.min(proximity(
        tourLocs,
        universal,
        skim,
        nZones))
    capUt = cap_utilization(
        shipments[tour[0], 4],
        shipments[tour, 6],
        carryingCapacity)
    numberOfStops = len(np.unique(tourLocs))
    vehicleTypeIs0 = (shipments[tour[0], 4] == 0) * 1
    vehicleTypeIs1 = (shipments[tour[0], 4] == 1) * 1
    maxNSTR = max_nstr(tour, shipments, nNSTR)
    nstrIs0 = (maxNSTR == 0) * 1
    nstrIs1 = (maxNSTR == 1) * 1
    nstrIs6 = (maxNSTR == 6) * 1
    nstrIs7 = (maxNSTR == 7) * 1
    nstrIs8 = (maxNSTR == 8) * 1

    # Calculate utility
    etUtility = (params[0] +
                 params[1] * (tourDuration) +
                 params[2] * capUt +
                 params[3] * prox +
                 params[4] * transship(tour, shipments) +
                 params[5] * np.log(numberOfStops) +
                 params[6] * lognode_loading(tour, shipments) +
                 params[7] * lognode_unloading(tour, shipments) +
                 params[8] * urbanzone(tour, shipments) +
                 params[9] * vehicleTypeIs0 +
                 params[10] * vehicleTypeIs1 +
                 params[11] * nstrIs0 +
                 params[12] * nstrIs1 +
                 params[13] * nstrIs6 +
                 params[14] * nstrIs7 +
                 params[15] * nstrIs8)

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
    shipments: np.ndarray
):
    '''
    Returns the chosen shipment based on Select Shipment MNL
    '''

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
    prox = proximity(tourLocs, universal, skim, nZones)

    # Which shipments belong to the same BasGoed-goods type
    # and have the same vehicle type as the tour
    sameBGGG = np.array(universal[:, 14] == bggg)
    sameVT = np.array(universal[:, 4] == vt)

    # Initialize feasible choice set, those shipments that
    # comply with constraints
    if isExternal == 1:
        selectShipConstraints = (
            (capUt < 1.1) &
            (sameBGGG) &
            (universal[:, 11] == origBG) &
            (universal[:, 12] == destBG))
    else:
        selectShipConstraints = (
            (capUt < 1.1) &
            (sameBGGG) &
            (universal[:, 13] == 0))

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
    carMarkers: np.ndarray,
    shipments: np.ndarray,
    skim: np.ndarray,
    nZones: int,
    maxNumShips: int,
    carryingCapacity: np.ndarray,
    dcZones: np.ndarray,
    nShipmentsPerCarrier: np.ndarray,
    nCarriers: int,
    logitParams_ETfirst: np.ndarray,
    logitParams_ETlater: np.ndarray,
    nNSTR: int,
    cars: np.ndarray
):
    '''
    Run the tour formation procedure for a set of carriers
    with a set of shipments.
    '''

    tours = [[] for car in range(nCarriers)]
    tourSequences = [[] for car in range(nCarriers)]
    nTours = np.zeros(nCarriers, dtype=int)
    tourIsExternal = [[] for car in range(nCarriers)]

    for car in cars:
        print(
            f'\tForming tours for carrier {car+1} of {nCarriers}...',
            end='\r')

        tourCount = 0

        # Universal choice set = all non-allocated shipments per carrier
        universalChoiceSet = shipments[
            carMarkers[car]:carMarkers[car + 1], :].copy()

        if car < len(dcZones):
            dc = dcZones[car]

            # Sort by shipment distance, this will help in constructing
            # more efficient/realistic tours
            shipmentDistances = skim[
                (universalChoiceSet[:, 1] - 1) * nZones +
                (universalChoiceSet[:, 1] - 1)]
            universalChoiceSet = np.c_[
                universalChoiceSet, shipmentDistances]
            universalChoiceSet = universalChoiceSet[
                universalChoiceSet[:, -1].argsort()]
            universalChoiceSet = universalChoiceSet[:, :-1]

        else:
            dc = None

        while len(universalChoiceSet) != 0:
            shipmentCount = 0

            # Loading and unloading locations of the shipments in the tour
            tourLocations = np.zeros(
                (min(nShipmentsPerCarrier[car], maxNumShips) * 2, 2),
                dtype=int)
            tourLocations[shipmentCount, 0] = universalChoiceSet[0, 1].copy()
            tourLocations[shipmentCount, 1] = universalChoiceSet[0, 2].copy()

            # First shipment in the tour is the first listed one in
            # universal choice set
            tour = np.zeros((1, 1), dtype=int)[0]
            tour[0] = universalChoiceSet[0, 0].copy()

            # Tour with only 1 shipment:
            # Sequence = go from loading to unloading
            tourSequence = np.zeros(
                min(nShipmentsPerCarrier[car], maxNumShips) * 2,
                dtype=int)
            tourSequence[shipmentCount] = (
                universalChoiceSet[0, 1].copy())
            tourSequence[shipmentCount + 1] = (
                universalChoiceSet[0, 2].copy())

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
                tourSequences[car].append([
                    x for x in tourSequence.copy() if x != 0])
                tourIsExternal[car].append(externalShipment)
                break

            # Shipments that goes to BG-zone outside the edges
            # of the NRM-zone system
            tourIsExternal[car].append(externalShipment)

            # Input for ET constraints and choice check
            tourLocs = tourLocations[0:(shipmentCount + 1)]
            tourDuration = get_tourdur(tourSequence, skim, nZones)
            prox = proximity(tourLocs, universalChoiceSet, skim, nZones)
            capUt = cap_utilization(
                shipments[tour[0], 4],
                shipments[tour, 6],
                carryingCapacity)

            etConstraintCheck = (
                (shipmentCount < maxNumShips) and
                (tourDuration < 9) and
                (capUt < 1) and
                (np.min(prox) < 100) and
                ~(is_concrete()))

            if etConstraintCheck:
                # End Tour? --> Yes or No
                etChoice = endtour_first(
                    tourDuration, capUt, tour, logitParams_ETfirst, shipments,
                    nNSTR)

                while (not etChoice) and len(universalChoiceSet) != 0:
                    tourLocs = tourLocations[0:(shipmentCount + 1)]
                    tourDuration = get_tourdur(
                        tourSequence, skim, nZones)
                    prox = proximity(
                        tourLocs, universalChoiceSet, skim, nZones)
                    capUt = cap_utilization(
                        shipments[tour[0], 4],
                        shipments[tour, 6],
                        carryingCapacity)

                    etConstraintCheck = (
                        (shipmentCount < maxNumShips) and
                        (tourDuration < 9) and
                        (capUt < 1) and
                        (np.min(prox) < 100) and
                        ~(is_concrete()))

                    if etConstraintCheck:

                        chosenShipment = selectshipment(
                            tour, tourLocs,
                            universalChoiceSet, skim,
                            nZones, carryingCapacity, shipments)

                        # If no feasible shipments
                        if np.any(chosenShipment == -2):
                            tourCount += 1
                            tours[car].append(tour.copy())
                            tourSequences[car].append([
                                x for x in tourSequence.copy() if x != 0])
                            break

                        else:
                            shipmentCount += 1

                            tour = np.append(tour, int(chosenShipment[0]))
                            tourLocations[shipmentCount, 0] = int(
                                chosenShipment[1])
                            tourLocations[shipmentCount, 1] = int(
                                chosenShipment[2])
                            tourSequence = nearest_neighbor(
                                tourLocations[0:(shipmentCount + 1)],
                                dc, skim, nZones)

                            shipmentToDelete = np.where(
                                universalChoiceSet[:, 0] == chosenShipment[0]
                            )[0][0]
                            universalChoiceSet = np.delete(
                                universalChoiceSet, shipmentToDelete, 0)

                            if len(universalChoiceSet) != 0:
                                tourLocs = tourLocations[0:(shipmentCount + 1)]
                                etChoice = endtour_later(
                                    tour, tourLocs, tourSequence,
                                    universalChoiceSet, skim, nZones,
                                    carryingCapacity, logitParams_ETlater,
                                    shipments, nNSTR)

                            else:
                                tourCount += 1
                                nTours[car] = tourCount
                                tours[car].append(tour.copy())
                                tourSequences[car].append([
                                    x for x in tourSequence.copy() if x != 0])
                                break

                    else:
                        tourCount += 1
                        tours[car].append(tour.copy())
                        tourSequences[car].append([
                            x for x in tourSequence.copy() if x != 0])
                        break

                else:
                    tourCount += 1
                    tours[car].append(tour.copy())
                    tourSequences[car].append([
                        x for x in tourSequence.copy() if x != 0])

            else:
                tourCount += 1
                tours[car].append(tour.copy())
                tourSequences[car].append([
                    x for x in tourSequence.copy() if x != 0])

    return [tours, tourSequences, nTours, tourIsExternal]


def get_cum_prob_combustion(
    varDict: dict,
    nCombType: int,
    nLS: int, nVT: int
):
    """
    Bepaal binnen elk voertuigtype en logistiek segment de cumulatieve
    kansen van de energiebronnen (uit ZEZ_SCENARIO).

    Args:
        varDict (dict): _description_
        nCombType (int): _description_
        nLS (int): _description_
        nVT (int): _description_

    Returns:
        list: Per voertuigtype een numpy.ndarray met kansen
        per voertuigtype (rij) en energiebron (kolom)
    """
    # Vehicle/combustion shares (for UCC scenario)
    sharesUCC = pd.read_csv(
        varDict['ZEZ_SCENARIO'], index_col='Segment')

    # Assume no consolidation potential and vehicle type switch
    # for dangerous goods
    sharesUCC = np.array(sharesUCC)[:-1, :-1]

    # Combustion type shares per vehicle type and
    # logistic segment
    cumProbCombustion = [
        np.zeros((nLS - 1, nCombType), dtype=float)
        for vt in range(nVT)]

    for ls in range(nLS - 1):
        # Truck_Small, Truck_Medium, Truck_Large,
        # TruckTrailer_Small, TruckTrailer_Large
        if np.all(sharesUCC[ls, 15:20] == 0):
            for vt in range(5):
                cumProbCombustion[vt][ls] = np.ones(nCombType)
        else:
            for vt in range(5):
                cumProbCombustion[vt][ls] = (
                    np.cumsum(sharesUCC[ls, 15:20]) /
                    np.sum(sharesUCC[ls, 15:20]))

        # TractorTrailer
        if np.all(sharesUCC[ls, 20:25] == 0):
            cumProbCombustion[5][ls] = np.ones(nCombType)
        else:
            cumProbCombustion[5][ls] = (
                np.cumsum(sharesUCC[ls, 20:25]) /
                np.sum(sharesUCC[ls, 20:25]))

        # SpecialVehicle
        if np.all(sharesUCC[ls, 30:35] == 0):
            cumProbCombustion[6][ls] = np.ones(nCombType)
        else:
            cumProbCombustion[6][ls] = (
                np.cumsum(sharesUCC[ls, 30:35]) /
                np.sum(sharesUCC[ls, 30:35]))

        # LZV
        if np.all(sharesUCC[ls, 20:25] == 0):
            cumProbCombustion[7][ls] = np.ones(nCombType)
        else:
            cumProbCombustion[7][ls] = (
                np.cumsum(sharesUCC[ls, 20:25]) /
                np.sum(sharesUCC[ls, 20:25]))

        # Van
        if np.all(sharesUCC[ls, 10:15] == 0):
            cumProbCombustion[8][ls] = np.ones(nCombType)
        else:
            cumProbCombustion[8][ls] = (
                np.cumsum(sharesUCC[ls, 10:15]) /
                np.sum(sharesUCC[ls, 10:15]))

        # LEVV
        if np.all(sharesUCC[ls, 0:5] == 0):
            cumProbCombustion[9][ls] = np.ones(nCombType)
        else:
            cumProbCombustion[9][ls] = (
                np.cumsum(sharesUCC[ls, 0:5]) /
                np.sum(sharesUCC[ls, 0:5]))

        # Moped
        if np.all(sharesUCC[ls, 5:10] == 0):
            cumProbCombustion[10][ls] = np.ones(nCombType)
        else:
            cumProbCombustion[10][ls] = (
                np.cumsum(sharesUCC[ls, 5:10]) /
                np.sum(sharesUCC[ls, 5:10]))
        # Waste
        if ls == 5:
            cumProbCombustion[6][ls] = (
                np.cumsum(sharesUCC[ls, 25:30]) /
                np.sum(sharesUCC[ls, 25:30]))  # SpecialVehicle

    return cumProbCombustion


def get_shipments(
    varDict: dict
):
    """
    Haal de zendingen uit module SHIP op en splits ze naar verschijningsvorm.

    Args:
        varDict (dict): _description_

    Returns:
        tuple: Met daarin:
            - pandas.DataFrame: Niet-container zendingen.
            - pandas.DataFrame: Container zendingen.
    """
    shipments = pd.read_csv(
        varDict['OUTPUTFOLDER'] + 'Shipments_' + varDict['LABEL'] + '.csv',
        sep=',')
    shipments['ORIG_LMS'] = shipments['ORIG_LMS'].astype(int)
    shipments['DEST_LMS'] = shipments['DEST_LMS'].astype(int)

    # Container shipments
    shipmentsCKM = shipments.copy()[shipments['CONTAINER'] == 1]
    shipmentsCKM.index = np.arange(len(shipmentsCKM))

    # Non-container shipments
    shipments = shipments[shipments['CONTAINER'] == 0]
    shipments.index = np.arange(len(shipments))

    return shipments, shipmentsCKM


def add_location_types_to_shipments(
    shipments: pd.DataFrame,
    zonesSted: np.ndarray,
    BGwithOverlapNRM: dict
):
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
    shipments['LOGNODE_LOADING'] = 0
    shipments.loc[(shipments['ORIG_DC'] != -9999), 'LOGNODE_LOADING'] = 2
    shipments.loc[(shipments['ORIG_TT'] != -9999), 'LOGNODE_LOADING'] = 1

    # Is the shipment UNloaded at a distribution center?
    shipments['LOGNODE_UNLOADING'] = 0
    shipments.loc[(shipments['DEST_DC'] != -9999), 'LOGNODE_UNLOADING'] = 2
    shipments.loc[(shipments['DEST_TT'] != -9999), 'LOGNODE_UNLOADING'] = 1

    # Is the loading or unloading point in an urbanized region?
    origNRM = np.array(shipments['ORIG_NRM'], dtype=int)
    destNRM = np.array(shipments['DEST_NRM'], dtype=int)
    shipments['URBAN'] = (
        (zonesSted[origNRM - 1]) | (zonesSted[destNRM - 1])).astype(int)

    # Gaat de zending naar buiten het bereik van de NRM-zonering of niet
    shipments['EXTERNAL'] = 0
    for i in shipments.index:
        isExternal = (
            (shipments.at[i, 'ORIG_BG'] not in BGwithOverlapNRM) or
            (shipments.at[i, 'DEST_BG'] not in BGwithOverlapNRM))
        if isExternal:
            shipments.at[i, 'EXTERNAL'] = 1

    return shipments


def assign_carrier_to_shipments(
    shipments: pd.DataFrame,
    varDict: dict,
    nDC: int, nCarNonDC: int,
    zezZones: np.ndarray,
    uccZones: np.ndarray,
    zezToUCC: dict
):
    """
    Wijs elke zendingen toe aan een vervoerder, voeg dit toe als extra kolom
    met header 'CARRIER' aan de zendingen.

    Args:
        varDict (dict): _description_
        shipments (pd.DataFrame): _description_
        nDC (int): _description_
        nCarNonDC (int): _description_
        zezZones (np.ndarray): _description_
        uccZones (np.ndarray): _description_
        zezToUCC (dict): _description_

    Returns:
        pandas.DataFrame: De zendingen met extra kolom CARRIER.
    """
    shipments['CARRIER'] = 0
    whereLoadDC = (shipments['ORIG_DC'] != -9999)
    whereUnloadDC = (shipments['DEST_DC'] != -9999)
    whereBothDC = (whereLoadDC) & (whereUnloadDC)
    whereNoDC = ~(whereLoadDC) & ~(whereUnloadDC)

    shipments.loc[whereLoadDC, 'CARRIER'] = shipments['ORIG_DC'][
        whereLoadDC]
    shipments.loc[whereUnloadDC, 'CARRIER'] = shipments['DEST_DC'][
        whereUnloadDC]
    shipments.loc[whereBothDC, 'CARRIER'] = [
        [shipments['ORIG_DC'][i], shipments['DEST_DC'][i]][
            np.random.randint(0, 2)]
        for i in shipments.loc[whereBothDC, :].index]
    shipments.loc[whereNoDC, 'CARRIER'] = (
        nDC + np.random.randint(0, nCarNonDC, np.sum(whereNoDC)))

    # Extra carrierIDs for shipments transported from
    # Urban Consolidation Centers
    if varDict['LABEL'] == 'UCC':
        whereToUCC = np.where(shipments['TO_UCC'] == 1)[0]
        whereFromUCC = np.where(shipments['FROM_UCC'] == 1)[0]

        for i in whereToUCC:
            origNRM = shipments.at[i, 'ORIG_NRM']

            if origNRM in zezZones:
                uccIndex = np.where(uccZones == zezToUCC[origNRM])[0][0]
                shipments.at[i, 'CARRIER'] = nDC + nCarNonDC + uccIndex

        for i in whereFromUCC:
            destNRM = shipments.at[i, 'DEST_NRM']

            if destNRM in zezZones:
                uccIndex = np.where(uccZones == zezToUCC[destNRM])[0][0]
                shipments.at[i, 'CARRIER'] = nDC + nCarNonDC + uccIndex

    return shipments


def split_shipments_with_capacity_exceedence(
    shipments: np.ndarray,
    carryingCapacity: np.ndarray
):
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
