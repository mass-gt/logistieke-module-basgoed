import numpy as np
import pandas as pd
import time
import datetime
from __functions__ import read_mtx, read_shape

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
        self.root.title("Progress Parcel Scheduling")
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

        root = args[0]
        varDict = args[1]

        if root != '':
            root.progressBar['value'] = 0

        start_time = time.time()

        log_file = open(
            varDict['OUTPUTFOLDER'] + "Logfile_ParcelDemand.log", "w")
        log_file.write(
            "Start simulation at: " +
            datetime.datetime.now().strftime("%y-%m-%d %H:%M") + "\n")

        if varDict['SEED'] != '':
            np.random.seed(varDict['SEED'])

        if root != '':
            root.progressBar['value'] = 0.1

        print('Importing data...')
        log_file.write('Importing data...\n')

        print('\tParcel demand...')

        parcels = get_parcels(varDict)

        print('\tZones...')

        zones = read_shape(varDict['ZONES'])
        zones = zones[zones['LAND'] == 1]
        zones.index = zones['SEGNR_2018']
        zonesX = np.array(zones['XCOORD'])
        zonesY = np.array(zones['YCOORD'])
        nZones = len(zones)

        print('\tParcel depots...')

        parcelNodes, coords = read_shape(
            varDict['PARCELNODES'], returnGeometry=True)
        parcelNodes['X'] = [
            coords[i]['coordinates'][0] for i in range(len(coords))]
        parcelNodes['Y'] = [
            coords[i]['coordinates'][1] for i in range(len(coords))]
        parcelNodes.index = parcelNodes['id'].astype(int)
        parcelNodes = parcelNodes.sort_index()
        parcelNodesCEP = dict(
            (parcelNodes.at[i, 'id'], parcelNodes.at[i, 'CEP'])
            for i in parcelNodes.index)
        depotIDs = list(parcelNodes['id'])

        # Change zoning to skim zones which run continuously from 0
        parcels['X'] = [zonesX[x - 1] for x in parcels['DEST_NRM'].values]
        parcels['Y'] = [zonesY[x - 1] for x in parcels['DEST_NRM'].values]

        if root != '':
            root.progressBar['value'] = 0.3

        # System input for scheduling
        parcelDepTime = np.array(pd.read_csv(
            varDict['DEPTIME_PARCELS']).iloc[:, 1])
        dropOffTime = varDict['PARCELS_DROPTIME'] / 3600

        if varDict['LABEL'] == 'UCC':
            carryingCapacity = pd.read_csv(
                varDict['VEHCAPACITY'], index_col='Vehicle Type')
            maxVehicleLoadLEVV = (
                carryingCapacity.at['LEVV', 'Tonnes'] /
                carryingCapacity.at['Van', 'Tonnes'] *
                varDict['PARCELS_MAXLOAD'])

        print('\tSkims...')

        skimTravTime = read_mtx(varDict['SKIMTIME'])
        skimDistance = read_mtx(varDict['SKIMDISTANCE'])
        nZonesSkim = int(len(skimTravTime)**0.5)

        # Alleen de zones in Nederland doen ertoe voor de pakkettenmodule
        skimTravTime = skimTravTime.reshape(nZonesSkim, nZonesSkim)
        skimTravTime = skimTravTime[:nZones, :]
        skimTravTime = skimTravTime[:, :nZones]
        skimDistance = skimDistance.reshape(nZonesSkim, nZonesSkim)
        skimDistance = skimDistance[:nZones, :]
        skimDistance = skimDistance[:, :nZones]

        # Intrazonal impedances
        for i in range(nZones):
            skimTravTime[i, i] = 0.7 * np.min(
                skimTravTime[i, skimTravTime[i, :] > 0])
        for i in range(nZones):
            skimDistance[i, i] = 0.7 * np.min(
                skimDistance[i, skimDistance[i, :] > 0])

        # Weer platslaan naar lange array
        skimTravTime = skimTravTime.flatten()
        skimDistance = skimDistance.flatten()

        if root != '':
            root.progressBar['value'] = 1.0

        print('Forming spatial clusters of parcels...')
        log_file.write('Forming spatial clusters of parcels...\n')

        # A measure of euclidean distance based on the coordinates
        skimEuclidean = 0.5 * ((
            (zonesX.repeat(nZones).reshape(nZones, nZones) -
             zonesX.repeat(nZones).reshape(nZones, nZones).transpose()) /
            1000)**2 + (
            (zonesY.repeat(nZones).reshape(nZones, nZones) -
             zonesY.repeat(nZones).reshape(nZones, nZones).transpose()) /
            1000)**2).flatten()
        skimEuclidean /= np.sum(skimEuclidean)

        # To prevent instability related to possible mistakes in skim,
        # use average of skim and euclidean distance
        # (both normalized to a sum of 1)
        skimClustering = skimDistance.copy()
        skimClustering /= np.sum(skimClustering)
        skimClustering += skimEuclidean

        del skimEuclidean

        if varDict['LABEL'] == 'UCC':

            # Divide parcels into the 4 tour types, namely:
            # 0: Depots to households
            # 1: Depots to UCCs
            # 2: From UCCs, by van
            # 3: From UCCs, by LEVV
            parcelsUCC = {}
            parcelsUCC[0] = pd.DataFrame(parcels[
                (parcels['FROM_UCC'] == 0) & (parcels['TO_UCC'] == 0)])
            parcelsUCC[1] = pd.DataFrame(parcels[
                (parcels['FROM_UCC'] == 0) & (parcels['TO_UCC'] == 1)])
            parcelsUCC[2] = pd.DataFrame(parcels[
                (parcels['FROM_UCC'] == 1) & (parcels['VEHTYPE'] == 8)])
            parcelsUCC[3] = pd.DataFrame(parcels[
                (parcels['FROM_UCC'] == 1) & (parcels['VEHTYPE'] >= 9)])

            # Cluster parcels based on proximity and
            # constrained by vehicle capacity
            for i in range(3):
                startValueProgress = 2.0 + i / 3 * (55.0 - 2.0)
                endValueProgress = 2.0 + (i + 1) / 3 * (55.0 - 2.0)

                print('\tTour type ' + str(i + 1) + '...')
                log_file.write('\tTour type ' + str(i + 1) + '...\n')

                parcelsUCC[i] = cluster_parcels(
                    parcelsUCC[i],
                    int(varDict['PARCELS_MAXLOAD']),
                    skimClustering,
                    root, startValueProgress, endValueProgress)

            # LEVV have smaller capacity
            startValueProgress = 55.0
            startValueProgress = 60.0

            print('\tTour type 4...')
            log_file.write('\tTour type 4...\n')

            parcelsUCC[3] = cluster_parcels(
                parcelsUCC[3],
                maxVehicleLoadLEVV,
                skimClustering,
                root, startValueProgress, endValueProgress)

            # Aggregate parcels based on depot, cluster and destination
            for i in range(4):
                if i <= 1:
                    parcelsUCC[i] = pd.pivot_table(
                        parcelsUCC[i],
                        values=['PARCEL_ID'],
                        index=['DEPOT_ID', 'Cluster', 'ORIG_NRM', 'DEST_NRM'],
                        aggfunc={'PARCEL_ID': 'count'})
                    parcelsUCC[i] = parcelsUCC[i].rename(
                        columns={'PARCEL_ID': 'N_PARCELS'})
                    parcelsUCC[i]['Depot'] = [
                        x[0] for x in parcelsUCC[i].index]
                    parcelsUCC[i]['Cluster'] = [
                        x[1] for x in parcelsUCC[i].index]
                    parcelsUCC[i]['Orig'] = [
                        x[2] for x in parcelsUCC[i].index]
                    parcelsUCC[i]['Dest'] = [
                        x[3] for x in parcelsUCC[i].index]

                else:
                    parcelsUCC[i] = pd.pivot_table(
                        parcelsUCC[i],
                        values=['PARCEL_ID'],
                        index=['ORIG_NRM', 'Cluster', 'DEST_NRM'],
                        aggfunc={'PARCEL_ID': 'count'})
                    parcelsUCC[i] = parcelsUCC[i].rename(
                        columns={'PARCEL_ID': 'N_PARCELS'})
                    parcelsUCC[i]['Depot'] = [
                        x[0] for x in parcelsUCC[i].index]
                    parcelsUCC[i]['Cluster'] = [
                        x[1] for x in parcelsUCC[i].index]
                    parcelsUCC[i]['Orig'] = [
                        x[0] for x in parcelsUCC[i].index]
                    parcelsUCC[i]['Dest'] = [
                        x[2] for x in parcelsUCC[i].index]
                parcelsUCC[i].index = np.arange(len(parcelsUCC[i]))

        if varDict['LABEL'] != 'UCC':
            # Cluster parcels based on proximity and
            # constrained by vehicle capacity
            startValueProgress = 2.0
            endValueProgress = 60.0
            parcels = cluster_parcels(
                parcels,
                int(varDict['PARCELS_MAXLOAD']),
                skimClustering,
                root, startValueProgress, endValueProgress)

            # Aggregate parcels based on depot, cluster and destination
            parcels = pd.pivot_table(
                parcels,
                values=['PARCEL_ID'],
                index=['DEPOT_ID', 'Cluster', 'ORIG_NRM', 'DEST_NRM'],
                aggfunc={'PARCEL_ID': 'count'})
            parcels = parcels.rename(columns={'PARCEL_ID': 'N_PARCELS'})
            parcels['Depot'] = [x[0] for x in parcels.index]
            parcels['Cluster'] = [x[1] for x in parcels.index]
            parcels['Orig'] = [x[2] for x in parcels.index]
            parcels['Dest'] = [x[3] for x in parcels.index]
            parcels.index = np.arange(len(parcels))

        del skimClustering

        # ----------- Scheduling of trips (UCC scenario) ---------------------

        if varDict['LABEL'] == 'UCC':

            # Depots to households
            message = (
                'Starting scheduling procedure for parcels ' +
                'from depots to households...')
            print(message)
            log_file.write(message + '\n')

            startValueProgress = 60.0
            endValueProgress = 80.0
            tourType = 0
            deliveries = create_schedules(
                parcelsUCC[0],
                dropOffTime,
                skimTravTime, skimDistance,
                parcelNodesCEP,
                parcelDepTime,
                tourType,
                root, startValueProgress, endValueProgress)

            # Depots to UCCs
            message = (
                'Starting scheduling procedure for parcels ' +
                'from depots to UCC...')
            print(message)
            log_file.write(message + '\n')

            startValueProgress = 80.0
            endValueProgress = 83.0
            tourType = 1
            deliveries1 = create_schedules(
                parcelsUCC[1],
                dropOffTime,
                skimTravTime, skimDistance,
                parcelNodesCEP,
                parcelDepTime,
                tourType,
                root, startValueProgress, endValueProgress)

            # Depots to UCCs (van)
            message = (
                'Starting scheduling procedure for parcels ' +
                'from UCCs (by van)...')
            print(message)
            log_file.write(message + '\n')

            startValueProgress = 83.0
            endValueProgress = 86.0
            tourType = 2
            deliveries2 = create_schedules(
                parcelsUCC[2],
                dropOffTime,
                skimTravTime, skimDistance,
                parcelNodesCEP,
                parcelDepTime,
                tourType,
                root, startValueProgress, endValueProgress)

            # Depots to UCCs (LEVV)
            message = (
                'Starting scheduling procedure for parcels ' +
                'from UCCs (by LEVV)...')
            print(message)
            log_file.write(message + '\n')

            startValueProgress = 86.0
            endValueProgress = 89.0
            tourType = 3
            deliveries3 = create_schedules(
                parcelsUCC[3],
                dropOffTime,
                skimTravTime, skimDistance,
                parcelNodesCEP,
                parcelDepTime,
                tourType,
                root, startValueProgress, endValueProgress)

            # Combine deliveries of all tour types
            deliveries = pd.concat([
                deliveries, deliveries1, deliveries2, deliveries3])
            deliveries.index = np.arange(len(deliveries))

        # ----------- Scheduling of trips (REF scenario) ----------------------

        if varDict['LABEL'] != 'UCC':
            print('Starting scheduling procedure for parcels...')
            log_file.write('Starting scheduling procedure for parcels...\n')

            startValueProgress = 60.0
            endValueProgress = 90.0
            tourType = 0

            deliveries = create_schedules(
                parcels,
                dropOffTime,
                skimTravTime, skimDistance,
                parcelNodesCEP, parcelDepTime,
                tourType,
                root, startValueProgress, endValueProgress)

        # ------------------ Export output table to CSV and SHP ---------------

        deliveries['TOUR_DEPTIME'] = [
            round(depTime, 3) for depTime in deliveries['TOUR_DEPTIME'].values]
        deliveries['TRIP_DEPTIME'] = [
            round(depTime, 3) for depTime in deliveries['TRIP_DEPTIME'].values]

        outFileName = (
            varDict['OUTPUTFOLDER'] +
            'ParcelSchedule_' +
            varDict['LABEL'] +
            '.csv')

        print(f"Writing scheduled trips to {outFileName}")
        log_file.write(f"Writing scheduled trips to {outFileName}\n")

        deliveries.to_csv(outFileName, index=False)
        nTrips = len(deliveries)

        if root != '':
            root.progressBar['value'] = 91.0

        print('Writing GeoJSON...')
        log_file.write('Writing GeoJSON...\n')

        # Initialize arrays with coordinates
        Ax = np.zeros(nTrips, dtype=int)
        Ay = np.zeros(nTrips, dtype=int)
        Bx = np.zeros(nTrips, dtype=int)
        By = np.zeros(nTrips, dtype=int)

        # Determine coordinates of LineString for each trip
        tripIDs = [x.split('_')[-1] for x in deliveries['TRIP_ID']]
        tourTypes = np.array(deliveries['TOUR_TYPE'], dtype=int)
        depotIDs = np.array(deliveries['DEPOT_ID'])

        for i in deliveries.index[:-1]:
            # First trip of tour
            if tripIDs[i] == '0' and tourTypes[i] <= 1:
                Ax[i] = parcelNodes['X'][depotIDs[i]]
                Ay[i] = parcelNodes['Y'][depotIDs[i]]
                Bx[i] = zonesX[deliveries['DEST_NRM'][i] - 1]
                By[i] = zonesY[deliveries['DEST_NRM'][i] - 1]
            # Last trip of tour
            elif tripIDs[i + 1] == '0' and tourTypes[i] <= 1:
                Ax[i] = zonesX[deliveries['ORIG_NRM'][i] - 1]
                Ay[i] = zonesY[deliveries['ORIG_NRM'][i] - 1]
                Bx[i] = parcelNodes['X'][depotIDs[i]]
                By[i] = parcelNodes['Y'][depotIDs[i]]
            # Intermediate trips of tour
            else:
                Ax[i] = zonesX[deliveries['ORIG_NRM'][i] - 1]
                Ay[i] = zonesY[deliveries['ORIG_NRM'][i] - 1]
                Bx[i] = zonesX[deliveries['DEST_NRM'][i] - 1]
                By[i] = zonesY[deliveries['DEST_NRM'][i] - 1]

        # Last trip of last tour
        i += 1
        if tourTypes[i] <= 1:
            Ax[i] = zonesX[deliveries['ORIG_NRM'][i] - 1]
            Ay[i] = zonesY[deliveries['ORIG_NRM'][i] - 1]
            Bx[i] = parcelNodes['X'][depotIDs[i]]
            By[i] = parcelNodes['Y'][depotIDs[i]]
        else:
            Ax[i] = zonesX[deliveries['ORIG_NRM'][i] - 1]
            Ay[i] = zonesY[deliveries['ORIG_NRM'][i] - 1]
            Bx[i] = zonesX[deliveries['DEST_NRM'][i] - 1]
            By[i] = zonesY[deliveries['DEST_NRM'][i] - 1]

        Ax = np.array(Ax, dtype=str)
        Ay = np.array(Ay, dtype=str)
        Bx = np.array(Bx, dtype=str)
        By = np.array(By, dtype=str)

        outFileName = (
            varDict['OUTPUTFOLDER'] +
            'ParcelSchedule_' +
            varDict['LABEL'] +
            '.geojson')

        with open(outFileName, 'w') as geoFile:
            geoFile.write(
                '{\n' + '"type": "FeatureCollection",\n' + '"features": [\n')

            for i in range(nTrips - 1):
                outputStr = (
                    '{ "type": "Feature", "properties": ' +
                    str(deliveries.loc[i, :].to_dict()).replace("'", '"') +
                    ', "geometry": { "type": "LineString", ' +
                    '"coordinates": [ [ ' +
                    Ax[i] + ', ' + Ay[i] + ' ], [ ' +
                    Bx[i] + ', ' + By[i] + ' ] ] } },\n')
                geoFile.write(outputStr)

                if i % int(nTrips / 10) == 0:
                    print(
                        '\t' + str(int(round((i / nTrips) * 100, 0))) + '%',
                        end='\r')
                    if root != '':
                        root.progressBar['value'] = (
                            91.0 + (98.0 - 91.0) * (i / nTrips))

            # Bij de laatste feature moet er geen komma aan het einde
            i += 1
            outputStr = (
                '{ "type": "Feature", "properties": ' +
                str(deliveries.loc[i, :].to_dict()).replace("'", '"') +
                ', "geometry": { "type": "LineString", ' +
                '"coordinates": [ [ ' +
                Ax[i] + ', ' + Ay[i] + ' ], [ ' +
                Bx[i] + ', ' + By[i] + ' ] ] } }\n')
            geoFile.write(outputStr)
            geoFile.write(']\n')
            geoFile.write('}')

        print(f'Parcel schedules written to {outFileName}')
        log_file.write(f'Parcel schedules written to {outFileName}\n')

        # ------------------------ Create and export trip matrices ------------

        print('Generating trip matrix...')
        log_file.write('Generating trip matrix...\n')

        cols = ['ORIG', 'DEST', 'N_TOT']
        deliveries['N_TOT'] = 1

        # Gebruik N_TOT om het aantal ritten per HB te bepalen,
        # voor elk logistiek segment, voertuigtype en totaal
        pivotTable = pd.pivot_table(
            deliveries,
            values=['N_TOT'],
            index=['ORIG_NRM', 'DEST_NRM'],
            aggfunc=np.sum)
        pivotTable['ORIG'] = [x[0] for x in pivotTable.index]
        pivotTable['DEST'] = [x[1] for x in pivotTable.index]
        pivotTable = pivotTable[cols]

        # Assume one intrazonal trip for each zone with
        # multiple deliveries visited in a tour
        intrazonalTrips = {}
        for i in deliveries[deliveries['N_PARCELS'] > 1].index:
            zone = deliveries.at[i, 'DEST_NRM']
            if zone in intrazonalTrips.keys():
                intrazonalTrips[zone] += 1
            else:
                intrazonalTrips[zone] = 1

        intrazonalKeys = list(intrazonalTrips.keys())
        for zone in intrazonalKeys:
            if (zone, zone) in pivotTable.index:
                pivotTable.at[(zone, zone), 'N_TOT'] += intrazonalTrips[zone]
                del intrazonalTrips[zone]

        intrazonalTripsDF = pd.DataFrame(
            np.zeros((len(intrazonalTrips), 3)), columns=cols)
        intrazonalTripsDF['ORIG'] = intrazonalTrips.keys()
        intrazonalTripsDF['DEST'] = intrazonalTrips.keys()
        intrazonalTripsDF['N_TOT'] = intrazonalTrips.values()
        pivotTable = pivotTable.append(intrazonalTripsDF)
        pivotTable = pivotTable.sort_values(['ORIG', 'DEST'])

        outFileName = (
            varDict['OUTPUTFOLDER'] +
            'TripMatrix_ParcelVans_NRM_' +
            varDict['LABEL'] +
            '.csv')

        pivotTable.to_csv(outFileName, index=False, sep=',')

        print(f'Trip matrix written to {outFileName}')
        log_file.write(f'Trip matrix written to {outFileName}\n')

#        deliveries.loc[deliveries['TripDepTime']>=24,'TripDepTime'] -= 24
#        deliveries.loc[deliveries['TripDepTime']>=24,'TripDepTime'] -= 24

        # --------------------------- End of module ---------------------------

        totaltime = round(time.time() - start_time, 2)
        log_file.write(
            "Total runtime: %s seconds\n" % (totaltime))
        log_file.write(
            "End simulation at: " +
            datetime.datetime.now().strftime("%y-%m-%d %H:%M") + "\n")
        log_file.close()

        if root != '':
            root.update_statusbar("Parcel Scheduling: Done")
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
                    "Parcel Scheduling: Execution failed!")
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


def get_parcels(
    varDict: dict
):
    """
    Lees de de CSV met de pakketvraag in en herstructureer deze zodat
    iedere rij 1 pakket is.

    Args:
        varDict (dict): _description_

    Returns:
        _type_: _description_
    """
    # Inlezen CSV met pakketvraag
    parcelsAgg = pd.read_csv(
        (
            varDict['OUTPUTFOLDER'] +
            'ParcelDemand_' +
            varDict['LABEL'] +
            '.csv'
        ),
        sep=',')
    nParcelsTotal = np.sum(parcelsAgg['N_PARCELS'])

    # Parcels herstructureren zodat iedere rij 1 pakket is
    parcelColumns = [
        'PARCEL_ID',
        'ORIG_NRM', 'DEST_NRM',
        'DEPOT_ID', 'CEP',
        'VEHTYPE']
    if varDict['LABEL'] == 'UCC':
        parcelColumns.append('FROM_UCC')
        parcelColumns.append('TO_UCC')
    parcels = np.zeros((nParcelsTotal, len(parcelColumns)), dtype=object)
    count = 0
    for i in range(len(parcelsAgg)):
        orig = parcelsAgg.at[i, 'ORIG_NRM']
        dest = parcelsAgg.at[i, 'DEST_NRM']
        nParcels = parcelsAgg.at[i, 'N_PARCELS']
        depotNo = parcelsAgg.at[i, 'DEPOT_ID']
        cep = parcelsAgg.at[i, 'CEP']
        vehType = parcelsAgg.at[i, 'VEHTYPE']

        parcels[count:(count + nParcels), 1] = orig
        parcels[count:(count + nParcels), 2] = dest
        parcels[count:(count + nParcels), 3] = depotNo
        parcels[count:(count + nParcels), 4] = cep
        parcels[count:(count + nParcels), 5] = vehType

        if varDict['LABEL'] == 'UCC':
            fromUCC = parcelsAgg.at[i, 'FROM_UCC']
            toUCC = parcelsAgg.at[i, 'TO_UCC']

            parcels[count:(count + nParcels), 6] = fromUCC
            parcels[count:(count + nParcels), 7] = toUCC

        count += nParcels

    # Stop in een DataFrame met de juiste kolomnamen en data types
    parcels = pd.DataFrame(parcels, columns=parcelColumns)
    parcels['PARCEL_ID'] = np.arange(nParcelsTotal)
    parcels = parcels.astype({
        'PARCEL_ID': int,
        'ORIG_NRM': int,
        'DEST_NRM': int,
        'DEPOT_ID': int,
        'CEP': str,
        'VEHTYPE': int})

    return parcels


def create_schedules(
    parcelsAgg: pd.DataFrame,
    dropOffTime: int,
    skimTravTime: np.ndarray, skimDistance: np.ndarray,
    parcelNodesCEP: dict,
    parcelDepTime: np.ndarray,
    tourType: int,
    root='', startValueProgress=0.0, endValueProgress=100.0
):
    """
    Vorm de rondritten en bewaar de informatie in een DataFrame.

    Args:
        parcelsAgg (pd.DataFrame): _description_
        dropOffTime (int): _description_
        skimTravTime (np.ndarray): _description_
        skimDistance (np.ndarray): _description_
        parcelNodesCEP (dict): _description_
        parcelDepTime (np.ndarray): _description_
        tourType (int): _description_
        root (str, optional): _description_. Defaults to ''.
        startValueProgress (float, optional): _description_. Defaults to 0.0.
        endValueProgress (float, optional): _description_. Defaults to 100.0.

    Returns:
        pd.DataFrame: De rondritten.
    """

    nZones = int(len(skimTravTime)**0.5)
    depots = np.unique(parcelsAgg['Depot'])
    nDepots = len(depots)

    print('\t0%', end='\r')

    tours = {}
    parcelsDelivered = {}
    departureTimes = {}
    depotCount = 0
    nTrips = 0

    for depot in np.unique(parcelsAgg['Depot']):
        depotParcels = parcelsAgg[parcelsAgg['Depot'] == depot]

        tours[depot] = {}
        parcelsDelivered[depot] = {}
        departureTimes[depot] = {}

        for cluster in np.unique(depotParcels['Cluster']):
            tour = []

            clusterParcels = depotParcels[depotParcels['Cluster'] == cluster]
            depotZone = list(clusterParcels['Orig'])[0]
            destZones = list(clusterParcels['Dest'])
            nParcelsPerZone = dict(zip(destZones, clusterParcels['N_PARCELS']))

            # Nearest neighbor
            tour.append(depotZone)
            for i in range(len(destZones)):
                distances = [
                    skimDistance[(tour[i] - 1) * nZones + (dest - 1)]
                    for dest in destZones]
                nextIndex = np.argmin(distances)
                tour.append(destZones[nextIndex])
                destZones.pop(nextIndex)
            tour.append(depotZone)

            # Shuffle the order of tour locations and accept the shuffle
            # if it reduces the tour distance
            nStops = len(tour)
            tour = np.array(tour, dtype=int)
            tourDist = np.sum(
                skimDistance[(tour[:-1] - 1) * nZones + (tour[1:] - 1)])
            if nStops > 4:
                for shiftLocA in range(1, nStops - 1):
                    for shiftLocB in range(1, nStops - 1):
                        if shiftLocA != shiftLocB:
                            swappedTour = tour.copy()
                            swappedTour[shiftLocA] = tour[shiftLocB]
                            swappedTour[shiftLocB] = tour[shiftLocA]
                            swappedTourDist = np.sum(
                                skimDistance[
                                    (swappedTour[:-1] - 1) * nZones +
                                    (swappedTour[1:] - 1)])

                            if swappedTourDist < tourDist:
                                tour = swappedTour.copy()
                                tourDist = swappedTourDist

            # Add current tour to dictionary with all formed tours
            tours[depot][cluster] = list(tour.copy())

            # Store the number of parcels delivered at each
            # location in the tour
            nParcelsPerStop = []
            for i in range(1, nStops - 1):
                nParcelsPerStop.append(nParcelsPerZone[tour[i]])
            nParcelsPerStop.append(0)
            parcelsDelivered[depot][cluster] = list(nParcelsPerStop.copy())

            # Determine the departure time of each trip in the tour
            departureTimesTour = [
                np.where(parcelDepTime > np.random.rand())[0][0] +
                np.random.rand()]
            for i in range(1, nStops - 1):
                orig = tour[i - 1]
                dest = tour[i]
                travTime = skimTravTime[
                    (orig - 1) * nZones + (dest - 1)] / 3600
                departureTimesTour.append(
                    departureTimesTour[i - 1] +
                    dropOffTime * nParcelsPerStop[i - 1] +
                    travTime)
            departureTimes[depot][cluster] = list(departureTimesTour.copy())

            nTrips += (nStops - 1)

        print(
            '\t' + str(round((depotCount + 1) / nDepots * 100, 1)) + '%',
            end='\r')

        if root != '':
            root.progressBar['value'] = (
                startValueProgress +
                (endValueProgress - startValueProgress - 1) *
                (depotCount + 1) / nDepots)

        depotCount += 1

    deliveriesCols = [
        'TOUR_TYPE', 'CEP', 'DEPOT_ID', 'TOUR_ID',
        'TRIP_ID', 'UNIQUE_ID', 'ORIG_NRM', 'DEST_NRM',
        'N_PARCELS', 'TRAVTIME', 'TOUR_DEPTIME', 'TRIP_DEPTIME']
    deliveries = np.zeros((nTrips, len(deliveriesCols)), dtype=object)

    tripcount = 0
    for depot in tours.keys():
        for tour in tours[depot].keys():
            for trip in range(len(tours[depot][tour]) - 1):
                orig = tours[depot][tour][trip]
                dest = tours[depot][tour][trip + 1]

                # Depot to HH (0) or UCC (1), UCC to HH by van (2)/LEVV (3)
                deliveries[tripcount, 0] = tourType

                # Name of the couriers
                if tourType <= 1:
                    deliveries[tripcount, 1] = parcelNodesCEP[depot]
                else:
                    deliveries[tripcount, 1] = 'ConsolidatedUCC'

                # Depot_ID, Tour_ID, Trip_ID,
                # Unique ID under consideration of tour type
                deliveries[tripcount, 2] = depot
                deliveries[tripcount, 3] = f'{depot}_{tour}'
                deliveries[tripcount, 4] = f'{depot}_{tour}_{trip}'
                deliveries[tripcount, 5] = f'{depot}_{tour}_{trip}_{tourType}'

                # Origin and destination zone
                deliveries[tripcount, 6] = orig
                deliveries[tripcount, 7] = dest

                # Number of parcels
                deliveries[tripcount, 8] = parcelsDelivered[depot][tour][trip]

                # Travel time in hrs
                deliveries[tripcount, 9] = skimTravTime[
                    (orig - 1) * nZones + (dest - 1)] / 3600

                # Departure of tour and trip
                deliveries[tripcount, 10] = departureTimes[depot][tour][0]
                deliveries[tripcount, 11] = departureTimes[depot][tour][trip]

                tripcount += 1

    # Place in DataFrame with the right data type per column
    deliveries = pd.DataFrame(deliveries, columns=deliveriesCols)
    dtypes = {
        'TOUR_TYPE': int, 'CEP': str,
        'DEPOT_ID': int, 'TOUR_ID': str,
        'TRIP_ID': str, 'UNIQUE_ID': str,
        'ORIG_NRM': int, 'DEST_NRM': int,
        'N_PARCELS': float, 'TRAVTIME': float,
        'TOUR_DEPTIME': float, 'TRIP_DEPTIME': float}
    for col in deliveriesCols:
        deliveries[col] = deliveries[col].astype(dtypes[col])

    vehTypes = ['Van', 'Van', 'Van', 'LEVV']
    origTypes = ['Depot', 'Depot', 'UCC', 'UCC']
    destTypes = ['HH', 'UCC', 'HH', 'HH']

    deliveries['VEHTYPE'] = vehTypes[tourType]
    deliveries['ORIGTYPE'] = origTypes[tourType]
    deliveries['DESTTYPE'] = destTypes[tourType]

    if root != '':
        root.progressBar['value'] = endValueProgress

    return deliveries


def cluster_parcels(
    parcels: pd.DataFrame,
    maxVehicleLoad: int,
    skimDistance: np.ndarray,
    root='', startValueProgress=0.0, endValueProgress=100.0
):
    '''
    Assign parcels to clusters based on spatial proximity
    with cluster size constraints.
    The cluster variable is added as extra column to the DataFrame.
    '''
    depotNumbers = np.unique(parcels['DEPOT_ID'])
    nParcels = len(parcels)
    nParcelsAssigned = 0
    firstClusterID = 0
    nZones = int(len(skimDistance)**0.5)

    parcels.index = np.arange(nParcels)

    parcelsCluster = - np.ones(nParcels)

    print('\t0%', end='\r')

    # First check for depot/destination combination with more than
    # 'maxVehicleLoad' parcels.
    # For these we don't need to use the clustering algorithm.
    counts = pd.pivot_table(
        parcels,
        values=['VEHTYPE'],
        index=['DEPOT_ID', 'DEST_NRM'],
        aggfunc=len)
    whereLargeCluster = list(
        counts.index[np.where(counts >= maxVehicleLoad)[0]])

    whereDepotDest = {}
    parcelsDepotID = np.array(parcels['DEPOT_ID'], dtype=int)
    parcelsDestZone = np.array(parcels['DEST_NRM'], dtype=int)
    for i in range(nParcels):
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
            nParcelsAssigned += maxVehicleLoad

            progress = nParcelsAssigned / nParcels
            print(
                '\t' + str(round(progress * 100, 1)) + '%',
                end='\r')

            if root != '':
                root.progressBar['value'] = (
                    startValueProgress +
                    (endValueProgress - startValueProgress - 1) * progress)

    parcels['Cluster'] = parcelsCluster

    # For each depot, cluster remaining parcels into batches of
    # 'maxVehicleLoad' parcels
    for depotNumber in depotNumbers:
        # Select parcels of the depot that are not assigned a cluster yet
        parcelsToFit = parcels[
            (parcels['DEPOT_ID'] == depotNumber) &
            (parcels['Cluster'] == -1)].copy()

        # Sort parcels descending based on distance to depot
        # so that at the end of the loop the remaining parcels
        # are all nearby the depot and form a somewhat reasonable
        # parcels cluster
        parcelsToFit['Distance'] = skimDistance[
            (parcelsToFit['ORIG_NRM'] - 1) * nZones +
            (parcelsToFit['DEST_NRM'] - 1)]
        parcelsToFit = parcelsToFit.sort_values('Distance', ascending=False)
        parcelsToFitIndex = list(parcelsToFit.index)
        parcelsToFit.index = np.arange(len(parcelsToFit))
        dests = np.array(parcelsToFit['DEST_NRM'])

        # How many tours are needed to deliver these parcels
        nTours = int(np.ceil(len(parcelsToFit) / maxVehicleLoad))

        # In the case of 1 tour it's simple, all parcels belong to the
        # same cluster
        if nTours == 1:
            parcels.loc[parcelsToFitIndex, 'Cluster'] = firstClusterID
            firstClusterID += 1
            nParcelsAssigned += len(parcelsToFit)

        # When there are multiple tours needed, the heuristic is
        # a little bit more complex
        else:
            clusters = np.ones(len(parcelsToFit), dtype=int) * -1

            for tour in range(nTours):
                # Select the first parcel for the new cluster that
                # is now initialized
                yetAssigned = (clusters != -1)
                notYetAssigned = np.where(~yetAssigned)[0]
                firstParcelIndex = notYetAssigned[0]
                clusters[firstParcelIndex] = firstClusterID

                # Find the nearest {maxVehicleLoad-1} parcels to
                # this first parcel that are not in a cluster yet
                distances = skimDistance[
                    (dests[firstParcelIndex] - 1) * nZones + (dests - 1)]
                distances[notYetAssigned[0]] = 99999
                distances[yetAssigned] = 99999
                whereFirstClusterID = (
                    np.argsort(distances)[:(maxVehicleLoad - 1)])
                clusters[whereFirstClusterID] = firstClusterID

                firstClusterID += 1

            # Group together remaining parcels, these are all nearby the depot
            yetAssigned = (clusters != -1)
            notYetAssigned = np.where(~yetAssigned)[0]
            clusters[notYetAssigned] = firstClusterID
            firstClusterID += 1

            parcels.loc[parcelsToFitIndex, 'Cluster'] = clusters
            nParcelsAssigned += len(parcelsToFit)

            progress = nParcelsAssigned / nParcels
            print(
                '\t' + str(round(progress * 100, 1)) + '%',
                end='\r')

            if root != '':
                root.progressBar['value'] = (
                    startValueProgress +
                    (endValueProgress - startValueProgress - 1) * progress)

    parcels['Cluster'] = parcels['Cluster'].astype(int)

    return parcels
