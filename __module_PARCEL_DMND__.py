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
        self.root.title("Progress Parcel Demand")
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
            varDict['OUTPUTFOLDER'] + "Logfile_ParcelDemand.log", "w")
        log_file.write(
            "Start simulation at: " +
            datetime.datetime.now().strftime("%y-%m-%d %H:%M") + "\n")

        outFilename = (
            varDict['OUTPUTFOLDER'] +
            'ParcelDemand_' +
            varDict['LABEL'] +
            '.csv')

        if varDict['SEED'] != '':
            np.random.seed(varDict['SEED'])

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

        print('Importing data...')
        log_file.write('Importing data...\n')

        print('\tZones...')

        zones = read_shape(varDict['ZONES'])
        zones = zones[zones['LAND'] == 1]
        zones.index = zones['SEGNR_2018']
        zonesX = np.array(zones['XCOORD'])
        zonesY = np.array(zones['YCOORD'])

        print('\tParcel depots...')

        parcelNodes, coords = read_shape(
            varDict['PARCELNODES'], returnGeometry=True)
        parcelNodes['X'] = [
            coords[i]['coordinates'][0] for i in range(len(coords))]
        parcelNodes['Y'] = [
            coords[i]['coordinates'][1] for i in range(len(coords))]
        parcelNodes.index = parcelNodes['id'].astype(int)
        parcelNodes = parcelNodes.sort_index()
        nParcelNodes = len(parcelNodes)

        cepShares = pd.read_csv(varDict['CEP_SHARES'], index_col=0)
        cepList = np.unique(parcelNodes['CEP'])
        cepNodes = [
            np.where(parcelNodes['CEP'] == str(cep))[0]
            for cep in cepList]
        cepNodeDict = {}
        for cepNo in range(len(cepList)):
            cepNodeDict[cepList[cepNo]] = cepNodes[cepNo]

        print('\tSkim...')

        skimDistance = read_mtx(varDict['SKIMDISTANCE'])
        nZones = int(len(skimDistance)**0.5)
        parcelSkim = np.zeros((nZones, nParcelNodes))

        # Skim with travel times between parcel nodes and all other zones
        i = 0
        for orig in parcelNodes['SEGNR_2018']:
            dest = 1 + np.arange(nZones)
            parcelSkim[:, i] = np.round(
                skimDistance[(orig - 1) * nZones + (dest - 1)] / 1000, 4)
            i += 1

        print('Generating parcels...')
        log_file.write('Generating parcels...\n')

        # Calculate number of parcels per zone based on number of households
        # and total number of parcels on an average day
        nParcelsPerZone = (
            (zones['HUISH' + yrAppendix] *
             varDict['PARCELS_PER_HH'] / varDict['PARCELS_SUCCESS_B2C']) +
            (zones['BANENTOT' + yrAppendix] *
             varDict['PARCELS_PER_EMPL'] / varDict['PARCELS_SUCCESS_B2B']))
        nParcelsPerZone = np.array(np.round(nParcelsPerZone), dtype=int)

        # Spread over couriers based on market shares
        zones['parcels'] = 0
        for cep in cepList:
            zones['parcels_' + str(cep)] = np.round(
                cepShares['ShareTotal'][cep] * nParcelsPerZone)
            zones['parcels_' + str(cep)] = (
                zones['parcels_' + str(cep)].astype(int))
            zones['parcels'] += zones['parcels_' + str(cep)]

        # Total number of parcels per courier
        nParcels = int(zones['parcels'].sum())

        # Maak een DataFrame met parcels per HB en depot
        parcels = generate_parcels(
            nParcels,
            zones,
            cepNodeDict, cepList,
            parcelSkim, parcelNodes)

        if varDict['LABEL'] == 'UCC':

            vtNamesUCC = [
                'LEVV', 'Moped',
                'Van',
                'Truck',
                'TractorTrailer',
                'WasteCollection',
                'SpecialConstruction']
            nLS = 8

            # Logistic segment is 6: parcels
            ls = 6

            # Write the REF parcel demand
            print(f"Writing parcels to {outFilename}")
            log_file.write(f"Writing parcels to {outFilename}\n")
            parcels.to_csv(outFilename, index=False)

            # Consolidation potential per logistic segment (for UCC scenario)
            probConsolidation = np.array(pd.read_csv(
                varDict['ZEZ_CONSOLIDATION'], index_col='Segment'))

            # Vehicle/combustion shares (for UCC scenario)
            sharesUCC = pd.read_csv(
                varDict['ZEZ_SCENARIO'], index_col='Segment')

            # Assume no consolidation potential and vehicle type switch
            # for dangerous goods
            sharesUCC = np.array(sharesUCC)[:-1, :-1]

            # Only vehicle shares (summed up combustion types)
            sharesVehUCC = np.zeros((nLS - 1, len(vtNamesUCC)))
            for ls in range(nLS - 1):
                sharesVehUCC[ls, 0] = np.sum(sharesUCC[ls, 0:5])
                sharesVehUCC[ls, 1] = np.sum(sharesUCC[ls, 5:10])
                sharesVehUCC[ls, 2] = np.sum(sharesUCC[ls, 10:15])
                sharesVehUCC[ls, 3] = np.sum(sharesUCC[ls, 15:20])
                sharesVehUCC[ls, 4] = np.sum(sharesUCC[ls, 20:25])
                sharesVehUCC[ls, 5] = np.sum(sharesUCC[ls, 25:30])
                sharesVehUCC[ls, 6] = np.sum(sharesUCC[ls, 30:35])
                sharesVehUCC[ls, :] = (
                    np.cumsum(sharesVehUCC[ls, :]) /
                    np.sum(sharesVehUCC[ls, :]))

            # Couple these vehicle types to Harmony vehicle types
            vehUccToVeh = {
                0: 9,
                1: 10,
                2: 8,
                3: 1,
                4: 5,
                5: 6,
                6: 6}

            print('Redirecting parcels via UCC...')
            log_file.write('Redirecting parcels via UCC...\n')

            parcels['FROM_UCC'] = 0
            parcels['TO_UCC'] = 0

            parcelsDestNRM = np.array(parcels['DEST_NRM'].astype(int))
            parcelsDepotID = np.array(parcels['DEPOT_ID'].astype(int))
            parcelsCEP = [str(x) for x in parcels['CEP'].values]

            zezZones = pd.read_csv(varDict['ZEZ_ZONES'], sep=',')
            zezArray = np.zeros(nZones, dtype=int)
            uccArray = np.zeros(nZones, dtype=int)
            zezArray[np.array(zezZones['SEGNR_2018'], dtype=int) - 1] = 1
            uccArray[np.array(zezZones['SEGNR_2018'], dtype=int) - 1] = (
                zezZones['UCC'])

            isDestZEZ = (
                (zezArray[parcelsDestNRM - 1] == 1) &
                (probConsolidation[ls][0] > np.random.rand(len(parcels))))
            whereDestZEZ = np.where(isDestZEZ)[0]

            newParcels = np.zeros(parcels.shape, dtype=object)

            count = 0

            for i in whereDestZEZ:
                trueDest = parcelsDestNRM[i]

                # Redirect to UCC
                parcels.at[i, 'DEST_NRM'] = uccArray[trueDest - 1]
                parcels.at[i, 'TO_UCC'] = 1

                # Add parcel set to ZEZ from UCC
                newParcels[count, 1] = uccArray[trueDest - 1]  # Origin
                newParcels[count, 2] = trueDest                # Destination
                newParcels[count, 3] = parcelsDepotID[i]       # Depot ID
                newParcels[count, 4] = parcelsCEP[i]           # Courier name
                newParcels[count, 6] = 1    # From UCC
                newParcels[count, 7] = 0    # To UCC

                # Vehicle type
                newParcels[count, 5] = vehUccToVeh[
                    np.where(sharesVehUCC[ls, :] > np.random.rand())[0][0]]

                count += 1

            newParcels = pd.DataFrame(newParcels)
            newParcels.columns = parcels.columns
            newParcels = newParcels.iloc[np.arange(count), :]

            dtypes = {
                'PARCEL_ID': int,
                'ORIG_NRM': int,
                'DEST_NRM': int,
                'DEPOT_ID': int,
                'CEP': str,
                'VEHTYPE': int,
                'FROM_UCC': int,
                'TO_UCC': int}

            for col in dtypes.keys():
                newParcels[col] = newParcels[col].astype(dtypes[col])

            parcels = parcels.append(newParcels)
            parcels.index = np.arange(len(parcels))
            parcels['PARCEL_ID'] = np.arange(1, len(parcels) + 1)

            nParcels = len(parcels)

        print(f"Writing parcels CSV to {outFilename}.csv")
        log_file.write(f"Writing parcels to {outFilename}\n")

        # Aggregate to number of parcels per zone and export to geojson
        if varDict['LABEL'] == 'UCC':
            parcelsShape = pd.pivot_table(
                parcels,
                values=['PARCEL_ID'],
                index=[
                    "DEPOT_ID",
                    'CEP',
                    'DEST_NRM',
                    'ORIG_NRM',
                    'VEHTYPE',
                    'FROM_UCC',
                    'TO_UCC'],
                aggfunc={
                    'DEPOT_ID': np.mean,
                    'CEP': 'first',
                    'ORIG_NRM': np.mean,
                    'DEST_NRM': np.mean,
                    'PARCEL_ID': 'count',
                    'VEHTYPE': np.mean,
                    'FROM_UCC': np.mean,
                    'TO_UCC': np.mean})
            parcelsShape = parcelsShape.rename(
                columns={'PARCEL_ID': 'N_PARCELS'})
            parcelsShape = parcelsShape.set_index(
                np.arange(len(parcelsShape)))
            parcelsShape = parcelsShape.reindex(
                columns=[
                    'ORIG_NRM',
                    'DEST_NRM',
                    'N_PARCELS',
                    'DEPOT_ID',
                    'CEP',
                    'VEHTYPE',
                    'FROM_UCC',
                    'TO_UCC'])
            parcelsShape = parcelsShape.astype({
                'DEPOT_ID': int,
                'ORIG_NRM': int,
                'DEST_NRM': int,
                'N_PARCELS': int,
                'VEHTYPE': int,
                'FROM_UCC': int,
                'TO_UCC': int})

        else:
            parcelsShape = pd.pivot_table(
                parcels,
                values=['PARCEL_ID'],
                index=[
                    "DEPOT_ID",
                    'CEP',
                    'DEST_NRM',
                    'ORIG_NRM',
                    'VEHTYPE'],
                aggfunc={
                    'DEPOT_ID': np.mean,
                    'CEP': 'first',
                    'ORIG_NRM': np.mean,
                    'DEST_NRM': np.mean,
                    'VEHTYPE': np.mean,
                    'PARCEL_ID': 'count'})
            parcelsShape = parcelsShape.rename(
                columns={'PARCEL_ID': 'N_PARCELS'})
            parcelsShape = parcelsShape.set_index(
                np.arange(len(parcelsShape)))
            parcelsShape = parcelsShape.reindex(
                columns=[
                    'ORIG_NRM',
                    'DEST_NRM',
                    'N_PARCELS',
                    'DEPOT_ID',
                    'CEP',
                    'VEHTYPE'])
            parcelsShape = parcelsShape.astype({
                'DEPOT_ID': int,
                'ORIG_NRM': int,
                'DEST_NRM': int,
                'N_PARCELS': int,
                'VEHTYPE': int})

        parcelsShape.to_csv(outFilename, index=False)

        outFilenameShp = outFilename[:-4] + '.geojson'
        print(f"Writing parcels GeoJSON to {outFilenameShp}")
        log_file.write(f"Writing shapefile to {outFilenameShp}\n")

        # Initialize arrays with coordinates
        Ax = np.zeros(len(parcelsShape), dtype=int)
        Ay = np.zeros(len(parcelsShape), dtype=int)
        Bx = np.zeros(len(parcelsShape), dtype=int)
        By = np.zeros(len(parcelsShape), dtype=int)

        # Determine coordinates of LineString for each trip
        depotIDs = np.array(parcelsShape['DEPOT_ID'])
        for i in parcelsShape.index:
            if varDict['LABEL'] == 'UCC':
                if parcelsShape.at[i, 'FROM_UCC'] == 1:
                    Ax[i] = zonesX[parcelsShape['ORIG_NRM'][i] - 1]
                    Ay[i] = zonesY[parcelsShape['ORIG_NRM'][i] - 1]
                    Bx[i] = zonesX[parcelsShape['DEST_NRM'][i] - 1]
                    By[i] = zonesY[parcelsShape['DEST_NRM'][i] - 1]
            else:
                Ax[i] = parcelNodes['X'][depotIDs[i]]
                Ay[i] = parcelNodes['Y'][depotIDs[i]]
                Bx[i] = zonesX[parcelsShape['DEST_NRM'][i] - 1]
                By[i] = zonesY[parcelsShape['DEST_NRM'][i] - 1]

        Ax = np.array(Ax, dtype=str)
        Ay = np.array(Ay, dtype=str)
        Bx = np.array(Bx, dtype=str)
        By = np.array(By, dtype=str)
        nRecords = len(parcelsShape)

        with open(outFilenameShp, 'w') as geoFile:
            geoFile.write(
                '{\n' + '"type": "FeatureCollection",\n' + '"features": [\n')
            for i in range(nRecords - 1):
                outputStr = (
                    '{ "type": "Feature", "properties": ' +
                    str(parcelsShape.loc[i, :].to_dict()).replace("'", '"') +
                    ', "geometry": { "type": "LineString", ' +
                    '"coordinates": [ [ ' +
                    Ax[i] + ', ' + Ay[i] +
                    ' ], [ ' +
                    Bx[i] + ', ' + By[i] +
                    ' ] ] } },\n')
                geoFile.write(outputStr)
                if i % 500 == 0:
                    print(
                        '\t' + str(round((i / nRecords) * 100, 1)) + '%',
                        end='\r')

            # Bij de laatste feature moet er geen komma aan het einde
            i += 1
            outputStr = (
                '{ "type": "Feature", "properties": ' +
                str(parcelsShape.loc[i, :].to_dict()).replace("'", '"') +
                ', "geometry": { "type": "LineString", ' +
                '"coordinates": [ [ ' +
                Ax[i] + ', ' + Ay[i] +
                ' ], [ ' +
                Bx[i] + ', ' + By[i] +
                ' ] ] } }\n')
            geoFile.write(outputStr)
            geoFile.write(']\n')
            geoFile.write('}')

        print('\t 100%', end='\r')

        totaltime = round(time.time() - start_time, 2)
        log_file.write(
            "Total runtime: %s seconds\n" % (totaltime))
        log_file.write(
            "End simulation at: " +
            datetime.datetime.now().strftime("%y-%m-%d %H:%M") + "\n")
        log_file.close()

        if root != '':
            root.update_statusbar("Parcel Demand: Done")
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
                    "Parcel Demand: Execution failed!")
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


def generate_parcels(
    nParcels: int,
    zones: pd.DataFrame,
    cepNodeDict: dict,
    cepList: list,
    parcelSkim: np.ndarray,
    parcelNodes: pd.DataFrame
):
    """
    Maak een DataFrame met een pakket op elke rij.
    Kolommen zijn PARCEL_ID, ORIG_NRM, DEST_NRM en DEPOT_ID.

    Args:
        nParcels (int): _description_
        zones (pd.DataFrame): _description_
        cepNodeDict (dict): _description_
        cepList (list): _description_
        parcelSkim (np.ndarray): _description_
        parcelNodes (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: De pakketten.
    """
    # Put parcel demand in Numpy array (faster indexing)
    cols = ['PARCEL_ID', 'ORIG_NRM', 'DEST_NRM', 'DEPOT_ID']
    parcels = np.zeros((nParcels, len(cols)), dtype=int)
    parcelsCep = np.array(['' for i in range(nParcels)], dtype=object)

    # Now determine for each zone and courier from which
    # depot the parcels are delivered
    count = 0
    for zoneID in zones['SEGNR_2018']:

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
                    parcelNodes['SEGNR_2018'][parcelNodeIndex[0] + 1])
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
    parcels['CEP'] = parcelsCep

    # Default vehicle type for parcel deliveries: vans
    parcels['VEHTYPE'] = 7

    return parcels
