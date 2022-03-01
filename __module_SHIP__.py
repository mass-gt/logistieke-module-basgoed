import numpy as np
import pandas as pd
import time
import datetime
import shapefile as shp
from numba import njit, float64
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
        self.root.title("Progress Shipment Synthesizer")
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


@njit
def choice_model_ssvt(
    costFreightKm: np.ndarray, costFreightHr: np.ndarray,
    goodsTypeBG: int,
    fromDC: int, toDC: int,
    travTime: float, distance: float,
    absoluteShipmentSizes: np.ndarray,
    truckCapacities: np.ndarray,
    B_TransportCosts: float, B_InventoryCosts: float,
    B_FromDC: float, B_ToDC: float,
    B_LongHaul_TruckTrailer: float, B_LongHaul_TractorTrailer: float,
    ASC_VT: float,
    nVT: int, nSS: int,
    container: bool = False
):

    # Determine the utility and probability for each alternative
    utilities = np.zeros(nVT * nSS, dtype=float64)

    inventoryCosts = absoluteShipmentSizes
    longHaul = int(distance > 100)

    for ss in range(nSS):
        for vt in range(nVT):
            index = ss * nVT + vt
            costPerHr = costFreightHr[goodsTypeBG - 1, vt]
            costPerKm = costFreightKm[goodsTypeBG - 1, vt]

            # If vehicle type is available for the current goods type
            if costPerHr > 0:
                transportCosts = costPerHr * travTime + costPerKm * distance

                # Multiply transport costs by number of required vehicles
                transportCosts *= np.ceil(
                    absoluteShipmentSizes[ss] / truckCapacities[vt])

                # Utility function
                if container and vt != 5:
                    utilities[index] = -100
                else:
                    utilities[index] = (
                        B_TransportCosts * transportCosts +
                        B_InventoryCosts * inventoryCosts[ss] +
                        B_FromDC * fromDC * (vt == 0) +
                        B_ToDC * toDC * (vt in [3, 4, 5]) +
                        B_LongHaul_TruckTrailer * longHaul * (vt in [3, 4]) +
                        B_LongHaul_TractorTrailer * longHaul * (vt == 5) +
                        ASC_VT[vt])
            else:
                utilities[index] = -100

    probs = np.exp(utilities) / np.sum(np.exp(utilities))
    cumProbs = np.cumsum(probs)

    # Sample one choice based on the cumulative probability distribution
    ssvt = draw_choice(cumProbs)

    return ssvt


@njit
def draw_choice(cumProbs: np.ndarray):
    '''
    Draw one choice from a list of cumulative probabilities.
    '''
    nAlt = len(cumProbs)

    rand = np.random.rand()
    for alt in range(nAlt):
        if cumProbs[alt] >= rand:
            return alt

    raise Exception(
        '\nError in function "draw_choice", random draw was ' +
        'outside range of cumulative probability distribution.')


def actually_run_module(args):

    try:

        root = args[0]
        varDict = args[1]

        start_time = time.time()

        log_file = open(
            varDict['OUTPUTFOLDER'] + "Logfile_ShipmentSynthesizer.log", "w")
        log_file.write(
            "Start simulation at: " +
            datetime.datetime.now().strftime("%y-%m-%d %H:%M") + "\n")

        if varDict['SEED'] != '':
            np.random.seed(varDict['SEED'])

        message = "Importing and preparing data..."
        print(message), log_file.write(message + "\n")
        if root != '':
            root.update_statusbar("Shipment Synthesizer: " + message)
            root.progressBar['value'] = 0

        dimNSTR = pd.read_csv(
            varDict['DIMFOLDER'] + 'nstr.txt',
            sep='\t')
        dimLS = pd.read_csv(
            varDict['DIMFOLDER'] + 'logistic_segment.txt',
            sep='\t')
        dimSS = pd.read_csv(
            varDict['DIMFOLDER'] + 'shipment_size.txt',
            sep='\t')
        dimVT = pd.read_csv(
            varDict['DIMFOLDER'] + 'vehicle_type.txt',
            sep='\t')
        dimFT = pd.read_csv(
            varDict['DIMFOLDER'] + 'flow_type.txt',
            sep='\t')
        dimGG = pd.read_csv(
            varDict['DIMFOLDER'] + 'basgoed_goods_type.txt',
            sep='\t')

        nNSTR = len(dimNSTR) - 1
        nLS = len(dimLS) - 1
        nSS = len(dimSS)
        nVT = np.sum(dimVT['IsRefTypeFreight'] == 1)
        nGG = len(dimGG)

        nFlowTypesInternal = np.sum(dimFT['IsExternal'] == 0)
        # nFlowTypesExternal = np.sum(dimFT['IsExternal'] == 1)

        ftFromDC = [3 - 1, 5 - 1, 7 - 1]
        ftToDC = [2 - 1, 5 - 1, 8 - 1]
        ftFromTT = [6 - 1, 8 - 1, 9 - 1]
        ftToTT = [4 - 1, 7 - 1, 9 - 1]

        # TODO: Aantal zones NL BasGoed uit setup-data afleiden
        nZonesNL_BG = 45

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

        forecast = (varDict['YEAR'] > varDict['BASEYEAR'])

        jobNames = [
            'LANDBOUW' + yrAppendix,
            'INDUSTRIE' + yrAppendix,
            'DETAIL' + yrAppendix,
            'DIENSTEN' + yrAppendix,
            'OVERHEID' + yrAppendix,
            'OVERIG' + yrAppendix]
        jobDict = dict((jobNames[i], i) for i in range(len(jobNames)))
        nSectors = len(jobNames)

        truckCapacities = np.array(pd.read_csv(
            varDict['VEHCAPACITY'],
            index_col=0))[:, 0] / 1000
        absoluteShipmentSizes = np.array(dimSS['Median'])

        if root != '':
            root.progressBar['value'] = 0.2

        # Conversion tables for goods types and sectors in make-use table
        convGG = np.array(pd.read_csv(
            varDict['BASGOED_PARAMETERS'] + "Goederengroep.asc",
            sep='\t'), dtype=int)
        convGG = dict((convGG[i, :]) for i in range(len(convGG)))

        convSector = np.array(pd.read_csv(
            varDict['SECTOR_TO_SECTOR'], sep=','))
        convSector[:, 1] = [jobDict[x + yrAppendix] for x in convSector[:, 1]]
        convSector = dict(
            (int(convSector[i, 0]), int(convSector[i, 1]))
            for i in range(len(convSector)))

        # Import make-use table BasGoed
        makeUseTable = np.array(pd.read_csv(
            varDict['BASGOED_PARAMETERS'] + "MakeUseEerstJr.asc",
            sep='\t'))

        # Convert the goods types and sectors
        for i in range(len(makeUseTable)):
            if makeUseTable[i, 0] in convGG.keys():
                makeUseTable[i, 0] = convGG[makeUseTable[i, 0]]
            else:
                makeUseTable[i, 0] = -1

            if makeUseTable[i, 1] in convSector.keys():
                makeUseTable[i, 1] = convSector[makeUseTable[i, 1]]
            else:
                makeUseTable[i, 1] = -1

        # Remove rows with no known converstion for the goods type or sector
        makeUseTable = makeUseTable[
            (makeUseTable[:, 0] >= 0) & (makeUseTable[:, 1] >= 0), :]

        # Create make and use distribution tables
        # (by BG-GG and NRM-industry-sector)
        makeDistribution = np.zeros((nGG, nSectors))
        useDistribution = np.zeros((nGG, nSectors))
        for i in range(len(makeUseTable)):
            gg = int(makeUseTable[i, 0])
            sector = int(makeUseTable[i, 1])
            makeWeight = makeUseTable[i, 2]
            useWeight = makeUseTable[i, 3]

            makeDistribution[gg - 1, sector] += makeWeight
            useDistribution[gg - 1,  sector] += useWeight

        for gg in range(nGG):
            sumMake = np.sum(makeDistribution[gg, :])
            sumUse = np.sum(useDistribution[gg, :])

            if sumMake > 0:
                makeDistribution[gg, :] /= sumMake
            else:
                makeDistribution[gg, :] = (np.ones(nSectors) / nSectors)

            if sumUse > 0:
                useDistribution[gg, :] /= sumUse
            else:
                useDistribution[gg, :] = (np.ones(nSectors) / nSectors)

        if root != '':
            root.progressBar['value'] = 0.4

        # Connectie BasGoed-GG en logistiek segment
        # (kansen, niet cumulatief)
        GGtoLScontProb = np.array(pd.read_csv(
            varDict['BGGG_TO_LS_CONT'],
            sep=',',
            header=None))
        GGtoLSncontProb = np.array(pd.read_csv(
            varDict['BGGG_TO_LS_NCONT'],
            sep=',',
            header=None))

        for gg in range(nGG):
            if np.sum(GGtoLScontProb[gg, :]) > 0:
                GGtoLScontProb[gg, :] /= np.sum(GGtoLScontProb[gg, :])
            else:
                GGtoLScontProb[gg, :] = (np.ones(nLS) / nLS)

            if np.sum(GGtoLSncontProb[gg, :]) > 0:
                GGtoLSncontProb[gg, :] /= np.sum(GGtoLSncontProb[gg, :])
            else:
                GGtoLSncontProb[gg, :] = (np.ones(nLS) / nLS)

        # Road tonnes calculated by BasGoed
        if forecast:
            filename = 'Forecast'
        else:
            filename = 'BaseMergedModes'

        tonnes = pd.read_csv(
            varDict['BASGOED_MS'] + f'{filename}1.matrix',
            sep='\t')[['orig', 'dest', 'road']]
        tonnes.columns = ['ORIG', 'DEST', 'WeightDay']
        tonnes['GG'] = 1

        for gg in range(2, nGG + 1):
            tonnesGG = pd.read_csv(
                varDict['BASGOED_MS'] + f'{filename}{gg}.matrix',
                sep='\t')[['orig', 'dest', 'road']]
            tonnesGG.columns = ['ORIG', 'DEST', 'WeightDay']
            tonnesGG['GG'] = gg
            tonnes = tonnes.append(tonnesGG.copy())

        tonnes['WeightDay'] /= varDict['YEARFACTOR']
        tonnes = tonnes.sort_values(by=['ORIG', 'DEST', 'GG'])
        tonnes.index = np.arange(len(tonnes))

        # Spreid tonnen niet-container over logistieke segmenten
        tonnesLS = np.zeros((nLS * len(tonnes), 5), dtype=float)
        row = 0
        for i in tonnes.index:
            orig = tonnes.at[i, 'ORIG']
            dest = tonnes.at[i, 'DEST']
            gg = tonnes.at[i, 'GG']
            weight = tonnes.at[i, 'WeightDay']

            for ls in range(nLS):
                fracLS = GGtoLSncontProb[gg - 1, ls]

                if fracLS > 0:
                    tonnesLS[row, 0] = orig
                    tonnesLS[row, 1] = dest
                    tonnesLS[row, 2] = gg
                    tonnesLS[row, 3] = ls
                    tonnesLS[row, 4] = weight * fracLS

                row += 1

        tonnesLS = tonnesLS[tonnesLS[:, 4] > 0]
        tonnes = pd.DataFrame(
            tonnesLS,
            columns=['ORIG', 'DEST', 'GG', 'LS', 'WeightDay'])
        intCols = ['ORIG', 'DEST', 'GG', 'LS']
        tonnes[intCols] = tonnes[intCols].astype(int)

        # Containerized road tonnes calculated by BasGoed
        if forecast:
            filename = 'Forecast'
        else:
            filename = 'Base'

        tonnesCKM = pd.read_csv(
            varDict['BASGOED_CKM'] + f'{filename}1.matrix',
            sep='\t')[['orig', 'dest', 'road']]
        tonnesCKM.columns = ['ORIG', 'DEST', 'WeightDay']
        tonnesCKM['GG'] = 1

        for gg in range(2, nGG + 1):
            tonnesGG = pd.read_csv(
                varDict['BASGOED_CKM'] + f'{filename}{gg}.matrix',
                sep='\t')[['orig', 'dest', 'road']]
            tonnesGG.columns = ['ORIG', 'DEST', 'WeightDay']
            tonnesGG['GG'] = gg
            tonnesCKM = tonnesCKM.append(tonnesGG.copy())

        tonnesCKM['WeightDay'] /= varDict['YEARFACTOR']
        tonnesCKM = tonnesCKM.sort_values(by=['ORIG', 'DEST', 'GG'])
        tonnesCKM.index = np.arange(len(tonnesCKM))

        # Spreid tonnen container over logistieke segmenten
        tonnesLS = np.zeros((nLS * len(tonnesCKM), 5), dtype=float)
        row = 0
        for i in tonnesCKM.index:
            orig = tonnesCKM.at[i, 'ORIG']
            dest = tonnesCKM.at[i, 'DEST']
            gg = tonnesCKM.at[i, 'GG']
            weight = tonnesCKM.at[i, 'WeightDay']

            for ls in range(nLS):
                fracLS = GGtoLScontProb[gg - 1, ls]

                if fracLS > 0:
                    tonnesLS[row, 0] = orig
                    tonnesLS[row, 1] = dest
                    tonnesLS[row, 2] = gg
                    tonnesLS[row, 3] = ls
                    tonnesLS[row, 4] = weight * fracLS

                row += 1

        tonnesLS = tonnesLS[tonnesLS[:, 4] > 0]
        tonnesCKM = pd.DataFrame(
            tonnesLS,
            columns=['ORIG', 'DEST', 'GG', 'LS', 'WeightDay'])
        intCols = ['ORIG', 'DEST', 'GG', 'LS']
        tonnesCKM[intCols] = tonnesCKM[intCols].astype(int)

        del tonnesLS, tonnesGG

        # Container statistics
        containerStats = pd.read_csv(
            varDict['CONTAINER'],
            sep=',',
            index_col='BG_GGR')
        containerSS = np.array(containerStats['SHIPSIZE_AVG'])[:-1]
        containerEmptyShare = (
            containerStats.iloc[-1, 1] / np.sum(containerStats.iloc[:-1, -1]))
        containerEmptySS = containerStats.at['LEEG', 'SHIPSIZE_AVG']

        # Koppeltabel NRM naar BG
        NRMtoBG = pd.read_csv(varDict['NRM_TO_BG'], sep=',')
        NRMtoBG = NRMtoBG.astype(int)
        dictNRMtoBG = dict(
            (NRMtoBG.at[i, 'SEGNR_2018'], NRMtoBG.at[i, 'ZONEBG2018'])
            for i in NRMtoBG.index)
        BGwithOverlapNRM = np.unique(NRMtoBG['ZONEBG2018'])

        # Koppeltabel BG naar NRM
        BGtoNRM = pd.read_csv(varDict['BG_TO_NRM'], sep=',')
        BGtoNRM = BGtoNRM.astype(int)
        BGtoNRM.index = BGtoNRM['ZONEBG2018']
        dictBGtoNRM = {}
        for i in BGtoNRM.index:
            if i in BGwithOverlapNRM:
                dictBGtoNRM[i] = np.array(
                    NRMtoBG.loc[NRMtoBG['ZONEBG2018'] == i, 'SEGNR_2018'])
            else:
                dictBGtoNRM[i] = np.array([BGtoNRM.at[i, 'SEGNR_2018']])

        # Connectie BasGoed-GG en NSTR (cumulatieve kansen)
        GGtoNSTRcont = np.array(pd.read_csv(
            varDict['BGGG_TO_NSTR_CONT'],
            sep=',',
            header=None))
        GGtoNSTRncont = np.array(pd.read_csv(
            varDict['BGGG_TO_NSTR_NCONT'],
            sep=',',
            header=None))

        for gg in range(nGG):
            if np.sum(GGtoNSTRcont[gg, :]) == 0:
                GGtoNSTRcont[gg, :] = np.ones(nNSTR)
            if np.sum(GGtoNSTRncont[gg, :]) == 0:
                GGtoNSTRncont[gg, :] = np.ones(nNSTR)

            GGtoNSTRcont[gg, :] = np.cumsum(GGtoNSTRcont[gg, :])
            GGtoNSTRncont[gg, :] = np.cumsum(GGtoNSTRncont[gg, :])

            GGtoNSTRcont[gg, :] /= GGtoNSTRcont[gg, -1]
            GGtoNSTRncont[gg, :] /= GGtoNSTRncont[gg, -1]

        if root != '':
            root.progressBar['value'] = 0.6

        # Import zones data
        zones = read_shape(varDict['ZONES'])
        zones.index = zones['SEGNR_2018']
        zonesX = np.array(zones['XCOORD'])
        zonesY = np.array(zones['YCOORD'])
        zonesLMS = np.array(zones['LMSVAM'], dtype=int)

        if root != '':
            root.progressBar['value'] = 1.5

        # Calculate urban density of zones
        urbanDensityCat = {}

        for i in zones.index:
            tmpNumInhabitants = zones.at[i, 'INWONERS' + yrAppendix]
            tmpSurface = zones.at[i, 'OPP']
            tmpUrbanDensity = 100 * (tmpNumInhabitants / tmpSurface)

            if tmpUrbanDensity < 500:
                urbanDensityCat[i] = 1
            elif tmpUrbanDensity < 1000:
                urbanDensityCat[i] = 2
            elif tmpUrbanDensity < 1500:
                urbanDensityCat[i] = 3
            elif tmpUrbanDensity < 2500:
                urbanDensityCat[i] = 4
            else:
                urbanDensityCat[i] = 5

        # Import logistic nodes data
        logNodes = pd.read_csv(varDict['DISTRIBUTIECENTRA'])
        logNodes = logNodes[~pd.isna(logNodes['SEGNR_2018'])]
        logNodes.index = np.arange(len(logNodes))
        logNodes['ZONEBG2018'] = [
            dictNRMtoBG[x] for x in logNodes['SEGNR_2018']]
        logNodesX = np.array(logNodes['X'])
        logNodesY = np.array(logNodes['Y'])
        logNodesBG = np.array(logNodes['ZONEBG2018'], dtype=int)
        logNodesNRM = np.array(logNodes['SEGNR_2018'], dtype=int)
        logNodesWP = [x if x != np.nan else 0 for x in logNodes['WP']]
        BGzonesWithLogNode = set(list(np.unique(logNodesBG)))

        # Import transshipment terminals data
        terminals = pd.read_csv(varDict['TERMINALS'], sep=',')
        terminals['ZONEBG2018'] = [
            dictNRMtoBG[x] for x in terminals['NRM_Moederzone']]
        terminalsX = np.array(terminals['X'])
        terminalsY = np.array(terminals['Y'])
        terminalsBG = np.array(terminals['ZONEBG2018'], dtype=int)
        terminalsNRM = np.array(terminals['NRM_Moederzone'], dtype=int)
        BGzonesWithTerminal = set(list(np.unique(terminalsBG)))

        if root != '':
            root.progressBar['value'] = 1.6

        # Prevent double counting of flows to/from TTs
        industrieFields = [
            'INDUSTRIE', 'INDUSTRIE_1', 'INDUSTRIE_2', 'INDUSTRIE_3']
        zones.loc[terminalsNRM, industrieFields] = 0.0

        # Prevent double counting of flows to/from DCs
        for i in range(len(logNodes)):
            zones.loc[logNodesNRM[i], industrieFields] -= logNodesWP[i]
        for field in industrieFields:
            zones[field] = [max(0.0, x) for x in zones[field].values]

        # Flowtype distribution (7 LSs and 12 flowtypes)
        ftSharesNL = np.array(pd.read_csv(
            varDict['FLOWTYPES'],
            index_col=0))[:nFlowTypesInternal, :]
        ftSharesExt = np.array(pd.read_csv(
            varDict['FLOWTYPES'],
            index_col=0))[nFlowTypesInternal:, :]

        # Account flowtypes for BG-zones without DC or TT (within NL)
        tonnesNL = tonnes[
            (tonnes['ORIG'] <= nZonesNL_BG) &
            (tonnes['DEST'] <= nZonesNL_BG)].copy()

        for ls in range(nLS):
            tonnesNL_LS = tonnesNL[tonnesNL['LS'] == ls]
            sumTonnesNL = np.sum(tonnesNL_LS['WeightDay'])

            whereFromDC = [
                i for i in tonnesNL_LS.index
                if tonnesNL_LS.at[i, 'ORIG'] in BGzonesWithLogNode]
            sumTonnesNLfromDC = np.sum(
                tonnesNL_LS.loc[whereFromDC, 'WeightDay'])
            fromDCfac = sumTonnesNL / sumTonnesNLfromDC
            ftSharesNL[ftFromDC, ls] *= fromDCfac

            whereToDC = [
                i for i in tonnesNL_LS.index
                if tonnesNL_LS.at[i, 'DEST'] in BGzonesWithLogNode]
            sumTonnesNLtoDC = np.sum(
                tonnesNL_LS.loc[whereToDC, 'WeightDay'])
            toDCfac = sumTonnesNL / sumTonnesNLtoDC
            ftSharesNL[ftToDC, ls] *= toDCfac

            whereFromTT = [
                i for i in tonnesNL_LS.index if
                tonnesNL_LS.at[i, 'ORIG'] in BGzonesWithTerminal]
            sumTonnesNLfromTT = np.sum(
                tonnesNL_LS.loc[whereFromTT, 'WeightDay'])
            fromTTfac = sumTonnesNL / sumTonnesNLfromTT
            ftSharesNL[ftFromTT, ls] *= fromTTfac

            whereToTT = [
                i for i in tonnesNL_LS.index if
                tonnesNL_LS.at[i, 'DEST'] in BGzonesWithTerminal]
            sumTonnesNLtoTT = np.sum(
                tonnesNL_LS.loc[whereToTT, 'WeightDay'])
            toTTfac = sumTonnesNL / sumTonnesNLtoTT
            ftSharesNL[ftToTT, ls] *= toTTfac

        # Account flowtypes for BG-zones without DC or TT (from/to NL)
        whereTonnesExt = (
            ((tonnes['ORIG'] > nZonesNL_BG) | (tonnes['DEST'] > nZonesNL_BG)) &
            ~((tonnes['ORIG'] > nZonesNL_BG) & (tonnes['DEST'] > nZonesNL_BG)))
        tonnesExt = tonnes[whereTonnesExt].copy()

        for ls in range(nLS):
            tonnesExtLS = tonnesExt[tonnesExt['LS'] == ls]

            sumTonnesExtToNL = np.sum(
                tonnesExtLS.loc[
                    tonnesExtLS['DEST'] <= nZonesNL_BG,
                    'WeightDay'])
            sumTonnesExtFromNL = np.sum(
                tonnesExtLS.loc[
                    tonnesExtLS['ORIG'] <= nZonesNL_BG,
                    'WeightDay'])

            whereExtFromDC = [
                i for i in tonnesExtLS.index
                if tonnesExtLS.at[i, 'ORIG'] in BGzonesWithLogNode]
            whereExtToDC = [
                i for i in tonnesExtLS.index
                if tonnesExtLS.at[i, 'DEST'] in BGzonesWithLogNode]
            sumTonnesExtFromDC = np.sum(
                tonnesExtLS.loc[whereExtFromDC, 'WeightDay'])
            sumTonnesExtToDC = np.sum(
                tonnesExtLS.loc[whereExtToDC, 'WeightDay'])

            fromDCfac = sumTonnesExtFromNL / sumTonnesExtFromDC
            toDCfac = sumTonnesExtToNL / sumTonnesExtToDC

            ftSharesExt[1, ls] *= (fromDCfac + toDCfac) / 2

            whereExtFromTT = [
                i for i in tonnesExtLS.index
                if tonnesExtLS.at[i, 'ORIG'] in BGzonesWithTerminal]
            whereExtToTT = [
                i for i in tonnesExtLS.index
                if tonnesExtLS.at[i, 'DEST'] in BGzonesWithTerminal]
            sumTonnesExtFromTT = np.sum(
                tonnesExtLS.loc[whereExtFromTT, 'WeightDay'])
            sumTonnesExtToTT = np.sum(
                tonnesExtLS.loc[whereExtToTT,   'WeightDay'])

            fromTTfac = sumTonnesExtFromNL / sumTonnesExtFromTT
            toTTfac = sumTonnesExtToNL / sumTonnesExtToTT

            ftSharesExt[2, ls] *= (fromTTfac + toTTfac) / 2

        # Flow types shares for containers (always from/to TT or between TTs)
        ftSharesCont = ftSharesNL.copy()
        ftSharesCont[[0, 1, 2, 4], :] = 0

        # Make cumulative probability distribution of flow types per LS
        for ls in range(nLS):
            ftSharesNL[:, ls] = np.cumsum(ftSharesNL[:, ls])
            ftSharesNL[:, ls] /= ftSharesNL[-1, ls]

            ftSharesExt[:, ls] = np.cumsum(ftSharesExt[:, ls])
            ftSharesExt[:, ls] /= ftSharesExt[-1, ls]

            ftSharesCont[:, ls] = np.cumsum(ftSharesCont[:, ls])
            ftSharesCont[:, ls] /= ftSharesCont[-1, ls]

        del tonnesNL, tonnesNL_LS, tonnesExt, tonnesExtLS
        tonnes = tonnes[['ORIG', 'DEST', 'GG', 'LS', 'WeightDay']]

        if root != '':
            root.progressBar['value'] = 1.7

        # Skim with travel times and distances
        skimTravTime = read_mtx(varDict['SKIMTIME'])
        skimDistance = read_mtx(varDict['SKIMDISTANCE'])
        nZones = int(len(skimTravTime)**0.5)

        skimTravTime[skimTravTime < 0] = 0
        skimDistance[skimDistance < 0] = 0

        # For zero times and distances assume half the value to the
        # nearest (non-zero) zone
        # (otherwise we get problem in the distance decay function)
        for orig in range(nZones):

            whereZero = np.where(
                skimTravTime[orig * nZones + np.arange(nZones)] == 0)[0]
            whereNonZero = np.where(
                skimTravTime[orig * nZones + np.arange(nZones)] != 0)[0]
            if len(whereZero) > 0:
                skimTravTime[orig * nZones + whereZero] = (
                    0.5 * np.min(skimTravTime[orig * nZones + whereNonZero]))

            whereZero = np.where(
                skimDistance[orig * nZones + np.arange(nZones)] == 0)[0]
            whereNonZero = np.where(
                skimDistance[orig * nZones + np.arange(nZones)] != 0)[0]
            if len(whereZero) > 0:
                skimDistance[orig * nZones + whereZero] = (
                    0.5 * np.min(skimDistance[orig * nZones + whereNonZero]))

        if root != '':
            root.progressBar['value'] = 2.5

        # Cost parameters by vehicle type with size (small/medium/large)
        costFreight = [None for vt in range(nVT)]
        costFreight[0] = np.array(pd.read_csv(
            varDict['COST_VRACHTWAGEN'],
            sep='\t'))[:, :3]
        costFreight[3] = np.array(pd.read_csv(
            varDict['COST_AANHANGER'],
            sep='\t'))[:, :3]
        costFreight[5] = np.array(pd.read_csv(
            varDict['COST_OPLEGGER'],
            sep='\t'))[:, :3]
        costFreight[6] = np.array(pd.read_csv(
            varDict['COST_SPECIAAL'],
            sep='\t'))[:, :3]
        costFreight[7] = np.array(pd.read_csv(
            varDict['COST_LZV'],
            sep='\t'))[:, :3]
        costFreight[8] = np.array(pd.read_csv(
            varDict['COST_BESTEL'],
            sep='\t'))[:, :3]
        costFreight[1] = costFreight[0]
        costFreight[2] = costFreight[0]
        costFreight[4] = costFreight[3]

        # Fill in missing goods types
        for vt in range(nVT):
            newArray = np.zeros((nGG, 3), dtype=float)

            for gg in range(nGG):
                if (gg + 1) not in costFreight[vt][:, 0]:
                    newArray[gg, :] = [gg + 1, 0, 0]
                else:
                    row = np.where(costFreight[vt][:, 0] == (gg + 1))[0]
                    newArray[gg, :] = costFreight[vt][row, :]

            costFreight[vt] = newArray

        # Restructure cost array
        costFreightKm = np.zeros((nGG, nVT), dtype=float)
        costFreightHr = np.zeros((nGG, nVT), dtype=float)
        for gg in range(nGG):
            for vt in range(nVT):
                costFreightKm[gg, vt] = costFreight[vt][gg, 1]
                costFreightHr[gg, vt] = costFreight[vt][gg, 2]

        # Estimated parameters MNL for combined shipment size and vehicle type
        paramsShipSizeVehType = pd.read_csv(
            varDict['PARAMS_SSVT'],
            index_col=0)

        if root != '':
            root.progressBar['value'] = 2.6

        if varDict['LABEL'] == 'UCC':
            # Consolidation potential per logistic segment (for UCC scenario)
            probConsolidation = np.array(pd.read_csv(
                varDict['ZEZ_CONSOLIDATION'],
                sep=',',
                index_col='Segment'))

            # Vehicle/combustion shares (for UCC scenario)
            sharesUCC = pd.read_csv(
                varDict['ZEZ_SCENARIO'],
                index_col='Segment')

            vtNamesUCC = [
                'LEVV', 'Moped',
                'Van', 'Truck',
                'TractorTrailer', 'WasteCollection',
                'SpecialConstruction']

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

            # Couple these vehicle types to HARMONY vehicle types
            vehUccToVeh = {
                0: 9,
                1: 10,
                2: 8,
                3: 1,
                4: 5,
                5: 6,
                6: 6}

            # Which zones are ZEZ and by which UCC-zone are they served
            zezZones = pd.read_csv(varDict['ZEZ_ZONES'], sep=',').astype(int)
            zezToUCC = dict(
                (zezZones.at[i, 'SEGNR_2018'], zezZones.at[i, 'UCC'])
                for i in zezZones.index)
            zezZones = np.array(zezZones['SEGNR_2018'], dtype=int)

        if root != '':
            root.progressBar['value'] = 2.8

        # Probabilities of a shipment being allocated to a particular DC/TT
        probDC = np.array(logNodes['oppervlak'], dtype=float)
        probTT = np.ones(len(terminals), dtype=float)

        if root != '':
            root.progressBar['value'] = 2.9

        if varDict['SHIPMENTS_REF'] == "":

            message = "Calculating zonal probabilities..."
            print(message), log_file.write(message + "\n")
            if root != '':
                root.update_statusbar("Shipment Synthesizer: " + message)

            # Determine probabilities of zones/DCs/TTs being origin
            # of a shipment
            origProbs = {}
            origProbsDC = {}
            origProbsTT = {}

            origZonesBG = np.unique(tonnes['ORIG'].append(tonnesCKM['ORIG']))
            for origZoneBG in origZonesBG:
                # Which NRM-zones fall under the current BG-zones
                origZonesNRM = dictBGtoNRM[origZoneBG]
                origJobsNRM = np.array(zones.loc[origZonesNRM, jobNames])

                # Probabilities consumer origins
                origProbs[origZoneBG] = {}
                for gg in range(nGG):
                    origProbs[origZoneBG][gg] = np.sum(
                        makeDistribution[gg, :] * origJobsNRM,
                        axis=1)
                    if np.sum(origProbs[origZoneBG][gg]) == 0:
                        origProbs[origZoneBG][gg] = np.ones(
                            len(origProbs[origZoneBG][gg]))
                    origProbs[origZoneBG][gg] = np.cumsum(
                        origProbs[origZoneBG][gg])
                    origProbs[origZoneBG][gg] /= origProbs[origZoneBG][gg][-1]

                # Probabilities DC origins
                if origZoneBG in BGzonesWithLogNode:
                    whereDCinOrig = np.where(logNodesBG == origZoneBG)[0]
                    origProbsDC[origZoneBG] = probDC[whereDCinOrig]
                    origProbsDC[origZoneBG] = np.cumsum(
                        origProbsDC[origZoneBG])
                    origProbsDC[origZoneBG] /= origProbsDC[origZoneBG][-1]

                # Probabilities TT origins
                if origZoneBG in BGzonesWithTerminal:
                    whereTTinOrig = np.where(terminalsBG == origZoneBG)[0]
                    origProbsTT[origZoneBG] = probTT[whereTTinOrig]
                    origProbsTT[origZoneBG] = np.cumsum(
                        origProbsTT[origZoneBG])
                    origProbsTT[origZoneBG] /= origProbsTT[origZoneBG][-1]

            if root != '':
                root.progressBar['value'] = 4

            # Determine probabilities of zones/DCs/TTs being
            # destination of a shipment
            destProbs = {}
            destProbsDC = {}
            destProbsTT = {}

            destZonesBG = np.unique(tonnes['DEST'].append(tonnesCKM['DEST']))
            for destZoneBG in destZonesBG:
                # Which NRM-zones fall under the current BG-zones
                destZonesNRM = dictBGtoNRM[destZoneBG]
                destJobsNRM = np.array(zones.loc[destZonesNRM, jobNames])

                # Probabilities consumer destinations
                destProbs[destZoneBG] = {}
                for gg in range(nGG):
                    destProbs[destZoneBG][gg] = np.sum(
                        useDistribution[gg, :] * destJobsNRM, axis=1)
                    if np.sum(destProbs[destZoneBG][gg]) == 0:
                        destProbs[destZoneBG][gg] = np.ones(
                            len(destProbs[destZoneBG][gg]))
                    destProbs[destZoneBG][gg] = np.cumsum(
                        destProbs[destZoneBG][gg])
                    destProbs[destZoneBG][gg] /= destProbs[destZoneBG][gg][-1]

                # Probabilities DC destinations
                if destZoneBG in BGzonesWithLogNode:
                    whereDCinDest = np.where(logNodesBG == destZoneBG)[0]
                    destProbsDC[destZoneBG] = probDC[whereDCinDest]
                    destProbsDC[destZoneBG] = np.cumsum(
                        destProbsDC[destZoneBG])
                    destProbsDC[destZoneBG] /= destProbsDC[destZoneBG][-1]

                # Probabilities TT destinations
                if destZoneBG in BGzonesWithTerminal:
                    whereTTinDest = np.where(terminalsBG == destZoneBG)[0]
                    destProbsTT[destZoneBG] = probTT[whereTTinDest]
                    destProbsTT[destZoneBG] = np.cumsum(
                        destProbsTT[destZoneBG])
                    destProbsTT[destZoneBG] /= destProbsTT[destZoneBG][-1]

            # Initialize shipment attributes as dictionaries
            ship_flowType = {}
            ship_goodsTypeNSTR = {}
            ship_goodsTypeBG = {}
            ship_logisticSegment = {}
            ship_containerized = {}
            ship_shipmentSize = {}
            ship_shipmentSizeCat = {}
            ship_vehicleType = {}
            ship_origZoneNRM = {}
            ship_destZoneNRM = {}
            ship_origZoneBG = {}
            ship_destZoneBG = {}
            ship_origTT = {}
            ship_origDC = {}
            ship_destTT = {}
            ship_destDC = {}
            ship_origX = {}
            ship_origY = {}
            ship_destX = {}
            ship_destY = {}

            message = "Synthesizing shipments MS..."
            print(message), log_file.write(message + "\n")
            percStart, percEnd = 5, 80
            if root != '':
                root.progressBar['value'] = percStart
                root.update_statusbar(
                    "Shipment Synthesizer: Synthesizing shipments MS (0.0%)")

            # For progress bar
            totalWeight = np.sum(tonnes['WeightDay'])
            allocatedWeight = 0

            # Initialize a counter for the procedure
            count = 0

            for i in tonnes.index:
                # BG-origin, -destination and -goods type of the current cell
                origZoneBG = tonnes.at[i, 'ORIG']
                destZoneBG = tonnes.at[i, 'DEST']
                goodsTypeBG = tonnes.at[i, 'GG']
                ls = tonnes.at[i, 'LS']

                # Keep track of allocated weight for current cell
                # (i.e. orig-dest-goods type combinations)
                allocatedWeightCurrentCell = 0
                totalWeightCurrentCell = tonnes.at[i, 'WeightDay']

                # Same origin BG-zone as previous loop?
                recalculateOrigZones = False
                if count == 0:
                    recalculateOrigZones = True
                elif origZoneBG != tonnes.at[i - 1, 'ORIG']:
                    recalculateOrigZones = True

                # Same destination BG-zone as previous loop?
                recalculateDestZones = False
                if count == 0:
                    recalculateDestZones = True
                elif destZoneBG != tonnes.at[i - 1, 'DEST']:
                    recalculateDestZones = True

                if recalculateOrigZones:
                    # Which NRM-zones fall under the current BG-origin-zone
                    origZonesNRM = dictBGtoNRM[origZoneBG]

                    # DC origins
                    if origZoneBG in BGzonesWithLogNode:
                        whereDCinOrig = np.where(logNodesBG == origZoneBG)[0]
                        DCsInOrig = True
                    else:
                        DCsInOrig = False

                    # TT origins
                    if origZoneBG in BGzonesWithTerminal:
                        whereTTinOrig = np.where(terminalsBG == origZoneBG)[0]
                        TTsInOrig = True
                    else:
                        TTsInOrig = False

                if recalculateDestZones:
                    # Which NRM-zones fall under the current
                    # BG-destination-zone
                    destZonesNRM = dictBGtoNRM[destZoneBG]

                    # DC destinations
                    if destZoneBG in BGzonesWithLogNode:
                        whereDCinDest = np.where(logNodesBG == destZoneBG)[0]
                        DCsInDest = True
                    else:
                        DCsInDest = False

                    # TT destinatons
                    if destZoneBG in BGzonesWithTerminal:
                        whereTTinDest = np.where(terminalsBG == destZoneBG)[0]
                        TTsInDest = True
                    else:
                        TTsInDest = False

                while allocatedWeightCurrentCell < totalWeightCurrentCell:

                    # Determine NSTR goods type
                    nstr = draw_choice(GGtoNSTRncont[goodsTypeBG - 1, :])

                    # Determine flow type to/from external zones
                    if origZoneBG > nZonesNL_BG or destZoneBG > nZonesNL_BG:
                        if (
                            origZoneBG > nZonesNL_BG and
                            destZoneBG > nZonesNL_BG
                        ):
                            ft = 10
                            fromExt, toExt = 1, 1

                        else:
                            ft = 10 + draw_choice(ftSharesExt[:, ls])

                            if origZoneBG > nZonesNL_BG:
                                fromExt, toExt = 1, 0

                            if destZoneBG > nZonesNL_BG:
                                fromExt, toExt = 0, 1

                    # Determine flow type within study area (=NL)
                    else:
                        ft = 1 + draw_choice(ftSharesNL[:, ls])
                        fromExt, toExt = 0, 0

                    # Determine origin zone: Flows from DC
                    if ft in [x + 1 for x in ftFromDC]:
                        if DCsInOrig:
                            origDC = whereDCinOrig[draw_choice(
                                origProbsDC[origZoneBG])]
                            origZoneNRM = logNodesNRM[origDC]
                            fromDC = 1
                            fromTT = 0
                        else:
                            ft = {3: 1, 5: 2, 7: 4}[ft]

                    # Determine origin zone: Flows from TT
                    if ft in [x + 1 for x in ftFromTT]:
                        if TTsInOrig:
                            origTT = whereTTinOrig[draw_choice(
                                origProbsTT[origZoneBG])]
                            origZoneNRM = terminalsNRM[origTT]
                            fromDC = 0
                            fromTT = 1
                        else:
                            ft = {6: 1, 8: 2, 9: 4}[ft]

                    # Determine origin zone: Flows from consumer
                    if ft in (1, 2, 4):
                        origZoneNRM = origZonesNRM[draw_choice(
                            origProbs[origZoneBG][goodsTypeBG - 1])]
                        fromDC = 0
                        fromTT = 0

                    # Determine destination zone: Flows to DC
                    if ft in [x + 1 for x in ftToDC]:
                        if DCsInDest:
                            destDC = whereDCinDest[draw_choice(
                                destProbsDC[destZoneBG])]
                            destZoneNRM = logNodesNRM[destDC]
                            toDC = 1
                            toTT = 0
                        else:
                            ft = {2: 1, 5: 3, 8: 6}[ft]

                    # Determine destination zone: Flows to TT
                    if ft in [x + 1 for x in ftToTT]:
                        if TTsInDest:
                            destTT = whereTTinDest[draw_choice(
                                destProbsTT[destZoneBG])]
                            destZoneNRM = terminalsNRM[destTT]
                            toDC = 0
                            toTT = 1
                        else:
                            ft = {4: 1, 7: 3, 9: 6}[ft]

                    # Determine destination zone: Flows to consumer
                    if ft in (1, 3, 6):
                        destZoneNRM = destZonesNRM[draw_choice(
                            destProbs[destZoneBG][goodsTypeBG - 1])]
                        toDC = 0
                        toTT = 0

                    # Flows to/from/between external zones: to/from DC
                    if ft == 11:

                        if fromExt == 1:
                            if DCsInDest:
                                origZoneNRM = origZonesNRM[draw_choice(
                                    origProbs[origZoneBG][goodsTypeBG - 1])]
                                destDC = whereDCinDest[draw_choice(
                                    destProbsDC[destZoneBG])]
                                destZoneNRM = logNodesNRM[destDC]
                                fromDC, toDC = 0, 1
                                fromTT, toTT = 0, 0
                            else:
                                ft = 10

                        elif toExt == 1:
                            if DCsInOrig:
                                origDC = whereDCinOrig[draw_choice(
                                    origProbsDC[origZoneBG])]
                                origZoneNRM = logNodesNRM[origDC]
                                destZoneNRM = destZonesNRM[draw_choice(
                                    destProbs[destZoneBG][goodsTypeBG - 1])]
                                fromDC, toDC = 1, 0
                                fromTT, toTT = 0, 0
                            else:
                                ft = 10

                    # Flows to/from/between external zones: to/from TT
                    if ft == 12:

                        if fromExt == 1:
                            if TTsInDest:
                                origZoneNRM = origZonesNRM[draw_choice(
                                    origProbs[origZoneBG][goodsTypeBG - 1])]
                                destTT = whereTTinDest[draw_choice(
                                    destProbsTT[destZoneBG])]
                                destZoneNRM = terminalsNRM[destTT]
                                fromDC, toDC = 0, 0
                                fromTT, toTT = 0, 1
                            else:
                                ft = 10

                        elif toExt == 1:
                            if TTsInOrig:
                                origTT = whereTTinOrig[draw_choice(
                                    origProbsTT[origZoneBG])]
                                origZoneNRM = terminalsNRM[origTT]
                                destZoneNRM = destZonesNRM[draw_choice(
                                    destProbs[destZoneBG][goodsTypeBG - 1])]
                                fromDC, toDC = 0, 0
                                fromTT, toTT = 1, 0
                            else:
                                ft = 10

                    # Flows to/from/between external zones: to/from consumer
                    if ft == 10:
                        origZoneNRM = origZonesNRM[draw_choice(
                            origProbs[origZoneBG][goodsTypeBG - 1])]
                        destZoneNRM = destZonesNRM[draw_choice(
                            destProbs[destZoneBG][goodsTypeBG - 1])]
                        fromDC, toDC = 0, 0
                        fromTT, toTT = 0, 0

                    # TODO: Select parcel nodes for logistic segment 6

                    # Replacement value for origDC/TT and destDC/TT if the
                    # flow doesn't go to or come from a DC/TT
                    if fromDC == 0:
                        origDC = -9999
                    if fromTT == 0:
                        origTT = -9999
                    if toDC == 0:
                        destDC = -9999
                    if toTT == 0:
                        destTT = -9999

                    # Selecting the logit parameters for this
                    # BasGoed goods type
                    paramsShipSizeVehTypeGG = paramsShipSizeVehType.iloc[
                        :, goodsTypeBG - 1]
                    B_TransportCosts = paramsShipSizeVehTypeGG[
                        'B_TransportCosts']
                    B_InventoryCosts = paramsShipSizeVehTypeGG[
                        'B_InventoryCosts']
                    B_FromDC = paramsShipSizeVehTypeGG[
                        'B_FromDC']
                    B_ToDC = paramsShipSizeVehTypeGG[
                        'B_ToDC']
                    B_LongHaul_TruckTrailer = paramsShipSizeVehTypeGG[
                        'B_LongHaul_TruckTrailer']
                    B_LongHaul_TractorTrailer = paramsShipSizeVehTypeGG[
                        'B_LongHaul_TractorTrailer']

                    ASC_VT = np.array([
                        paramsShipSizeVehTypeGG[f'ASC_VT_{i+1}']
                        for i in range(nVT)], dtype=float)

                    # Determine values for attributes in the
                    # utility function of the shipment size/vehicle type MNL
                    travTime = skimTravTime[
                        (origZoneNRM - 1) * nZones + (destZoneNRM - 1)] / 3600
                    distance = skimDistance[
                        (origZoneNRM - 1) * nZones + (destZoneNRM - 1)] / 1000

                    # Draw one choice for shipment size and vehicle type
                    ssvt = choice_model_ssvt(
                        costFreightKm, costFreightHr,
                        goodsTypeBG, fromDC, toDC,
                        travTime, distance,
                        absoluteShipmentSizes, truckCapacities,
                        B_TransportCosts, B_InventoryCosts, B_FromDC, B_ToDC,
                        B_LongHaul_TruckTrailer, B_LongHaul_TractorTrailer,
                        ASC_VT,
                        nVT, nSS)

                    # The chosen shipment size category
                    ssChosen = int(np.floor(ssvt / nVT))
                    shipmentSizeCat = ssChosen
                    shipmentSize = min(
                        absoluteShipmentSizes[ssChosen],
                        totalWeightCurrentCell - allocatedWeightCurrentCell)

                    # The chosen vehicle type
                    vehicleType = ssvt - ssChosen * nVT

                    # Store the shipment attributes in dictionaries
                    ship_origZoneBG[count] = origZoneBG
                    ship_destZoneBG[count] = destZoneBG
                    ship_origZoneNRM[count] = origZoneNRM
                    ship_destZoneNRM[count] = destZoneNRM
                    ship_origDC[count] = origDC
                    ship_destDC[count] = destDC
                    ship_origTT[count] = origTT
                    ship_destTT[count] = destTT
                    ship_goodsTypeBG[count] = goodsTypeBG
                    ship_goodsTypeNSTR[count] = nstr
                    ship_logisticSegment[count] = ls
                    ship_flowType[count] = ft
                    ship_shipmentSizeCat[count] = shipmentSizeCat
                    ship_shipmentSize[count] = shipmentSize
                    ship_vehicleType[count] = vehicleType
                    ship_containerized[count] = 0

                    # Origin coordinates
                    if fromDC == 0:
                        if fromTT == 0:
                            ship_origX[count] = zonesX[origZoneNRM - 1]
                            ship_origY[count] = zonesY[origZoneNRM - 1]
                        else:
                            ship_origX[count] = terminalsX[origTT]
                            ship_origY[count] = terminalsY[origTT]
                    else:
                        ship_origX[count] = logNodesX[origDC]
                        ship_origY[count] = logNodesY[origDC]

                    # Destination coordinates
                    if toDC == 0:
                        if toTT == 0:
                            ship_destX[count] = zonesX[destZoneNRM - 1]
                            ship_destY[count] = zonesY[destZoneNRM - 1]
                        else:
                            ship_destX[count] = terminalsX[destTT]
                            ship_destY[count] = terminalsY[destTT]
                    else:
                        ship_destX[count] = logNodesX[destDC]
                        ship_destY[count] = logNodesY[destDC]

                    # Update allocated weight in for current cell
                    allocatedWeightCurrentCell += ship_shipmentSize[count]
                    allocatedWeight += ship_shipmentSize[count]

                    count += 1

                    if count % 500 == 0:
                        progress = round(
                            100 * allocatedWeight / totalWeight, 1)
                        print('\t' + str(progress) + '%', end='\r')
                        if root != '':
                            root.progressBar['value'] = (
                                percStart + (
                                    (percEnd - percStart) *
                                    (allocatedWeight / totalWeight)))
                            root.update_statusbar(
                                "Shipment Synthesizer: " +
                                f"Synthesizing shipments MS ({progress}%)")

            nShipsMS = count

            message = "Synthesizing shipments CKM..."
            print(message), log_file.write(message + "\n")
            percStart, percEnd = 80, 90
            if root != '':
                root.progressBar['value'] = percStart
                root.update_statusbar(
                    "Shipment Synthesizer: Synthesizing shipments CKM (0.0%)")

            # For progress bar
            totalWeight = np.sum(tonnesCKM['WeightDay'])
            allocatedWeight = 0

            for i in tonnesCKM.index:
                # BG-origin, -destination and -goods type of the current cell
                origZoneBG = tonnesCKM.at[i, 'ORIG']
                destZoneBG = tonnesCKM.at[i, 'DEST']
                goodsTypeBG = tonnesCKM.at[i, 'GG']
                ls = tonnesCKM.at[i, 'LS']

                # Total weight for current cell
                # (i.e. orig-dest-goods type combination)
                totalWeightCurrentCell = tonnesCKM.at[i, 'WeightDay']

                # Split weight by empty and loaded containers
                totalWeightCurrentCellLoaded = (
                    (1 - containerEmptyShare) * totalWeightCurrentCell)
                totalWeightCurrentCellEmpty = (
                    containerEmptyShare * totalWeightCurrentCell)

                # Determine number of shipments
                nShipsCurrentCellLoaded = int(np.round(
                    totalWeightCurrentCellLoaded /
                    containerSS[goodsTypeBG - 1]))
                nShipsCurrentCellLoaded = max(1, nShipsCurrentCellLoaded)

                nShipsCurrentCellEmpty = int(np.round(
                    totalWeightCurrentCellEmpty / containerEmptySS))

                nShipsCurrentCell = (
                    nShipsCurrentCellLoaded + nShipsCurrentCellEmpty)

                if nShipsCurrentCellLoaded > 0:
                    shipSizeCurrentCellLoaded = (
                        totalWeightCurrentCellLoaded / nShipsCurrentCellLoaded)
                else:
                    shipSizeCurrentCellLoaded = containerSS[goodsTypeBG - 1]

                if nShipsCurrentCellEmpty > 0:
                    shipSizeCurrentCellEmpty = (
                        totalWeightCurrentCellEmpty / nShipsCurrentCellEmpty)
                else:
                    shipSizeCurrentCellEmpty = containerEmptySS

                # Same origin BG-zone as previous loop?
                recalculateOrigZones = False
                if count == nShipsMS:
                    recalculateOrigZones = True
                elif origZoneBG != tonnesCKM.at[i - 1, 'ORIG']:
                    recalculateOrigZones = True

                # Same destination BG-zone as previous loop?
                recalculateDestZones = False
                if count == nShipsMS:
                    recalculateDestZones = True
                elif destZoneBG != tonnesCKM.at[i - 1, 'DEST']:
                    recalculateDestZones = True

                if recalculateOrigZones:
                    # Which NRM-zones fall under the current BG-origin-zone
                    origZonesNRM = dictBGtoNRM[origZoneBG]

                    # DC origins
                    if origZoneBG in BGzonesWithLogNode:
                        whereDCinOrig = np.where(logNodesBG == origZoneBG)[0]
                        DCsInOrig = True
                    else:
                        DCsInOrig = False

                    # TT origins
                    if origZoneBG in BGzonesWithTerminal:
                        whereTTinOrig = np.where(terminalsBG == origZoneBG)[0]
                        TTsInOrig = True
                    else:
                        TTsInOrig = False

                if recalculateDestZones:
                    # Which NRM-zones fall under the current
                    # BG-destination-zone
                    destZonesNRM = dictBGtoNRM[destZoneBG]

                    # DC destinations
                    if destZoneBG in BGzonesWithLogNode:
                        whereDCinDest = np.where(logNodesBG == destZoneBG)[0]
                        DCsInDest = True
                    else:
                        DCsInDest = False

                    # TT destinatons
                    if destZoneBG in BGzonesWithTerminal:
                        whereTTinDest = np.where(terminalsBG == destZoneBG)[0]
                        TTsInDest = True
                    else:
                        TTsInDest = False

                for ship in range(nShipsCurrentCell):

                    # Determine NSTR goods type
                    nstr = draw_choice(GGtoNSTRcont[goodsTypeBG - 1, :])

                    # Flow type to/from external zones (always to/from TT)
                    if origZoneBG > nZonesNL_BG or destZoneBG > nZonesNL_BG:
                        if (
                            origZoneBG > nZonesNL_BG and
                            destZoneBG > nZonesNL_BG
                        ):
                            ft = 10
                            fromExt, toExt = 1, 1

                        else:
                            ft = 12

                            if origZoneBG > nZonesNL_BG:
                                fromExt, toExt = 1, 0

                            if destZoneBG > nZonesNL_BG:
                                fromExt, toExt = 0, 1

                    # Determine flow type within study area (=NL)
                    else:
                        ft = 1 + draw_choice(ftSharesCont[:, ls])
                        fromExt = 0
                        toExt = 0

                    # Determine origin zone: Flows from DC
                    if ft in [x + 1 for x in ftFromDC]:
                        if DCsInOrig:
                            origDC = whereDCinOrig[draw_choice(
                                origProbsDC[origZoneBG])]
                            origZoneNRM = logNodesNRM[origDC]
                            fromDC = 1
                            fromTT = 0
                        else:
                            ft = {3: 1, 5: 2, 7: 4}[ft]

                    # Determine origin zone: Flows from TT
                    if ft in [x + 1 for x in ftFromTT]:
                        if TTsInOrig:
                            origTT = whereTTinOrig[draw_choice(
                                origProbsTT[origZoneBG])]
                            origZoneNRM = terminalsNRM[origTT]
                            fromDC = 0
                            fromTT = 1
                        else:
                            ft = {6: 1, 8: 2, 9: 4}[ft]

                    # Determine origin zone: Flows from consumer
                    if ft in (1, 2, 4):
                        origZoneNRM = origZonesNRM[draw_choice(
                            origProbs[origZoneBG][goodsTypeBG - 1])]
                        fromDC = 0
                        fromTT = 0

                    # Determine destination zone: Flows to DC
                    if ft in [x + 1 for x in ftToDC]:
                        if DCsInDest:
                            destDC = whereDCinDest[draw_choice(
                                destProbsDC[destZoneBG])]
                            destZoneNRM = logNodesNRM[destDC]
                            toDC = 1
                            toTT = 0
                        else:
                            ft = {2: 1, 5: 3, 8: 6}[ft]

                    # Determine destination zone: Flows to TT
                    if ft in [x + 1 for x in ftToTT]:
                        if TTsInDest:
                            destTT = whereTTinDest[draw_choice(
                                destProbsTT[destZoneBG])]
                            destZoneNRM = terminalsNRM[destTT]
                            toDC = 0
                            toTT = 1
                        else:
                            ft = {4: 1, 7: 3, 9: 6}[ft]

                    # Determine destination zone: Flows to consumer
                    if ft in (1, 3, 6):
                        destZoneNRM = destZonesNRM[draw_choice(
                            destProbs[destZoneBG][goodsTypeBG - 1])]
                        toDC = 0
                        toTT = 0

                    # Flows to/from/between external zones: to/from DC
                    if ft == 11:

                        if fromExt == 1:
                            if DCsInDest:
                                origZoneNRM = origZonesNRM[draw_choice(
                                    origProbs[origZoneBG][goodsTypeBG - 1])]
                                destDC = whereDCinDest[draw_choice(
                                    destProbsDC[destZoneBG])]
                                destZoneNRM = logNodesNRM[destDC]
                                fromDC, toDC = 0, 1
                                fromTT, toTT = 0, 0
                            else:
                                ft = 10

                        elif toExt == 1:
                            if DCsInOrig:
                                origDC = whereDCinOrig[draw_choice(
                                    origProbsDC[origZoneBG])]
                                origZoneNRM = logNodesNRM[origDC]
                                destZoneNRM = destZonesNRM[draw_choice(
                                    destProbs[destZoneBG][goodsTypeBG - 1])]
                                fromDC, toDC = 1, 0
                                fromTT, toTT = 0, 0
                            else:
                                ft = 10

                    # Flows to/from/between external zones: to/from TT
                    if ft == 12:

                        if fromExt == 1:
                            if TTsInDest:
                                origZoneNRM = origZonesNRM[draw_choice(
                                    origProbs[origZoneBG][goodsTypeBG - 1])]
                                destTT = whereTTinDest[draw_choice(
                                    destProbsTT[destZoneBG])]
                                destZoneNRM = terminalsNRM[destTT]
                                fromDC, toDC = 0, 0
                                fromTT, toTT = 0, 1
                            else:
                                ft = 10

                        elif toExt == 1:
                            if TTsInOrig:
                                origTT = whereTTinOrig[draw_choice(
                                    origProbsTT[origZoneBG])]
                                origZoneNRM = terminalsNRM[origTT]
                                destZoneNRM = destZonesNRM[draw_choice(
                                    destProbs[destZoneBG][goodsTypeBG - 1])]
                                fromDC, toDC = 0, 0
                                fromTT, toTT = 1, 0
                            else:
                                ft = 10

                    # Flows to/from/between external zones: to/from consumer
                    if ft == 10:
                        origZoneNRM = origZonesNRM[draw_choice(
                            origProbs[origZoneBG][goodsTypeBG - 1])]
                        destZoneNRM = destZonesNRM[draw_choice(
                            destProbs[destZoneBG][goodsTypeBG - 1])]
                        fromDC, toDC = 0, 0
                        fromTT, toTT = 0, 0

                    # Replacement value for origDC/TT and destDC/TT if the
                    # flow doesn't go to or come from a DC/TT
                    if fromDC == 0:
                        origDC = -9999
                    if fromTT == 0:
                        origTT = -9999
                    if toDC == 0:
                        destDC = -9999
                    if toTT == 0:
                        destTT = -9999

                    # The chosen shipment size category
                    ssChosen = int(np.floor(ssvt / nVT))
                    shipmentSizeCat = ssChosen
                    shipmentSize = min(
                        absoluteShipmentSizes[ssChosen],
                        totalWeightCurrentCell - allocatedWeightCurrentCell)

                    # The chosen vehicle type
                    # (always tractor+trailer for containerized shipments)
                    vehicleType = 5

                    # The shipment size
                    if ship < nShipsCurrentCellLoaded:
                        shipmentSize = shipSizeCurrentCellLoaded
                        gg = goodsTypeBG
                    else:
                        shipmentSize = shipSizeCurrentCellEmpty
                        gg = -1
                        nstr = -1

                    # Store the shipment attributes in dictionaries
                    ship_origZoneBG[count] = origZoneBG
                    ship_destZoneBG[count] = destZoneBG
                    ship_origZoneNRM[count] = origZoneNRM
                    ship_destZoneNRM[count] = destZoneNRM
                    ship_origDC[count] = origDC
                    ship_destDC[count] = destDC
                    ship_origTT[count] = origTT
                    ship_destTT[count] = destTT
                    ship_goodsTypeBG[count] = gg
                    ship_goodsTypeNSTR[count] = nstr
                    ship_logisticSegment[count] = ls
                    ship_flowType[count] = ft
                    ship_shipmentSizeCat[count] = -99999
                    ship_shipmentSize[count] = shipmentSize
                    ship_vehicleType[count] = vehicleType
                    ship_containerized[count] = 1

                    # Origin coordinates
                    if fromDC == 0:
                        if fromTT == 0:
                            ship_origX[count] = zonesX[origZoneNRM - 1]
                            ship_origY[count] = zonesY[origZoneNRM - 1]
                        else:
                            ship_origX[count] = terminalsX[origTT]
                            ship_origY[count] = terminalsY[origTT]
                    else:
                        ship_origX[count] = logNodesX[origDC]
                        ship_origY[count] = logNodesY[origDC]

                    # Destination coordinates
                    if toDC == 0:
                        if toTT == 0:
                            ship_destX[count] = zonesX[destZoneNRM - 1]
                            ship_destY[count] = zonesY[destZoneNRM - 1]
                        else:
                            ship_destX[count] = terminalsX[destTT]
                            ship_destY[count] = terminalsY[destTT]
                    else:
                        ship_destX[count] = logNodesX[destDC]
                        ship_destY[count] = logNodesY[destDC]

                    # Update allocated weight and counter
                    allocatedWeight += ship_shipmentSize[count]
                    count += 1

                    if count % 500 == 0:
                        progress = round(
                            100 * allocatedWeight / totalWeight, 1)
                        print('\t' + str(progress) + '%', end='\r')
                        if root != '':
                            root.progressBar['value'] = (
                                percStart + (
                                    (percEnd - percStart) *
                                    (allocatedWeight / totalWeight)))
                            root.update_statusbar(
                                "Shipment Synthesizer: " +
                                f"Synthesizing shipments CKM ({progress}%)")

            nShips = count

            print('\t100.0%', end='\r')
            if root != '':
                root.progressBar['value'] = percStart + (percEnd - percStart)
                root.update_statusbar(
                    "Shipment Synthesizer: " +
                    "Synthesizing shipments CKM (100.0%)")

            # Shipment attributes in a list instead of a dictionary
            origZoneBG = list(ship_origZoneBG.values())
            destZoneBG = list(ship_destZoneBG.values())
            origZoneNRM = list(ship_origZoneNRM.values())
            destZoneNRM = list(ship_destZoneNRM.values())
            origDC = list(ship_origDC.values())
            destDC = list(ship_destDC.values())
            origTT = list(ship_origTT.values())
            destTT = list(ship_destTT.values())
            goodsTypeBG = list(ship_goodsTypeBG.values())
            goodsTypeNSTR = list(ship_goodsTypeNSTR.values())
            logisticSegment = list(ship_logisticSegment.values())
            flowType = list(ship_flowType.values())
            shipmentSizeCat = list(ship_shipmentSizeCat.values())
            shipmentSize = list(ship_shipmentSize.values())
            vehicleType = list(ship_vehicleType.values())
            containerized = list(ship_containerized.values())

            shipCols = [
                "SHIP_ID",
                "ORIG_BG", "DEST_BG",
                "ORIG_NRM", "DEST_NRM",
                "ORIG_LMS", "DEST_LMS",
                "ORIG_DC", "DEST_DC",
                "ORIG_TT", "DEST_TT",
                "BG_GG", "NSTR",
                "LOGSEG", "FLOWTYPE",
                "WEIGHT_CAT", "WEIGHT",
                "VEHTYPE", "CONTAINER"]

            shipments = pd.DataFrame(np.zeros((nShips, len(shipCols))))
            shipments.columns = shipCols

            shipments['SHIP_ID'] = np.arange(nShips)
            shipments['ORIG_BG'] = origZoneBG
            shipments['DEST_BG'] = destZoneBG
            shipments['ORIG_NRM'] = origZoneNRM
            shipments['DEST_NRM'] = destZoneNRM
            shipments['ORIG_LMS'] = [zonesLMS[x - 1] for x in origZoneNRM]
            shipments['DEST_LMS'] = [zonesLMS[x - 1] for x in destZoneNRM]
            shipments['ORIG_DC'] = origDC
            shipments['DEST_DC'] = destDC
            shipments['ORIG_TT'] = origTT
            shipments['DEST_TT'] = destTT
            shipments['BG_GG'] = goodsTypeBG
            shipments['NSTR'] = goodsTypeNSTR
            shipments['LOGSEG'] = logisticSegment
            shipments['FLOWTYPE'] = flowType
            shipments['WEIGHT_CAT'] = shipmentSizeCat
            shipments['WEIGHT'] = shipmentSize
            shipments['VEHTYPE'] = vehicleType
            shipments['CONTAINER'] = containerized

        else:
            if root != '':
                root.progressBar['value'] = 80

            # Import the reference shipments
            shipments = pd.read_csv(varDict['SHIPMENTS_REF'], sep=',')
            nShips = len(shipments)

            if root != '':
                root.progressBar['value'] = 83

            # Also get the geometries
            temp, shipCoords = read_shape(
                varDict['SHIPMENTS_REF'][:-4] + '.shp',
                returnGeometry=True)

            shipCoords = [
                shipCoords[i]['coordinates'] for i in range(len(shipCoords))]
            ship_origX = dict(
                (i, shipCoords[i][0][0]) for i in range(len(shipCoords)))
            ship_origY = dict(
                (i, shipCoords[i][0][1]) for i in range(len(shipCoords)))
            ship_destX = dict(
                (i, shipCoords[i][1][0]) for i in range(len(shipCoords)))
            ship_destY = dict(
                (i, shipCoords[i][1][1]) for i in range(len(shipCoords)))

            del temp

            if root != '':
                root.progressBar['value'] = 93

        # Get the datatypes right
        intCols = [
            "SHIP_ID",
            "ORIG_BG", "DEST_BG",
            "ORIG_NRM", "DEST_NRM",
            "ORIG_LMS", "DEST_LMS",
            "ORIG_DC", "DEST_DC",
            "ORIG_TT", "DEST_TT",
            "BG_GG", "NSTR",
            "LOGSEG", "FLOWTYPE",
            "WEIGHT_CAT",
            "VEHTYPE", "CONTAINER"]
        floatCols = ['WEIGHT']

        shipments[intCols] = shipments[intCols].astype(int)
        shipments[floatCols] = shipments[floatCols].astype(float)

        # Redirect shipments via UCCs and change vehicle type
        if varDict['LABEL'] == 'UCC':
            if varDict['SHIPMENTS_REF'] == "":
                message = (
                    'Exporting REF shipments to ' +
                    varDict['OUTPUTFOLDER'] + "Shipments_REF.csv")
                print(message), log_file.write(message + "\n")

                if root != '':
                    root.progressBar['value'] = 93
                    root.update_statusbar(
                        "Shipment Synthesizer: " +
                        "Exporting REF shipments to CSV")

                shipments.to_csv(varDict['OUTPUTFOLDER'] + 'Shipments_REF.csv')

            message = "Redirecting shipments via UCC..."
            print(message), log_file.write(message + "\n")
            if root != '':
                root.progressBar['value'] = 94
                root.update_statusbar(
                    "Shipment Synthesizer: " +
                    "Redirecting shipments via UCC")

            shipments['FROM_UCC'] = 0
            shipments['TO_UCC'] = 0

            whereOrigZEZ = np.array([
                i for i in shipments[shipments['ORIG_BG'] <= nZonesNL_BG].index
                if shipments.at[i, 'ORIG_NRM'] in zezZones], dtype=int)
            whereDestZEZ = np.array([
                i for i in shipments[shipments['DEST_BG'] <= nZonesNL_BG].index
                if shipments.at[i, 'DEST_NRM'] in zezZones], dtype=int)
            setWhereOrigZEZ = set(whereOrigZEZ)
            setWhereDestZEZ = set(whereDestZEZ)

            whereBothZEZ = [
                i for i in shipments.index
                if i in setWhereOrigZEZ and i in setWhereDestZEZ]

            newShipments = pd.DataFrame(np.zeros(shipments.shape))
            newShipments.columns = shipments.columns
            newShipments[intCols] = newShipments[intCols].astype(int)
            newShipments[floatCols] = newShipments[floatCols].astype(float)

            count = 0

            for i in whereOrigZEZ:

                if i not in setWhereDestZEZ:
                    ls = int(shipments['LOGSEG'][i])

                    if probConsolidation[ls][0] > np.random.rand():
                        trueOrigin = int(shipments['ORIG_NRM'][i])
                        newOrigin = zezToUCC[trueOrigin]

                        # Redirect to UCC
                        shipments.at[i, 'ORIG_NRM'] = newOrigin
                        shipments.at[i, 'FROM_UCC'] = 1
                        ship_origX[i] = zonesX[newOrigin - 1]
                        ship_origY[i] = zonesY[newOrigin - 1]

                        # Add shipment from ZEZ to UCC
                        newShipments.loc[count, :] = list(
                            shipments.loc[i, :].copy())
                        newShipments.at[count, 'ORIG_NRM'] = trueOrigin
                        newShipments.at[count, 'DEST_NRM'] = newOrigin
                        newShipments.at[count, 'FROM_UCC'] = 0
                        newShipments.at[count, 'TO_UCC'] = 1
                        newShipments.at[count, 'VEHTYPE'] = vehUccToVeh[
                            draw_choice(sharesVehUCC[ls, :])]
                        ship_origX[nShips + count] = zonesX[trueOrigin - 1]
                        ship_origY[nShips + count] = zonesY[trueOrigin - 1]
                        ship_destX[nShips + count] = zonesX[newOrigin - 1]
                        ship_destY[nShips + count] = zonesY[newOrigin - 1]

                        count += 1

            for i in whereDestZEZ:

                if i not in setWhereOrigZEZ:
                    ls = int(shipments['LOGSEG'][i])

                    if probConsolidation[ls][0] > np.random.rand():
                        trueDest = int(shipments['DEST_NRM'][i])
                        newDest = zezToUCC[trueDest]

                        # Redirect to UCC
                        shipments.at[i, 'DEST_NRM'] = newDest
                        shipments.at[i, 'TO_UCC'] = 1
                        ship_destX[i] = zonesX[newDest - 1]
                        ship_destY[i] = zonesY[newDest - 1]

                        # Add shipment to ZEZ from UCC
                        newShipments.loc[count, :] = list(
                            shipments.loc[i, :].copy())
                        newShipments.at[count, 'ORIG_NRM'] = newDest
                        newShipments.at[count, 'DEST_NRM'] = trueDest
                        newShipments.at[count, 'FROM_UCC'] = 1
                        newShipments.at[count, 'TO_UCC'] = 0
                        newShipments.at[count, 'VEHTYPE'] = vehUccToVeh[
                            draw_choice(sharesVehUCC[ls, :])]
                        ship_origX[nShips + count] = zonesX[newDest - 1]
                        ship_origY[nShips + count] = zonesY[newDest - 1]
                        ship_destX[nShips + count] = zonesX[trueDest - 1]
                        ship_destY[nShips + count] = zonesY[trueDest - 1]

                        count += 1

            # Also change vehicle type and rerouting for shipments
            # that go from a ZEZ area to a ZEZ area
            for i in whereBothZEZ:
                ls = int(shipments['LOGSEG'][i])

                # Als het binnen dezelfde gemeente (i.e. dezelfde ZEZ) blijft,
                # dan hoeven we alleen maar het voertuigtype aan te passen
                # Assume dangerous goods keep the same vehicle type
                if (
                    zones['GEM2019'][shipments['ORIG_NRM'][i]] ==
                    zones['GEM2019'][shipments['DEST_NRM'][i]]
                ):
                    if ls != 7:
                        shipments.at[i, 'VEHTYPE'] = vehUccToVeh[
                            draw_choice(sharesVehUCC[ls, :])]

                # Als het van de ene ZEZ naar de andere ZEZ gaat,
                # maken we 3 legs: ZEZ1--> UCC1, UCC1-->UCC2, UCC2-->ZEZ2
                else:
                    if probConsolidation[ls][0] > np.random.rand():
                        trueOrigin = int(shipments['ORIG_NRM'][i])
                        trueDest = int(shipments['DEST_NRM'][i])
                        newOrigin = zezToUCC[trueOrigin]
                        newDest = zezToUCC[trueDest]

                        # Redirect to UCC
                        shipments.at[i, 'ORIG_NRM'] = newOrigin
                        shipments.at[i, 'FROM_UCC'] = 1
                        ship_origX[i] = zonesX[newOrigin - 1]
                        ship_origY[i] = zonesY[newOrigin - 1]

                        # Add shipment from ZEZ1 to UCC1
                        newShipments.loc[count, :] = list(
                            shipments.loc[i, :].copy())
                        newShipments.at[count, 'ORIG_NRM'] = trueOrigin
                        newShipments.at[count, 'DEST_NRM'] = newOrigin
                        newShipments.at[count, 'FROM_UCC'] = 0
                        newShipments.at[count, 'TO_UCC'] = 1
                        newShipments.at[count, 'VEHTYPE'] = vehUccToVeh[
                            draw_choice(sharesVehUCC[ls, :])]
                        ship_origX[nShips + count] = zonesX[trueOrigin - 1]
                        ship_origY[nShips + count] = zonesY[trueOrigin - 1]
                        ship_destX[nShips + count] = zonesX[newOrigin - 1]
                        ship_destY[nShips + count] = zonesY[newOrigin - 1]

                        count += 1

                        # Redirect to UCC
                        shipments.at[i, 'DEST_NRM'] = newDest
                        shipments.at[i, 'TO_UCC'] = 1
                        ship_destX[i] = zonesX[newDest - 1]
                        ship_destY[i] = zonesY[newDest - 1]

                        # Add shipment from UCC2 to ZEZ2
                        newShipments.loc[count, :] = list(
                            shipments.loc[i, :].copy())
                        newShipments.at[count, 'ORIG_NRM'] = newDest
                        newShipments.at[count, 'DEST_NRM'] = trueDest
                        newShipments.at[count, 'FROM_UCC'] = 1
                        newShipments.at[count, 'TO_UCC'] = 0
                        newShipments.at[count, 'VEHTYPE'] = vehUccToVeh[
                            draw_choice(sharesVehUCC[ls, :])]
                        ship_origX[nShips + count] = zonesX[newDest - 1]
                        ship_origY[nShips + count] = zonesY[newDest - 1]
                        ship_destX[nShips + count] = zonesX[trueDest - 1]
                        ship_destY[nShips + count] = zonesY[trueDest - 1]

                        count += 1

            newShipments = newShipments.iloc[np.arange(count), :]

            newShipments['ORIG_LMS'] = [
                zonesLMS[int(x - 1)]for x in newShipments['ORIG_NRM'].values]
            newShipments['DEST_LMS'] = [
                zonesLMS[int(x - 1)] for x in newShipments['DEST_NRM'].values]

            shipments = shipments.append(newShipments)
            shipments['ORIG_LMS'] = shipments['ORIG_LMS'].astype(int)
            shipments['DEST_LMS'] = shipments['DEST_LMS'].astype(int)
            nShips = len(shipments)
            shipments['SHIP_ID'] = np.arange(nShips)
            shipments.index = np.arange(nShips)

        message = (
            'Exporting ' +
            varDict['LABEL'] +
            ' shipments to ' +
            varDict['OUTPUTFOLDER'] + f"Shipments_{varDict['LABEL']}.csv")
        print(message), log_file.write(message + "\n")
        if root != '':
            root.progressBar['value'] = 94
            root.update_statusbar(
                "Shipment Synthesizer: " +
                "Exporting shipments to CSV")

        shipments.to_csv(
            varDict['OUTPUTFOLDER'] + f"Shipments_{varDict['LABEL']}.csv",
            index=False)

        # Export as shapefile
        print("Writing Shapefile..."), log_file.write("Writing Shapefile...\n")
        percStart, percEnd = 95, 100
        if root != '':
            root.progressBar['value'] = percStart
            root.update_statusbar(
                "Shipment Synthesizer: " +
                "Writing Shapefile")

        Ax = list(ship_origX.values())
        Ay = list(ship_origY.values())
        Bx = list(ship_destX.values())
        By = list(ship_destY.values())

        # Initialize shapefile fields
        filename = (
            varDict['OUTPUTFOLDER'] +
            f"Shipments_{varDict['LABEL']}.shp")
        w = shp.Writer(filename)
        w.field('SHIP_ID', 'N', size=6, decimal=0)
        w.field('ORIG_BG', 'N', size=3, decimal=0)
        w.field('DEST_BG', 'N', size=3, decimal=0)
        w.field('ORIG_NRM', 'N', size=4, decimal=0)
        w.field('DEST_NRM', 'N', size=4, decimal=0)
        w.field('ORIG_LMS', 'N', size=4, decimal=0)
        w.field('DEST_LMS', 'N', size=4, decimal=0)
        w.field('ORIG_DC', 'N', size=5, decimal=0)
        w.field('DEST_DC', 'N', size=5, decimal=0)
        w.field('ORIG_TT', 'N', size=5, decimal=0)
        w.field('DEST_TT', 'N', size=5, decimal=0)
        w.field('BG_GG', 'N', size=2, decimal=0)
        w.field('NSTR', 'N', size=2, decimal=0)
        w.field('LOGSEG', 'N', size=2, decimal=0)
        w.field('FLOWTYPE', 'N', size=2, decimal=0)
        w.field('WEIGHT_CAT', 'N', size=2, decimal=0)
        w.field('WEIGHT', 'N', size=4, decimal=2)
        w.field('VEHTYPE', 'N', size=2, decimal=0)
        w.field('CONTAINER', 'N', size=2, decimal=0)
        if varDict['LABEL'] == 'UCC':
            w.field('FROM_UCC', 'N', size=2, decimal=0)
            w.field('TO_UCC', 'N', size=2, decimal=0)

        dbfData = np.array(shipments, dtype=object)
        for i in range(nShips):
            # Add geometry
            w.line([[[Ax[i], Ay[i]],
                     [Bx[i], By[i]]]])

            # Add data fields
            w.record(*dbfData[i, :])

            if i % 500 == 0:
                print('\t' + str(round((i / nShips) * 100, 1)) + '%', end='\r')

                if root != '':
                    root.progressBar['value'] = (
                        percStart + (percEnd - percStart) * i / nShips)

        w.close()

        print('\t100.0%', end='\r')

        totaltime = round(time.time() - start_time, 2)
        log_file.write("Total runtime: %s seconds\n" % (totaltime))
        log_file.write(
            "End simulation at: " +
            datetime.datetime.now().strftime("%y-%m-%d %H:%M") + "\n")
        log_file.close()

        if root != '':
            root.update_statusbar("Shipment Synthesizer: Done")
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
                    "Shipment Synthesizer: Execution failed!")
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
