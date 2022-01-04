# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 09:33:15 2019

@author: STH
"""
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
from __functions__ import read_mtx, read_shape
import psutil

# Modules nodig voor de user interface
import tkinter as tk
from tkinter.ttk import Progressbar
import zlib
import base64
import tempfile
from threading import Thread


#%% Main

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
        self.width  = 500
        self.height = 60
        self.bg     = 'black'
        self.fg     = 'white'
        self.font   = 'Verdana'
        
        # Create a GUI window
        self.root = tk.Tk()
        self.root.title("Progress Emission Calculation")
        self.root.geometry(f'{self.width}x{self.height}+0+200')
        self.root.resizable(False, False)
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height, bg=self.bg)
        self.canvas.place(x=0, y=0)
        self.statusBar = tk.Label(self.root, text="", anchor='w', borderwidth=0, fg='black')
        self.statusBar.place(x=2, y=self.height-22, width=self.width, height=22)
        
        # Remove the default tkinter icon from the window
        icon = zlib.decompress(base64.b64decode('eJxjYGAEQgEBBiDJwZDBy''sAgxsDAoAHEQCEGBQaIOAg4sDIgACMUj4JRMApGwQgF/ykEAFXxQRc='))
        _, self.iconPath = tempfile.mkstemp()
        with open(self.iconPath, 'wb') as iconFile:
            iconFile.write(icon)
        self.root.iconbitmap(bitmap=self.iconPath)
        
        # Create a progress bar
        self.progressBar = Progressbar(self.root, length=self.width-20)
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



    def error_screen(self, text='', event=None, size=[800,50], title='Error message'):
        '''
        Pop up a window with an error message
        '''
        windowError = tk.Toplevel(self.root)
        windowError.title(title)
        windowError.geometry(f'{size[0]}x{size[1]}+0+{200+50+self.height}')
        windowError.minsize(width=size[0], height=size[1])
        windowError.iconbitmap(default=self.iconPath)
        labelError = tk.Label(windowError, text=text, anchor='w', justify='left')
        labelError.place(x=10, y=10)  
        
        

    def run_module(self, event=None):
        Thread(target=actually_run_module, args=self.args, daemon=True).start()



def actually_run_module(args):
    '''
    Emission Calculation: Main body of the script where all calculations are performed. 
    '''
    try:      
        
        start_time = time.time()
        
        root    = args[0]
        varDict = args[1]
        
        if root != '':
            root.progressBar['value'] = 0
                
        # Define folders relative to current datapath
        datapathO = varDict['OUTPUTFOLDER']
        datapathP = varDict['PARAMFOLDER']
        zonesPath   = varDict['ZONES']
        costMSpath  = varDict['COST_ROAD_MS']
        costCKMpath = varDict['COST_ROAD_CKM']
        costVanPath = varDict['COST_BESTEL']
        skimDistancePath = varDict['SKIMDISTANCE']
        vehCapacityPath  = varDict['VEHCAPACITY']
        emBuitenwegLeegPath = varDict['EM_BUITENWEG_LEEG']
        emBuitenwegVolPath = varDict['EM_BUITENWEG_VOL']
        emSnelwegLeegPath = varDict['EM_SNELWEG_LEEG']
        emSnelwegVolPath = varDict['EM_SNELWEG_VOL']
        emStadLeegPath = varDict['EM_STAD_LEEG']
        emStadVolPath = varDict['EM_STAD_VOL']
        label = varDict['LABEL']
        nCPU  = varDict['N_CPU']

        seed = varDict['SEED']

        exportShp  = True
        
        log_file = open(datapathO + "Logfile_EmissionCalculation.log", 'w')
        log_file.write("Start simulation at: "+datetime.datetime.now().strftime("%y-%m-%d %H:%M")+"\n")

        if seed != '':
            np.random.seed(seed)

        # To convert emissions to kilograms
        emissionDivFac = [1000, 1000000, 1000000, 1000] 
        etDict    = {0:'CO2', 1:'SO2', 2:'PM', 3:'NOX'}
        etInvDict = {'CO2':0, 'SO2':1, 'PM':2, 'NOX':3}
        indexCO2 = etInvDict['CO2']
        indexSO2 = etInvDict['SO2']
        indexPM  = etInvDict['PM' ]
        indexNOX = etInvDict['NOX']
            
        nLS = 8 + 1 # Number of logistic segments (+ parcel module)
        nET = 4     # Number of emission types
        nVT = 11
        
        # Which vehicle type can be used in the parcel module        
        vehTypesParcels = [8, 9]
        
        # Carrying capacity in kg
        carryingCapacity = np.array(pd.read_csv(vehCapacityPath, index_col='Vehicle Type'))
        
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

        if nCPU not in ['', '""', "''"]:
            try:
                nCPU = int(nCPU)

                if nCPU > mp.cpu_count():
                    nCPU = max(1, min(mp.cpu_count() - 1, maxCPU))
                    text = ('N_CPU parameter too high. Only ' + str(mp.cpu_count()) +
                            ' CPUs available. Defaulting to ' + str(nCPU) + 'CPUs.')
                    print(text)
                    log_file.write(text + '\n')

                if nCPU < 1:
                    nCPU = max(1, min(mp.cpu_count() - 1, maxCPU))

            except:
                nCPU = max(1, min(mp.cpu_count() - 1, maxCPU))
                text = 'Could not convert N_CPU parameter to an integer. Defaulting to ' + str(nCPU) + 'CPUs.'
                print(text)
                log_file.write(text + '\n')

        else:
            nCPU = max(1,min(mp.cpu_count()-1,maxCPU))
        
        # Aantal routes waarover gespreid wordt per HB
        nMultiRoute = 1

        if root != '':
            root.progressBar['value'] = 0.5
            
        # ------------------- Importing and preprocessing network ---------------------------------------
        print("Importing and preprocessing network...")
        log_file.write("Importing and preprocessing network...\n")
        
        print('\tReading zones...')
        zones = read_shape(zonesPath)
        zones = zones[['SEGNR_2018','N','O','W','Z', 'XCOORD','YCOORD']]
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
        nodeDict    = dict((i, nodes.iloc[i,0]) for i in range(nNodes))
        invNodeDict = dict((nodes.iloc[i,0], i) for i in range(nNodes))
        
        nodes['NODENR' ] = [invNodeDict[x] for x in nodes['NODENR'].values]
        nodes.index = np.arange(nNodes)
        
        links['KNOOP_A'] = [invNodeDict[x] for x in links['KNOOP_A'].values]
        links['KNOOP_B'] = [invNodeDict[x] for x in links['KNOOP_B'].values]

        # Van meter naar kilometer
        links['DISTANCE'] /= 1000
        
        # Van seconde naar uur
        links['T0_FREIGHT'] /= 3600
        links['T0_VAN'    ] /= 3600
        
        if root != '':
            root.progressBar['value'] = 3.0
            
        print('\tReading other input files...')

        # Cost parameters freight
        costFreightMS  = np.array(pd.read_csv(costMSpath, sep='\t'))[:,[0,2,3]]
        costFreightCKM = np.array(pd.read_csv(costCKMpath, sep='\t'))[:,[0,1,2]]
        shipments = pd.read_csv(datapathO + 'Shipments_' + label + '.csv')
        ggSharesMS  = pd.pivot_table(shipments[shipments['CONTAINER']==0], values='WEIGHT', index='BG_GG', aggfunc=sum)
        ggSharesCKM = pd.pivot_table(shipments[shipments['CONTAINER']==1], values='WEIGHT', index='BG_GG', aggfunc=sum)
        sumWeight = (np.sum(ggSharesMS) + np.sum(ggSharesCKM))
        
        ggSharesMS.index  = [int(x) for x in ggSharesMS.index]
        ggSharesCKM.index = [int(x) for x in ggSharesCKM.index]

        del shipments
        
        # Calculate average cost weighted by weight per goods type and containerized/non-containerized
        costPerKmFreight = 0.0
        costPerHrFreight = 0.0

        for gg in costFreightMS[:,0]:
            if int(gg) in ggSharesMS.index:
                row = np.where(costFreightMS[:,0] == gg)[0][0]
                costPerKmFreight += (costFreightMS[row,1] * ggSharesMS.at[int(gg), 'WEIGHT'])
                costPerHrFreight += (costFreightMS[row,2] * ggSharesMS.at[int(gg), 'WEIGHT'])
                
        for gg in costFreightCKM[:,0]:
            if int(gg) in ggSharesCKM.index:
                row = np.where(costFreightCKM[:,0] == gg)[0][0]
                costPerKmFreight += (costFreightCKM[row,1] * ggSharesCKM.at[int(gg), 'WEIGHT'])
                costPerHrFreight += (costFreightCKM[row,2] * ggSharesCKM.at[int(gg), 'WEIGHT'])
        
        costPerKmFreight /= sumWeight
        costPerHrFreight /= sumWeight
        costPerKmFreight = float(costPerKmFreight)
        costPerHrFreight = float(costPerHrFreight)      
        
        # Cost parameters van
        costVan  = np.array(pd.read_csv(costVanPath, sep='\t'))[:,[0,1,2]]
        costPerKmVan = np.average(costVan[:,1])
        costPerHrVan = np.average(costVan[:,2])
        
        maxNumConnections = 8
        linkDict = -1 * np.ones((max(links['KNOOP_A']) + 1, 2 * maxNumConnections), dtype=int)
        for i in links.index:
            aNode = links['KNOOP_A'][i]
            bNode = links['KNOOP_B'][i]
            
            for col in range(maxNumConnections):
                if linkDict[aNode][col] == -1:
                    linkDict[aNode][col] = bNode
                    linkDict[aNode][col + maxNumConnections] = i
                    break
        
        # Travel times and travel costs
        links['COST_FREIGHT'] = costPerKmFreight * links['DISTANCE'] + costPerHrFreight * links['T0_FREIGHT']
        links['COST_VAN'    ] = costPerKmVan     * links['DISTANCE'] + costPerHrVan     * links['T0_VAN']
        
        # Set connector travel costs high so these are not chosen other than for entering/leaving network
        links.loc[links['LINKTYPE']==99, 'COST_FREIGHT'] = 10000
        links.loc[links['LINKTYPE']==99, 'COST_VAN'    ] = 10000
        
        # Set travel costs for forbidden-for-freight-links high so these are not chosen for freight
        links.loc[links['DOELSTROOK']==2, 'COST_FREIGHT'] = 10000
        
        # Set travel costs for only-freight-links high so these are not chosen for vans
        links.loc[links['DOELSTROOK']==3, 'COST_VAN'] = 10000
        
        # Set travel times on links in ZEZ Rotterdam high so these are only used to go to UCC and not for through traffic
        costFreightHybrid = links['COST_FREIGHT'].copy()
        costVanHybrid = links['COST_VAN'].copy()
        if label == 'UCC':
            links.loc[links['ZEZ']==1, 'COST_FREIGHT'] += 10000
            links.loc[links['ZEZ']==1, 'COST_VAN'] += 10000
            
        # Initialize empty fields with emissions and traffic intensity per link (also save list with all field names)
        volCols = ['N_LS0','N_LS1','N_LS2','N_LS3','N_LS4','N_LS5','N_LS6','N_LS7','N_LS8',
                   'N_VAN_S','N_VAN_C',
                   'N_VEH0','N_VEH1','N_VEH2','N_VEH3',
                   'N_VEH4','N_VEH5','N_VEH6','N_VEH7',
                   'N_VEH8','N_VEH9','N_VEH10',
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
            
            
        # ----------------- Information for the emission calculations ------------------------------------
        # Lees emissiefactoren in (kolom 0=CO2, 1=SO2, 2=PM, 3=NOX)
        emissionsBuitenwegLeeg  = np.array(pd.read_csv(emBuitenwegLeegPath, index_col='Voertuigtype'))
        emissionsBuitenwegVol   = np.array(pd.read_csv(emBuitenwegVolPath, index_col='Voertuigtype'))
        emissionsSnelwegLeeg    = np.array(pd.read_csv(emSnelwegLeegPath, index_col='Voertuigtype'))
        emissionsSnelwegVol     = np.array(pd.read_csv(emSnelwegVolPath, index_col='Voertuigtype'))
        emissionsStadLeeg       = np.array(pd.read_csv(emStadLeegPath, index_col='Voertuigtype'))
        emissionsStadVol        = np.array(pd.read_csv(emStadVolPath, index_col='Voertuigtype'))
        
        # Average of small and large tractor+trailer
        emissionsBuitenwegLeeg[7,:] = (emissionsBuitenwegLeeg[7,:]  + emissionsBuitenwegLeeg[8,:]) / 2
        emissionsBuitenwegVol[ 7,:] = (emissionsBuitenwegVol[ 7,:]  + emissionsBuitenwegVol[ 8,:]) / 2
        emissionsSnelwegLeeg[  7,:] = (emissionsSnelwegLeeg[  7,:]  + emissionsSnelwegLeeg[  8,:]) / 2
        emissionsSnelwegVol[   7,:] = (emissionsSnelwegVol[   7,:]  + emissionsSnelwegVol[   8,:]) / 2
        emissionsStadLeeg[     7,:] = (emissionsStadLeeg[     7,:]  + emissionsStadLeeg[     8,:]) / 2
        emissionsStadVol[      7,:] = (emissionsStadVol[      7,:]  + emissionsStadVol[      8,:]) / 2

        # Average of small and large van
        emissionsBuitenwegLeeg[0,:] = (emissionsBuitenwegLeeg[0,:]  + emissionsBuitenwegLeeg[1,:]) / 2
        emissionsBuitenwegVol[ 0,:] = (emissionsBuitenwegVol[ 0,:]  + emissionsBuitenwegVol[ 1,:]) / 2
        emissionsSnelwegLeeg[  0,:] = (emissionsSnelwegLeeg[  0,:]  + emissionsSnelwegLeeg[  1,:]) / 2
        emissionsSnelwegVol[   0,:] = (emissionsSnelwegVol[   0,:]  + emissionsSnelwegVol[   1,:]) / 2
        emissionsStadLeeg[     0,:] = (emissionsStadLeeg[     0,:]  + emissionsStadLeeg[     1,:]) / 2
        emissionsStadVol[      0,:] = (emissionsStadVol[      0,:]  + emissionsStadVol[      1,:]) / 2
        
        # To which vehicle type in the emission factors (value) does each of our vehicle types (key) belong
        vtDict = {0:2, 1:3, 2:5, 3:4, 4:6, 5:7, 6:9, 7:9, 8:0, 9:0, 10:0}                
            
        
        # -------------------- Importing freight and parcel trips ---------------------------------------        
        print('\tReading freight and parcel trips...')
                
        # Import trips csv
        allTrips = pd.read_csv(datapathO + "Tours_" + label + ".csv")
        allTrips.loc[allTrips['TRIP_DEPTIME']>=24,'TRIP_DEPTIME'] -= 24
        allTrips.loc[allTrips['TRIP_DEPTIME']>=24,'TRIP_DEPTIME'] -= 24
        capUt = ((allTrips['TRIP_WEIGHT'] * 1000) /
                 carryingCapacity[np.array(allTrips['VEHTYPE'], dtype=int)][:,0])
        allTrips['CAP_UT'] = capUt
        allTrips['INDEX' ] = allTrips.index
        allTrips = allTrips[['CARRIER_ID','ORIG_NRM','DEST_NRM',
                             'VEHTYPE','CAP_UT','LOG_SEG','COMBTYPE','INDEX']]
        
        # Import parcel schedule csv         
        allParcelTrips = pd.read_csv(datapathO + "ParcelSchedule_" + label + ".csv")
        allParcelTrips = allParcelTrips.rename(columns={'ORIG_NRM':'ORIG', 
                                                        'DEST_NRM':'DEST'})
        allParcelTrips.loc[allParcelTrips['TRIP_DEPTIME']>=24,'TRIP_DEPTIME'] -= 24
        allParcelTrips.loc[allParcelTrips['TRIP_DEPTIME']>=24,'TRIP_DEPTIME'] -= 24
        
        # Assume on average 50% of loading capacity used for parcel deliveries
        allParcelTrips['CAP_UT'  ] = 0.5
        
        # Recode vehicle type from string to number
        allParcelTrips['VEHTYPE' ] = [{'Van':8, 'LEVV':9}[vt] for vt in allParcelTrips['VEHTYPE']]
        
        # Fuel as basic combustion type
        allParcelTrips['COMBTYPE'] = 0 
        
        # Trips coming from UCC to ZEZ use electric
        allParcelTrips.loc[allParcelTrips['ORIGTYPE']=='UCC', 'COMBTYPE'] = 1
        allParcelTrips['LOG_SEG' ] = 6
        allParcelTrips['INDEX'   ] = allParcelTrips.index
        allParcelTrips = allParcelTrips[['DEPOT_ID', 'ORIG',   'DEST',
                                         'VEHTYPE',  'CAP_UT', 'LOG_SEG',
                                         'COMBTYPE', 'INDEX']]
        
        # Determine linktypes (urban/rural/highway)
        stadLinkTypes      = [6, 7]
        buitenwegLinkTypes = [3, 4, 5]
        snelwegLinkTypes   = [1, 2]
        
        whereStad      = [links['LINKTYPE'][i] in stadLinkTypes      for i in links.index]
        whereBuitenweg = [links['LINKTYPE'][i] in buitenwegLinkTypes for i in links.index]
        whereSnelweg   = [links['LINKTYPE'][i] in snelwegLinkTypes   for i in links.index]      
       
        roadtypeArray = np.zeros((len(links)))
        roadtypeArray[whereStad     ] = 1
        roadtypeArray[whereBuitenweg] = 2
        roadtypeArray[whereSnelweg  ] = 3
        stadArray      = (roadtypeArray==1)
        buitenwegArray = (roadtypeArray==2)
        snelwegArray   = (roadtypeArray==3)
        distArray = np.array(links['DISTANCE'])
        ZEZarray  = np.array(links['ZEZ']==1, dtype=int)
        NLarray   = np.array(links['NL' ]==1, dtype=int)
                
        # Bring ORIG and DEST to the front of the list of column names
        newColOrder = volCols.copy()
        newColOrder.insert(0,'DEST_NRM')
        newColOrder.insert(0,'ORIG_NRM')
        
        # Trip matrices per time-of-day
        tripMatrix = pd.read_csv(datapathO + f'TripMatrix_Freight_NRM_{label}.csv', sep=',')
        tripMatrix['N_LS8'  ] = 0
        tripMatrix['N_VAN_S'] = 0
        tripMatrix['N_VAN_C'] = 0
        tripMatrix['N_LS8_VEH8'] = 0
        tripMatrix['N_LS8_VEH9'] = 0
        tripMatrix = tripMatrix[newColOrder]            
        tripMatrix = np.array(tripMatrix)
            
        # Parcels trip matrices per time-of-day
        tripMatrixParcels = pd.read_csv(datapathO + f'TripMatrix_ParcelVans_NRM_{label}.csv', sep=',')
        tripMatrixParcels = np.array(tripMatrixParcels)
        
        # For which origin zones do we need to find the routes
        origSelection  = np.arange(nZones)
        nOrigSelection = len(origSelection)
                                   
        # Initialize arrays for intensities and emissions
        linkTripsArray    = np.zeros((nLinks,len(volCols)))
        linkVanTripsArray = np.zeros((nLinks, 2))
        linkEmissionsArray    = [np.zeros((nLinks, nET)) for ls in range(nLS)]
        linkVanEmissionsArray = [np.zeros((nLinks, nET)) for ls in ['VAN_S','VAN_C']]
       
        if root != '':
            root.progressBar['value'] = 5.0

        # ----------------------- Freight ----------------------------------------

        tripMatrixOrigins = set(tripMatrix[:, 0])        
        trips = np.array(allTrips[allTrips['VEHTYPE']!=8])

        if nMultiRoute >= 2:
            np.random.seed(100)
            whereTripsByIter = [[] for r in range(nMultiRoute)]
            for i in range(len(trips)):
                whereTripsByIter[np.random.randint(nMultiRoute)].append(i)
        elif nMultiRoute == 1:
            whereTripsByIter = [np.arange(len(trips))]
        else:
            raise Exception('nMultiRoute should be >= 1, now it is ' + str(nMultiRoute))

        # Vaste seed voor spreiding reiskosten op links
        if nMultiRoute >= 2:
            linksRandArray = [[] for r in range(nMultiRoute)]
            linksA = np.array(links['KNOOP_A'])
            linksB = np.array(links['KNOOP_B'])
            for i in range(len(links)):
                np.random.seed(linksA[i] + linksB[i])
                for r in range(nMultiRoute):
                    linksRandArray[r].append(np.random.rand())

        if seed != '':
            np.random.seed(seed)

        if root != '':
            root.progressBar['value'] = 6.0

        tripsCO2       = {}
        tripsCO2_NL    = {}
        tripsDist      = {}
        tripsDist_NL   = {}
        parcelTripsCO2 = {}
        
        # From which nodes do we need to perform the shortest path algoritm
        indices = origSelection

        # From which nodes does every CPU perform the shortest path algorithm
        indicesPerCPU       = [[indices[cpu::nCPU], cpu]            for cpu in range(nCPU)]
        origSelectionPerCPU = [np.arange(nOrigSelection)[cpu::nCPU] for cpu in range(nCPU)]

        # Whether a separate route search needs to be done for hybrid vehicles or not
        doHybridRoutes = np.any(trips[:, 6] == 3)
        
        for r in range(nMultiRoute):

            # ----------------------- Route search (freight) ----------------------------------------

            print(f"Route search (freight part {r + 1})...")
            log_file.write(f"Route search (freight part {r + 1})...\n")   
                
            # Route search freight
            if nCPU > 1:    
                
                # The network with costs between nodes (freight)
                csgraphFreight = lil_matrix((nNodes, nNodes))
                tmp = np.array(links['COST_FREIGHT'])
                if nMultiRoute > 1:
                    tmp *= 0.95 + 0.1 * linksRandArray[r]
                csgraphFreight[np.array(links['KNOOP_A']), np.array(links['KNOOP_B'])] = tmp
                
                # Initialize a pool object that spreads tasks over different CPUs
                p = mp.Pool(nCPU)
                
                # Execute the Dijkstra route search
                prevFreightPerCPU = p.map(functools.partial(get_prev, csgraphFreight, nNodes), indicesPerCPU)
    
                # Wait for completion of processes
                p.close()
                p.join()
                
                # Combine the results from the different CPUs
                # Matrix with for each node the previous node on the shortest path
                prevFreight = np.zeros((nOrigSelection, nNodes), dtype=int)
                for cpu in range(nCPU):
                    for i in range(len(indicesPerCPU[cpu][0])):
                        prevFreight[origSelectionPerCPU[cpu][i], :] = prevFreightPerCPU[cpu][i, :]
    
                # Make some space available on the RAM
                del prevFreightPerCPU
                
            else:
    
                # The network with costs between nodes (freight)
                csgraphFreight = lil_matrix((nNodes, nNodes))
                tmp = np.array(links['COST_FREIGHT'])
                if nMultiRoute > 1:
                    tmp *= 0.95 + 0.1 * linksRandArray[r]
                csgraphFreight[np.array(links['KNOOP_A']), np.array(links['KNOOP_B'])] = tmp
                
                # Execute the Dijkstra route search
                prevFreight = get_prev(csgraphFreight, nNodes, [indices, 0])
    
            if root != '':
                root.progressBar['value'] = 8.0 + (r + 1) * 10.0 + r * 10.0

            # Make some space available on the RAM
            del csgraphFreight

            # Route search freight (for clean vehicles in the UCC scenario)
            if doHybridRoutes:
                print(f"Route search (freight with hybrid combustion part {r + 1})...")
                log_file.write(f"Route search (freight with hybrid combustion part {r + 1})...\n")  

                # Route search freight
                if nCPU > 1:    
                    
                    # The network with costs between nodes (freight)
                    csgraphFreightHybrid = lil_matrix((nNodes, nNodes))
                    tmp = costFreightHybrid
                    if nMultiRoute > 1:
                        tmp *= 0.95 + 0.1 * linksRandArray[r]
                    csgraphFreightHybrid[np.array(links['KNOOP_A']), np.array(links['KNOOP_B'])] = tmp
                    
                    # Initialize a pool object that spreads tasks over different CPUs
                    p = mp.Pool(nCPU)
                    
                    # Execute the Dijkstra route search
                    prevFreightHybridPerCPU = p.map(functools.partial(get_prev, csgraphFreightHybrid, nNodes), indicesPerCPU)
        
                    # Wait for completion of processes
                    p.close()
                    p.join()
                    
                    # Combine the results from the different CPUs
                    # Matrix with for each node the previous node on the shortest path
                    prevFreightHybrid = np.zeros((nOrigSelection, nNodes), dtype=int)
                    for cpu in range(nCPU):
                        for i in range(len(indicesPerCPU[cpu][0])):
                            prevFreightHybrid[origSelectionPerCPU[cpu][i], :] = prevFreightHybridPerCPU[cpu][i, :]
        
                    # Make some space available on the RAM
                    del prevFreightHybridPerCPU
                    
                else:
        
                    # The network with costs between nodes (freight)
                    csgraphFreightHybrid = lil_matrix((nNodes, nNodes))
                    tmp = costFreightHybrid
                    if nMultiRoute > 1:
                        tmp *= 0.95 + 0.1 * linksRandArray[r]
                    csgraphFreightHybrid[np.array(links['KNOOP_A']), np.array(links['KNOOP_B'])] = tmp
                    
                    # Execute the Dijkstra route search
                    prevFreightHybrid = get_prev(csgraphFreightHybrid, nNodes, [indices, 0])
        
                if root != '':
                    root.progressBar['value'] = 8.0 + (r + 1) * 10.0 + r * 10.0
    
                # Make some space available on the RAM
                del csgraphFreightHybrid

            # --------------------- Emissions and intensities (freight) --------------------------------------     
        
            print(f"Calculating emissions and traffic intensities (freight part {r + 1})")
            log_file.write(f"Calculating emissions and traffic intensities (freight part {r + 1})\n")
            
            iterTrips = trips[whereTripsByIter[r], :]
    
            whereODL = {}
            for i in range(len(tripMatrix)):
                orig = tripMatrix[i, 0]
                dest = tripMatrix[i, 1]
                for ls in range(nLS):
                    whereODL[(orig,dest,ls)] = []
            for i in range(len(iterTrips)):
                orig = iterTrips[i, 1]
                dest = iterTrips[i, 2]
                ls   = iterTrips[i, 5]
                whereODL[(orig,dest,ls)].append(i)
    
            for i in range(nOrigSelection):
                origZone = i + 1              
                
                print('\tOrigin ' + str(origZone), end='\r')
                
                if origZone in tripMatrixOrigins:
                    destZoneIndex = np.where(tripMatrix[:,0] == origZone)[0]
                    destZones = tripMatrix[destZoneIndex, 1]
                    
                    # Regular routes
                    routes = [get_route(i, destZone-1, prevFreight, linkDict)
                              for destZone in destZones]
                    
                    # Routes for hybrid vehicles (no need for penalty on ZEZ-links here)
                    if doHybridRoutes:
                        hybridRoutes = [get_route(i, destZone-1, prevFreightHybrid, linkDict)
                                        for destZone in destZones]

                    # Schrijf de volumes op de links
                    for j in range(len(destZones)):
                        destZone = destZones[j]
                        route = routes[j]
                        
                        # Get route and part of route that is stad/buitenweg/snelweg and ZEZ/non-ZEZ
                        routeStad      = route[stadArray[route]]
                        routeBuitenweg = route[buitenwegArray[route]]
                        routeSnelweg   = route[snelwegArray[route]]
                        
                        # Het gedeelte van de route in NL
                        routeStadNL      = np.where(NLarray[routeStad     ] == 1)[0]
                        routeBuitenwegNL = np.where(NLarray[routeBuitenweg] == 1)[0]
                        routeSnelwegNL   = np.where(NLarray[routeSnelweg  ] == 1)[0]
    
                        distStad      = distArray[routeStad]
                        distBuitenweg = distArray[routeBuitenweg]
                        distSnelweg   = distArray[routeSnelweg]
                        distTotal   = np.sum(distArray[route])
                        distTotalNL = np.sum(distArray[np.where(NLarray[route] == 1)[0]])

                        if doHybridRoutes:
                            hybridRoute = hybridRoutes[j]
                            
                            # Get route and part of route that is stad/buitenweg/snelweg and ZEZ/non-ZEZ
                            hybridRouteStad      = hybridRoute[stadArray[hybridRoute]]
                            hybridRouteBuitenweg = hybridRoute[buitenwegArray[hybridRoute]]
                            hybridRouteSnelweg   = hybridRoute[snelwegArray[hybridRoute]]
                            hybridZEZstad      = (ZEZarray[hybridRouteStad     ] == 1)
                            hybridZEZbuitenweg = (ZEZarray[hybridRouteBuitenweg] == 1)
                            hybridZEZsnelweg   = (ZEZarray[hybridRouteSnelweg  ] == 1)
                            
                            # Het gedeelte van de route in NL
                            hybridRouteStadNL      = np.where(NLarray[hybridRouteStad     ] == 1)[0]
                            hybridRouteBuitenwegNL = np.where(NLarray[hybridRouteBuitenweg] == 1)[0]
                            hybridRouteSnelwegNL   = np.where(NLarray[hybridRouteSnelweg  ] == 1)[0]
        
                            hybridDistStad      = distArray[hybridRouteStad]
                            hybridDistBuitenweg = distArray[hybridRouteBuitenweg]
                            hybridDistSnelweg   = distArray[hybridRouteSnelweg]
                            hybridDistTotal   = np.sum(distArray[hybridRoute])
                            hybridDistTotalNL = np.sum(distArray[np.where(NLarray[hybridRoute] == 1)[0]])

                        # Bereken en schrijf de intensiteiten/emissies op de links
                        for ls in range(nLS):
                            # Welke trips worden allemaal gemaakt op de HB van de huidige iteratie van de ij-loop
                            currentTrips  = iterTrips[whereODL[(origZone,destZone,ls)], :]
                            nCurrentTrips = len(currentTrips)
                            
                            if nCurrentTrips > 0:
                                capUt = currentTrips[:, 4]
                                vt    = [vtDict[x] for x in currentTrips[:, 3]]
        
                                emissionFacStad      = [None for et in range(nET)]
                                emissionFacBuitenweg = [None for et in range(nET)]
                                emissionFacSnelweg   = [None for et in range(nET)]
    
                                for et in range(nET):                            
                                    lowerFacStad = emissionsStadLeeg[vt, et]
                                    upperFacStad = emissionsStadVol[vt, et]
                                    emissionFacStad[et] = (lowerFacStad +
                                                           capUt * (upperFacStad - lowerFacStad))
    
                                    lowerFacBuitenweg = emissionsBuitenwegLeeg[vt, et]
                                    upperFacBuitenweg = emissionsBuitenwegVol[vt, et]
                                    emissionFacBuitenweg[et] = (lowerFacBuitenweg +
                                                                capUt * (upperFacBuitenweg - lowerFacBuitenweg))
    
                                    lowerFacSnelweg = emissionsSnelwegLeeg[vt, et]
                                    upperFacSnelweg = emissionsSnelwegVol[vt, et]
                                    emissionFacSnelweg[et] = (lowerFacSnelweg +
                                                              capUt * (upperFacSnelweg - lowerFacSnelweg))

                                for trip in range(nCurrentTrips):
                                    vt = int(currentTrips[trip, 3])
                                    ct = int(currentTrips[trip, 6])

                                    # If combustion type is fuel or bio-fuel                             
                                    if ct in [0, 4]:
                                        stadEmissions      = np.zeros((len(routeStad), nET))
                                        buitenwegEmissions = np.zeros((len(routeBuitenweg), nET))
                                        snelwegEmissions   = np.zeros((len(routeSnelweg), nET))
        
                                        for et in range(nET):                              
                                            stadEmissions[:, et]      = distStad      * emissionFacStad[et][trip]
                                            buitenwegEmissions[:, et] = distBuitenweg * emissionFacBuitenweg[et][trip]
                                            snelwegEmissions[:, et]   = distSnelweg   * emissionFacSnelweg[et][trip]
                                            
                                        linkEmissionsArray[ls][routeStad, :]      += stadEmissions
                                        linkEmissionsArray[ls][routeBuitenweg, :] += buitenwegEmissions
                                        linkEmissionsArray[ls][routeSnelweg, :]   += snelwegEmissions

                                        linkTripsArray[route, nLS + 2 + vt] += 1
                                        linkTripsArray[route, ls] += 1
                                        linkTripsArray[route, -1] += 1

                                    # If combustion type is hybrid
                                    elif ct == 3:
                                        stadEmissions      = np.zeros((len(hybridRouteStad), nET))
                                        buitenwegEmissions = np.zeros((len(hybridRouteBuitenweg), nET))
                                        snelwegEmissions   = np.zeros((len(hybridRouteSnelweg), nET))
        
                                        for et in range(nET):                              
                                            stadEmissions[:, et]      = hybridDistStad      * emissionFacStad[et][trip]
                                            buitenwegEmissions[:, et] = hybridDistBuitenweg * emissionFacBuitenweg[et][trip]
                                            snelwegEmissions[:, et]   = hybridDistSnelweg   * emissionFacSnelweg[et][trip]
                                            
                                        # No emissions in ZEZ part of route
                                        stadEmissions[hybridZEZstad, :] = 0
                                        buitenwegEmissions[hybridZEZbuitenweg, :] = 0
                                        snelwegEmissions[hybridZEZsnelweg, :] = 0
                                            
                                        linkEmissionsArray[ls][hybridRouteStad, :]      += stadEmissions
                                        linkEmissionsArray[ls][hybridRouteBuitenweg, :] += buitenwegEmissions
                                        linkEmissionsArray[ls][hybridRouteSnelweg, :]   += snelwegEmissions

                                        linkTripsArray[hybridRoute, nLS + 2 + vt] += 1
                                        linkTripsArray[hybridRoute, ls] += 1
                                        linkTripsArray[hybridRoute, -1] += 1
                                    
                                    # Clean combustion types (no emissions)
                                    else:
                                        linkTripsArray[route, nLS + 2 + vt] += 1
                                        linkTripsArray[route, ls] += 1
                                        linkTripsArray[route, -1] += 1

                                    tripIndex = currentTrips[trip, -1]
        
                                    # Total CO2 for each trip
                                    if ct in [0, 4]:
                                        tripsCO2[tripIndex] =  (np.sum(stadEmissions[:, 0]) + 
                                                                np.sum(buitenwegEmissions[:, 0]) +
                                                                np.sum(snelwegEmissions[:, 0]))
                                        tripsCO2_NL[tripIndex] = (np.sum(stadEmissions[routeStadNL, 0]) + 
                                                                  np.sum(buitenwegEmissions[routeBuitenwegNL, 0]) + 
                                                                  np.sum(snelwegEmissions[routeSnelwegNL, 0]))
                                    elif ct == 3:
                                        tripsCO2[tripIndex] =  (np.sum(stadEmissions[:, 0]) + 
                                                                np.sum(buitenwegEmissions[:, 0]) +
                                                                np.sum(snelwegEmissions[:, 0]))
                                        tripsCO2_NL[tripIndex] = (np.sum(stadEmissions[hybridRouteStadNL, 0]) + 
                                                                  np.sum(buitenwegEmissions[hybridRouteBuitenwegNL, 0]) + 
                                                                  np.sum(snelwegEmissions[hybridRouteSnelwegNL, 0]))
                                    else:
                                        tripsCO2[tripIndex] = 0
                                        tripsCO2_NL[tripIndex] = 0
                                        
                                    # Total distance for each trip
                                    if ct == 3:
                                        tripsDist[tripIndex] = hybridDistTotal
                                        tripsDist_NL[tripIndex] = hybridDistTotalNL
                                    else:
                                        tripsDist[tripIndex] = distTotal
                                        tripsDist_NL[tripIndex] = distTotalNL
                                        
                if root != '':
                    root.progressBar['value'] = 8.0 + (r + 1) * 10.0 + 10.0 * i / nOrigSelection
                        
            del prevFreight

            if doHybridRoutes:
                del prevFreightHybrid

        # ----------------------- Vans ----------------------------------------

        trips = np.array(allTrips[allTrips['VEHTYPE']==8])

        if nMultiRoute >= 2:
            whereTripsByIter = [[] for r in range(nMultiRoute)]
            for i in range(len(trips)):
                whereTripsByIter[np.random.randint(nMultiRoute)].append(i)
        elif nMultiRoute == 1:
            whereTripsByIter = [np.arange(len(trips))]
        else:
            raise Exception('nMultiRoute should be >= 1, now it is ' + str(nMultiRoute))

        if nMultiRoute >= 2:
            whereParcelTripsByIter = [[] for r in range(nMultiRoute)]
            for i in range(len(allParcelTrips)):
                whereParcelTripsByIter[np.random.randint(nMultiRoute)].append(i)
        elif nMultiRoute == 1:
            whereParcelTripsByIter = [np.arange(len(allParcelTrips))]
    
        for r in range(nMultiRoute):

            # ----------------------- Route search (van) ----------------------------------------  
                
            print(f"Route search (vans part {r + 1})...")
            log_file.write(f"Route search (vans part {r + 1})...\n")   

            # Route search freight
            if nCPU > 1:
                # From which nodes does every CPU perform the shortest path algorithm
                indicesPerCPU       = [[indices[cpu::nCPU], cpu]            for cpu in range(nCPU)]
                origSelectionPerCPU = [np.arange(nOrigSelection)[cpu::nCPU] for cpu in range(nCPU)]

                # The network with costs between nodes (freight)
                csgraphVan = lil_matrix((nNodes, nNodes))
                tmp = np.array(links['COST_VAN'])
                if nMultiRoute > 1:
                    tmp *= 0.95 + 0.1 * linksRandArray[r]
                csgraphVan[np.array(links['KNOOP_A']), np.array(links['KNOOP_B'])] = tmp
                
                # Initialize a pool object that spreads tasks over different CPUs
                p = mp.Pool(nCPU)
                
                # Execute the Dijkstra route search
                prevVanPerCPU = p.map(functools.partial(get_prev, csgraphVan, nNodes), indicesPerCPU)
    
                # Wait for completion of processes
                p.close()
                p.join()
                
                # Combine the results from the different CPUs
                # Matrix with for each node the previous node on the shortest path
                prevVan = np.zeros((nOrigSelection, nNodes), dtype=int)
                for cpu in range(nCPU):
                    for i in range(len(indicesPerCPU[cpu][0])):
                        prevVan[origSelectionPerCPU[cpu][i], :] = prevVanPerCPU[cpu][i, :]
    
                # Make some space available on the RAM
                del prevVanPerCPU
            
            else:   

                # The network with costs between nodes (freight)
                csgraphVan = lil_matrix((nNodes, nNodes))
                tmp = np.array(links['COST_VAN'])
                if nMultiRoute > 1:
                    tmp *= 0.95 + 0.1 * linksRandArray[r]
                csgraphVan[np.array(links['KNOOP_A']), np.array(links['KNOOP_B'])] = tmp
                
                # Execute the Dijkstra route search
                prevVan = get_prev(csgraphVan, nNodes, [indices, 0])
    

            if root != '':
                root.progressBar['value'] = 40.0 + (r + 1) * 10.0 + r * 10.0    

            # Make some space available on the RAM
            del csgraphVan

            # Route search freight (for clean vehicles in the UCC scenario)
            if doHybridRoutes:
                print(f"Route search (vans with hybrid combustion part {r + 1})...")
                log_file.write(f"Route search (vans with hybrid combustion part {r + 1})...\n")  

                # Route search van
                if nCPU > 1:    
                    
                    # The network with costs between nodes (van)
                    csgraphVanHybrid = lil_matrix((nNodes, nNodes))
                    tmp = costVanHybrid
                    if nMultiRoute > 1:
                        tmp *= 0.95 + 0.1 * linksRandArray[r]
                    csgraphVanHybrid[np.array(links['KNOOP_A']), np.array(links['KNOOP_B'])] = tmp
                    
                    # Initialize a pool object that spreads tasks over different CPUs
                    p = mp.Pool(nCPU)
                    
                    # Execute the Dijkstra route search
                    prevVanHybridPerCPU = p.map(functools.partial(get_prev, csgraphVanHybrid, nNodes), indicesPerCPU)
        
                    # Wait for completion of processes
                    p.close()
                    p.join()
                    
                    # Combine the results from the different CPUs
                    # Matrix with for each node the previous node on the shortest path
                    prevVanHybrid = np.zeros((nOrigSelection, nNodes), dtype=int)
                    for cpu in range(nCPU):
                        for i in range(len(indicesPerCPU[cpu][0])):
                            prevVanHybrid[origSelectionPerCPU[cpu][i], :] = prevVanHybridPerCPU[cpu][i, :]
        
                    # Make some space available on the RAM
                    del prevVanHybridPerCPU
                    
                else:
        
                    # The network with costs between nodes (van)
                    csgraphVanHybrid = lil_matrix((nNodes, nNodes))
                    tmp = costVanHybrid
                    if nMultiRoute > 1:
                        tmp *= 0.95 + 0.1 * linksRandArray[r]
                    csgraphVanHybrid[np.array(links['KNOOP_A']), np.array(links['KNOOP_B'])] = tmp
                    
                    # Execute the Dijkstra route search
                    prevVanHybrid = get_prev(csgraphVanHybrid, nNodes, [indices, 0])
        
                if root != '':
                    root.progressBar['value'] = 8.0 + (r + 1) * 10.0 + r * 10.0
    
                # Make some space available on the RAM
                del csgraphVanHybrid

            # --------------------- Emissions and intensities (freight) --------------------------------------     
            
            print(f"Calculating emissions and traffic intensities (vans part {r + 1})")
            log_file.write(f"Calculating emissions and traffic intensities (vans part {r + 1})\n")
    
            print('\tGoods vans...')
            log_file.write('\tGoods vans...\n')
                
            iterTrips = trips[whereTripsByIter[r], :]
    
            whereODL = {}
            for i in range(len(tripMatrix)):
                orig = tripMatrix[i, 0]
                dest = tripMatrix[i, 1]
                for ls in range(nLS):
                    whereODL[(orig,dest,ls)] = []
            for i in range(len(iterTrips)):
                orig = iterTrips[i, 1]
                dest = iterTrips[i, 2]
                ls   = iterTrips[i, 5]
                whereODL[(orig,dest,ls)].append(i)
    
            for i in range(nOrigSelection):
                origZone = i + 1
                
                print('\t\tOrigin ' + str(origZone), end='\r')
                
                if origZone in tripMatrixOrigins:
                    destZoneIndex = np.where(tripMatrix[:,0] == origZone)[0]
                    destZones = tripMatrix[destZoneIndex, 1]
                    
                    routes = [get_route(i, destZone-1, prevVan, linkDict)
                              for destZone in destZones]

                    # Routes for hybrid vehicles (no need for penalty on ZEZ-links here)
                    if doHybridRoutes:
                        hybridRoutes = [get_route(i, destZone-1, prevVanHybrid, linkDict)
                                        for destZone in destZones]

                    # Schrijf de volumes op de links
                    for j in range(len(destZones)):
                        destZone = destZones[j]
                        route = routes[j]
                        
                        # Get route and part of route that is stad/buitenweg/snelweg and ZEZ/non-ZEZ
                        routeStad      = route[stadArray[route]]
                        routeBuitenweg = route[buitenwegArray[route]]
                        routeSnelweg   = route[snelwegArray[route]]
                        
                        # Het gedeelte van de route in NL
                        routeStadNL      = np.where(NLarray[routeStad     ] == 1)[0]
                        routeBuitenwegNL = np.where(NLarray[routeBuitenweg] == 1)[0]
                        routeSnelwegNL   = np.where(NLarray[routeSnelweg  ] == 1)[0]
    
                        distStad      = distArray[routeStad]
                        distBuitenweg = distArray[routeBuitenweg]
                        distSnelweg   = distArray[routeSnelweg]
                        distTotal   = np.sum(distArray[route])
                        distTotalNL = np.sum(distArray[np.where(NLarray[route] == 1)[0]])

                        if doHybridRoutes:
                            hybridRoute = hybridRoutes[j]
                            
                            # Get route and part of route that is stad/buitenweg/snelweg and ZEZ/non-ZEZ
                            hybridRouteStad      = hybridRoute[stadArray[hybridRoute]]
                            hybridRouteBuitenweg = hybridRoute[buitenwegArray[hybridRoute]]
                            hybridRouteSnelweg   = hybridRoute[snelwegArray[hybridRoute]]
                            hybridZEZstad      = (ZEZarray[hybridRouteStad     ] == 1)
                            hybridZEZbuitenweg = (ZEZarray[hybridRouteBuitenweg] == 1)
                            hybridZEZsnelweg   = (ZEZarray[hybridRouteSnelweg  ] == 1)
                            
                            # Het gedeelte van de route in NL
                            hybridRouteStadNL      = np.where(NLarray[hybridRouteStad     ] == 1)[0]
                            hybridRouteBuitenwegNL = np.where(NLarray[hybridRouteBuitenweg] == 1)[0]
                            hybridRouteSnelwegNL   = np.where(NLarray[hybridRouteSnelweg  ] == 1)[0]
        
                            hybridDistStad      = distArray[hybridRouteStad]
                            hybridDistBuitenweg = distArray[hybridRouteBuitenweg]
                            hybridDistSnelweg   = distArray[hybridRouteSnelweg]
                            hybridDistTotal   = np.sum(distArray[hybridRoute])
                            hybridDistTotalNL = np.sum(distArray[np.where(NLarray[hybridRoute] == 1)[0]])

                        # Bereken en schrijf de intensiteiten/emissies op de links
                        for ls in range(nLS):
                            # Welke trips worden allemaal gemaakt op de HB van de huidige iteratie van de ij-loop
                            currentTrips  = iterTrips[whereODL[(origZone,destZone,ls)], :]
                            nCurrentTrips = len(currentTrips)
                            
                            if nCurrentTrips > 0:
                                capUt = currentTrips[:, 4]
                                vt    = vtDict[8]
        
                                emissionFacStad      = [None for et in range(nET)]
                                emissionFacBuitenweg = [None for et in range(nET)]
                                emissionFacSnelweg   = [None for et in range(nET)]
    
                                for et in range(nET):                            
                                    lowerFacStad = emissionsStadLeeg[vt, et]
                                    upperFacStad = emissionsStadVol[vt, et]
                                    emissionFacStad[et] = (lowerFacStad +
                                                           capUt * (upperFacStad - lowerFacStad))
    
                                    lowerFacBuitenweg = emissionsBuitenwegLeeg[vt, et]
                                    upperFacBuitenweg = emissionsBuitenwegVol[vt, et]
                                    emissionFacBuitenweg[et] = (lowerFacBuitenweg +
                                                                capUt * (upperFacBuitenweg - lowerFacBuitenweg))
    
                                    lowerFacSnelweg = emissionsSnelwegLeeg[vt, et]
                                    upperFacSnelweg = emissionsSnelwegVol[vt, et]
                                    emissionFacSnelweg[et] = (lowerFacSnelweg +
                                                              capUt * (upperFacSnelweg - lowerFacSnelweg))
                                
                                for trip in range(nCurrentTrips):
                                    vt = int(currentTrips[trip, 3])
                                    ct = int(currentTrips[trip, 6])
                                          
                                    # If combustion type is fuel or bio-fuel
                                    if ct in [0, 4]:
                                        stadEmissions      = np.zeros((len(routeStad), nET))
                                        buitenwegEmissions = np.zeros((len(routeBuitenweg), nET))
                                        snelwegEmissions   = np.zeros((len(routeSnelweg), nET))
        
                                        for et in range(nET):                              
                                            stadEmissions[:, et]      = distStad      * emissionFacStad[et][trip]
                                            buitenwegEmissions[:, et] = distBuitenweg * emissionFacBuitenweg[et][trip]
                                            snelwegEmissions[:, et]   = distSnelweg   * emissionFacSnelweg[et][trip]
                                            
                                        linkEmissionsArray[ls][routeStad, :]      += stadEmissions
                                        linkEmissionsArray[ls][routeBuitenweg, :] += buitenwegEmissions
                                        linkEmissionsArray[ls][routeSnelweg, :]   += snelwegEmissions

                                        linkTripsArray[route, nLS + 2 + vt] += 1
                                        linkTripsArray[route, ls] += 1
                                        linkTripsArray[route, -1] += 1

                                    # If combustion type is hybrid
                                    elif ct == 3:
                                        stadEmissions      = np.zeros((len(hybridRouteStad), nET))
                                        buitenwegEmissions = np.zeros((len(hybridRouteBuitenweg), nET))
                                        snelwegEmissions   = np.zeros((len(hybridRouteSnelweg), nET))
        
                                        for et in range(nET):                              
                                            stadEmissions[:, et]      = hybridDistStad      * emissionFacStad[et][trip]
                                            buitenwegEmissions[:, et] = hybridDistBuitenweg * emissionFacBuitenweg[et][trip]
                                            snelwegEmissions[:, et]   = hybridDistSnelweg   * emissionFacSnelweg[et][trip]
                                            
                                        # No emissions in ZEZ part of route
                                        stadEmissions[hybridZEZstad, :] = 0
                                        buitenwegEmissions[hybridZEZbuitenweg, :] = 0
                                        snelwegEmissions[hybridZEZsnelweg, :] = 0
                                            
                                        linkEmissionsArray[ls][hybridRouteStad, :]      += stadEmissions
                                        linkEmissionsArray[ls][hybridRouteBuitenweg, :] += buitenwegEmissions
                                        linkEmissionsArray[ls][hybridRouteSnelweg, :]   += snelwegEmissions

                                        linkTripsArray[hybridRoute, nLS + 2 + vt] += 1
                                        linkTripsArray[hybridRoute, ls] += 1
                                        linkTripsArray[hybridRoute, -1] += 1
                                    
                                    # Clean combustion types (no emissions)
                                    else:
                                        linkTripsArray[route, nLS + 2 + vt] += 1
                                        linkTripsArray[route, ls] += 1
                                        linkTripsArray[route, -1] += 1

                                    tripIndex = currentTrips[trip, -1]
        
                                    # Total CO2 for each trip
                                    if ct in [0, 4]:
                                        tripsCO2[tripIndex] =  (np.sum(stadEmissions[:, 0]) + 
                                                                np.sum(buitenwegEmissions[:, 0]) +
                                                                np.sum(snelwegEmissions[:, 0]))
                                        tripsCO2_NL[tripIndex] = (np.sum(stadEmissions[routeStadNL, 0]) + 
                                                                  np.sum(buitenwegEmissions[routeBuitenwegNL, 0]) + 
                                                                  np.sum(snelwegEmissions[routeSnelwegNL, 0]))
                                    elif ct == 3:
                                        tripsCO2[tripIndex] =  (np.sum(stadEmissions[:, 0]) + 
                                                                np.sum(buitenwegEmissions[:, 0]) +
                                                                np.sum(snelwegEmissions[:, 0]))
                                        tripsCO2_NL[tripIndex] = (np.sum(stadEmissions[hybridRouteStadNL, 0]) + 
                                                                  np.sum(buitenwegEmissions[hybridRouteBuitenwegNL, 0]) + 
                                                                  np.sum(snelwegEmissions[hybridRouteSnelwegNL, 0]))
                                    else:
                                        tripsCO2[tripIndex] = 0
                                        tripsCO2_NL[tripIndex] = 0
                                        
                                    # Total distance for each trip
                                    if ct == 3:
                                        tripsDist[tripIndex] = hybridDistTotal
                                        tripsDist_NL[tripIndex] = hybridDistTotalNL
                                    else:
                                        tripsDist[tripIndex] = distTotal
                                        tripsDist_NL[tripIndex] = distTotalNL
    
                if root != '':
                    root.progressBar['value'] = 40.0 + (r + 1) * 15.0 + 4.0 * i / nOrigSelection
                        
            del whereODL
        
                            
            # -------------------- Emissions and intensities (parcel vans) ------------------------------------
            
            print('\tParcel vans...')
            log_file.write('\tParcels vans...\n')
            
            # Logistic segment: parcel deliveries
            ls = 8 
            
            # Assume half of vehicle capacity (in terms of weight) used
            capUt = 0.5
            
            for vt in vehTypesParcels:
                parcelTrips = allParcelTrips.loc[whereParcelTripsByIter[r], :]
                parcelTrips = parcelTrips.loc[(parcelTrips['VEHTYPE']==vt), :]
                parcelTrips = np.array(parcelTrips)
                
                emissionFacStadCO2      = (emissionsStadLeeg[vtDict[vt],      indexCO2] + capUt * (emissionsStadVol[vtDict[vt],      indexCO2] - emissionsStadLeeg[vtDict[vt],      indexCO2]))
                emissionFacBuitenwegCO2 = (emissionsBuitenwegLeeg[vtDict[vt], indexCO2] + capUt * (emissionsBuitenwegVol[vtDict[vt], indexCO2] - emissionsBuitenwegLeeg[vtDict[vt], indexCO2]))
                emissionFacSnelwegCO2   = (emissionsSnelwegLeeg[vtDict[vt],   indexCO2] + capUt * (emissionsSnelwegVol[vtDict[vt],   indexCO2] - emissionsSnelwegLeeg[vtDict[vt],   indexCO2]))
                emissionFacStadSO2      = (emissionsStadLeeg[vtDict[vt],      indexSO2] + capUt * (emissionsStadVol[vtDict[vt],      indexSO2] - emissionsStadLeeg[vtDict[vt],      indexSO2]))
                emissionFacBuitenwegSO2 = (emissionsBuitenwegLeeg[vtDict[vt], indexSO2] + capUt * (emissionsBuitenwegVol[vtDict[vt], indexSO2] - emissionsBuitenwegLeeg[vtDict[vt], indexSO2]))
                emissionFacSnelwegSO2   = (emissionsSnelwegLeeg[vtDict[vt],   indexSO2] + capUt * (emissionsSnelwegVol[vtDict[vt],   indexSO2] - emissionsSnelwegLeeg[vtDict[vt],   indexSO2]))
                emissionFacStadPM       = (emissionsStadLeeg[vtDict[vt],      indexPM ] + capUt * (emissionsStadVol[vtDict[vt],      indexPM ] - emissionsStadLeeg[vtDict[vt],      indexPM ]))
                emissionFacBuitenwegPM  = (emissionsBuitenwegLeeg[vtDict[vt], indexPM ] + capUt * (emissionsBuitenwegVol[vtDict[vt], indexPM ] - emissionsBuitenwegLeeg[vtDict[vt], indexPM ]))
                emissionFacSnelwegPM    = (emissionsSnelwegLeeg[vtDict[vt],   indexPM ] + capUt * (emissionsSnelwegVol[vtDict[vt],   indexPM ] - emissionsSnelwegLeeg[vtDict[vt],   indexPM ]))
                emissionFacStadNOX      = (emissionsStadLeeg[vtDict[vt],      indexNOX] + capUt * (emissionsStadVol[vtDict[vt],      indexNOX] - emissionsStadLeeg[vtDict[vt],      indexNOX]))
                emissionFacBuitenwegNOX = (emissionsBuitenwegLeeg[vtDict[vt], indexNOX] + capUt * (emissionsBuitenwegVol[vtDict[vt], indexNOX] - emissionsBuitenwegLeeg[vtDict[vt], indexNOX]))
                emissionFacSnelwegNOX   = (emissionsSnelwegLeeg[vtDict[vt],   indexNOX] + capUt * (emissionsSnelwegVol[vtDict[vt],   indexNOX] - emissionsSnelwegLeeg[vtDict[vt],   indexNOX]))        
        
                emissionFacStad = np.array([emissionFacStadCO2, emissionFacStadSO2,
                                            emissionFacStadPM,  emissionFacStadNOX])
                emissionFacBuitenweg = np.array([emissionFacBuitenwegCO2, emissionFacBuitenwegSO2,
                                                 emissionFacBuitenwegPM,  emissionFacBuitenwegNOX])
                emissionFacSnelweg = np.array([emissionFacSnelwegCO2, emissionFacSnelwegSO2,
                                               emissionFacSnelwegPM,  emissionFacSnelwegNOX])

                if len(parcelTrips) > 0:
                    if doHybridRoutes:
                        args = [parcelTrips,
                                tripMatrixParcels,
                                prevVan, prevVanHybrid,
                                distArray, stadArray, buitenwegArray, snelwegArray, ZEZarray,
                                indexCO2, indexSO2, indexPM, indexNOX,
                                emissionFacStad, emissionFacBuitenweg, emissionFacSnelweg,
                                nLinks, nET, linkDict, 
                                indices, 
                                doHybridRoutes]
                    else:
                        args = [parcelTrips,
                                tripMatrixParcels,
                                prevVan, None,
                                distArray, stadArray, buitenwegArray, snelwegArray, ZEZarray,
                                indexCO2, indexSO2, indexPM, indexNOX,
                                emissionFacStad, emissionFacBuitenweg, emissionFacSnelweg,
                                nLinks, nET, linkDict, 
                                indices, 
                                doHybridRoutes]
                    parcelResults = emissions_parcels(*args)
                    
                    # Intensiteiten
                    nTrips = parcelResults[0] 
                    linkTripsArray[:, ls]             += nTrips # Number of trips for LS8 (=parcel deliveries)
                    linkTripsArray[:, nLS+2+nVT+vt-8] += nTrips # De parcel demand trips per voertuigtype
                    linkTripsArray[:, nLS+2+vt]       += nTrips # Number of trips for vehicle type
                    linkTripsArray[:, -1]             += nTrips # Total number of trips
                    
                    # Emissies
                    linkEmissionsArray[ls] += parcelResults[1]
                    
                    # CO2 per trip
                    for trip in parcelResults[2].keys():
                        parcelTripsCO2[int(trip)] = parcelResults[2][trip]
         
                    del parcelResults
    
            if root != '':
                root.progressBar['value'] = 40.0 + (r + 1) * 15.0 + 5.0

            # ------------------- Emissions and intensities (serv/constr vans) -------------------------------- 
            
            print('\tService & construction vans...')
            log_file.write('\tService & construction vans...\n')

            # Van trips for service and construction purposes
            try:
                vanTripsService      = read_mtx(datapathO + 'TripsVanService.mtx')
                vanTripsConstruction = read_mtx(datapathO + 'TripsVanConstruction.mtx')
                vanTripsFound = True

            except FileNotFoundError:
                log_file.write('Could not find TripsVanService.mtx and/or TripsVanConstruction.mtx' +
                               ' in specified outputfolder. Hence no service/construction vans are' +
                               ' assigned to the network.')
                print('Could not find TripsVanService.mtx and/or TripsVanConstruction.mtx' +
                      ' in specified outputfolder. Hence no service/construction vans are' +
                      ' assigned to the network.')
                vanTripsFound = False
                
            if vanTripsFound:
                # Select half of ODs per multiroute iteration
                if nMultiRoute > 1:
                    vanTripsService[r::nMultiRoute] = 0
                    vanTripsConstruction[r::nMultiRoute] = 0
                
                # Reshape to square array
                vanTripsService      = vanTripsService.reshape(nZones,nZones)
                vanTripsConstruction = vanTripsConstruction.reshape(nZones,nZones)
    
                vt    = 8   # Vehicle type: Van
                capUt = 0.5 # Assume half of loading capacity used
                
                emissionFacStadCO2 = (emissionsStadLeeg[vtDict[vt], indexCO2] +
                                      capUt * (emissionsStadVol[vtDict[vt], indexCO2] - emissionsStadLeeg[vtDict[vt], indexCO2]))
                emissionFacStadSO2 = (emissionsStadLeeg[vtDict[vt],  indexSO2] +
                                      capUt * (emissionsStadVol[vtDict[vt], indexSO2] - emissionsStadLeeg[vtDict[vt], indexSO2]))
                emissionFacStadPM  = (emissionsStadLeeg[vtDict[vt], indexPM ] +
                                      capUt * (emissionsStadVol[vtDict[vt], indexPM ] - emissionsStadLeeg[vtDict[vt], indexPM ]))
                emissionFacStadNOX = (emissionsStadLeeg[vtDict[vt], indexNOX] +
                                      capUt * (emissionsStadVol[vtDict[vt], indexNOX] - emissionsStadLeeg[vtDict[vt], indexNOX]))
                emissionFacBuitenwegCO2 = (emissionsBuitenwegLeeg[vtDict[vt], indexCO2] +
                                           capUt * (emissionsBuitenwegVol[vtDict[vt], indexCO2] - emissionsBuitenwegLeeg[vtDict[vt], indexCO2]))
                emissionFacBuitenwegSO2 = (emissionsBuitenwegLeeg[vtDict[vt], indexSO2] +
                                           capUt * (emissionsBuitenwegVol[vtDict[vt], indexSO2] - emissionsBuitenwegLeeg[vtDict[vt], indexSO2]))
                emissionFacBuitenwegPM  = (emissionsBuitenwegLeeg[vtDict[vt], indexPM ] +
                                           capUt * (emissionsBuitenwegVol[vtDict[vt], indexPM ] - emissionsBuitenwegLeeg[vtDict[vt], indexPM ]))
                emissionFacBuitenwegNOX = (emissionsBuitenwegLeeg[vtDict[vt], indexNOX] +
                                           capUt * (emissionsBuitenwegVol[vtDict[vt], indexNOX] - emissionsBuitenwegLeeg[vtDict[vt], indexNOX]))
                emissionFacSnelwegCO2 = (emissionsSnelwegLeeg[vtDict[vt], indexCO2] +
                                         capUt * (emissionsSnelwegVol[vtDict[vt], indexCO2] - emissionsSnelwegLeeg[vtDict[vt], indexCO2]))
                emissionFacSnelwegSO2 = (emissionsSnelwegLeeg[vtDict[vt], indexSO2] +
                                         capUt * (emissionsSnelwegVol[vtDict[vt], indexSO2] - emissionsSnelwegLeeg[vtDict[vt], indexSO2]))
                emissionFacSnelwegPM  = (emissionsSnelwegLeeg[vtDict[vt], indexPM ] +
                                         capUt * (emissionsSnelwegVol[vtDict[vt], indexPM ] - emissionsSnelwegLeeg[vtDict[vt], indexPM ]))
                emissionFacSnelwegNOX = (emissionsSnelwegLeeg[vtDict[vt], indexNOX] +
                                         capUt * (emissionsSnelwegVol[vtDict[vt], indexNOX] - emissionsSnelwegLeeg[vtDict[vt], indexNOX]))        
        
                emissionFacStad = np.array([emissionFacStadCO2, emissionFacStadSO2,
                                            emissionFacStadPM,  emissionFacStadNOX])
                emissionFacBuitenweg = np.array([emissionFacBuitenwegCO2, emissionFacBuitenwegSO2,
                                                 emissionFacBuitenwegPM,  emissionFacBuitenwegNOX])
                emissionFacSnelweg = np.array([emissionFacSnelwegCO2, emissionFacSnelwegSO2,
                                               emissionFacSnelwegPM,  emissionFacSnelwegNOX])
                        
                indicesPerCPU = []
                for cpu in range(nCPU):            
                    indicesPerCPU.append([])
                    
                    indicesPerCPU[cpu].append(indices[cpu::nCPU])
                    indicesPerCPU[cpu].append(vanTripsService[indices[cpu::nCPU],:])     
                    indicesPerCPU[cpu].append(vanTripsConstruction[indices[cpu::nCPU],:])
                    if doHybridRoutes:
                        indicesPerCPU[cpu].append(prevVanHybrid[indices[cpu::nCPU],:])
                    else:
                        indicesPerCPU[cpu].append(prevVan[indices[cpu::nCPU],:])

                # Make some space available on the RAM
                del vanTripsService, vanTripsConstruction, prevVan
                if doHybridRoutes:
                    del prevVanHybrid
                
                if nCPU > 1:
                    # Initialize a pool object that spreads tasks over different CPUs
                    p = mp.Pool(nCPU)
                    
                    # Calculate emissions for service / construction vans
                    args = [emissions_vans,
                            distArray, stadArray, buitenwegArray, snelwegArray,
                            indexCO2, indexSO2, indexPM, indexNOX,
                            emissionFacStad, emissionFacBuitenweg, emissionFacSnelweg,
                            nLinks, nET, linkDict]
                    vanResultPerCPU = p.map(functools.partial(*args), indicesPerCPU)

                    # Wait for completion of processes
                    p.close()
                    p.join()
                    
                else:
                    # Calculate emissions for service / construction vans
                    args = [distArray, stadArray, buitenwegArray, snelwegArray,
                            indexCO2, indexSO2, indexPM, indexNOX,
                            emissionFacStad, emissionFacBuitenweg, emissionFacSnelweg,
                            nLinks, nET, linkDict,
                            indicesPerCPU[0]]
                    vanResultPerCPU = [emissions_vans(*args)]

                if root != '':
                    root.progressBar['value'] = 40.0 + (r + 1) * 15.0 + 15.0 * i / nOrigSelection

                del indicesPerCPU

                # Combine the results from the different CPUs
                for cpu in range(nCPU):
                    linkVanTripsArray        += vanResultPerCPU[cpu][0]
                    linkVanEmissionsArray[0] += vanResultPerCPU[cpu][1][0]
                    linkVanEmissionsArray[1] += vanResultPerCPU[cpu][1][1]
                
                del vanResultPerCPU

                        
        # -------------------- Writing emissions into network -------------------------------- 
        
        # Write the intensities and emissions into the links-DataFrame
        for field in intensityFields:
            links[field] = 0.0
        links.loc[:, volCols] += linkTripsArray.astype(int)
        
        # Total emissions and per logistic segment
        for ls in range(nLS):
            for et in range(nET):
                links[etDict[et]                  ] += (linkEmissionsArray[ls][:, et] / emissionDivFac[et])
                links[etDict[et] + '_LS' + str(ls)] += (linkEmissionsArray[ls][:, et] / emissionDivFac[et])

        del linkTripsArray, linkEmissionsArray

        if root != '':
            root.progressBar['value'] = 90.0

        if vanTripsFound:
            # Number of van trips
            linkVanTripsArray = np.round(linkVanTripsArray, 3)
            links.loc[:,'N_VAN_S'] = linkVanTripsArray[:, 0]
            links.loc[:,'N_VAN_C'] = linkVanTripsArray[:, 1]
            links.loc[:,'N_VEH7'] += linkVanTripsArray[:, 0]
            links.loc[:,'N_VEH7'] += linkVanTripsArray[:, 1]
            links.loc[:,'N_TOT'] += linkVanTripsArray[:, 0]
            links.loc[:,'N_TOT'] += linkVanTripsArray[:, 1]
        
            # Emissions from van trips
            for et in range(nET):
                links[etDict[et] + '_' + 'VAN_S'] = (linkVanEmissionsArray[0][:, et] / emissionDivFac[et])
                links[etDict[et] + '_' + 'VAN_C'] = (linkVanEmissionsArray[1][:, et] / emissionDivFac[et])
                links[etDict[et]] += (linkVanEmissionsArray[0][:, et] / emissionDivFac[et])
                links[etDict[et]] += (linkVanEmissionsArray[1][:, et] / emissionDivFac[et])
            
            del linkVanTripsArray, linkVanEmissionsArray        
        
        if root != '':
            root.progressBar['value'] = 91.0
            
            
        # ----------------------- Enriching tours and shipments -------------------        
        try:
            print("Writing emissions into Tours and ParcelSchedule...")
            log_file.write("Writing emissions into Tours and ParcelSchedule...\n")

            tours = pd.read_csv(datapathO + 'Tours_' + label + '.csv')
            tours['CO2'   ] = [tripsCO2[i]    for i in tours.index]
            tours['CO2_NL'] = [tripsCO2_NL[i] for i in tours.index]
            tours['CO2'   ] = np.round(tours['CO2'   ] / emissionDivFac[0], 3)
            tours['CO2_NL'] = np.round(tours['CO2_NL'] / emissionDivFac[0], 3)
            tours['AFSTAND'   ] = [tripsDist[i]    for i in tours.index]
            tours['AFSTAND_NL'] = [tripsDist_NL[i] for i in tours.index]
            tours.to_csv(datapathO + 'Tours_' + label + '.csv', index=False)
        
            parcelTours        = pd.read_csv(datapathO + 'ParcelSchedule_' + label + '.csv')
            parcelTours['CO2'] = [parcelTripsCO2[i] for i in parcelTours.index]
            parcelTours['CO2'] = np.round(parcelTours['CO2'] / emissionDivFac[0], 3)
            parcelTours.to_csv(datapathO + 'ParcelSchedule_' + label + '.csv', index=False)

            if root != '':
                root.progressBar['value'] = 94.0
            
            print("Writing emissions into Shipments...")
            log_file.write("Writing emissions into Shipments...\n")        

            # Calculate emissions at the tour level instead of trip level
            tours['TOUR_ID'] = [str(tours.at[i,'CARRIER_ID']) + '_' + str(tours.at[i,'TOUR_ID']) for i in tours.index]
            toursCO2 = pd.pivot_table(tours, values=['CO2'], index=['TOUR_ID'], aggfunc=np.sum)
            tourIDDict = dict(np.transpose(np.vstack((toursCO2.index, np.arange(len(toursCO2))))))
            toursCO2 = np.array(toursCO2['CO2'])
            
            # Read the shipments
            shipments = pd.read_csv(datapathO + 'Shipments_AfterScheduling_' + label + '.csv')
            shipments = shipments.sort_values('TOUR_ID')
            shipments.index = np.arange(len(shipments))
            
            # For each tour, which shipments belong to it
            tourIDs = [tourIDDict[x] for x in shipments['TOUR_ID']]
            shipIDs = []
            currentShipIDs = [0]
            for i in range(1,len(shipments)):
                if tourIDs[i-1] == tourIDs[i]:
                    currentShipIDs.append(i)
                else:
                    shipIDs.append(currentShipIDs.copy())
                    currentShipIDs = [i]
            shipIDs.append(currentShipIDs.copy())
                
            # Network distance of each shipment
            skimDistance = read_mtx(skimDistancePath)
            origNRM = np.array(shipments['ORIG_NRM'], dtype=int)
            destNRM = np.array(shipments['DEST_NRM'], dtype=int)
            shipDist = skimDistance[(origNRM - 1) * nZones + (destNRM - 1)]            
            
            # Divide CO2 of each tour over its shipments based on distance
            shipCO2  = np.zeros(len(shipments))         
            for tourID in np.unique(tourIDs):
                currentDists = shipDist[shipIDs[tourID]]
                currentCO2   = toursCO2[tourID]
                if np.sum(currentDists) == 0:
                    shipCO2[shipIDs[tourID]] = 0
                else:                        
                    shipCO2[shipIDs[tourID]] = currentDists / np.sum(currentDists) * currentCO2
            shipments['CO2'] = shipCO2
            
            # Export enriched shipments with CO2 field
            shipments = shipments.sort_values('SHIP_ID')
            shipments.index = np.arange(len(shipments))        
            shipments.to_csv(datapathO + 'Shipments_AfterScheduling_' + label + '.csv', index=False)
            
        except:
            print("Writing emissions into Tours/ParcelSchedule/Shipments failed!")
            log_file.write("Writing emissions into Tours/ParcelSchedule/Shipments failed!" + '\n')

            try:
                import sys
                print(sys.exc_info()[0]),
                log_file.write(str(sys.exc_info()[0])),
                log_file.write("\n")
                import traceback
                print(traceback.format_exc()),
                log_file.write(str(traceback.format_exc())),
                log_file.write("\n")
            except:
                pass

        if root != '':
            root.progressBar['value'] = 95.0
            
            
        # ----------------------- Export loaded network to shapefile -------------------
        if exportShp:
            print("Exporting network to .shp...")
            log_file.write("Exporting network to .shp...\n")            
            
            # Set travel times of connectors at 0 for in the output network shape
            links.loc[links['LINKTYPE'] == 99, 'T0_FREIGHT'] = 0.0
            links.loc[links['LINKTYPE'] == 99, 'T0_VAN'    ] = 0.0

            links['KNOOP_A'] = [nodeDict[x] for x in links['KNOOP_A']]
            links['KNOOP_B'] = [nodeDict[x] for x in links['KNOOP_B']]

            # Van kilometer terug naar meter
            links['DISTANCE'] *= 1000
            
            # Van uur terug naar secondes
            links['T0_FREIGHT'] *= 3600
            links['T0_VAN'    ] *= 3600
            
            # Initialize shapefile fields
            w = shp.Writer(datapathO + f'links_loaded_{label}.shp')
            w.field('LINKNR',      'N', size=7, decimal=0)
            w.field('KNOOP_A',     'N', size=7, decimal=0)
            w.field('KNOOP_B',     'N', size=7, decimal=0)
            w.field('SNEL_FF',     'N', size=3, decimal=0) 
            w.field('SNEL',        'N', size=3, decimal=0) 
            w.field('CAP',         'N', size=5, decimal=0) 
            w.field('LINKTYPE',    'N', size=2, decimal=0)
            w.field('DISTANCE',    'N', size=8, decimal=0)
            w.field('DOELSTROOK',  'N', size=1, decimal=0) 
            w.field('NRM',         'N', size=1, decimal=0)
            w.field('ZEZ',         'N', size=1, decimal=0)
            w.field('NL',          'N', size=1, decimal=0)
            w.field('T0_FREIGHT',  'N', size=7, decimal=1)
            w.field('T0_VAN',      'N', size=7, decimal=1)
            w.field('COST_FREIGHT','N', size=8, decimal=3)
            w.field('COST_VAN',    'N', size=8, decimal=3)
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
            for i in range(nLinks):
                # Add geometry
                line = []
                line.append([nodesX[linksA[i]], nodesY[linksA[i]]])
                line.append([nodesX[linksB[i]], nodesY[linksB[i]]])
                w.line([line])
                
                # Add data fields
                w.record(*dbfData[i,:])
                                
                if i%500 == 0:
                    print('\t' + str(round((i / nLinks)*100, 1)) + '%', end='\r')    

                    if root != '':
                        root.progressBar['value'] = 93.0 + (100.0 - 93.0) * i / nLinks
                        
            w.close()
        
        print('\t100%', end='\r')
        

        # --------------------------- End of module ---------------------------
                
        totaltime = round(time.time() - start_time, 2)
        print("Total runtime: %s seconds\n" % (totaltime))
        log_file.write("Total runtime: %s seconds\n" % (totaltime))  
        log_file.write("End simulation at: "+datetime.datetime.now().strftime("%y-%m-%d %H:%M")+"\n")
        log_file.close()    

        if root != '':
            root.update_statusbar("Emission Calculation: Done")
            root.progressBar['value'] = 100
            
            # 0 means no errors in execution
            root.returnInfo = [0, [0,0]]
            
            return root.returnInfo
        
        else:
            return [0, [0,0]]
            
        
    except BaseException:
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
                root.update_statusbar("Emission Calculation: Execution failed!")
                errorMessage = ('Execution failed!\n\n' +
                                str(root.returnInfo[1][0]) +
                                '\n\n' +
                                str(root.returnInfo[1][1]))
                root.error_screen(text=errorMessage, size=[900,350])                
            
            else:
                return root.returnInfo
        else:
            return [1, [sys.exc_info()[0], traceback.format_exc()]]
        
        

#%% Route search functions
            
def get_prev(csgraph, nNodes, indices):
    '''
    For each origin zone and destination node, determine the previously visited node on the shortest path.
    '''
    whichCPU = indices[1]
    indices  = indices[0]
    nOrigSelection = len(indices)
    
    prev = np.zeros((nOrigSelection,nNodes), dtype=int)        
    for i in range(nOrigSelection):
        prev[i,:] = scipy.sparse.csgraph.dijkstra(csgraph, 
                                                  indices=indices[i], 
                                                  return_predecessors=True)[1]
        
        if whichCPU == 0:
            if i%int(round(nOrigSelection/20,0)) == 0:
                print('\t' + str(int(round((i / nOrigSelection)*100, 0))) + '%', end='\r')          
    
    del csgraph

    return prev


#%%

@njit
def get_route(orig, dest, prev, linkDict, maxNumConnections=8):
    '''
    Deduce the paths from the prev object.
    Returns path sequence in terms of link IDs.
    '''
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


#%% emissions_parcels
    
def emissions_parcels(trips,
                      tripMatrixParcels,
                      prevVan, prevVanHybrid,
                      distArray, stadArray, buitenwegArray, snelwegArray, ZEZarray,
                      indexCO2, indexSO2, indexPM, indexNOX,
                      emissionFacStad, emissionFacBuitenweg, emissionFacSnelweg,
                      nLinks, nET, linkDict,
                      indices,
                      doHybridRoutes):
    '''
    Calculate the emissions and intensities of the parcel vans.
    ''' 
    linkTripsArray     = np.zeros(nLinks)
    linkEmissionsArray = np.zeros((nLinks, nET))
    parcelTripsCO2 = {}
    
    for i in range(len(indices)):
        print('\t\tOrigin ' + '{:>5}'.format(indices[i] + 1), end='\r')
        
        origZone = indices[i]
        destZoneIndex = np.where(tripMatrixParcels[:,0]==(origZone+1))[0]
                        
        if len(destZoneIndex) > 0:            
            # Schrijf de volumes op de links
            for j in destZoneIndex:
                destZone = tripMatrixParcels[j, 1] - 1
                
                if doHybridRoutes:
                    route = get_route(origZone, destZone, prevVan, linkDict)
                    hybridRoute = get_route(origZone, destZone, prevVanHybrid, linkDict)
                else:
                    route = get_route(origZone, destZone, prevVan, linkDict)

                # Welke trips worden allemaal gemaakt op de HB van de huidige iteratie van de ij-loop
                currentTrips = trips[(trips[:,1]==(origZone+1)) & (trips[:,2]==(destZone+1)), :]

                # Route per wegtype
                routeStad      = route[stadArray[route]]
                routeBuitenweg = route[buitenwegArray[route]]
                routeSnelweg   = route[snelwegArray[route]]

                # Emissies voor een enkele trip
                stadEmissions      = np.zeros((len(routeStad),nET))
                buitenwegEmissions = np.zeros((len(routeBuitenweg),nET))
                snelwegEmissions   = np.zeros((len(routeSnelweg),nET))
                for et in range(nET):
                    stadEmissions[:,et]      = distArray[routeStad]      * emissionFacStad[et]
                    buitenwegEmissions[:,et] = distArray[routeBuitenweg] * emissionFacBuitenweg[et]
                    snelwegEmissions[:,et]   = distArray[routeSnelweg]   * emissionFacSnelweg[et]
                
                if doHybridRoutes:
                    # Route per wegtype
                    hybridRouteStad      = hybridRoute[stadArray[hybridRoute]]
                    hybridRouteBuitenweg = hybridRoute[buitenwegArray[hybridRoute]]
                    hybridRouteSnelweg   = hybridRoute[snelwegArray[hybridRoute]]
    
                    # Welke links liggen in het ZE-gebied
                    hybridZEZstad      = (ZEZarray[hybridRouteStad     ] == 1)
                    hybridZEZbuitenweg = (ZEZarray[hybridRouteBuitenweg] == 1)
                    hybridZEZsnelweg   = (ZEZarray[hybridRouteSnelweg  ] == 1)
    
                    # Emissies voor een enkele trip
                    hybridStadEmissions      = np.zeros((len(hybridRouteStad),nET))
                    hybridBuitenwegEmissions = np.zeros((len(hybridRouteBuitenweg),nET))
                    hybridSnelwegEmissions   = np.zeros((len(hybridRouteSnelweg),nET))
                    for et in range(nET):
                        hybridStadEmissions[:,et]      = distArray[hybridRouteStad]      * emissionFacStad[et]
                        hybridBuitenwegEmissions[:,et] = distArray[hybridRouteBuitenweg] * emissionFacBuitenweg[et]
                        hybridSnelwegEmissions[:,et]   = distArray[hybridRouteSnelweg]   * emissionFacSnelweg[et]

                # Bereken en schrijf de emissies op de links                                                               
                for trip in range(len(currentTrips)):
                    ct = int(currentTrips[trip, 6])
                    
                    if ct == 3:
                        linkTripsArray[hybridRoute] += 1
                    else:
                        linkTripsArray[route] += 1

                    # If combustion type is fuel or bio-fuel                             
                    if ct in [0, 4]:
                        stadEmissionsTrip      = stadEmissions
                        buitenwegEmissionsTrip = buitenwegEmissions
                        snelwegEmissionsTrip   = snelwegEmissions
                                        
                        for et in range(nET):                        
                            linkEmissionsArray[routeStad,      et] += stadEmissionsTrip[:,et]
                            linkEmissionsArray[routeBuitenweg, et] += buitenwegEmissionsTrip[:,et]
                            linkEmissionsArray[routeSnelweg,   et] += snelwegEmissionsTrip[:,et]

                        # CO2-emissions for the current trip
                        parcelTripsCO2[currentTrips[trip,-1]] = (np.sum(stadEmissionsTrip[:,indexCO2]) + 
                                                                 np.sum(buitenwegEmissionsTrip[:,indexCO2]) + 
                                                                 np.sum(snelwegEmissionsTrip[:,indexCO2]))
                    # If hybrid combustion
                    elif ct == 3:
                        stadEmissionsTrip      = hybridStadEmissions
                        buitenwegEmissionsTrip = hybridBuitenwegEmissions
                        snelwegEmissionsTrip   = hybridSnelwegEmissions
                        
                        stadEmissionsTrip[hybridZEZstad]           = 0
                        buitenwegEmissionsTrip[hybridZEZbuitenweg] = 0
                        snelwegEmissionsTrip[hybridZEZsnelweg]     = 0
                                        
                        for et in range(nET):                        
                            linkEmissionsArray[hybridRouteStad,      et] += stadEmissionsTrip[:,et]
                            linkEmissionsArray[hybridRouteBuitenweg, et] += buitenwegEmissionsTrip[:,et]
                            linkEmissionsArray[hybridRouteSnelweg,   et] += snelwegEmissionsTrip[:,et]

                        # CO2-emissions for the current trip
                        parcelTripsCO2[currentTrips[trip,-1]] = (np.sum(stadEmissionsTrip[:,indexCO2]) + 
                                                                 np.sum(buitenwegEmissionsTrip[:,indexCO2]) + 
                                                                 np.sum(snelwegEmissionsTrip[:,indexCO2]))

                    else:
                        parcelTripsCO2[currentTrips[trip,-1]] = 0

    del prevVan

    return [linkTripsArray, linkEmissionsArray, parcelTripsCO2]



#%% emissions_vans
    
def emissions_vans(distArray, stadArray, buitenwegArray, snelwegArray, 
                   indexCO2, indexSO2, indexPM, indexNOX,
                   emissionFacStad, emissionFacBuitenweg, emissionFacSnelweg,
                   nLinks, nET, linkDict,
                   indices):
    '''
    Calculate the emissions and intensities of the service and construction vans.
    '''
    
    vanTripsService      = indices[1]
    vanTripsConstruction = indices[2]
    prevVan              = indices[3]
    indices              = indices[0]

    linkVanTripsArray     = np.zeros((nLinks, 2))
    linkVanEmissionsArray = [np.zeros((nLinks, nET)) for ls in ['VAN_S', 'VAN_C']]
        
    for i in range(len(indices)):
        print('\t\tOrigin ' + '{:>5}'.format(indices[i] + 1), end='\r')

        # For which destination zones to search routes
        destZones = np.where((vanTripsService[i, :] > 0) | (vanTripsConstruction[i, :] > 0))[0]
        
        # Dijkstra route search
        routes = [get_route(i, j, prevVan, linkDict) for j in destZones]
        
        j = 0
        for destZone in destZones:                            
            nTripsService      = vanTripsService[i, destZone]
            nTripsConstruction = vanTripsConstruction[i, destZone]
            
            route = np.array(list(routes[j]), dtype=int)
            
            # Route per wegtype
            routeStad      = route[stadArray[route]]
            routeBuitenweg = route[buitenwegArray[route]]
            routeSnelweg   = route[snelwegArray[route]]
            
            # Afstanden van de links op de route
            distStad      = distArray[routeStad]
            distBuitenweg = distArray[routeBuitenweg]
            distSnelweg   = distArray[routeSnelweg]
            
            # Emissies voor een enkele trip
            stadEmissions      = np.zeros((len(routeStad), nET))
            buitenwegEmissions = np.zeros((len(routeBuitenweg), nET))
            snelwegEmissions   = np.zeros((len(routeSnelweg), nET))
            for et in range(nET):
                stadEmissions[:, et]      = distStad      * emissionFacStad[et]
                buitenwegEmissions[:, et] = distBuitenweg * emissionFacBuitenweg[et]
                snelwegEmissions[:, et]   = distSnelweg   * emissionFacSnelweg[et]
                    
            # Van: Service segment                                        
            if nTripsService > 0:
                # Number of trips made on each link on the route
                linkVanTripsArray[route, 0] += nTripsService

                # Emissions on each link on the route
                stadEmissionsService      = stadEmissions      * nTripsService
                buitenwegEmissionsService = buitenwegEmissions * nTripsService
                snelwegEmissionsService   = snelwegEmissions   * nTripsService
                                                            
                for et in range(nET):
                    linkVanEmissionsArray[0][routeStad,      et] += stadEmissionsService[:, et]
                    linkVanEmissionsArray[0][routeBuitenweg, et] += buitenwegEmissionsService[:, et]
                    linkVanEmissionsArray[0][routeSnelweg,   et] += snelwegEmissionsService[:, et]
        
            # Van: Construction segment
            if nTripsConstruction > 0:
                # Number of trips made on each link on the route
                linkVanTripsArray[route,1] += nTripsConstruction

                # Emissions on each link on the route
                stadEmissionsConstruction      = stadEmissions      * nTripsConstruction
                buitenwegEmissionsConstruction = buitenwegEmissions * nTripsConstruction
                snelwegEmissionsConstruction   = snelwegEmissions   * nTripsConstruction
                                                            
                for et in range(nET):
                    linkVanEmissionsArray[1][routeStad,      et] += stadEmissionsConstruction[:, et]
                    linkVanEmissionsArray[1][routeBuitenweg, et] += buitenwegEmissionsConstruction[:, et]
                    linkVanEmissionsArray[1][routeSnelweg,   et] += snelwegEmissionsConstruction[:, et]

            j += 1
    
    del prevVan, vanTripsService, vanTripsConstruction

    return [linkVanTripsArray, linkVanEmissionsArray]

       
    
#%% For if you want to run the module from this script itself (instead of calling it from the GUI module)
        
if __name__ == '__main__':
    
    varDict = {}
    
    DATAFOLDER = 'P:/Projects_Active/21020 WVL BasGoed Logistiek wegvervoer en zero emissie/Work/Dataverzameling/'
    
    varDict['OUTPUTFOLDER'] = 'P:/Projects_Active/21020 WVL BasGoed Logistiek wegvervoer en zero emissie/Work/Runs/test4/'
    varDict['PARAMFOLDER']  = 'P:/Projects_Active/21020 WVL BasGoed Logistiek wegvervoer en zero emissie/Work/Parameterbestanden TFS/'
    
    varDict['BASGOEDFOLDER']       = DATAFOLDER + 'BasGoed/2018/'    
    
    varDict['SKIMTIME']            = DATAFOLDER + 'NRM netwerk/NRM moedernetwerk/Skims/SkimVrachtTijdNRM.mtx'
    varDict['SKIMDISTANCE']        = DATAFOLDER + 'NRM netwerk/NRM moedernetwerk/Skims/SkimVrachtAfstandNRM.mtx'
    varDict['ZONES']               = DATAFOLDER + 'NRM zones/ZON_Moederbestand_Hoog_versie_23_nov_2020.shp'   
    varDict['PARCELNODES']         = DATAFOLDER + 'Pakketsorteercentra/Pakketsorteercentra.shp'
    varDict['DISTRIBUTIECENTRA']   = DATAFOLDER + 'Distributiecentra/Distributiecentra.csv'
    varDict['TERMINALS']           = DATAFOLDER + 'Overslagzones/TerminalsWithCoordinates.csv'
    varDict['SECTOR_TO_SECTOR']    = DATAFOLDER + 'Koppeltabellen/SectorMakeUse_SectorNRM.csv'
    varDict['CONTAINER']           = DATAFOLDER + 'Container/ContainerStats.csv'
    varDict['CEP_SHARES']          = varDict['PARAMFOLDER'] + 'CEPshares.csv'
    varDict['VEHCAPACITY']         = varDict['PARAMFOLDER'] + 'CarryingCapacity.csv'
    
    varDict['NRM_TO_BG']          = DATAFOLDER + 'Koppeltabellen/NRM_ZONEBG.csv'
    varDict['BG_TO_NRM']          = DATAFOLDER + 'Koppeltabellen/ZONEBG_NRM.csv'
    varDict['BGGG_TO_NSTR_CONT']  = DATAFOLDER + 'Koppeltabellen/BGG_NSTR_Container.csv'
    varDict['BGGG_TO_NSTR_NCONT'] = DATAFOLDER + 'Koppeltabellen/BGG_NSTR_NietContainer.csv'
    varDict['BGGG_TO_LS_CONT']    = DATAFOLDER + 'Koppeltabellen/BGG_LS_Container.csv'
    varDict['BGGG_TO_LS_NCONT']   = DATAFOLDER + 'Koppeltabellen/BGG_LS_NietContainer.csv'
    
    varDict['COST_ROAD_MS']        = varDict['BASGOEDFOLDER'] + 'Parameters/modechoice_cost_weg_base.cff'
    varDict['COST_ROAD_CKM']       = varDict['BASGOEDFOLDER'] + 'Parameters/ckm_cost_road_base.cff'
    varDict['COST_AANHANGER']      = varDict['BASGOEDFOLDER'] + 'Parameters/rw_kosten_aanhanger_base.cff'
    varDict['COST_BESTEL']         = varDict['BASGOEDFOLDER'] + 'Parameters/rw_kosten_bestel_base.cff'
    varDict['COST_LZV']            = varDict['BASGOEDFOLDER'] + 'Parameters/rw_kosten_lzv_base.cff'
    varDict['COST_OPLEGGER']       = varDict['BASGOEDFOLDER'] + 'Parameters/rw_kosten_oplegger_base.cff'
    varDict['COST_SPECIAAL']       = varDict['BASGOEDFOLDER'] + 'Parameters/rw_kosten_speciaal_base.cff'
    varDict['COST_VRACHTWAGEN']    = varDict['BASGOEDFOLDER'] + 'Parameters/rw_kosten_vrachtwagen_base.cff'
    
    varDict['YEARFACTOR'] = 256
    
    varDict['PARCELS_PER_HH']	= 0.155
    varDict['PARCELS_PER_EMPL'] = 0.055
    varDict['PARCELS_MAXLOAD']	= 150
    varDict['PARCELS_DROPTIME'] = 120
    varDict['PARCELS_SUCCESS_B2C']   = 0.75
    varDict['PARCELS_SUCCESS_B2B']   = 0.95
    varDict['PARCELS_GROWTHFREIGHT'] = 1.0
    
    varDict['NODES'] = varDict['OUTPUTFOLDER'] + 'nodes.csv'
    varDict['LINKS'] = varDict['OUTPUTFOLDER'] + 'links.csv'
    
    varDict['N_MULTIROUTE' ] = 2
    varDict['SHIPMENTS_REF'] = ''
    varDict['N_CPU'] = ''
    
    varDict['LABEL'] = 'REF'
        
    # Run the module
    root = ''
    main(varDict)
    


