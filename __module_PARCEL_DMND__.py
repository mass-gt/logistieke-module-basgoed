# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 14:20:50 2020

@author: modelpc
"""

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
        self.width  = 500
        self.height = 60
        self.bg     = 'black'
        self.fg     = 'white'
        self.font   = 'Verdana'
        
        # Create a GUI window
        self.root = tk.Tk()
        self.root.title("Progress Parcel Demand")
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

    
    try:
        # -------------------- Define datapaths -----------------------------------
        
        start_time = time.time()
        
        root    = args[0]
        varDict = args[1]
                
        if root != '':
            root.progressBar['value'] = 0

        datapathO = varDict['OUTPUTFOLDER']
        zonesPath        = varDict['ZONES']
        skimDistancePath = varDict['SKIMDISTANCE']
        parcelNodesPath  = varDict['PARCELNODES']
        cepSharesPath    = varDict['CEP_SHARES']  
        label            = varDict['LABEL']
        year             = varDict['YEAR']

        parcelsPerHH     = varDict['PARCELS_PER_HH']
        parcelsPerEmpl   = varDict['PARCELS_PER_EMPL']
        parcelSuccessB2B = varDict['PARCELS_SUCCESS_B2B']
        parcelSuccessB2C = varDict['PARCELS_SUCCESS_B2C']

        zezConsolidationPath = varDict['ZEZ_CONSOLIDATION']
        zezScenarioPath      = varDict['ZEZ_SCENARIO']
        zezZonesPath         = varDict['ZEZ_ZONES']

        seed = varDict['SEED']

        log_file = open(datapathO + "Logfile_ParcelDemand.log", "w")
        log_file.write("Start simulation at: " + datetime.datetime.now().strftime("%y-%m-%d %H:%M")+"\n")

        if seed != '':
            np.random.seed(seed)

        # Plek van de zonale attributen
        # Alle attributen komen 4 x voor
        # De volgorde is: 2030, 2040, 2050, 2018
        # Het basisjaar staat dus achteraan.
        if year == 2018:
            yrAppendix = '_3'
        elif year == 2030:
            yrAppendix = ''
        elif year == 2040:
            yrAppendix = '_1'
        elif year == 2050:
            yrAppendix = '_2'
        else:
            raise Exception('YEAR parameter is not 2018, 2030, 2040 or 2050, but ' +
                            str(year) + ' instead.')

        # ---------------------------- Import data --------------------------------
        print('Importing data...'), log_file.write('Importing data...\n')
        print('\tZones...')
        zones = read_shape(zonesPath)
        zones = zones[zones['LAND']==1]
        zones.index = zones['SEGNR_2018']
        zonesX = np.array(zones['XCOORD'])
        zonesY = np.array(zones['YCOORD'])
        
        print('\tParcel depots...')
        parcelNodes, coords = read_shape(parcelNodesPath, returnGeometry=True)
        parcelNodes['X']    = [coords[i]['coordinates'][0] for i in range(len(coords))]
        parcelNodes['Y']    = [coords[i]['coordinates'][1] for i in range(len(coords))]
        parcelNodes.index   = parcelNodes['id'].astype(int)
        parcelNodes         = parcelNodes.sort_index()    
        nParcelNodes        = len(parcelNodes)
           
        cepShares = pd.read_csv(cepSharesPath, index_col=0)
        cepList   = np.unique(parcelNodes['CEP'])
        cepNodes = [np.where(parcelNodes['CEP']==str(cep))[0] for cep in cepList]
        cepNodeDict = {}
        for cepNo in range(len(cepList)):
            cepNodeDict[cepList[cepNo]] = cepNodes[cepNo]
        
        
        # ------------------ Get skim data and make parcel skim for REF --------------------
        print('\tSkim...')
        skimDistance = read_mtx(skimDistancePath)
        nZones       = int(len(skimDistance)**0.5)
        parcelSkim   = np.zeros((nZones, nParcelNodes))
            
        # Skim with travel times between parcel nodes and all other zones
        i = 0
        for orig in parcelNodes['SEGNR_2018']:
            dest = 1 + np.arange(nZones)
            parcelSkim[:,i] = np.round(skimDistance[(orig-1)*nZones+(dest-1)] / 1000, 4)     
            i += 1
        
        # ---- Generate parcels for each zone based on households and employment ----------
        # ------------- and select a parcel node for each parcel --------------------------
        print('Generating parcels...'), log_file.write('Generating parcels...\n')
        
        # Calculate number of parcels per zone based on number of households and total number of parcels on an average day
        zones['parcels']  = (zones['HUISH'    + yrAppendix] * parcelsPerHH   / parcelSuccessB2C)
        zones['parcels'] += (zones['BANENTOT' + yrAppendix] * parcelsPerEmpl / parcelSuccessB2B)
        zones['parcels']  = np.array(np.round(zones['parcels'],0), dtype=int)
        
        # Spread over couriers based on market shares
        for cep in cepList:
            zones['parcels_' + str(cep)] = np.round(cepShares['ShareTotal'][cep] * zones['parcels'], 0)
            zones['parcels_' + str(cep)] = zones['parcels_' + str(cep)].astype(int)
        
        # Total number of parcels per courier
        nParcels  = int(zones[["parcels_"+str(cep) for cep in cepList]].sum().sum())
        
        # Put parcel demand in Numpy array (faster indexing)
        cols    = ['PARCEL_ID', 'ORIG_NRM', 'DEST_NRM', 'DEPOT_ID']
        parcels = np.zeros((nParcels,len(cols)), dtype=int)
        parcelsCep = np.array(['' for i in range(nParcels)], dtype=object)
        
        # Now determine for each zone and courier from which depot the parcels are delivered
        count = 0
        for zoneID in zones['SEGNR_2018']:
            
            if zones.at[zoneID,'parcels'] > 0:
            
                for cep in cepList:
                    
#                    if len(cepNodeDict[cep]) > 1:
#                        # Check the two nearest DCs from the current CEP based on distance in skim
#                        distances = parcelSkim[zoneID-1, cepNodeDict[cep]]
#                        indices   = distances.argsort()[:2]
#                        
#                        # If the second nearest is not too far away (50km), spread parcels over both DCs
#                        if distances[indices[1]] < 50:
#                            parcelNodeIndex = [cepNodeDict[cep][i] for i in indices]
#                            
#                        # Else only choose the nearest DC
#                        else:
#                            parcelNodeIndex = [cepNodeDict[cep][indices[0]]]
#                            
#                    else:
#                        # Select DC of current CEP based shortest distance in skim
#                        parcelNodeIndex = [cepNodeDict[cep][parcelSkim[zoneID-1, cepNodeDict[cep]].argmin()]]
                
                    # Select DC of current CEP based shortest distance in skim
                    parcelNodeIndex = [cepNodeDict[cep][parcelSkim[zoneID-1, cepNodeDict[cep]].argmin()]]
                    
                    # Fill the df parcels with parcels, zone after zone. Parcels consist of ID, D and O zone and parcel node number
                    # in ongoing df from index count-1 the next x=no. of parcels rows, fill the cell in the column Parcel_ID with a number 
                    n = zones.loc[zoneID,'parcels_' + str(cep)]
                    
                    # Parcel_ID
                    parcels[count:count+n,0]  = np.arange(count+1, count+1+n,dtype=int)  
                    
                    # ORIG_NRM
                    parcels[count:count+n,1]  = parcelNodes['SEGNR_2018'][parcelNodeIndex[0]+1] 
                    
                    # DEST_NRM
                    parcels[count:count+n,2]  = zoneID
                    
                    # DEPOT_ID
                    if len(parcelNodeIndex) > 1:
                        start   = count
                        halfway = int(count + np.floor(n/2))
                        end     = count + n
                        parcels[start:halfway,3] = parcelNodeIndex[0] + 1
                        parcels[halfway:end,  3] = parcelNodeIndex[1] + 1
                    else:
                        parcels[count:count+n,3] = parcelNodeIndex[0] + 1
                    
                    # CEP
                    parcelsCep[count:count+n] = cep                                          
                
                    count += zones['parcels_' + str(cep)][zoneID]
        
        # Put the parcel demand data back in a DataFrame
        parcels = pd.DataFrame(parcels, columns=cols)
        parcels['CEP'] = parcelsCep
        
        # Default vehicle type for parcel deliveries: vans
        parcels['VEHTYPE'] = 7
      

        # ------------------- Rerouting through UCCs in the UCC-scenario ------------------
        if label == 'UCC': 
            
            vtNamesUCC = ['LEVV','Moped','Van','Truck','TractorTrailer','WasteCollection','SpecialConstruction']
            nLogSeg = 8
            
            # Logistic segment is 6: parcels
            ls = 6

            # Write the REF parcel demand
            print(f"Writing parcels to {datapathO}ParcelDemand_{label}.csv")
            log_file.write(f"Writing parcels to {datapathO}ParcelDemand_{label}.csv\n")
            parcels.to_csv(f"{datapathO}ParcelDemand_{label}.csv", index=False)  

            # Consolidation potential per logistic segment (for UCC scenario)
            probConsolidation = np.array(pd.read_csv(zezConsolidationPath, index_col='Segment'))
            
            # Vehicle/combustion shares (for UCC scenario)
            sharesUCC  = pd.read_csv(zezScenarioPath, index_col='Segment')
        
            # Assume no consolidation potential and vehicle type switch for dangerous goods
            sharesUCC = np.array(sharesUCC)[:-1,:-1]
            
            # Only vehicle shares (summed up combustion types)
            sharesVehUCC = np.zeros((nLogSeg-1,len(vtNamesUCC)))
            for ls in range(nLogSeg-1):
                sharesVehUCC[ls,0] = np.sum(sharesUCC[ls,0:5])
                sharesVehUCC[ls,1] = np.sum(sharesUCC[ls,5:10])
                sharesVehUCC[ls,2] = np.sum(sharesUCC[ls,10:15])
                sharesVehUCC[ls,3] = np.sum(sharesUCC[ls,15:20])
                sharesVehUCC[ls,4] = np.sum(sharesUCC[ls,20:25])
                sharesVehUCC[ls,5] = np.sum(sharesUCC[ls,25:30])            
                sharesVehUCC[ls,6] = np.sum(sharesUCC[ls,30:35])
                sharesVehUCC[ls,:] = np.cumsum(sharesVehUCC[ls,:]) / np.sum(sharesVehUCC[ls,:])

            # Couple these vehicle types to Harmony vehicle types
            vehUccToVeh = {0:9, 1:10, 2:8, 3:1, 4:5, 5:6, 6:6}              

            print('Redirecting parcels via UCC...'), log_file.write('Redirecting parcels via UCC...\n')
            
            parcels['FROM_UCC'] = 0
            parcels['TO_UCC'  ] = 0
          
            destZones    = np.array(parcels['DEST_NRM'].astype(int))
            depotNumbers = np.array(parcels['DEPOT_ID'].astype(int))
            
            zezZones = pd.read_csv(zezZonesPath, sep=',')
            zezArray = np.zeros(nZones, dtype=int)
            uccArray = np.zeros(nZones, dtype=int)
            zezArray[np.array(zezZones['SEGNR_2018'], dtype=int) - 1] = 1
            uccArray[np.array(zezZones['SEGNR_2018'], dtype=int) - 1] = zezZones['UCC']

            isDestZEZ = ((zezArray[destZones - 1] == 1) &
                         (probConsolidation[ls][0] > np.random.rand(len(parcels))))
            whereDestZEZ = np.where(isDestZEZ)[0]
                        
            newParcels = np.zeros(parcels.shape, dtype=object)

            count = 0
            
            for i in whereDestZEZ:
                                  
                trueDest = destZones[i]
                
                # Redirect to UCC
                parcels.at[i,'DEST_NRM'] = uccArray[trueDest - 1]
                parcels.at[i,'TO_UCC'] = 1
                
                # Add parcel set to ZEZ from UCC
                newParcels[count, 1] = uccArray[trueDest - 1] # Origin
                newParcels[count, 2] = trueDest               # Destination
                newParcels[count, 3] = depotNumbers[i]        # Depot ID
                newParcels[count, 4] = parcelsCep[i]          # Courier name
                newParcels[count, 5] = vehUccToVeh[np.where(sharesVehUCC[ls,:] > np.random.rand())[0][0]] # Vehicle type
                newParcels[count, 6] = 1                      # From UCC
                newParcels[count, 7] = 0                      # To UCC
                
                count += 1

            newParcels = pd.DataFrame(newParcels)
            newParcels.columns = parcels.columns            
            newParcels = newParcels.iloc[np.arange(count),:]
            
            dtypes = {'PARCEL_ID':int, 'ORIG_NRM':int, 'DEST_NRM':int, 'DEPOT_ID':int, \
                      'CEP':str,       'VEHTYPE':int,  'FROM_UCC':int, 'TO_UCC':int}
            for col in dtypes.keys():
                newParcels[col] = newParcels[col].astype(dtypes[col])
            
            parcels = parcels.append(newParcels)        
            parcels.index = np.arange(len(parcels))
            parcels['PARCEL_ID'] = np.arange(1, len(parcels) + 1)
            
            nParcels = len(parcels)
        
            
        # ------------------------- Prepare output -------------------------------- 
        print(f"Writing parcels CSV to     {datapathO}ParcelDemand_{label}.csv"), 
        log_file.write(f"Writing parcels to {datapathO}ParcelDemand_{label}.csv\n")
                
        # Aggregate to number of parcels per zone and export to geojson
        if label == 'UCC':     
            parcelsShape = pd.pivot_table(parcels, 
                                          values=['PARCEL_ID'],
                                          index=["DEPOT_ID", 'CEP','DEST_NRM', 'ORIG_NRM', 'VEHTYPE', 'FROM_UCC', 'TO_UCC'],
                                          aggfunc = {'DEPOT_ID':np.mean, 
                                                     'CEP':'first',  
                                                     'ORIG_NRM':np.mean, 
                                                     'DEST_NRM':np.mean, 
                                                     'PARCEL_ID':'count', 
                                                     'VEHTYPE':np.mean,
                                                     'FROM_UCC': np.mean,
                                                     'TO_UCC': np.mean})
            parcelsShape = parcelsShape.rename(columns={'PARCEL_ID':'N_PARCELS'})
            parcelsShape = parcelsShape.set_index(np.arange(len(parcelsShape)))
            parcelsShape = parcelsShape.reindex(columns=[ 'ORIG_NRM','DEST_NRM', 'N_PARCELS', 'DEPOT_ID', 'CEP','VEHTYPE', 'FROM_UCC', 'TO_UCC'])
            parcelsShape = parcelsShape.astype({'DEPOT_ID': int,
                                                'ORIG_NRM': int,
                                                'DEST_NRM': int,
                                                'N_PARCELS': int,
                                                'VEHTYPE': int,
                                                'FROM_UCC': int,
                                                'TO_UCC': int})
                 
        else:
            parcelsShape = pd.pivot_table(parcels, 
                                          values=['PARCEL_ID'], 
                                          index=["DEPOT_ID", 'CEP','DEST_NRM', 'ORIG_NRM', 'VEHTYPE'],
                                          aggfunc = {'DEPOT_ID': np.mean,
                                                     'CEP':'first',
                                                     'ORIG_NRM': np.mean,
                                                     'DEST_NRM': np.mean,
                                                     'VEHTYPE':np.mean,
                                                     'PARCEL_ID': 'count'})
            parcelsShape = parcelsShape.rename(columns={'PARCEL_ID':'N_PARCELS'})
            parcelsShape = parcelsShape.set_index(np.arange(len(parcelsShape)))
            parcelsShape = parcelsShape.reindex(columns=[ 'ORIG_NRM','DEST_NRM', 'N_PARCELS', 'DEPOT_ID', 'CEP','VEHTYPE'])
            parcelsShape = parcelsShape.astype({'DEPOT_ID': int, 
                                                'ORIG_NRM': int, 
                                                'DEST_NRM': int, 
                                                'N_PARCELS': int, 
                                                'VEHTYPE':int})

        parcelsShape.to_csv(f"{datapathO}ParcelDemand_{label}.csv", index=False)

        print(f"Writing parcels GeoJSON to {datapathO}ParcelDemand_{label}.geojson")
        log_file.write(f"Writing shapefile to {datapathO}ParcelDemand_{label}.geojson\n")
        
        # Initialize arrays with coordinates
        Ax = np.zeros(len(parcelsShape), dtype=int)
        Ay = np.zeros(len(parcelsShape), dtype=int)
        Bx = np.zeros(len(parcelsShape), dtype=int)
        By = np.zeros(len(parcelsShape), dtype=int)
        
        # Determine coordinates of LineString for each trip
        depotIDs = np.array(parcelsShape['DEPOT_ID'])
        for i in parcelsShape.index:
            if label == 'UCC' and parcelsShape.at[i, 'FROM_UCC'] == 1:
                    Ax[i] = zonesX[parcelsShape['ORIG_NRM'][i]-1]
                    Ay[i] = zonesY[parcelsShape['ORIG_NRM'][i]-1]
                    Bx[i] = zonesX[parcelsShape['DEST_NRM'][i]-1]
                    By[i] = zonesY[parcelsShape['DEST_NRM'][i]-1]
            else:
                Ax[i] = parcelNodes['X'][depotIDs[i]]
                Ay[i] = parcelNodes['Y'][depotIDs[i]]
                Bx[i] = zonesX[parcelsShape['DEST_NRM'][i]-1]
                By[i] = zonesY[parcelsShape['DEST_NRM'][i]-1]
                
        Ax = np.array(Ax, dtype=str)
        Ay = np.array(Ay, dtype=str)
        Bx = np.array(Bx, dtype=str)
        By = np.array(By, dtype=str)
        nRecords = len(parcelsShape)
        
        with open(datapathO + f"ParcelDemand_{label}.geojson", 'w') as geoFile:
            geoFile.write('{\n' + '"type": "FeatureCollection",\n' + '"features": [\n')
            for i in range(nRecords-1):
                outputStr = ""
                outputStr = outputStr + '{ "type": "Feature", "properties": '
                outputStr = outputStr + str(parcelsShape.loc[i,:].to_dict()).replace("'",'"')
                outputStr = outputStr + ', "geometry": { "type": "LineString", "coordinates": [ [ '
                outputStr = outputStr + Ax[i] + ', ' + Ay[i] + ' ], [ '
                outputStr = outputStr + Bx[i] + ', ' + By[i] + ' ] ] } },\n'
                geoFile.write(outputStr)
                if i%500 == 0:
                    print('\t' + str(round((i / nRecords)*100, 1)) + '%', end='\r')
                    
            # Bij de laatste feature moet er geen komma aan het einde
            i += 1
            outputStr = ""
            outputStr = outputStr + '{ "type": "Feature", "properties": '
            outputStr = outputStr + str(parcelsShape.loc[i,:].to_dict()).replace("'",'"')
            outputStr = outputStr + ', "geometry": { "type": "LineString", "coordinates": [ [ '
            outputStr = outputStr + Ax[i] + ', ' + Ay[i] + ' ], [ '
            outputStr = outputStr + Bx[i] + ', ' + By[i] + ' ] ] } }\n'
            geoFile.write(outputStr)
            geoFile.write(']\n')
            geoFile.write('}')

        print('\t 100%', end='\r')
        
        # ---------------------- Sluit logfile -------------------------------------------------    
        totaltime = round(time.time() - start_time, 2)
        log_file.write("Total runtime: %s seconds\n" % (totaltime))  
        log_file.write("End simulation at: "+datetime.datetime.now().strftime("%y-%m-%d %H:%M")+"\n")
        log_file.close()    
        
        if root != '':
            root.update_statusbar("Parcel Demand: Done")
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
                root.update_statusbar("Parcel Demand: Execution failed!")
                errorMessage = 'Execution failed!\n\n' + str(root.returnInfo[1][0]) + '\n\n' + str(root.returnInfo[1][1])
                root.error_screen(text=errorMessage, size=[900,350])                
            
            else:
                return root.returnInfo
        else:
            return [1, [sys.exc_info()[0], traceback.format_exc()]]
 
    
    
#%% For if you want to run the module from this script itself (instead of calling it from the GUI module)
        
if __name__ == '__main__':
    
    varDict = {}
    
    DATAFOLDER = 'P:/Projects_Active/21020 WVL BasGoed Logistiek wegvervoer en zero emissie/Work/Dataverzameling/'
    
    varDict['OUTPUTFOLDER'] = 'P:/Projects_Active/21020 WVL BasGoed Logistiek wegvervoer en zero emissie/Work/Runs/test1/'
    varDict['PARAMFOLDER']	= 'P:/Projects_Active/21020 WVL BasGoed Logistiek wegvervoer en zero emissie/Work/Parameterbestanden TFS/'
    
    varDict['BASGOEDFOLDER']       = DATAFOLDER + 'BasGoed/UitvoerMS/'    
    
    varDict['SKIMTIME']            = DATAFOLDER + 'NRM netwerk/NRM moedernetwerk/Skims/SkimBestelTijdNRM.mtx'
    varDict['SKIMDISTANCE']        = DATAFOLDER + 'NRM netwerk/NRM moedernetwerk/Skims/SkimBestelAfstandNRM.mtx'
    varDict['ZONES']               = DATAFOLDER + 'NRM zones/ZON_Moederbestand_Hoog_versie_23_nov_2020.shp'   
    varDict['PARCELNODES']         = DATAFOLDER + 'Pakketsorteercentra/Pakketsorteercentra.shp'
    varDict['DISTRIBUTIECENTRA']   = DATAFOLDER + 'OppervlakteDCs_NRM.csv'
    varDict['COST_VEHTYPE']        = varDict['PARAMFOLDER'] + 'Cost_VehType_2016.csv'
    varDict['COST_SOURCING']       = varDict['PARAMFOLDER'] + 'Cost_Sourcing_2016.csv'
    varDict['MRDH_TO_NUTS3']       = varDict['PARAMFOLDER'] + 'MRDHtoNUTS32013.csv'
    varDict['NUTS3_TO_MRDH']       = varDict['PARAMFOLDER'] + 'NUTS32013toMRDH.csv'
    varDict['CEP_SHARES']          = varDict['PARAMFOLDER'] + 'CEPshares.csv'
    
    varDict['YEARFACTOR'] = 256
    
    varDict['PARCELS_PER_HH']	= 0.155
    varDict['PARCELS_PER_EMPL'] = 0.055
    varDict['PARCELS_MAXLOAD']	= 150
    varDict['PARCELS_DROPTIME'] = 120
    varDict['PARCELS_SUCCESS_B2C']   = 0.75
    varDict['PARCELS_SUCCESS_B2B']   = 0.95
    varDict['PARCELS_GROWTHFREIGHT'] = 1.0
    
    varDict['SHIPMENTS_REF'] = ""
    varDict['N_CPU'] = ""
    
    varDict['LABEL'] = 'REF'
        
    # Run the module
    root = ''
    main(varDict)
    

    