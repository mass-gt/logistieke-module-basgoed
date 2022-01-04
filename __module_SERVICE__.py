import pandas as pd
import numpy as np
import time
import datetime
from __functions__ import read_shape, read_mtx, write_mtx

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
        self.root.title("Progress Service Vans")
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
        
        root    = args[0]
        varDict = args[1]
         
        datapathO         = varDict['OUTPUTFOLDER'] 
        pathZones         = varDict['ZONES']
        pathParcelNodes   = varDict['PARCELNODES']
        pathSkimTravTime  = varDict['SKIMTIME']
        pathSkimDistance  = varDict['SKIMDISTANCE']
        pathCostVan       = varDict['COST_BESTEL']
        pathDistanceDecay = varDict['SERVICE_DISTANCEDECAY']
        pathCoeffPA       = varDict['SERVICE_PA']
        pathDistributieCentra = varDict['DISTRIBUTIECENTRA']
        yearFactor        = varDict['YEARFACTOR']
        year              = varDict['YEAR']

        start_time = time.time()
        
        log_file = open(datapathO + "Logfile_ServiceTrips.log", "w")
        log_file.write("Start simulation at: " + datetime.datetime.now().strftime("%y-%m-%d %H:%M")+"\n")
        
        tolerance = 0.005
        maxIter   = 50
        
        avgParcelDepotSurface = 5000
        
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

        sectorNames = ['LANDBOUW'   + yrAppendix,
                       'INDUSTRIE'  + yrAppendix,
                       'DETAIL'     + yrAppendix,
                       'DIENSTEN'   + yrAppendix,
                       'OVERIG'     + yrAppendix]


        # --------------------------- Import data -----------------------------------
        
        print('Importing data...')
        log_file.write('Importing data...\n')
        if root != '':
            root.update_statusbar('Importing data...')
        
        # Import cost parameters
        costVan  = np.array(pd.read_csv(pathCostVan, sep='\t'))[:,[0,1,2]]
        costPerKilometer = np.average(costVan[:,1])
        costPerHour      = np.average(costVan[:,2])
        
        # Import distance decay parameters
        distanceDecay = pd.read_csv(pathDistanceDecay, index_col=0)
        alphaService = distanceDecay.at['Service', 'ALPHA']
        betaService  = distanceDecay.at['Service', 'BETA']
        alphaBouw = distanceDecay.at['Construction', 'ALPHA']
        betaBouw  = distanceDecay.at['Construction', 'BETA']
        
        # Import zones data
        zones = read_shape(pathZones)
        zones.index = zones['SEGNR_2018']
        nZones = len(zones)
        zonesNL = np.where(zones['LAND']==1)[0]
        
        # Import regression coefficients
        regrCoeffs = pd.read_csv(pathCoeffPA, sep=',', index_col=[0])
            
        # Parcel nodes
        parcelNodes = read_shape(pathParcelNodes)

        # Import logistic nodes data
        logNodes = pd.read_csv(pathDistributieCentra)
        logNodes = logNodes[~pd.isna(logNodes['SEGNR_2018'])]
        logNodes.index = np.arange(len(logNodes))
        
        if root != '':
            root.progressBar['value'] = 0.5
            
        # Skim with travel times and distances
        skimTravTime = read_mtx(pathSkimTravTime)
        skimDistance = read_mtx(pathSkimDistance)

        if root != '':
            root.progressBar['value'] = 2.0
            
            
        # ---------------------- Productions and attractions -------------------------
 
        print('Calculating productions and attractions...')
        log_file.write('Calculating productions and attractions...\n')

        if root != '':
            root.update_statusbar('Calculating productions and attractions...')
            
        # Surface of DCs per zone
        surfaceDC = np.zeros(nZones, dtype=float)
        for i in range(len(logNodes)):
            zone    = logNodes.at[i, 'SEGNR_2018']
            surface = logNodes.at[i, 'oppervlak']
            surfaceDC[int(zone-1)] += surface      
        
        # Surface of parcel nodes per zone
        surfaceParcelDepot = np.zeros(nZones, dtype=float)
        for i in range(len(parcelNodes)):
            zone = parcelNodes.at[i, 'SEGNR_2018']
            surfaceParcelDepot[int(zone-1)] += avgParcelDepotSurface
        
        # Jobs per sector
        jobs = {}
        for sector in sectorNames:
            jobs[sector] = np.array(zones[sector], dtype=float)
        
        # Number of inhabitants
        population = np.array(zones['INWONERS' + yrAppendix])        
                
        # Determine produced trips per zone for service and construction
        prodService = np.zeros(nZones, dtype=int)
        prodBouw    = np.zeros(nZones, dtype=int)
        
        # For the zones in the study area (ZH)
        for i in zonesNL:
            prodService[i] = regrCoeffs.at['Service','DC_OPP'    ] * surfaceDC[i]          + \
                             regrCoeffs.at['Service','PARCEL_OPP'] * surfaceParcelDepot[i] + \
                             regrCoeffs.at['Service','LANDBOUW'  ] * jobs['LANDBOUW'  + yrAppendix][i]  + \
                             regrCoeffs.at['Service','INDUSTRIE' ] * jobs['INDUSTRIE' + yrAppendix][i]  + \
                             regrCoeffs.at['Service','DETAIL'    ] * jobs['DETAIL'    + yrAppendix][i]  + \
                             regrCoeffs.at['Service','DIENSTEN'  ] * jobs['DIENSTEN'  + yrAppendix][i]  + \
                             regrCoeffs.at['Service','OVERIG'    ] * jobs['OVERIG'    + yrAppendix][i]  + \
                             regrCoeffs.at['Service','INWONERS'  ] * population[i]

            prodBouw[i]    = regrCoeffs.at['Construction','DC_OPP'    ] * surfaceDC[i]          + \
                             regrCoeffs.at['Construction','PARCEL_OPP'] * surfaceParcelDepot[i] + \
                             regrCoeffs.at['Construction','LANDBOUW'  ] * jobs['LANDBOUW'  + yrAppendix][i]  + \
                             regrCoeffs.at['Construction','INDUSTRIE' ] * jobs['INDUSTRIE' + yrAppendix][i]  + \
                             regrCoeffs.at['Construction','DETAIL'    ] * jobs['DETAIL'    + yrAppendix][i]  + \
                             regrCoeffs.at['Construction','DIENSTEN'  ] * jobs['DIENSTEN'  + yrAppendix][i]  + \
                             regrCoeffs.at['Construction','OVERIG'    ] * jobs['OVERIG'    + yrAppendix][i]  + \
                             regrCoeffs.at['Construction','INWONERS'  ] * population[i]
        
        if root != '':
            root.progressBar['value'] = 4.0
            
        
        # ------------------------- Trip distribution -------------------------------

        print('Trip distribution...')
        log_file.write('Trip distribution...\n')
        if root != '':
            root.update_statusbar('Trip distribution...')
                        
        print('\tConstructing initial matrix...')
        log_file.write('\tConstructing initial matrix...\n')
        
        # Travel costs
        skimCost = costPerHour * (skimTravTime / 3600) + costPerKilometer * (skimDistance / 1000)
        skimCost = skimCost.reshape(nZones,nZones)
        
        # Intrazonal costs (half of costs to nearest zone in terms of travel costs)
        for i in range(nZones):
            skimCost[i,i] = 0.5 * np.min(skimCost[i,skimCost[i,:]>0])
            
        # Travel resistance
        matrixService = 100 / (1 + (np.exp(alphaService) * skimCost ** betaService))
        matrixBouw    = 100 / (1 + (np.exp(alphaBouw)    * skimCost ** betaBouw))
        
        # Multiply by productions and then attractions to get start matrix (assumed: productions = attractions)
        matrixService *= np.tile(prodService, (len(prodService), 1))
        matrixBouw    *= np.tile(prodBouw,    (len(prodBouw   ), 1))
        
        matrixService *= np.tile(prodService, (len(prodService), 1)).transpose()
        matrixBouw    *= np.tile(prodBouw,    (len(prodBouw   ), 1)).transpose()
                
        if root != '':
            root.progressBar['value'] = 6.0
            
        print('\tDistributing service trips...')
        log_file.write('\tDistributing service trips...\n')
        matrixService = distribute_trips(prodService,
                                         matrixService,
                                         tolerance, maxIter,
                                         yearFactor,
                                         log_file, root, 4, 46)


        print('\tDistributing construction trips...')
        log_file.write('\tDistributing construction trips...\n')
        matrixBouw = distribute_trips(prodBouw, 
                                      matrixBouw,
                                      tolerance, maxIter,
                                      yearFactor,
                                      log_file, root, 46, 86)      

        if root != '':
            root.progressBar['value'] = 86.0

            
        # ------------------------ Writing OD trip matrices -------------------------
               
        print('\tWriting trip matrices...'), log_file.write('Writing trip matrices...\n')
        if root != '':
            root.update_statusbar('Writing trip matrices...')
                    
        write_mtx(datapathO + 'TripsVanService.mtx', matrixService.flatten(), nZones)
        
        if root != '':
            root.progressBar['value'] = 93.0
                
        write_mtx(datapathO + 'TripsVanConstruction.mtx', matrixBouw.flatten(), nZones)  
        
        if root != '':
            root.progressBar['value'] = 100.0
            
            
        # --------------------------- End of module ---------------------------------
            
        totaltime = round(time.time() - start_time, 2)
        print('Finished. Run time: ' + str(round(totaltime,2)) + ' seconds')
        log_file.write("Total runtime: %s seconds\n" % (totaltime))  
        log_file.write("End simulation at: "+datetime.datetime.now().strftime("%y-%m-%d %H:%M")+"\n")
        log_file.close()    

        if root != '':
            root.update_statusbar("Service Vans: Done")
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
                root.update_statusbar("Service Vans: Execution failed!")
                errorMessage = 'Execution failed!\n\n' + str(root.returnInfo[1][0])
                errorMessage = errorMessage + '\n\n' + str(root.returnInfo[1][1])
                root.error_screen(text=errorMessage, size=[900,350])                
            
            else:
                return root.returnInfo
        else:
            return [1, [sys.exc_info()[0], traceback.format_exc()]]



def distribute_trips(prod, matrix, tolerance, maxIter, yearFactor,
                     log_file, root, startProgress, endProgress):
    '''
    Perform trip distribution given the zonal productions and start matrix. 
    '''
    itern = 0
    conv  = tolerance + 100
    
    nZones = len(matrix)
    
    while (itern < maxIter) and (conv > tolerance):
        itern += 1
        print('\t\tIteration ' + str(itern))
        log_file.write('\t\tIteration ' + str(itern) + '\n')
        
        maxColScaleFac = 0
        totalRows = np.sum(matrix, axis=0)
        
        for j in range(nZones):
            total = totalRows[j]
    
            if total > 0:
                scaleFacCol = prod[j] / total
    
                if abs(scaleFacCol)> abs(maxColScaleFac):
                    maxColScaleFac = scaleFacCol
    
                matrix[:,j] *= scaleFacCol
    
        maxRowScaleFac = 0
        totalCols = np.sum(matrix, axis=1)
        
        for i in range(nZones):        
            total = totalCols[i]
    
            if total > 0:
                scaleFacRow = prod[i] / total
    
                if abs(scaleFacRow)> abs(maxRowScaleFac):
                    maxRowScaleFac = scaleFacRow
    
                matrix[i,:] *= scaleFacRow
    
        conv = max(abs(maxColScaleFac - 1),abs(maxRowScaleFac - 1))            
        print('\t\tConvergence ' + str(round(conv, 4)))
        log_file.write('\t\tConvergence ' + str(round(conv, 4)) + '\n')
        
        if root != '':
            root.progressBar['value'] = (startProgress + 
                                         0.75 * (endProgress - startProgress) * itern / maxIter)

    if conv > tolerance:
        print('Warning! Convergence is lower than the tolerance criterion, more iterations might be needed.')
        log_file.write('Warning! Convergence is lower than the tolerance criterion, more iterations might be needed.' + '\n')
                 
    print('\t\tRounding...'), log_file.write('\t\tRounding...' + '\n')
    
    # From year to day and remove very small ODs
    matrix /= yearFactor
    matrix = np.round(matrix, 3)
    matrix *= yearFactor

    # Perform the distribution again after rounding
    itern = 0
    conv  = tolerance + 100

    while (itern < maxIter) and (conv > tolerance):
        itern += 1
        print('\t\tIteration ' + str(itern))
        log_file.write('\t\tIteration ' + str(itern) + '\n')

        maxColScaleFac = 0
        totalRows = np.sum(matrix, axis=0)
        
        for j in range(nZones):
            total = totalRows[j]
    
            if total > 0:
                scaleFacCol = prod[j] / total
    
                if abs(scaleFacCol)> abs(maxColScaleFac):
                    maxColScaleFac = scaleFacCol
    
                matrix[:,j] *= scaleFacCol
    
        maxRowScaleFac = 0
        totalCols = np.sum(matrix, axis=1)
        
        for i in range(nZones):        
            total = totalCols[i]
    
            if total > 0:
                scaleFacRow = prod[i] / total
    
                if abs(scaleFacRow)> abs(maxRowScaleFac):
                    maxRowScaleFac = scaleFacRow
    
                matrix[i,:] *= scaleFacRow
    
        conv = max(abs(maxColScaleFac - 1),abs(maxRowScaleFac - 1))           
        print('\t\tConvergence ' + str(round(conv, 4)))
        log_file.write('\t\tConvergence ' + str(round(conv, 4)) + '\n')
        
        if root != '':
            root.progressBar['value'] = (startProgress +
                                         0.75 * (endProgress - startProgress) +
                                         0.25 * (endProgress - startProgress) * itern / maxIter)
            
    matrix /= yearFactor
    
    return matrix

    


#%% For if you want to run the module from this script itself (instead of calling it from the GUI module)
        
if __name__ == '__main__':
    
    varDict = {}
    
    DATAFOLDER = 'P:/Projects_Active/21020 WVL BasGoed Logistiek wegvervoer en zero emissie/Work/Dataverzameling/'
    
    varDict['OUTPUTFOLDER'] = 'P:/Projects_Active/21020 WVL BasGoed Logistiek wegvervoer en zero emissie/Work/Runs/test2/'
    varDict['PARAMFOLDER']	= 'P:/Projects_Active/21020 WVL BasGoed Logistiek wegvervoer en zero emissie/Work/Parameterbestanden TFS/'
    
    varDict['BASGOEDFOLDER']       = DATAFOLDER + 'BasGoed/'    
    
    varDict['SKIMTIME']            = DATAFOLDER + 'NRM netwerk/NRM moedernetwerk/Skims/SkimBestelTijdNRM.mtx'
    varDict['SKIMDISTANCE']        = DATAFOLDER + 'NRM netwerk/NRM moedernetwerk/Skims/SkimBestelAfstandNRM.mtx'
    varDict['ZONES']               = DATAFOLDER + 'NRM zones/ZON_Moederbestand_Hoog_versie_23_nov_2020.shp'   
    varDict['PARCELNODES']         = DATAFOLDER + 'Pakketsorteercentra/Pakketsorteercentra.shp'
    varDict['DISTRIBUTIECENTRA']   = DATAFOLDER + 'Distributiecentra/Distributiecentra.csv'
    varDict['TERMINALS']           = DATAFOLDER + 'Overslagzones/TerminalsWithCoordinates.csv'
    varDict['SECTOR_TO_SECTOR']    = 'P:/Projects_Active/21020 WVL BasGoed Logistiek wegvervoer en zero emissie/Work/Dataverzameling/Koppeltabellen/SectorMakeUse_SectorNRM.csv'
    varDict['CONTAINER']           = 'P:/Projects_Active/21020 WVL BasGoed Logistiek wegvervoer en zero emissie/Work/Dataverzameling/Container/ContainerStats.csv'
    varDict['CEP_SHARES']          = varDict['PARAMFOLDER'] + 'CEPshares.csv'
    varDict['VEHCAPACITY']         = varDict['PARAMFOLDER'] + 'CarryingCapacity.csv'
    
    varDict['NRM_TO_BG']          = 'P:/Projects_Active/21020 WVL BasGoed Logistiek wegvervoer en zero emissie/Work/Dataverzameling/Koppeltabellen/NRM_ZONEBG.csv'
    varDict['BG_TO_NRM']          = 'P:/Projects_Active/21020 WVL BasGoed Logistiek wegvervoer en zero emissie/Work/Dataverzameling/Koppeltabellen/ZONEBG_NRM.csv'
    varDict['BGGG_TO_NSTR_CONT']  = 'P:/Projects_Active/21020 WVL BasGoed Logistiek wegvervoer en zero emissie/Work/Dataverzameling/Koppeltabellen/BGG_NSTR_Container.csv'
    varDict['BGGG_TO_NSTR_NCONT'] = 'P:/Projects_Active/21020 WVL BasGoed Logistiek wegvervoer en zero emissie/Work/Dataverzameling/Koppeltabellen/BGG_NSTR_NietContainer.csv'
    varDict['BGGG_TO_LS_CONT']    = 'P:/Projects_Active/21020 WVL BasGoed Logistiek wegvervoer en zero emissie/Work/Dataverzameling/Koppeltabellen/BGG_LS_Container.csv'
    varDict['BGGG_TO_LS_NCONT']   = 'P:/Projects_Active/21020 WVL BasGoed Logistiek wegvervoer en zero emissie/Work/Dataverzameling/Koppeltabellen/BGG_LS_NietContainer.csv'
    
    varDict['COST_ROAD_MS']        = varDict['BASGOEDFOLDER'] + 'Parameters/modechoice_cost_weg_base.cff'
    varDict['COST_ROAD_CKM']       = varDict['BASGOEDFOLDER'] + 'Parameters/ckm_cost_road_base.cff'
    varDict['COST_AANHANGER']      = varDict['BASGOEDFOLDER'] + 'Parameters/rw_kosten_aanhanger_base.cff'
    varDict['COST_BESTEL']         = varDict['BASGOEDFOLDER'] + 'Parameters/rw_kosten_bestel_base.cff'
    varDict['COST_LZV']            = varDict['BASGOEDFOLDER'] + 'Parameters/rw_kosten_lzv_base.cff'
    varDict['COST_OPLEGGER']       = varDict['BASGOEDFOLDER'] + 'Parameters/rw_kosten_oplegger_base.cff'
    varDict['COST_SPECIAAL']       = varDict['BASGOEDFOLDER'] + 'Parameters/rw_kosten_speciaal_base.cff'
    varDict['COST_VRACHTWAGEN']    = varDict['BASGOEDFOLDER'] + 'Parameters/rw_kosten_vrachtwagen_base.cff'

    varDict['SERVICE_PA']            = varDict['PARAMFOLDER'] + 'Params_PA_SERVICE.csv'
    varDict['SERVICE_DISTANCEDECAY'] = varDict['PARAMFOLDER'] + 'Params_DistanceDecay_SERVICE.csv'
    
    varDict['YEARFACTOR'] = 256
    
    varDict['PARCELS_PER_HH']	     = 0.112
    varDict['PARCELS_PER_EMPL']      = 0.041
    varDict['PARCELS_MAXLOAD']	     = 180
    varDict['PARCELS_DROPTIME']      = 120
    varDict['PARCELS_SUCCESS_B2C']   = 0.75
    varDict['PARCELS_SUCCESS_B2B']   = 0.95
    varDict['PARCELS_GROWTHFREIGHT'] = 1.0
    
    # varDict['SHIPMENTS_REF'] = varDict['OUTPUTFOLDER'] + 'Shipments_REF.csv'
    varDict['SHIPMENTS_REF'] = ''
    varDict['N_CPU'] = ""
    
    varDict['LABEL'] = 'REF'
    varDict['YEAR'] = 2018

    # Run the module
    root = ''
    main(varDict)

        
        

