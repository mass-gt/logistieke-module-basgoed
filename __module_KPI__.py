# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 09:42:34 2021

@author: STH
"""


import numpy as np
import pandas as pd
import time
import datetime
from dbfread import DBF
from __functions__ import read_mtx, read_shape

# Modules nodig voor de user interface
import tkinter as tk
from tkinter.ttk import Progressbar
import zlib
import base64
import tempfile
from threading import Thread



#%% GUI

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
        self.root.title("Progress KPI Module")
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







#%% Main

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
        skimDistancePath = varDict['SKIMDISTANCE']
        pathZones        = varDict['ZONES']
        label            = varDict['LABEL']
        
        log_file = open(datapathO + "Logfile_KPI.log", 'w')
        log_file.write("Start simulation at: "+datetime.datetime.now().strftime("%y-%m-%d %H:%M")+"\n")
            
        # To convert emissions to kilograms
        emissionDivFac = [1000, 1000000, 1000000, 1000] 
        etDict    = {0:'CO2', 1:'SO2', 2:'PM', 3:'NOX'}
        etInvDict = {'CO2':0, 'SO2':1, 'PM':2, 'NOX':3}
        
        # Label of the vehicle types and logistic segments
        vtNames = ['Truck (small)', 'Truck (medium)', 'Truck (large)', 
                   'Truck+trailer (small)', 'Truck+trailer (large)', 
                   'Tractor+trailer', 
                   'Special vehicle',
                   'LZV',
                   'Van', 'LEVV', 'Moped']        
        lsNames = ['Vracht: Food (general cargo)', 
                   'Vracht: Miscellaneous (general cargo)', 
                   'Vracht: Temperature controlled', 
                   'Vracht: Facility logistics', 
                   'Vracht: Construction logistics', 
                   'Vracht: Waste', 
                   'Vracht: Parcel (consolidated flows)', 
                   'Vracht: Dangerous']
        bgNames =  ['1: Landbouw- bosbouw- en visserijproducten',
                    '2: Steenkool bruinkool en cokes',
                    '3: Ruwe aardolie en aardgas',
                    '4: Ertsen',
                    '5: Zout zand grind klei',
                    '6: Aardolieproducten',
                    '7: Chemische producten',
                    '8: Kunststoffen/rubber',
                    '9: Basismetalen en metaalproducten',
                    '10: Overige minerale producten',
                    '11: Voedings- en genotsmiddelen',
                    '12: Machines elektronica en transportmiddelen',
                    '13: Overige']
        
        nVT = len(vtNames)
        nLS = len(lsNames)
        nET = len(etDict)
        nBG = len(bgNames)
        
        sep = ','
        
        # ---------------------------- Import data ----------------------------
        print('Importing data...')

        print('\tReading skims...')
        skimDistance = read_mtx(skimDistancePath)
        nZones       = int(len(skimDistance)**0.5)
        
        print('\tReading zones...')
        zones = read_shape(pathZones)
        zonesLMS = np.array(zones['LMSVAM'], dtype=int)
        buitenlandZones = np.where(zones['LAND']>1)[0]
        
        print('\tReading shipments...')
        shipments = pd.read_csv(datapathO + f'Shipments_{label}.csv', sep=',')
        shipments['VEHTYPE'] = shipments['VEHTYPE'].astype(int)

        print('\tReading tours...')
        trips = pd.read_csv(datapathO + f'Tours_{label}.csv', sep=',')
        tours = trips[trips['TRIP_ID']==0]
        tours.index = np.arange(len(tours))
        
        print('\tReading parcels...')
        parcels = pd.read_csv(datapathO + f'ParcelDemand_{label}.csv', sep=',')
        
        print('\tReading parcel tours...')
        parcelTrips = pd.read_csv(datapathO + f'ParcelSchedule_{label}.csv', sep=',')
        parcelTrips['AFSTAND'] = skimDistance[(parcelTrips['ORIG_NRM']-1)*nZones+(parcelTrips['DEST_NRM']-1)] / 1000
        parcelTours = parcelTrips[[x.split('_')[-1]=='0' for x in parcelTrips['TRIP_ID']]]
        
        print('\tReading service/construction van trips...')
        # Van trips for service and construction purposes
        vanTripsService      = read_mtx(datapathO + 'TripsVanService.mtx')
        vanTripsConstruction = read_mtx(datapathO + 'TripsVanConstruction.mtx')
        
        print('\tReading loaded network links...')
        tmp = DBF(datapathO + f'links_loaded_{label}.dbf', load=True).records
        linksCols = list(tmp[0].keys())
        links = np.zeros((len(tmp), len(linksCols)))
        for i in range(len(tmp)):
            links[i, :] = list(tmp[i].values())
        links = pd.DataFrame(links, columns=linksCols)
        links['DISTANCE'] /= 1000

        del tmp

        # ------------------------- Writing statistics ------------------------
        
        print('Calculating and writing statistics...')
        
        with open(datapathO + 'KPI.csv', 'w') as f:
            
            # Vehicle type counts
            print('\tVehicle types...')
            
            countsShipVT = shipments['VEHTYPE'].value_counts().astype(str)
            countsTripVT = trips['VEHTYPE'].value_counts().astype(str)
            countsTourVT = tours['VEHTYPE'].value_counts().astype(str)
            countsParcelTripVT = parcelTrips['VEHTYPE'].value_counts().astype(str)
            countsParcelTourVT = parcelTours['VEHTYPE'].value_counts().astype(str)
            
            weightsVT = pd.pivot_table(shipments, index='VEHTYPE', values='WEIGHT', aggfunc=sum).astype(str)

            f.write('\n')
            f.write(sep + 'Voertuigtype'     + sep + 
                          'Vracht: Vervoerde tonnen' + sep +
                          'Vracht: Aantal zendingen' + sep +
                          'Vracht: Aantal ritten'    + sep +
                          'Vracht: Aantal tours'     + sep +
                          'Pakket: Aantal ritten' + sep +
                          'Pakket: Aantal tours'  + sep +
                          'Service/Bouw: Aantal ritten' + '\n')
            for vt in range(nVT):
                f.write(sep + vtNames[vt] + sep)

                try:
                    f.write(weightsVT.at[vt,'WEIGHT'] + sep)
                except:
                    f.write('0' + sep)
                    
                try:
                    f.write(countsShipVT[vt] + sep)
                except:
                    f.write('0' + sep)
        
                try:
                    f.write(countsTripVT[vt] + sep)
                except:
                    f.write('0' + sep)    
        
                try:
                    f.write(countsTourVT[vt] + sep)
                except:
                    f.write('0' + sep)
                
                try:
                    f.write(countsParcelTripVT[vtNames[vt]] + sep)
                except:
                    f.write('0' + sep)                

                try:
                    f.write(countsParcelTourVT[vtNames[vt]] + sep)
                except:
                    f.write('0' + sep)
                    
                if vtNames[vt] == 'Van':
                    f.write(str(np.sum(vanTripsService + vanTripsConstruction)) + sep)
                else:
                    f.write('0' + sep)
                    
                f.write('\n')
                                    
            # Logistic segment counts
            print('\tLogistic segments...')
            
            countsShipLS = shipments['LOGSEG'].value_counts().astype(str)
            countsTripLS = trips['LOG_SEG'].value_counts().astype(str)
            countsTourLS = tours['LOG_SEG'].value_counts().astype(str)                   
                    
            weightsLS = pd.pivot_table(shipments, index='LOGSEG', values='WEIGHT', aggfunc=sum).astype(str)
                    
            f.write('\n')
            f.write(sep + 'Logistiek segment'     + sep + 
                          'Vervoerde tonnen' + sep +
                          'Aantal zendingen' + sep +
                          'Aantal ritten'    + sep +
                          'Aantal tours'     + '\n')
            for ls in range(nLS):
                f.write(sep + lsNames[ls] + sep)

                try:
                    f.write(weightsLS.at[ls,'WEIGHT'] + sep)
                except:
                    f.write('0' + sep)
                    
                try:
                    f.write(countsShipLS[ls] + sep)
                except:
                    f.write('0' + sep)
        
                try:
                    f.write(countsTripLS[ls] + sep)
                except:
                    f.write('0' + sep)    
        
                try:
                    f.write(countsTourLS[ls] + sep)
                except:
                    f.write('0' + sep)
                    
                f.write('\n')
            
            f.write(sep + 'Pakket: Parcel deliveries' + sep)
            f.write('n.v.t.' + sep)
            f.write(str(np.sum(parcels['N_PARCELS'])) + sep)
            f.write(str(len(parcelTrips)) + sep)
            f.write(str(len(parcelTours)) + sep)
            
            f.write('\n')
            
            f.write(sep + 'Service/Bouw: Service vans' + sep)
            f.write('n.v.t.' + sep)
            f.write('n.v.t.' + sep)
            f.write(str(np.sum(vanTripsService)) + sep)
            f.write('n.v.t.' + sep)

            f.write('\n')
            
            f.write(sep + 'Service/Bouw: Construction vans' + sep)
            f.write('n.v.t.' + sep)
            f.write('n.v.t.' + sep)
            f.write(str(np.sum(vanTripsConstruction)) + sep)
            f.write('n.v.t.' + sep)
            
            f.write('\n')


            # BasGoed-goederengroep counts
            print('\tBasGoed-goederengroepen...')
            countsShipBG = shipments['BG_GG'].value_counts().astype(str)
            countsTripBG = trips['BG_GG'].value_counts().astype(str)
            countsTourBG = tours['BG_GG'].value_counts().astype(str)                   
                    
            weightsBG = pd.pivot_table(shipments,
                                       index='BG_GG',
                                       values='WEIGHT',
                                       aggfunc=sum).astype(str)
                    
            f.write('\n')
            f.write(sep + 'BasGoed-goederengroep' + sep + 
                          'Vervoerde tonnen' + sep +
                          'Aantal zendingen' + sep +
                          'Aantal ritten'    + sep +
                          'Aantal tours'     + '\n')
            for bg in range(1,nBG+1):
                f.write(sep + bgNames[bg-1] + sep)

                try:
                    f.write(weightsBG.at[bg,'WEIGHT'] + sep)
                except:
                    f.write('0' + sep)
                    
                try:
                    f.write(countsShipBG[bg] + sep)
                except:
                    f.write('0' + sep)
        
                try:
                    f.write(countsTripBG[bg] + sep)
                except:
                    f.write('0' + sep)    
        
                try:
                    f.write(countsTourBG[bg] + sep)
                except:
                    f.write('0' + sep)
                    
                f.write('\n')
                
            bg = -1
            f.write(sep + 'Lege ritten' + sep)
            f.write('n.v.t.' + sep)
            f.write('n.v.t.' + sep)
            try:
                f.write(countsTripBG[bg] + sep)
            except:
                f.write('0' + sep)    
            f.write('n.v.t.' + sep)                
            f.write('\n')            



            # Voertuigkilometers en emissies
            print('\tAverage trip length...')
            shipments['AFSTAND'] = [skimDistance[int(shipments.at[i, 'ORIG_NRM'] - 1) * nZones + 
                                                 int(shipments.at[i, 'DEST_NRM'] - 1)]
                                    for i in shipments.index]
            shipments['AFSTAND'] /= 1000
            
            f.write('\n')
            f.write(sep + 'Segment' + sep +
                          'Gemiddelde ritlengte [km]' + '\n')
            
            f.write(sep + 'Vracht (ritafstand - alle ritten)' + sep)
            f.write(str(np.average(trips['AFSTAND'])) + sep)
            f.write('\n')
            
            f.write(sep + 'Vracht (ritafstand - binnenlandse ritten)' + sep)
            f.write(str(np.average(trips.loc[(trips['ORIG_BG']<=45) & (trips['DEST_BG']<=45), 'AFSTAND'])))
            f.write('\n')

            f.write(sep + 'Vracht (zendingsafstand - alle ritten)' + sep)
            f.write(str(np.average(shipments['AFSTAND'])) + sep)
            f.write('\n')
            
            f.write(sep + 'Vracht (zendingsafstand - binnenlandse ritten)' + sep)
            f.write(str(np.average(shipments.loc[(shipments['ORIG_LMS']<=413) & (shipments['DEST_LMS']<=413), 'AFSTAND'])))
            f.write('\n')
            
            f.write(sep + 'Pakket (rondritlengte)' + sep)
            f.write(str(np.sum(parcelTrips['AFSTAND'])/len(parcelTours)) + sep)
            f.write('\n')

            f.write(sep + 'Service (alle ritten)' + sep)
            f.write(str(np.sum(vanTripsService*skimDistance/1000)/np.sum(vanTripsService)) + sep)
            f.write('\n')

            f.write(sep + 'Service (binnenlandse ritten)' + sep)
            vanTripsServiceNL = vanTripsService.copy().reshape(nZones,nZones)
            for i in buitenlandZones:
                vanTripsServiceNL[i, :] = 0.0
                vanTripsServiceNL[:, i] = 0.0     
            vanTripsServiceNL = vanTripsServiceNL.flatten()
            f.write(str(np.sum(vanTripsServiceNL*skimDistance/1000)/np.sum(vanTripsServiceNL)) + sep)
            f.write('\n')
            del vanTripsServiceNL
            
            f.write(sep + 'Bouw (alle ritten)' + sep)
            f.write(str(np.sum(vanTripsConstruction*skimDistance/1000)/np.sum(vanTripsConstruction)) + sep)
            f.write('\n')

            f.write(sep + 'Bouw (binnenlandse ritten)' + sep)
            vanTripsConstructionNL = vanTripsConstruction.copy().reshape(nZones,nZones)
            for i in buitenlandZones:
                vanTripsConstructionNL[i, :] = 0.0
                vanTripsConstructionNL[:, i] = 0.0
            vanTripsConstructionNL = vanTripsConstructionNL.flatten()
            f.write(str(np.sum(vanTripsConstructionNL*skimDistance/1000)/np.sum(vanTripsConstructionNL)) + sep)
            f.write('\n')
            del vanTripsConstructionNL
            
            # Voertuigkilometers en emissies
            print('\tNetwerk-KPIs...')
            links['N_VRACHT'] = links['N_TOT'] - links['N_LS8'] - links['N_VAN_S'] - links['N_VAN_C']
            for et in range(nET):                
                links[etDict[et]+'_VRACHT'] = links[etDict[et]] - links[etDict[et]+'_LS8'] - links[etDict[et]+'_VAN_S'] - links[etDict[et]+'_VAN_C']
            
            linksNL = links[links['NL']==1]

            f.write('\n')
            f.write(sep + 'Segment' + sep +
                          'Voertuigkilometers (totaal)' + sep +
                          'Voertuigkilometers (NL)'     + sep +
                          'CO2 [kg] (totaal)' + sep +
                          'CO2 [kg] (NL)'     + sep +
                          'SO2 [kg] (totaal)' + sep +
                          'SO2 [kg] (NL)'     + sep +
                          'PM [kg] (totaal)'  + sep +
                          'PM [kg] (NL)'      + sep +
                          'NOX [kg] (totaal)' + sep +
                          'NOX [kg] (NL)'     + sep +                          
                          '\n')
            
            f.write(sep + 'Vracht' + sep)
            f.write(str(np.sum(links['DISTANCE'] * links['N_VRACHT'])) + sep)
            f.write(str(np.sum(linksNL['DISTANCE'] * linksNL['N_VRACHT'])) + sep)
            for et in range(nET):
                f.write(str(np.sum(links[etDict[et]+'_VRACHT'])) + sep)
                f.write(str(np.sum(linksNL[etDict[et]+'_VRACHT'])) + sep)
            f.write('\n')
            
            f.write(sep + 'Pakket' + sep)
            f.write(str(np.sum(links['DISTANCE'] * links['N_LS8'])) + sep)
            f.write(str(np.sum(linksNL['DISTANCE'] * linksNL['N_LS8'])) + sep)
            for et in range(nET):
                f.write(str(np.sum(links[etDict[et]+'_LS8'])) + sep)
                f.write(str(np.sum(linksNL[etDict[et]+'_LS8'])) + sep)                
            f.write('\n')
            
            f.write(sep + 'Service' + sep)
            f.write(str(np.sum(links['DISTANCE'] * links['N_VAN_S'])) + sep)
            f.write(str(np.sum(linksNL['DISTANCE'] * linksNL['N_VAN_S'])) + sep)
            for et in range(nET):
                f.write(str(np.sum(links[etDict[et]+'_VAN_S'])) + sep)
                f.write(str(np.sum(linksNL[etDict[et]+'_VAN_S'])) + sep)                  
            f.write('\n')
            
            f.write(sep + 'Bouw' + sep)
            f.write(str(np.sum(links['DISTANCE'] * links['N_VAN_C'])) + sep)
            f.write(str(np.sum(linksNL['DISTANCE'] * linksNL['N_VAN_C'])) + sep)
            for et in range(nET):
                f.write(str(np.sum(links[etDict[et]+'_VAN_C'])) + sep)
                f.write(str(np.sum(linksNL[etDict[et]+'_VAN_C'])) + sep)                     
            f.write('\n')           

            # Voertuigkilometers en emissies
            print('\tNetwerk-KPIs (per vehicle type)...')     
            f.write('\n')
            f.write(sep + 'Vracht: Voertuigtype'        + sep +
                          'Voertuigkilometers (totaal)' + sep +
                          'Voertuigkilometers (NL)'     + sep +
                          'CO2 [kg] (totaal)' + sep +
                          'CO2 [kg] (NL)'     + '\n')
            
            for vt in range(nVT):
                tripsVT = trips[trips['VEHTYPE']==vt]
                
                f.write(sep + vtNames[vt] + sep)
                f.write(str(np.sum(tripsVT['AFSTAND'   ])) + sep)
                f.write(str(np.sum(tripsVT['AFSTAND_NL'])) + sep)
                f.write(str(np.sum(tripsVT['CO2'   ])) + sep)
                f.write(str(np.sum(tripsVT['CO2_NL'])) + sep)
                f.write('\n')

            f.write('\n')
            f.write(sep + 'Vracht: Logistiek segment'   + sep +
                          'Voertuigkilometers (totaal)' + sep +
                          'Voertuigkilometers (NL)'     + sep +
                          'CO2 [kg] (totaal)' + sep +
                          'CO2 [kg] (NL)'     + '\n')
            
            for ls in range(nLS):
                tripsLS = trips[trips['LOG_SEG']==ls]
                
                f.write(sep + lsNames[ls] + sep)
                f.write(str(np.sum(tripsLS['AFSTAND'   ])) + sep)
                f.write(str(np.sum(tripsLS['AFSTAND_NL'])) + sep)
                f.write(str(np.sum(tripsLS['CO2'   ])) + sep)
                f.write(str(np.sum(tripsLS['CO2_NL'])) + sep)
                f.write('\n')
                
            f.write('\n')
            f.write(sep + 'Vracht: BasGoed-goederengroep' + sep +
                          'Voertuigkilometers (totaal)'   + sep +
                          'Voertuigkilometers (NL)'       + sep +
                          'CO2 [kg] (totaal)' + sep +
                          'CO2 [kg] (NL)'     + '\n')
            
            for bg in range(1,nBG+1):
                tripsBG = trips[trips['BG_GG']==bg]
                
                f.write(sep + bgNames[bg-1] + sep)
                f.write(str(np.sum(tripsBG['AFSTAND'   ])) + sep)
                f.write(str(np.sum(tripsBG['AFSTAND_NL'])) + sep)
                f.write(str(np.sum(tripsBG['CO2'   ])) + sep)
                f.write(str(np.sum(tripsBG['CO2_NL'])) + sep)
                f.write('\n')                
            
            tripsBG = trips[trips['BG_GG']==-1]     
            f.write(sep + 'Lege ritten' + sep)
            f.write(str(np.sum(tripsBG['AFSTAND'   ])) + sep)
            f.write(str(np.sum(tripsBG['AFSTAND_NL'])) + sep)
            f.write(str(np.sum(tripsBG['CO2'   ])) + sep)
            f.write(str(np.sum(tripsBG['CO2_NL'])) + sep)
            f.write('\n')                       

        print('\tCreating LMSVAM-matrix...')
        nZonesLMS = max(zones['LMSVAM'])
        mat = np.zeros((nZonesLMS * nZonesLMS, 2 + 4))
        mat[:, [0, 1]] = np.array(np.meshgrid(np.arange(nZonesLMS) + 1, 
                                              np.arange(nZonesLMS) + 1)).T.reshape(-1,2)
        
        # Vracht trips
        origsLMS = np.array(trips['ORIG_LMS'], dtype=int)
        destsLMS = np.array(trips['DEST_LMS'], dtype=int)
        for i in range(len(trips)):
            row = (origsLMS[i] - 1) * nZonesLMS + (destsLMS[i] - 1)
            mat[row, 2] += 1
            
        # Bouw/service trips
        for i in range(nZones):
            for j in range(nZones):
                nTripsService      = vanTripsService[i * nZones + j]
                nTripsConstruction = vanTripsConstruction[i * nZones + j]

                if nTripsService > 0 or nTripsConstruction > 0:
                    origLMS = zonesLMS[i]
                    destLMS = zonesLMS[j]
                    row = (origLMS - 1) * nZonesLMS + (destLMS - 1)
                    mat[row, 3] += nTripsService
                    mat[row, 4] += nTripsConstruction      

        # Parcel trips
        origsNRM = np.array(parcelTrips['ORIG_NRM'], dtype=int)
        destsNRM = np.array(parcelTrips['DEST_NRM'], dtype=int)
        for i in range(len(parcelTrips)):
            origLMS = zonesLMS[origsNRM[i] - 1]
            destLMS = zonesLMS[destsNRM[i] - 1]
            row = (origLMS - 1) * nZonesLMS + (destLMS - 1)
            mat[row, 5] += 1
            
        sumMat = np.sum(mat[:, 2:], axis=1)
        mat = mat[np.where(sumMat > 0)[0], :]
        
        mat = pd.DataFrame(mat, columns=['ORIG_LMS', 'DEST_LMS',
                                         'NTRIPS_FREIGHT', 
                                         'NTRIPS_VAN_S', 'NTRIPS_VAN_C',
                                         'NTRIPS_VAN_P'])
        mat.to_csv(datapathO + 'TripMatrix_All_LMSVAM.csv', index=False)
            
        # --------------------------- End of module ---------------------------
                
        totaltime = round(time.time() - start_time, 2)
        log_file.write("Total runtime: %s seconds\n" % (totaltime))  
        log_file.write("End simulation at: "+datetime.datetime.now().strftime("%y-%m-%d %H:%M")+"\n")
        log_file.close()    

        if root != '':
            root.update_statusbar("KPI Module: Done")
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
                root.update_statusbar("KPI Module: Execution failed!")
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
    
    varDict['OUTPUTFOLDER'] = 'P:/Projects_Active/21020 WVL BasGoed Logistiek wegvervoer en zero emissie/Work/Runs/test4/'
    varDict['PARAMFOLDER']	= 'P:/Projects_Active/21020 WVL BasGoed Logistiek wegvervoer en zero emissie/Work/Parameterbestanden TFS/'
    
    varDict['BASGOEDFOLDER']       = DATAFOLDER + 'BasGoed/Uitvoer MS/'    
    
    varDict['SKIMTIME']            = DATAFOLDER + 'NRM netwerk/NRM moedernetwerk/Skims/SkimVrachtTijdNRM.mtx'
    varDict['SKIMDISTANCE']        = DATAFOLDER + 'NRM netwerk/NRM moedernetwerk/Skims/SkimVrachtAfstandNRM.mtx'
    varDict['ZONES']               = DATAFOLDER + 'NRM zones/ZON_Moederbestand_Hoog_versie_23_nov_2020.shp'   
    varDict['PARCELNODES']         = DATAFOLDER + 'Pakketsorteercentra/Pakketsorteercentra.shp'
    varDict['DISTRIBUTIECENTRA']   = DATAFOLDER + 'Distributiecentra/Distributiecentra.csv'
    varDict['TERMINALS']           = DATAFOLDER + 'Overslagzones/TerminalsWithCoordinates.csv'
    varDict['NRM_TO_BASGOED']      = DATAFOLDER + 'Koppeltabellen/NRM_ZONEBG.csv'
    varDict['BASGOED_TO_NRM']      = DATAFOLDER + 'Koppeltabellen/ZONEBG_NRM.csv'
    varDict['BASGOEDGG_TO_NSTR']   = DATAFOLDER + 'Koppeltabellen/BGG_NSTR.csv'
    varDict['COST_VEHTYPE']        = varDict['PARAMFOLDER'] + 'Cost_VehType_2016.csv'
    varDict['COST_SOURCING']       = varDict['PARAMFOLDER'] + 'Cost_Sourcing_2016.csv'
    varDict['CEP_SHARES']          = varDict['PARAMFOLDER'] + 'CEPshares.csv'
    
    varDict['YEARFACTOR'] = 256
    
    varDict['PARCELS_PER_HH']	     = 0.112
    varDict['PARCELS_PER_EMPL']      = 0.041
    varDict['PARCELS_MAXLOAD']	     = 180
    varDict['PARCELS_DROPTIME']      = 120
    varDict['PARCELS_SUCCESS_B2C']   = 0.75
    varDict['PARCELS_SUCCESS_B2B']   = 0.95
    varDict['PARCELS_GROWTHFREIGHT'] = 1.0
    
    varDict['NODES_N'] = DATAFOLDER + 'NRM netwerk/NRM noord autonetwerk 2018/NOD2_218n2101.DAT'
    varDict['NODES_O'] = DATAFOLDER + 'NRM netwerk/NRM oost autonetwerk 2018/NOD2_218o2101.DAT'
    varDict['NODES_W'] = DATAFOLDER + 'NRM netwerk/NRM west autonetwerk 2018/NOD2_218w2101.DAT'
    varDict['NODES_Z'] = DATAFOLDER + 'NRM netwerk/NRM zuid autonetwerk 2018/NOD2_218z2101.DAT'
    varDict['LINKS_N'] = DATAFOLDER + 'NRM netwerk/NRM noord autonetwerk 2018/LNK2_218n2101.DAT'
    varDict['LINKS_O'] = DATAFOLDER + 'NRM netwerk/NRM oost autonetwerk 2018/LNK2_218o2101.DAT'
    varDict['LINKS_W'] = DATAFOLDER + 'NRM netwerk/NRM west autonetwerk 2018/LNK2_218w2101.DAT'
    varDict['LINKS_Z'] = DATAFOLDER + 'NRM netwerk/NRM zuid autonetwerk 2018/LNK2_218z2101.DAT'
    
    varDict['N_MULTIROUTE' ] = 2
    varDict['SHIPMENTS_REF'] = ''
    varDict['N_CPU'] = ""
    
    varDict['LABEL'] = 'REF'
        
    # Run the module
    root = ''
    main(varDict)