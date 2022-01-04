# -*- coding: utf-8 -*-
"""
Created on Fri May 29 10:47:15 2020

@author: STH
"""

# Importeer de modules van het model
import __module_SHIP__
import __module_TOUR__
import __module_PARCEL_DMND__
import __module_PARCEL_SCHD__
import __module_SERVICE__
import __module_TRAF__
import __module_KPI__

# Libraries nodig voor de user interface
import tkinter as tk
from tkinter import filedialog
from tkinter.ttk import Progressbar
import os.path
from sys import argv
import zlib
import base64
import tempfile
from threading import Thread
import multiprocessing as mp
import sys
import traceback
import datetime
import numpy as np
import psutil


#%% Class and functions: Graphical User Interface
        
class Root:
    
    def __init__(self):        
        '''
        Initialize a GUI object
        '''
        # Get directory of the script
        self.datapath = os.path.dirname(os.path.realpath(argv[0]))
        self.datapath = self.datapath.replace(os.sep, '/') + '/'

        self.moduleNames = ['SHIP','TOUR',
                            'PARCEL_DMND','PARCEL_SCHD',
                            'SERVICE',
                            'TRAF',
                            'KPI']
        
        # Set graphics parameters
        self.width  = 950
        self.height = 120
        self.bg     = 'black'
        self.fg     = 'white'
        self.font   = 'Verdana'
        
        # Create a GUI window
        self.root = tk.Tk()
        self.root.title("Logistieke Module BasGoed (Prototype 9 december 2021)")
        self.root.geometry(f'{self.width}x{self.height}+0+0')
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
        
        # Create control file label and entry
        self.labelControlFile = tk.Label(self.root, 
                                         text='Control file:', 
                                         width=20, height=1, 
                                         anchor='w', 
                                         bg=self.bg, fg=self.fg, 
                                         font=(self.font,8))
        self.labelControlFile.place(x=5, y=9)        
        self.controlFile = tk.StringVar(self.root, 
                                        self.datapath)
        self.entryControlFile= tk.Entry(self.root, 
                                        textvariable=self.controlFile, 
                                        width=120, 
                                        font=(self.font,8))
        self.entryControlFile.place(x=80, y=10)
        
        # Create button to search for control file
        self.searchButton = tk.Button(self.root, 
                                      text="...", 
                                      command=self.file_dialog, 
                                      font=(self.font,5), 
                                      width=2)
        self.searchButton.place(x=905, y=12)

        # Create button to run the main function
        self.runButton = tk.Button(self.root, 
                                   text="Run", 
                                   width=15, height=3, 
                                   command=self.run_main, 
                                   font=(self.font,8))
        self.runButton.place(x=425, y=35)
        
        self.progressBar = Progressbar(self.root, 
                                       length=int(self.width/2))
        self.progressBar.place(x=int(self.width/2), y=self.height-22)
        
        # If the control file is passed as an argument in a batch job or in the command line prompt
        if len(argv) > 1:
            self.controlFile.set(argv[1])
            self.run_main()
            
        # Keep GUI active until closed    
        self.root.mainloop()  



    def update_statusbar(self, text):
        self.statusBar.configure(text=text)


        
    def reset_statusbar(self, event=None):
        self.statusBar.configure(text="")  
 

       
    def file_dialog(self):
        '''
        Open up a file dialog
        '''
        self.filename = filedialog.askopenfilename(initialdir= "/", 
                                                   title="Select the .ini control file", 
                                                   filetype=(("Control files (.ini)","*.ini"),
                                                             ("All files","*.*")) )
        self.controlFile.set(self.filename)



    def run_main(self, event=None):
        Thread(target=self.actually_run_main, daemon=True).start()
       
        
        
    def actually_run_main(self):
        '''
        Run the actual emission calculation
        '''
        # Welke parameters zijn numeriek of stellen een directory/filenaam voor
        numericVars = ["PARCELS_PER_HH", "PARCELS_PER_EMPL", 
                       "PARCELS_MAXLOAD", "PARCELS_DROPTIME",
                       "PARCELS_SUCCESS_B2C", "PARCELS_SUCCESS_B2B", 
                       "PARCELS_GROWTHFREIGHT",
                       "YEARFACTOR", 
                       "N_CPU", "SEED",
                       "YEAR", "BASEYEAR"]
        dirVars     = ["OUTPUTFOLDER", "PARAMFOLDER",
                       "BASGOED_PARAMETERS", "BASGOED_CKM", "BASGOED_MS"]
        moduleVars  = ["MODULES"]
        fileVars    = ["SKIMTIME", "SKIMDISTANCE",
                       "ZONES", 
                       "LINKS", "NODES",
                       "PARCELNODES", 
                       "DISTRIBUTIECENTRA", "TERMINALS",
                       "SERVICE_PA", "SERVICE_DISTANCEDECAY",
                       "SECTOR_TO_SECTOR",
                       "CONTAINER",
                       "CEP_SHARES",
                       "FLOWTYPES",
                       "VEHCAPACITY",
                       "NRM_TO_BG", "BG_TO_NRM",
                       "BGGG_TO_NSTR_CONT", "BGGG_TO_NSTR_NCONT", 
                       "BGGG_TO_LS_CONT", "BGGG_TO_LS_NCONT",
                       "COST_ROAD_MS", "COST_ROAD_CKM",
                       "COST_AANHANGER", "COST_BESTEL", "COST_LZV",
                       "COST_OPLEGGER", "COST_SPECIAAL", "COST_VRACHTWAGEN",
                       "DEPTIME_PARCELS", "DEPTIME_FREIGHT",
                       "PARAMS_ET_FIRST", "PARAMS_ET_LATER", "PARAMS_SSVT",
                       "EM_BUITENWEG_LEEG", "EM_BUITENWEG_VOL",
                       "EM_SNELWEG_LEEG", "EM_SNELWEG_VOL",
                       "EM_STAD_LEEG", "EM_STAD_VOL",
                       "ZEZ_CONSOLIDATION", "ZEZ_SCENARIO", "ZEZ_ZONES",
                       "SHIPMENTS_REF"]
        optionalVars = ["SHIPMENTS_REF",
                        "ZEZ_CONSOLIDATION", "ZEZ_SCENARIO", "ZEZ_ZONES",
                        "N_CPU", "SEED"]
        otherVars    = ["LABEL"]
        
        # Lijst met alle mogelijke sleutels in de control file
        self.varStrings = np.unique(numericVars + dirVars + moduleVars +
                                    fileVars + optionalVars + otherVars)
        nVars = len(self.varStrings)
        
        # Op welke index staat de OUTPUTFOLDER sleutel
        whereOutputFolder  = [i for i in range(nVars) if self.varStrings[i]=="OUTPUTFOLDER" ][0]
        whereParamFolder   = [i for i in range(nVars) if self.varStrings[i]=="PARAMFOLDER"  ][0]
        
        # Hier stoppen we de waardes behorende bij de sleutels
        varValues = ["" for i in range(nVars)]
        
        acceptedRunYears = [2018, 2030, 2040, 2050]

        run = True          # Wel of niet runnen, wordt op False gezet als bijv. bestanden niet gevonden kunnen worden
        writeLog = True     # Wel of geen logfile schrijven, wordt op False gezet als de outputfolder niet bestaat
        errorMessage = ""   # Hier verzamelen we alle foutmeldingen in
        
        try:
           
            with open(self.controlFile.get(), 'r') as f:
                lines = f.readlines()
                
                for line in lines:
                    
                    if len(line.split('=')) > 1:
                        
                        if line[0] != '#':
                        
                            key   = line.split('=')[0]
                            value = line.split('=')[1]
                            
                            # Sta spaties en tabs voor/na de sleutel/waarde toe
                            while key[0] == ' ' or key[0] == '\t':
                                key = key[1:]
                                
                            while key[-1] == ' ' or key[-1] == '\t':
                                key = key[0:-1]
                                
                            while value[0] == ' ' or value[0] == '\t':
                                value = value[1:]
    
                            while value[-1] == ' ' or value[-1] == '\t':
                                value = value[0:-1]
                                
                            print(key + ' = ' + value.replace('\n',""))
                            
                            # Read the arguments in the control file
                            for i in range(nVars):                                
                                if key.upper() == self.varStrings[i]:
                                    
                                    # For numeric arguments, check if they can be converted from string to float
                                    if self.varStrings[i] in numericVars:
                                        value = value.replace("'", "").replace('"', "").replace('\n',"")
                                        try:
                                            varValues[i] = float(value)
                                        except:
                                            if value != "":
                                                varValues[i] = value
                                                errorMessage = errorMessage + 'Fill in a numeric value for ' + self.varStrings[i]
                                                errorMessage = errorMessage + ', could not convert following value to a number: '
                                                errorMessage = errorMessage + value.replace("'", "").replace('"', "").replace('\n',"") + "\n"
                                                run = False                                                
                                    
                                    # The argument which states which modules should be run
                                    elif self.varStrings[i] in moduleVars:
                                        varValues[i] = value.replace(os.sep, '/').replace("'", "").replace('"', "").replace('\n',"").replace(' ','')
                                        varValues[i] = varValues[i].split(',')
                                        for j in range(len(varValues[i])):
                                            if varValues[i][j] not in self.moduleNames:
                                                errorMessage = errorMessage + 'Module ' + varValues[i][j] + ' does not exist.' + '\n'
                                                run = False

                                    # For string arguments, also replace possible '\' by '/'
                                    else:
                                        varValues[i] = value.replace(os.sep, '/').replace("'", "").replace('"', "").replace('\n',"")

                                        if self.varStrings[i] in dirVars:
                                            if varValues[i][-1] != '/':
                                                varValues[i] = varValues[i] + '/'
                                        
                                        if self.varStrings[i] in dirVars or self.varStrings[i] in fileVars:
                                            if self.varStrings[i] not in ['OUTPUTFOLDER','PARAMFOLDER']:
                                                temp = varValues[i].split("<<")
                                                if len(temp) > 1:
                                                    temp = temp[1].split(">>")
                                                    
                                                    if temp[0] == 'OUTPUTFOLDER':
                                                        varValues[i] = varValues[whereOutputFolder] + temp[1]                            
                                                    if temp[0] == 'PARAMFOLDER':
                                                        varValues[i] = varValues[whereParamFolder] + temp[1]                                     
                                            
                            # Warning for unknown argument in control file
                            if key.upper() not in self.varStrings:
                                errorMessage = errorMessage + 'Unknown parameter in control file: ' + key + "\n"
                                run = False
                                    
                            
            for i in range(nVars):                   
                # Warnings for non-existing directories
                if self.varStrings[i] in dirVars:
                    if not os.path.isdir(varValues[i]) and varValues[i] != "":
                        errorMessage = errorMessage + 'The folder for parameter ' + self.varStrings[i]
                        errorMessage = errorMessage + ' does not exist: ' + "'" + varValues[i] + "'" + "\n"
                        run = False                                                

                        # Als de opgegeven outputfolder niet bestaat, dan kunnen we er ook geen logfile in schrijven
                        if self.varStrings[i] == "OUTPUTFOLDER":
                            writeLog = False
                            
                # Warnings for non-existing files
                if self.varStrings[i] in fileVars:
                    if not os.path.isfile(varValues[i]) and varValues[i] != "":
                        errorMessage = errorMessage + 'The file for parameter ' + self.varStrings[i]
                        errorMessage = errorMessage + ' does not exist: ' + "'" + varValues[i] + "'" + "\n"
                        run = False   
                            
                if self.varStrings[i] == 'YEAR':
                    if not varValues[i] in acceptedRunYears:
                        errorMessage = errorMessage + 'Fill in 2018, 2030, 2040 or 2050 for field YEAR. Got '
                        errorMessage = errorMessage + "'" + str(varValues[i]) + "'" + " instead.\n"
                        run = False 
                        
            # Warnings for omitted arguments in control file
            for i in range(nVars):
                if varValues[i] == "" and self.varStrings[i] not in optionalVars:
                    errorMessage = errorMessage + 'Warning, no value given for parameter ' + self.varStrings[i]
                    errorMessage = errorMessage + ' in the controle file.' + "\n"
                    run = False                    
                
        except:
            errorMessage = 'Could not find or read the following control file: ' + "'" + self.controlFile.get() + "'"
            errorMessage = errorMessage + '\n\n' + str(sys.exc_info()[0])
            errorMessage = errorMessage + '\n\n' + str(traceback.format_exc())
            run = False
            writeLog = False                

        # Open de logfile en schrijf de header, de opgegeven argumenten en eventuele foutmeldingen
        if writeLog:
            self.logFileName = varValues[whereOutputFolder] + "Logfile_20" + datetime.datetime.now().strftime("%y%m%d_%H%M%S") + ".log"
            
            with open(self.logFileName, "w") as f:            
                f.write('########################################################################################\n')
                f.write('### Logistieke Module BasGoed                                                        ###\n')
                f.write('### Prototype 9 December 2021                                                        ###\n')
                f.write('########################################################################################\n')
                f.write('\n')
    
                f.write('########################################################################################\n')
                f.write('### Settings                                                                         ###\n')
                f.write('########################################################################################\n')
                f.write('Control file: ' + self.controlFile.get() + '\n')
                for i in range(len(varValues)):
                    f.write(self.varStrings[i] + ' = ' + str(varValues[i]) + '\n')
                f.write('\n')
                
                if not run:
                    f.write('########################################################################################\n')
                    f.write('### Errors while reading control file                                                ###\n')
                    f.write('########################################################################################\n')
                    f.write(errorMessage)
                    
        if run:
            varDict = {}
            for i in range(nVars):
                varDict[self.varStrings[i]] = varValues[i]

            self.statusBar.configure(text="Start calculations...")
            result = self.main(varDict)
            self.statusBar.configure(text="")
            
            if result[0] == 1:
                self.statusBar.configure(text="Error in calculation! (Tool will be closed automatically after 15 seconds.)")         
                self.root.after(15000, lambda: self.root.destroy())
                    
            else:
                self.statusBar.configure(text="Calculations finished.")
                self.progressBar['value'] = 100
                if len(varDict['MODULES'])==1 and varDict['MODULES'][0]=='INVOERCONTROLE':
                    self.root.after(15000, lambda: self.root.destroy())
                else:
                    self.root.after(1000, lambda: self.root.destroy())
                
        else:
            self.statusBar.configure(text="Run was not started. See error message. ")
            errorMessage = "Could not start the run for the following reasons: \n\n" + errorMessage
            self.error_screen(text=errorMessage, size=[950,150])
          
                    
            
    def error_screen(self, text='', event=None, size=[950,350], title='Foutmelding'):
        '''
        Pop up a window with an error message
        '''
        windowError = tk.Toplevel(self.root)
        windowError.title(title)
        windowError.geometry(f'{size[0]}x{size[1]}+0+{self.height+50}')
        windowError.minsize(width=size[0], height=size[1])
        windowError.iconbitmap(bitmap=self.iconPath)
        labelError = tk.Label(windowError, text=text, anchor='w', justify='left')
        labelError.place(x=10, y=10)
        
        
    def main(self, varDict):
        
        run = True
        result = [0, 0]
        
        with open(self.logFileName, "a") as f:        

            f.write('########################################################################################\n')
            f.write('### Progress                                                                         ###\n')
            f.write('########################################################################################\n')

            args = [self, varDict]

                    
            if run and 'SHIP' in varDict['MODULES']:
                print('---------------------------------------------------------------------------------')
                print('----------------------- Shipment synthesizer ------------------------------------')
                print('---------------------------------------------------------------------------------')
                f.write("Shipment synthesizer" + '\n')
                f.write("\tStarted at:    " + datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")+"\n")
                
                self.statusBar.configure(text='Running Shipment Synthesizer...')
                                         
                result = __module_SHIP__.actually_run_module(args)
                
                if result[0] == 1:   
                    errorMessage = []
                    errorMessage.append('\nError in shipment synthesizer module!\n\n' )
                    errorMessage.append('See the log-file: ' + str(self.logFileName).split('/')[-1] + '\n\n')
                    errorMessage.append(str(result[1][0]) + '\n' + str(result[1][1]) + '\n\n')
                    self.error_screen(text=errorMessage[0] + errorMessage[1] + errorMessage[2])
                    run = False
                    f.write(errorMessage[0] + errorMessage[2])    
                else:
                    f.write("\tFinished at: " + datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")+"\n\n")
                
            if run and 'TOUR' in varDict['MODULES']:
                print('---------------------------------------------------------------------------------')
                print('------------------------- Tour Formation ----------------------------------------')
                print('---------------------------------------------------------------------------------')
                f.write("Tour formation" + '\n')
                f.write("\tStarted at:    " + datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")+"\n")
    
                self.statusBar.configure(text='Running Tour Formation...')
                
                result = __module_TOUR__.actually_run_module(args)
                
                if result[0] == 1:   
                    errorMessage = []
                    errorMessage.append('\nError in tour formation module!\n\n' )
                    errorMessage.append('See the log-file: ' + str(self.logFileName).split('/')[-1] + '\n\n')
                    errorMessage.append(str(result[1][0]) + '\n' + str(result[1][1]) + '\n\n')
                    self.error_screen(text=errorMessage[0] + errorMessage[1] + errorMessage[2])
                    run = False
                    f.write(errorMessage[0] + errorMessage[2])    
                else:
                    f.write("\tFinished at: " + datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")+"\n\n")

            if run and 'PARCEL_DMND' in varDict['MODULES']:
                print('---------------------------------------------------------------------------------')
                print('------------------------- Parcel Demand -----------------------------------------')
                print('---------------------------------------------------------------------------------')
                f.write("Parcel demand" + '\n')
                f.write("\tStarted at:    " + datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")+"\n")
                
                self.statusBar.configure(text='Running Parcel Demand...')
                
                result = __module_PARCEL_DMND__.actually_run_module(args)
                
                if result[0] == 1:   
                    errorMessage = []
                    errorMessage.append('\nError in parcel demand module!\n\n' )
                    errorMessage.append('See the log-file: ' + str(self.logFileName).split('/')[-1] + '\n\n')
                    errorMessage.append(str(result[1][0]) + '\n' + str(result[1][1]) + '\n\n')
                    self.error_screen(text=errorMessage[0] + errorMessage[1] + errorMessage[2])
                    run = False
                    f.write(errorMessage[0] + errorMessage[2])    
                else:
                    f.write("\tFinished at: " + datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")+"\n\n")

            if run and 'PARCEL_SCHD' in varDict['MODULES']:
                print('---------------------------------------------------------------------------------')
                print('----------------------- Parcel Scheduling ---------------------------------------')
                print('---------------------------------------------------------------------------------')
                f.write("Parcel scheduling" + '\n')
                f.write("\tStarted at:    " + datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")+"\n")
                    
                self.statusBar.configure(text='Running Parcel Scheduling...')
                
                result = __module_PARCEL_SCHD__.actually_run_module(args)
                
                if result[0] == 1:   
                    errorMessage = []
                    errorMessage.append('\nError in parcel scheduling module!\n\n' )
                    errorMessage.append('See the log-file: ' + str(self.logFileName).split('/')[-1] + '\n\n')
                    errorMessage.append(str(result[1][0]) + '\n' + str(result[1][1]) + '\n\n')
                    self.error_screen(text=errorMessage[0] + errorMessage[1] + errorMessage[2])
                    run = False
                    f.write(errorMessage[0] + errorMessage[2])    
                else:
                    f.write("\tFinished at: " + datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")+"\n\n")

            if run and 'SERVICE' in varDict['MODULES']:
                print('---------------------------------------------------------------------------------')
                print('-------------------- Vans service/construction ----------------------------------')
                print('---------------------------------------------------------------------------------')
                f.write("Vans service/construction" + '\n')
                f.write("\tStarted at:    " + datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")+"\n")
                    
                self.statusBar.configure(text='Running Vans service/construction...')
                
                result = __module_SERVICE__.actually_run_module(args)
                
                if result[0] == 1:   
                    errorMessage = []
                    errorMessage.append('\nError in service vans module!\n\n' )
                    errorMessage.append('See the log-file: ' + str(self.logFileName).split('/')[-1] + '\n\n')
                    errorMessage.append(str(result[1][0]) + '\n' + str(result[1][1]) + '\n\n')
                    self.error_screen(text=errorMessage[0] + errorMessage[1] + errorMessage[2])
                    run = False
                    f.write(errorMessage[0] + errorMessage[2])    
                else:
                    f.write("\tFinished at: " + datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")+"\n\n")
                    
            if run and 'TRAF' in varDict['MODULES']:
                print('---------------------------------------------------------------------------------')
                print('----------------------- Traffic Assignment --------------------------------------')
                print('---------------------------------------------------------------------------------')
                f.write("Traffic assignment" + '\n')
                f.write("\tStarted at:    " + datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")+"\n")
                
                self.statusBar.configure(text='Running Traffic Assignment...')
                
                result = __module_TRAF__.actually_run_module(args)
                
                if result[0] == 1:   
                    errorMessage = []
                    errorMessage.append('\nError in traffic assignment module!\n\n' )
                    errorMessage.append('See the log-file: ' + str(self.logFileName).split('/')[-1] + '\n\n')
                    errorMessage.append(str(result[1][0]) + '\n' + str(result[1][1]) + '\n\n')
                    self.error_screen(text=errorMessage[0] + errorMessage[1] + errorMessage[2])
                    run = False
                    f.write(errorMessage[0] + errorMessage[2])    
                else:
                    f.write("\tFinished at: " + datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")+"\n\n")

            if run and 'KPI' in varDict['MODULES']:
                print('---------------------------------------------------------------------------------')
                print('----------------------- Output Indicators ---------------------------------------')
                print('---------------------------------------------------------------------------------')
                f.write("Output indicators" + '\n')
                f.write("\tStarted at:    " + datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")+"\n")
                
                self.statusBar.configure(text='Running KPI module...')
                
                result = __module_KPI__.actually_run_module(args)
                
                if result[0] == 1:   
                    errorMessage = []
                    errorMessage.append('\nError in output indicator module!\n\n' )
                    errorMessage.append('See the log-file: ' + str(self.logFileName).split('/')[-1] + '\n\n')
                    errorMessage.append(str(result[1][0]) + '\n' + str(result[1][1]) + '\n\n')
                    self.error_screen(text=errorMessage[0] + errorMessage[1] + errorMessage[2])
                    run = False
                    f.write(errorMessage[0] + errorMessage[2])    
                else:
                    f.write("\tFinished at: " + datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")+"\n\n")
                    
            if run:
                f.write('Finished all calculations.')
                self.statusBar.configure(text='Finished all calculations.')
            
        return result 
        
        
        
#%% Run the script
        
if __name__ == '__main__':
    mp.freeze_support()
    root = Root()
    
    
    
    