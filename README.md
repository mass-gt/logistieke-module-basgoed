# logistieke-module-basgoed

This repository contains the code for the logistic module of the Dutch strategic freight model BasGoed, owned by Rijkswaterstaat.

## Setting up a run
To calculate a scenario, run the `__GUI__.py` script and enter the path to the .ini-file with configuration settings. 
An example of such an .ini-file is shown at the bottom. In this file you specify parameters and the paths to the input files to be used in the model run. 

Besides the Python Standard Library, make sure you have the following libraries installed:
- numpy==1.19.1
- pandas==1.0.5
- scipy==1.5.0
- pyshp==2.1.0
- shapely==1.7.0
- numba==0.53.0

Finally, when you are using the Spyder IDE for running your Python scripts, make sure to have selected `Execute in an external system terminal` under `Tools-->Preferences-->Run-->Console`. This is necessary to make the scripts work that use parallelization of processes (tour formation module and traffic assignment module). 

# Example ini-file
```
# Which modules to run (options: SHIP,TOUR,PARCEL_DMND,PARCEL_SCHD,SERVICE,TRAF,KPI)
MODULES = TRAF,KPI

LABEL = REF
BASEYEAR = 2014
YEAR     = 2018
YEARFACTOR = 256

OUTPUTFOLDER  = C:/.../Runs/test4/
PARAMFOLDER   = C:/.../Parameterbestanden TFS/
BASGOED_PARAMETERS = C:/.../BasGoed/2018/Parameters/
BASGOED_CKM 	   = C:/.../BasGoed/2018/Uitvoer CKM/
BASGOED_MS  	   = C:/.../BasGoed/2018/Uitvoer MS/

SKIMTIME            = C:/.../Invoer/Skims/SkimVrachtTijdNRM.mtx
SKIMDISTANCE        = C:/.../Invoer//Skims/SkimVrachtAfstandNRM.mtx
ZONES               = C:/.../Invoer//ZON_Moederbestand_Hoog_versie_23_nov_2020.shp 
PARCELNODES         = C:/.../Invoer//Pakketsorteercentra.shp
DISTRIBUTIECENTRA   = C:/.../Invoer//Distributiecentra.csv
TERMINALS           = C:/.../Invoer//TerminalsWithCoordinates.csv
SECTOR_TO_SECTOR    = C:/.../Invoer//SectorMakeUse_SectorNRM.csv
CONTAINER           = C:/.../Invoer//ContainerStats.csv

CEP_SHARES 			= <<PARAMFOLDER>>CEPshares.csv
VEHCAPACITY 		= <<PARAMFOLDER>>CarryingCapacity.csv
FLOWTYPES 			= <<PARAMFOLDER>>LogFlowtype_Shares.csv
DEPTIME_PARCELS 	= <<PARAMFOLDER>>departureTimeParcelsCDF.csv
DEPTIME_FREIGHT 	= <<PARAMFOLDER>>departuretimePDF.csv
PARAMS_ET_FIRST 	= <<PARAMFOLDER>>Params_EndTourFirst.csv
PARAMS_ET_LATER 	= <<PARAMFOLDER>>Params_EndTourLater.csv
PARAMS_SSVT 		= <<PARAMFOLDER>>Params_ShipSize_VehType.csv

NRM_TO_BG          = C:/.../Invoer/Koppeltabellen/NRM_ZONEBG.csv
BG_TO_NRM          = C:/.../Invoer/Koppeltabellen/ZONEBG_NRM.csv
BGGG_TO_NSTR_CONT  = C:/.../Invoer/Koppeltabellen/BGG_NSTR_Container.csv
BGGG_TO_NSTR_NCONT = C:/.../Invoer/Koppeltabellen/BGG_NSTR_NietContainer.csv
BGGG_TO_LS_CONT    = C:/.../Invoer/Koppeltabellen/BGG_LS_Container.csv
BGGG_TO_LS_NCONT   = C:/.../Invoer/Koppeltabellen/BGG_LS_NietContainer.csv

COST_ROAD_MS        = C:/.../BasGoed/2018/Parameters/modechoice_cost_weg_base.cff
COST_ROAD_CKM       = C:/.../BasGoed/2018/Parameters/ckm_cost_road_base.cff
COST_AANHANGER      = C:/.../BasGoed/2018/Parameters/rw_kosten_aanhanger_base.cff
COST_BESTEL         = C:/.../BasGoed/2018/Parameters/rw_kosten_bestel_base.cff
COST_LZV            = C:/.../BasGoed/2018/Parameters/rw_kosten_lzv_base.cff
COST_OPLEGGER       = C:/.../BasGoed/2018/Parameters/rw_kosten_oplegger_base.cff
COST_SPECIAAL       = C:/.../BasGoed/2018/Parameters/rw_kosten_speciaal_base.cff
COST_VRACHTWAGEN    = C:/.../BasGoed/2018/Parameters/rw_kosten_vrachtwagen_base.cff

SERVICE_PA            = <<PARAMFOLDER>>Params_PA_SERVICE.csv
SERVICE_DISTANCEDECAY = <<PARAMFOLDER>>Params_DistanceDecay_SERVICE.csv

NODES = C:/.../nodes.csv
LINKS = C:/.../links.shp
EM_BUITENWEG_LEEG 	= <<PARAMFOLDER>>EmissieFactoren_BUITENWEG_LEEG.csv
EM_BUITENWEG_VOL	= <<PARAMFOLDER>>EmissieFactoren_BUITENWEG_VOL.csv
EM_SNELWEG_LEEG 	= <<PARAMFOLDER>>EmissieFactoren_SNELWEG_LEEG.csv
EM_SNELWEG_VOL 		= <<PARAMFOLDER>>EmissieFactoren_SNELWEG_VOL.csv
EM_STAD_LEEG 		= <<PARAMFOLDER>>EmissieFactoren_STAD_LEEG.csv
EM_STAD_VOL 		= <<PARAMFOLDER>>EmissieFactoren_STAD_VOL.csv

PARCELS_PER_HH	 = 0.155
PARCELS_PER_EMPL = 0.055
PARCELS_MAXLOAD	 = 150
PARCELS_DROPTIME = 120
PARCELS_SUCCESS_B2C   = 0.75
PARCELS_SUCCESS_B2B   = 0.95
PARCELS_GROWTHFREIGHT = 1.0

#ZEZ_CONSOLIDATION 	= <<PARAMFOLDER>>ConsolidationPotential.csv
#ZEZ_SCENARIO		= <<PARAMFOLDER>>ZEZscenario.csv
#ZEZ_ZONES   		= <<PARAMFOLDER>>ZEZzones.csv
#SHIPMENTS_REF = ''

#N_CPU = ''
```

