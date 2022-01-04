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
from shapely.geometry import Point, Polygon, MultiPolygon
from __functions__ import read_shape, read_nodes, read_links



#%% In te vullen parameters

pathOutput  = 'P:/Projects_Active/21020 WVL BasGoed Logistiek wegvervoer en zero emissie/Work/Runs/test4/'
pathZones   = 'P:/Projects_Active/21020 WVL BasGoed Logistiek wegvervoer en zero emissie/Work/Dataverzameling/NRM zones/ZON_Moederbestand_Hoog_versie_23_nov_2020.shp'
pathShapeNL = 'P:/Projects_Active/21020 WVL BasGoed Logistiek wegvervoer en zero emissie/Work/Parameterbestanden TFS/NL_shape.shp'
pathZEZ     = 'P:/Projects_Active/21020 WVL BasGoed Logistiek wegvervoer en zero emissie/Work/Parameterbestanden TFS/ZEZzones.csv'

NETWORKFOLDER = 'P:/Projects_Active/21020 WVL BasGoed Logistiek wegvervoer en zero emissie/Work/Dataverzameling/NRM netwerk/'
pathNodesN = NETWORKFOLDER + 'NRM noord autonetwerk 2018/NOD2_218n2101.DAT'
pathNodesO = NETWORKFOLDER + 'NRM oost autonetwerk 2018/NOD2_218o2101.DAT'
pathNodesW = NETWORKFOLDER + 'NRM west autonetwerk 2018/NOD2_218w2101.DAT'
pathNodesZ = NETWORKFOLDER + 'NRM zuid autonetwerk 2018/NOD2_218z2101.DAT'
pathLinksN = NETWORKFOLDER + 'NRM noord autonetwerk 2018/GLD2_218.DAT'
pathLinksO = NETWORKFOLDER + 'NRM oost autonetwerk 2018/GLD2_218.DAT'
pathLinksW = NETWORKFOLDER + 'NRM west autonetwerk 2018/GLD2_218.DAT'
pathLinksZ = NETWORKFOLDER + 'NRM zuid autonetwerk 2018/GLD2_218.DAT'

# pathExcludeLinksZEZ = 'P:/Projects_Active/21020 WVL BasGoed Logistiek wegvervoer en zero emissie/Work/Parameterbestanden TFS/ExcludeLinksZEZ.csv'
pathExcludeLinksZEZ = ''
pathLinksZEZ = 'P:/Projects_Active/21020 WVL BasGoed Logistiek wegvervoer en zero emissie/Work/Parameterbestanden TFS/linksZEZ.csv'

maxSpeedFreight = 85    


#%% In te vullen parameters

#pathOutput  = 'P:/Projects_Active/21020 WVL BasGoed Logistiek wegvervoer en zero emissie/Work/Runs/test3_2030H/'
#pathZones   = 'P:/Projects_Active/21020 WVL BasGoed Logistiek wegvervoer en zero emissie/Work/Dataverzameling/NRM zones/ZON_Moederbestand_Hoog_versie_23_nov_2020.shp'
#pathShapeNL = 'P:/Projects_Active/21020 WVL BasGoed Logistiek wegvervoer en zero emissie/Work/Parameterbestanden TFS/NL_shape.shp'
#pathZEZ     = 'P:/Projects_Active/21020 WVL BasGoed Logistiek wegvervoer en zero emissie/Work/Parameterbestanden TFS/ZEZzones.csv'
#
#NETWORKFOLDER = 'P:/Projects_Active/21020 WVL BasGoed Logistiek wegvervoer en zero emissie/Work/Dataverzameling/NRM netwerk/'
#pathNodesN = NETWORKFOLDER + 'NRM noord autonetwerk 2030/NOD2_230n2101.DAT'
#pathNodesO = NETWORKFOLDER + 'NRM oost autonetwerk 2030/NOD2_230o2101.DAT'
#pathNodesW = NETWORKFOLDER + 'NRM west autonetwerk 2030/NOD2_230w2101.DAT'
#pathNodesZ = NETWORKFOLDER + 'NRM zuid autonetwerk 2030/NOD2_230z2101.DAT'
#pathLinksN = NETWORKFOLDER + 'NRM noord autonetwerk 2030/GLD2_230.DAT'
#pathLinksO = NETWORKFOLDER + 'NRM oost autonetwerk 2030/GLD2_230.DAT'
#pathLinksW = NETWORKFOLDER + 'NRM west autonetwerk 2030/GLD2_230.DAT'
#pathLinksZ = NETWORKFOLDER + 'NRM zuid autonetwerk 2030/GLD2_230.DAT'
#
#pathExcludeLinksZEZ = ''
#
#maxSpeedFreight = 85   


#%% Main

def main():
    try:
    
        log_file = open(pathOutput + "Logfile_NetworkPreparation.log", 'w')
        log_file.write("Start simulation at: "+datetime.datetime.now().strftime("%y-%m-%d %H:%M")+"\n")
                    
        start_time = time.time()
        
        # ------------------- Importing and preprocessing network ---------------------------------------
        print("Importing and preprocessing network..."), log_file.write("Importing and preprocessing network...\n")
        
        print('\tReading zones...')
        zones, zonesGeometry = read_shape(pathZones, returnGeometry=True)
        zones = zones[['SEGNR_2018',
                       'N', 'O', 'W', 'Z', 
                       'UNIEK_ID', 'LANDSDEEL_1',
                       'XCOORD','YCOORD']]

        # Read shape of the Netherlands
        temp, nlShape = read_shape(pathShapeNL, returnGeometry=True)
        nlShape = nlShape[0]
        if nlShape['type'] == 'MultiPolygon':               
            nlShape = [[Polygon(nlShape['coordinates'][i][j]) for j in range(len(nlShape['coordinates'][i]))]
                                    for i in range(len(nlShape['coordinates']))]
            for i in range(1,len(nlShape)):
                for j in range(len(nlShape[i])):
                    nlShape[0].append(nlShape[i][j])
            nlShape = MultiPolygon(nlShape[0])
        else:
            nlShape = Polygon(nlShape['coordinates'][0])
        del temp
            
        print('\tReading nodes...')
        nodesN = read_nodes(pathNodesN)
        nodesO = read_nodes(pathNodesO)
        nodesW = read_nodes(pathNodesW)
        nodesZ = read_nodes(pathNodesZ)

        print('\tReading links...')
        linksN, isLoadedN = read_links(pathLinksN)
        linksO, isLoadedO = read_links(pathLinksO)
        linksW, isLoadedW = read_links(pathLinksW)
        linksZ, isLoadedZ = read_links(pathLinksZ)
        
        linksN['NRM'] = 1
        linksO['NRM'] = 2
        linksW['NRM'] = 3
        linksZ['NRM'] = 4
           
        # Recoding node numbers in links file (if unloaded network)
        if (not isLoadedN) and (not isLoadedO) and (not isLoadedW) and (not isLoadedZ):
            # Noord
            linksN['KNOOP_A'] = np.array(nodesN['NODENR'], dtype=int)[linksN['KNOOP_A'].values-1]
            linksN['KNOOP_B'] = np.array(nodesN['NODENR'], dtype=int)[linksN['KNOOP_B'].values-1]
            
            # Oost
            linksO['KNOOP_A'] = np.array(nodesO['NODENR'], dtype=int)[linksO['KNOOP_A'].values-1]
            linksO['KNOOP_B'] = np.array(nodesO['NODENR'], dtype=int)[linksO['KNOOP_B'].values-1]
            
            # West
            linksW['KNOOP_A'] = np.array(nodesW['NODENR'], dtype=int)[linksW['KNOOP_A'].values-1]
            linksW['KNOOP_B'] = np.array(nodesW['NODENR'], dtype=int)[linksW['KNOOP_B'].values-1]
            
            # Zuid
            linksZ['KNOOP_A'] = np.array(nodesZ['NODENR'], dtype=int)[linksZ['KNOOP_A'].values-1]            
            linksZ['KNOOP_B'] = np.array(nodesZ['NODENR'], dtype=int)[linksZ['KNOOP_B'].values-1]
    
        elif not (isLoadedN and isLoadedO and isLoadedW and isLoadedZ):
            raise Exception("\nThe 4 road network links files contain both loaded and unloaded networks." + 
                                "\nExpecting only one type of network (i.e. all loaded or all unloaded).")

        # Per studiegebied en zone, op welke indices vinden we de connectors
        # in de links DataFrames
        connectorsN = {}
        connectorsO = {}
        connectorsW = {}
        connectorsZ = {}

        for i in np.where(linksN['LINKTYPE'] == 99)[0]:
            zoneNr = linksN.at[i, 'KNOOP_A']
            if zoneNr > 10000:
                zoneNr = linksN.at[i, 'KNOOP_B']

            try:
                connectorsN[zoneNr].append(linksN.loc[i, :])
            except KeyError:
                connectorsN[zoneNr] = [linksN.loc[i, :]]

        for i in np.where(linksO['LINKTYPE'] == 99)[0]:
            zoneNr = linksO.at[i, 'KNOOP_A']
            if zoneNr > 10000:
                zoneNr = linksO.at[i, 'KNOOP_B']

            try:
                connectorsO[zoneNr].append(linksO.loc[i, :])
            except KeyError:
                connectorsO[zoneNr] = [linksO.loc[i, :]]

        for i in np.where(linksW['LINKTYPE'] == 99)[0]:
            zoneNr = linksW.at[i, 'KNOOP_A']
            if zoneNr > 10000:
                zoneNr = linksW.at[i, 'KNOOP_B']

            try:
                connectorsW[zoneNr].append(linksW.loc[i, :])
            except KeyError:
                connectorsW[zoneNr] = [linksW.loc[i, :]]

        for i in np.where(linksZ['LINKTYPE'] == 99)[0]:
            zoneNr = linksZ.at[i, 'KNOOP_A']
            if zoneNr > 10000:
                zoneNr = linksZ.at[i, 'KNOOP_B']

            try:
                connectorsZ[zoneNr].append(linksZ.loc[i, :])
            except KeyError:
                connectorsZ[zoneNr] = [linksZ.loc[i, :]]

        print('\tCombining nodes...')
        
        nodes = nodesN.copy()
        nodes = nodes[nodes['NODENR']>10000]
        
        nodes = nodes.append(nodesO[nodesO['NODENR']>10000])
        nodes = nodes.append(nodesW[nodesW['NODENR']>10000])
        nodes = nodes.append(nodesZ[nodesZ['NODENR']>10000])
        nodes.index = np.arange(len(nodes))
        
        nodes = nodes.sort_values('NODENR')
        nodes.index = np.arange(len(nodes))
        
        toDrop = []
        nodeNr = np.array(nodes['NODENR'], dtype=int)
        for i in range(1,len(nodes)):
            if nodeNr[i] == nodeNr[i-1]:
                toDrop.append(i)
                
        nodes = nodes.drop(index=toDrop)
        
        centroids = zones[['SEGNR_2018', 'XCOORD','YCOORD']]
        centroids.columns = ['NODENR', 'X', 'Y']
        
        nodes = nodes.append(centroids)
        nodes = nodes.sort_values('NODENR')
        nodes.index = nodes['NODENR']
        nNodes = len(nodes)
        
        print('\tCombining links...')
        
        links = linksN.copy()
        links = links[links['LINKTYPE']!=99]
        
        links = links.append(linksO[linksO['LINKTYPE']!=99])
        links = links.append(linksW[linksW['LINKTYPE']!=99])
        links = links.append(linksZ[linksZ['LINKTYPE']!=99])
        links.index = np.arange(len(links))
        
        links['LINK_ID'] = links['KNOOP_A'].astype(str) + '_' + links['KNOOP_B'].astype(str)
        links['LINKNR' ] = np.arange(len(links))
        
        links = links.sort_values('LINK_ID')
        links.index = np.arange(len(links))
        
        # Find which links are duplicate
        toDrop = []
        linkId       = np.array(links['LINK_ID' ], dtype=str)
        linksSnel    = np.array(links['SNEL'    ], dtype=float)
        linksGebCode = np.array(links['GEB_CODE'], dtype=int)
        linksNRM     = np.array(links['NRM'     ], dtype=int)
        
        linkDuplicates = {}
        for i in range(1,len(links)):
            if linkId[i] == linkId[i-1]:
                toDrop.append(i)
                try:
                    linkDuplicates[str(linkId[i])].append(i)
                except:
                    linkDuplicates[str(linkId[i])] = [i-1,i]
                
        for duplicateRows in linkDuplicates.values():
            firstRow = duplicateRows[0]
            if np.any(linksGebCode[duplicateRows]==1):
                whereStudyArea = duplicateRows[np.where(linksGebCode[duplicateRows]==1)[0][0]]
                links.at[firstRow, 'SNEL'] = linksSnel[whereStudyArea]
                links.at[firstRow, 'NRM' ] = linksNRM[whereStudyArea]
            else:                
                # Average the speed of the networks on duplicate links
                links.at[firstRow, 'SNEL'] = np.average(linksSnel[duplicateRows])
                links.at[firstRow, 'NRM' ] = 0
           
        # Keep only the first of the duplicate links
        links = links.drop(index=toDrop)
        links = links[['LINKNR',
                       'KNOOP_A',     'KNOOP_B',    'SNEL_FF',
                       'SNEL',        'CAP',        'LINKTYPE',
                       'DISTANCE',    'DOELSTROOK', 'NRM']]
        links = links.sort_values('LINKNR')
        links.index = np.arange(len(links))
    
        del linkDuplicates, toDrop
            
        print('\tAdding connector links..')
        # Voor het buitengebied: Choose the nearest node (euclidean distance)
        nodesX      = np.array(nodes.loc[nodes['NODENR']>10000, 'X']) / 1000
        nodesY      = np.array(nodes.loc[nodes['NODENR']>10000, 'Y']) / 1000
        nodesNODENR = np.array(nodes.loc[nodes['NODENR']>10000, 'NODENR'], dtype=int)

        connectorList = []
        for i in zones.index:
            regionNRM = str(zones.at[i, 'LANDSDEEL_1']).upper()
            segnr2018 = int(zones.at[i, 'SEGNR_2018'])
            
            if regionNRM == 'NOORD':
                zoneNr = zones.at[i, 'N']
                for connector in connectorsN[zoneNr]:
                    if connector['KNOOP_A'] < 10000:
                        connector.at['KNOOP_A'] = segnr2018
                    else:
                        connector.at['KNOOP_B'] = segnr2018
                    connectorList.append(connector)

            elif regionNRM == 'OOST':
                zoneNr = zones.at[i, 'O']
                for connector in connectorsO[zoneNr]:
                    if connector['KNOOP_A'] < 10000:
                        connector.at['KNOOP_A'] = segnr2018
                    else:
                        connector.at['KNOOP_B'] = segnr2018
                    connectorList.append(connector)
        
            elif regionNRM == 'WEST':
                zoneNr = zones.at[i, 'W']
                for connector in connectorsW[zoneNr]:
                    if connector['KNOOP_A'] < 10000:
                        connector.at['KNOOP_A'] = segnr2018
                    else:
                        connector.at['KNOOP_B'] = segnr2018
                    connectorList.append(connector)

            elif regionNRM == 'ZUID':
                zoneNr = zones.at[i, 'Z']
                for connector in connectorsZ[zoneNr]:
                    if connector['KNOOP_A'] < 10000:
                        connector.at['KNOOP_A'] = segnr2018
                    else:
                        connector.at['KNOOP_B'] = segnr2018
                    connectorList.append(connector)
                  
            # Maak zelf een connector aan in het buitengebied
            else:
                zoneNr = zones.at[i, 'N']
                x = zones.at[i,'XCOORD'] / 1000
                y = zones.at[i,'YCOORD'] / 1000

                distances = ((x - nodesX)**2 + (y - nodesY)**2)**0.5
                distances[segnr2018]
                whereMin = np.argmin(distances)
                connectingNode = nodesNODENR[whereMin]
                connectingDist = distances[whereMin] * 1000
                
                connectorAB = [connectingNode,
                               segnr2018,
                               50,
                               50,
                               99999,
                               99,
                               connectingDist,
                               0,
                               0,
                               0]

                connectorBA = [segnr2018,
                               connectingNode,
                               50,
                               50,
                               99999,
                               99,
                               connectingDist,
                               0,
                               0,
                               0]
    
                connectorList.append(connectorAB)
                connectorList.append(connectorBA)   
        
        nRows = len(connectorList)
        nCols = len(connectorList[0])
        cols = list(connectorList[0].index)
        connectorDF = pd.DataFrame(np.zeros((nRows, nCols)), columns=cols)
        for i in range(nRows):
            connectorDF.loc[i, :] = connectorList[i]
        connectorDF['LINKNR'] = np.max(links['LINKNR']) + 1 + np.arange(len(connectorDF))

        # Add connectors to the links array
        links = links.append(connectorDF)
        links = links[['LINKNR',
                       'KNOOP_A',     'KNOOP_B',    'SNEL_FF',
                       'SNEL',        'CAP',        'LINKTYPE',
                       'DISTANCE',    'DOELSTROOK', 'NRM']]
        links = links.sort_values('LINKNR')
        nLinks = len(links)
        links.index = np.arange(nLinks)
    
        # Hercoderen nodes
        nodeDict    = dict((i, nodes.iloc[i,0]) for i in range(nNodes))
        invNodeDict = dict((nodes.iloc[i,0], i) for i in range(nNodes))
        
        nodes['NODENR' ] = [invNodeDict[x] for x in nodes['NODENR'].values]
        nodes.index = np.arange(nNodes)
        
        links['KNOOP_A'] = [invNodeDict[x] for x in links['KNOOP_A'].values]
        links['KNOOP_B'] = [invNodeDict[x] for x in links['KNOOP_B'].values]
    
        del linksN, linksO, linksW, linksZ
        del nodesN, nodesO, nodesW, nodesZ
        del connectorDF, connectorList, connector
        del connectorsN, connectorsO, connectorsW, connectorsZ
            
        # Do the spatial coupling of ZEZ zones to the links
        links['ZEZ'] = 0
        links['NL']  = 0
        
        if pathExcludeLinksZEZ != '':
            excludeLinksZEZ = np.array(pd.read_csv(pathExcludeLinksZEZ), dtype=int)
            
        if pathLinksZEZ != '':
            linksZEZ = np.array(pd.read_csv(pathLinksZEZ), dtype=int)

        print('\tChecking which nodes are located in a ZE-zone...')
        log_file.write('\tChecking which nodes are located in a ZE-zone...\n')
    
        nodesX = np.array(nodes['X'], dtype=int)
        nodesY = np.array(nodes['Y'], dtype=int)
        linksA = np.array(links['KNOOP_A'], dtype=int)
        linksB = np.array(links['KNOOP_B'], dtype=int)
        linksType = np.array(links['LINKTYPE'], dtype=int)
        
        # Read ZE-zones
        zezZones = pd.read_csv(pathZEZ, sep=',')
        zezZones = np.array(zezZones['SEGNR_2018'], dtype=int)
                    
        shapelyNodes  = [Point(nodesX[i], nodesY[i]) for i in range(nNodes)]
        
        # Get zones as shapely MultiPolygon/Polygon objects
        shapelyZones = []
        for x in zonesGeometry:
            if x['type'] == 'MultiPolygon':
                temp = [Polygon(x['coordinates'][0][i]) for i in range(len(x['coordinates'][0]))]
                shapelyZones.append(MultiPolygon(temp))
            else:
                shapelyZones.append(Polygon(x['coordinates'][0]))
        shapelyZonesZEZ = [shapelyZones[i-1] for i in zezZones]
            
        # Check in which zone each node is located
        zezNodes = np.zeros(nNodes, dtype=int)   
        nlNodes  = np.zeros(nNodes, dtype=int) 
        for i in range(nNodes):
    
            if nlShape.contains(shapelyNodes[i]):
                nlNodes[i] = 1
                
                # No ZE-zones outside NL
                if pathLinksZEZ == '':
                    for shapelyZone in shapelyZonesZEZ:
                        if shapelyZone.contains(shapelyNodes[i]):
                            zezNodes[i] = 1
                            break
            
            if i%500 == 0:
                print('\t\t' + str(round((i / nNodes)*100, 1)) + '%', end='\r')
    
        print('\tChecking which links are located in a ZE-zone...')
        log_file.write('\tChecking links nodes are located in a ZE-zone...\n')
        
        # Check if link is located in ZEZ or in NL
        zezLinks = np.zeros(nLinks, dtype=int)
        nlLinks  = np.zeros(nLinks, dtype=int)
        for i in range(nLinks):
            # Geen snelwegen als ZEZ-links
            if linksType[i] != 1:
                if zezNodes[linksA[i]] == 1 or zezNodes[linksB[i]] == 1:
                    zezLinks[i] = 1
            
            if nlNodes[linksA[i]] == 1 or nlNodes[linksB[i]] == 1:
                nlLinks[i] = 1
    
        links['ZEZ'] = zezLinks
        links['NL']  = nlLinks
        
        if pathExcludeLinksZEZ != '':
            links.index = links['LINKNR']
            for i in excludeLinksZEZ:
                try:
                    links.at[i, 'ZEZ'] = 0
                except:
                    print('Waarschuwing: LINKNR ' + str(i) + ' uit ExcludeLinksZEZ niet gevonden.')
            links.index = np.arange(nLinks)

        if pathLinksZEZ != '':
            links.index = links['LINKNR']
            links.loc[linksZEZ[:, 0], 'ZEZ'] = linksZEZ[:, 1]
            links.index = np.arange(nLinks)
            
            print('\t\tFound ' + str(np.sum(links['ZEZ'])) + ' links located in a ZE-zone.')
            print('\t\tFound ' + str(np.sum(nlLinks     )) + ' links located in NL.')
        
        else:
            print('\t\tFound ' + str(np.sum(zezLinks)) + ' links located in a ZE-zone.')
            print('\t\tFound ' + str(np.sum(nlLinks )) + ' links located in NL.')
        
        del shapelyNodes, shapelyZones, zonesGeometry, zezNodes, zezLinks, nlNodes, nlLinks    
        del nodesX, nodesY, linksA, linksB
            
        # Assume a speed of 50 km/h if there are links with speed < 10
        nSpeedZero = np.sum(links['SNEL']<10)
        if nSpeedZero > 0:
            links.loc[links['SNEL']<10, 'SNEL'] = 10 
            print(         '\tWarning: ' + str(nSpeedZero) + ' links found with freight speed (SNEL) < 10 km/h. Adjusting those to 10 km/h.')
            log_file.write('\tWarning: ' + str(nSpeedZero) + ' links found with freight speed (SNEL) < 10 km/h. Adjusting those to 10 km/h.' + '\n')
    
        nSpeedInf = np.sum(links['SNEL']==np.inf)
        if nSpeedInf > 0:
            links.loc[links['SNEL']==np.inf, 'SNEL'] = 50 
            print(         '\tWarning: ' + str(nSpeedInf) + ' links found with freight speed (SNEL) is inf km/h. Adjusting those to 50 km/h.')
            log_file.write('\tWarning: ' + str(nSpeedInf) + ' links found with freight speed (SNEL) is inf km/h. Adjusting those to 50 km/h.' + '\n')
    
        nSpeedNan = np.sum(links['SNEL']==np.nan)
        if nSpeedNan > 0:
            links.loc[links['SNEL']==np.nan, 'SNEL'] = 50 
            print(         '\tWarning: ' + str(nSpeedNan) + ' links found with freight speed (SNEL) is nan km/h. Adjusting those to 50 km/h.')
            log_file.write('\tWarning: ' + str(nSpeedNan) + ' links found with freight speed (SNEL) is nan km/h. Adjusting those to 50 km/h.' + '\n')
    
        # Travel times
        links['T0_FREIGHT'] = 3600 * (links['DISTANCE'] / 1000) / [min(maxSpeedFreight, links.at[i,'SNEL']) for i in links.index]
        links['T0_VAN'    ] = 3600 * (links['DISTANCE'] / 1000) / links['SNEL']
        
            
        # ----------------------- Export loaded network to shapefile -------------------
        print("Exporting network to .shp..."), log_file.write("Exporting network to .shp...\n")            
        
        nodes['NODENR'] = [nodeDict[x] for x in nodes['NODENR']]
        nodes.to_csv(pathOutput + 'nodes.csv', sep=',', index=False)
        
        # Set travel times of connectors at 0 for in the output network shape
        links.loc[links['LINKTYPE']==99,'T0_FREIGHT'] = 0.0
        links.loc[links['LINKTYPE']==99,'T0_VAN'    ] = 0.0
    
        links['KNOOP_A'] = [nodeDict[x] for x in links['KNOOP_A']]
        links['KNOOP_B'] = [nodeDict[x] for x in links['KNOOP_B']]
        
        # Vervang eventuele NaN's
        links.loc[pd.isna(links['ZEZ']), 'ZEZ'] = 0
        
        # Initialize shapefile fields
        w = shp.Writer(pathOutput + 'links.shp')
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
    
        w.close()
        
        print('\t100%', end='\r')
        
    
        # --------------------------- End of module ---------------------------
                
        totaltime = round(time.time() - start_time, 2)
        print("Total runtime: %s seconds\n" % (totaltime))
        log_file.write("Total runtime: %s seconds\n" % (totaltime))  
        log_file.write("End simulation at: "+datetime.datetime.now().strftime("%y-%m-%d %H:%M")+"\n")
        log_file.close()
            
        
    except:
        import sys
        print(sys.exc_info()[0])
        log_file.write(str(sys.exc_info()[0])), log_file.write("\n")
        
        import traceback
        print(traceback.format_exc())
        log_file.write(str(traceback.format_exc())), log_file.write("\n")
        log_file.write("Execution failed!")
        log_file.close()
        
        input()
        
       
    
#%% 
        
if __name__ == '__main__':
    
    main()
    


