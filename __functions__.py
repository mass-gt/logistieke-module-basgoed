# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 12:17:13 2020

@author: modelpc
"""
import pandas as pd
import numpy as np
import shapefile as shp
import array
import os.path



def read_mtx(mtxfile):  
    '''
    Read a binary mtx-file (skimTijd and skimAfstand)
    '''
    mtxData = array.array('f')  # i for integer
    mtxData.fromfile(open(mtxfile, 'rb'), os.path.getsize(mtxfile) // mtxData.itemsize)
    
    # The number of zones is in the first byte
    mtxData = np.array(mtxData, dtype=float)[1:]
    
    return mtxData



def write_mtx(filename, mat, aantalZones):
    '''
    Write an array into a binary file
    '''
    mat     = np.append(aantalZones,mat)
    matBin  = array.array('f')
    matBin.fromlist(list(mat))
    matBin.tofile(open(filename, 'wb'))



def read_shape(shapePath, encoding='latin1', returnGeometry=False):
    '''
    Read the shapefile with zones (using pyshp --> import shapefile as shp)
    '''
    # Load the shape
    sf = shp.Reader(shapePath, encoding=encoding)
    records = sf.records()
    if returnGeometry:
        geometry = sf.__geo_interface__
        geometry = geometry['features']
        geometry = [geometry[i]['geometry'] for i in range(len(geometry))]
    fields = sf.fields
    sf.close()
    
    # Get information on the fields in the DBF
    columns  = [x[0] for x in fields[1:]]
    colTypes = [x[1:] for x in fields[1:]]
    nRecords = len(records)
    
    # Check for headers that appear twice
    for col in range(len(columns)):
        name = columns[col]
        whereName = [i for i in range(len(columns)) if columns[i]==name]
        if len(whereName) > 1:
            for i in range(1,len(whereName)):
                columns[whereName[i]] = str(columns[whereName[i]]) + '_' + str(i)
                
    # Put all the data records into a NumPy array (much faster than Pandas DataFrame)
    shape = np.zeros((nRecords,len(columns)), dtype=object)
    for i in range(nRecords):
        shape[i,:] = records[i][0:]
    
    # Then put this into a Pandas DataFrame with the right headers and data types
    shape = pd.DataFrame(shape, columns=columns)
    for col in range(len(columns)):
        if colTypes[col][0] == 'C':
            shape[columns[col]] = shape[columns[col]].astype(str)
        else:
            shape.loc[pd.isna(shape[columns[col]]), columns[col]] = -99999
            if colTypes[col][-1] > 0:
                shape[columns[col]] = shape[columns[col]].astype(float)
            else:
                shape[columns[col]] = shape[columns[col]].astype(int)
            
    if returnGeometry:
        return (shape, geometry)
    else:
        return shape



def read_nodes(pathNodes):
    '''
    Read NRM nodes.
    '''
    fieldWidths = [6,6,10,10]
    colSpecs    = [(0,fieldWidths[0])]
    for i in range(len(fieldWidths)-1):
        colSpecs.append((colSpecs[i][1],colSpecs[i][1]+fieldWidths[i+1]))
        
    with open(pathNodes, 'r') as f:
        data = f.readlines()
        
    nRows = len(data)
    nCols = len(fieldWidths)
    nodes = np.zeros((nRows, nCols), dtype=object)
    
    for row in range(nRows):
        for col in range(nCols):
            nodes[row,col] = data[row][colSpecs[col][0]:colSpecs[col][1]]

    nodes = pd.DataFrame(nodes, columns=['NODENR','DEL','X','Y'])
    nodes = nodes.drop(columns=['DEL'])
    nodes = nodes.astype(int)
    
    return nodes



def read_links(pathLinks):
    '''
    Read NRM links.
    '''
    # Check op dataformaat (eerste 2 regels)
    with open(pathLinks, 'r') as f:
        firstRow  = f.readline()
        secondRow = f.readline()
        
    if len(secondRow) == 61:
        loaded = False
    elif len(firstRow) == 225:
        loaded = True
    else:
        raise BaseException("\nUnexpected file format for road network links (" + 
                            str(pathLinks) + ")\n" +
                            "Expected either a file with a width of 225 characters (loaded network) or " +
                            "a file with a width of 61 characters (unloaded network).")

    # Lees het hele bestand nu in
    with open(pathLinks, 'r') as f:
        data = f.readlines()
            
    # Geladen netwerk
    if loaded:
        fieldWidths = [6,6,7,7,7,7,7,7,7,7,7,7,2,3,2,6,4,7,2,4,4,10,10,10,10,3,3,8,8,8,8,8,4,3,2,3,10]
        colSpecs    = [(0,fieldWidths[0])]
        for i in range(len(fieldWidths)-1):
            colSpecs.append((colSpecs[i][1],colSpecs[i][1]+fieldWidths[i+1]))        
        
        nRows = len(data)
        nCols = len(fieldWidths)
        links = np.zeros((nRows, nCols), dtype=object)

        for row in range(nRows):
            for col in range(nCols):
                links[row,col] = data[row][colSpecs[col][0]:colSpecs[col][1]]        
        
        columns = ['Anode',         'Bnode',     'Flow',         'Q_bb',     'Fcap',
                   'Q_in',          'N_bb',      'FFcost',       'Wcost',    'Dcost',
                   'Cost',          'Dist',      'Lane',         'HWN',      'HOV',
                   'FBlk',          'WvType',    'FlCpW',        'Ltype',    'IvF',
                   'Hef',           'Nbb_in',    'Nek of Nprim', 'Nbb_rest', 'Buffer',
                   'Leidend',       'Leidlink',  'WvDist',       'Fileduur',  'VVU',
                   'VrachtTijdFF',  'VrachtTijd','WetSnel',      'NRM-type',  'Geb-C',
                   'DEL1',          'DEL2']
        
        links = pd.DataFrame(links, columns=columns)
        links['SNEL_MOD'] = 3.6 * links['Dist'].astype(float) / links['VrachtTijd'].astype(float)
        links = links[['Anode', 'Bnode', 'WetSnel', 'SNEL_MOD', 'Fcap', 'NRM-type', 'Dist', 'HOV', 'Geb-C']]
        
        links.columns = ['KNOOP_A',     'KNOOP_B',    'SNEL_FF',
                         'SNEL',        'CAP',        'LINKTYPE',
                         'DISTANCE',    'DOELSTROOK', 'GEB_CODE']
        
        for col in ['KNOOP_A', 'KNOOP_B', 'SNEL_FF', 'CAP', 
                    'LINKTYPE','DISTANCE', 'DOELSTROOK', 'GEB_CODE']:
            links[col] = links[col].astype(float).astype(int)
        links['SNEL'] = links['SNEL'].astype(float)
        links = links.fillna(-99999)
        
        return (links, loaded)
       
    # Ongeladen netwerk
    else:        
        fieldWidths = [6,6,5,3,5,2,3,7,2,2,2,2,2,2,2,2,7]
        colSpecs    = [(0,fieldWidths[0])]
        for i in range(len(fieldWidths)-1):
            colSpecs.append((colSpecs[i][1],colSpecs[i][1]+fieldWidths[i+1]))
            
        data = data[1:]
            
        nRows = len(data)
        nCols = len(fieldWidths)
        links = np.zeros((nRows, nCols), dtype=object)
        
        for row in range(nRows):
            for col in range(nCols):
                links[row,col] = data[row][colSpecs[col][0]:colSpecs[col][1]]
                
        links = pd.DataFrame(links, columns=['KNOOP_A',  'KNOOP_B', 'SNEL_FF',
                                             'SNEL',     'CAP',     'LINKTYPE',
                                             '1',
                                             'DISTANCE',
                                             '2','3','4','DOELSTROOK','5','6','7','8','9'])
        links = links.drop(columns=[str(x+1) for x in range(9)])
        links['SNEL'] = links['SNEL_FF']
        links['GEB_CODE'] = 0
        links = links.fillna(-99999)
        links = links.astype(int)
           
        return (links, loaded)    