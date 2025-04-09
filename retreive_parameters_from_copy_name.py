# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 15:43:57 2022

@author: Caterina

given copy name retreives copy parameters analyzing the string of the copy key

    

"""

import numpy as np
import glob
import json


#----------------------------------------------------------------------

'''
INPUT
'''

# output folder: dove leggere il fil JSON del DB e dove scrivere il file JSON del DB ottenuto

copyName="95810005_iadap_0.1_Idep_0.75_Idep0_0.25_c_0.75_d_0.5_R_0_0_-10_0_25_0_p_1"


'''
MAIN --------------------------------
'''
def retrive_parameters_from_copy_name(originalPyramidalDatabase,copyName):

    '''
    # outputdictionary
    '''
    neuronCopiesDatabase={} # result
    
    filenameSplitted = copyName.split('_')
    
    #corrente = filenameSplitted[3][:-2]
    
    nomeNeurone = filenameSplitted[0]
    iadapCoeff = np.float64(filenameSplitted[2])
    idepCoeff = np.float64(filenameSplitted[4])
    idep0Coeff = np.float64(filenameSplitted[6])
    cCoeff = np.float64(filenameSplitted[8])
    dCoeff = np.float64(filenameSplitted[10])
    R1coeff = np.float64(filenameSplitted[12])
    R2coeff = np.float64(filenameSplitted[13])
    R3coeff = np.float64(filenameSplitted[14])
    R4coeff = np.float64(filenameSplitted[15])
    R5coeff = np.float64(filenameSplitted[16])
    R6coeff = np.float64(filenameSplitted[17])
    pCoeff= np.float64(filenameSplitted[19])
    
    
    neuronID = nomeNeurone
    
    neuronCopiesDatabase[neuronID]={}
    if iadapCoeff==0 and idepCoeff==1 and idep0Coeff==1 and cCoeff==1 and dCoeff==1 and R1coeff==0 and R2coeff==0 and R3coeff==0 and R4coeff==0 and R5coeff==0 and R6coeff==0 and pCoeff==1:
        neuronCopiesDatabase[neuronID]['name'] = nomeNeurone
    
    else:
        neuronCopiesDatabase[neuronID]['name'] = nomeNeurone + '_copy'
        #print('copia')
    
    neuronCopiesDatabase[neuronID]['type'] = originalPyramidalDatabase[nomeNeurone]['type']
    
    #try:        
    neuronCopiesDatabase[neuronID]['parameters']={}
        # inserisce i dati ne DB
    neuronCopiesDatabase[neuronID]['parameters']['EL']=originalPyramidalDatabase[nomeNeurone]['parameters']['EL']
    
    neuronCopiesDatabase[neuronID]['parameters']['Vres']=originalPyramidalDatabase[nomeNeurone]['parameters']['Vres']
    
    neuronCopiesDatabase[neuronID]['parameters']['VTM']=originalPyramidalDatabase[nomeNeurone]['parameters']['VTM']
    
    neuronCopiesDatabase[neuronID]['parameters']['Cm']=originalPyramidalDatabase[nomeNeurone]['parameters']['Cm']
    
    neuronCopiesDatabase[neuronID]['parameters']['Ith']=originalPyramidalDatabase[nomeNeurone]['parameters']['Ith']
    
    neuronCopiesDatabase[neuronID]['parameters']['tao']=originalPyramidalDatabase[nomeNeurone]['parameters']['tao']
    
    neuronCopiesDatabase[neuronID]['parameters']['sc']=originalPyramidalDatabase[nomeNeurone]['parameters']['sc']
    
    neuronCopiesDatabase[neuronID]['parameters']['alpha']=originalPyramidalDatabase[nomeNeurone]['parameters']['alpha']
    
    neuronCopiesDatabase[neuronID]['parameters']['bet']=originalPyramidalDatabase[nomeNeurone]['parameters']['bet']
        
    neuronCopiesDatabase[neuronID]['parameters']['delta1']=originalPyramidalDatabase[nomeNeurone]['parameters']['delta1']
        
    neuronCopiesDatabase[neuronID]['parameters']['Idep_ini']=originalPyramidalDatabase[nomeNeurone]['parameters']['Idep_ini']*idepCoeff
      
    neuronCopiesDatabase[neuronID]['parameters']['Idep_ini_vr']=originalPyramidalDatabase[nomeNeurone]['parameters']['Idep_ini_vr']*idep0Coeff
       
    neuronCopiesDatabase[neuronID]['parameters']['psi']=originalPyramidalDatabase[nomeNeurone]['parameters']['psi']
        
    neuronCopiesDatabase[neuronID]['parameters']['time scale']=originalPyramidalDatabase[nomeNeurone]['parameters']['time scale']
    
    neuronCopiesDatabase[neuronID]['parameters']['A']=originalPyramidalDatabase[nomeNeurone]['parameters']['A']
    
    neuronCopiesDatabase[neuronID]['parameters']['B']=originalPyramidalDatabase[nomeNeurone]['parameters']['B']
    
    neuronCopiesDatabase[neuronID]['parameters']['C']=originalPyramidalDatabase[nomeNeurone]['parameters']['C']*cCoeff
    
    neuronCopiesDatabase[neuronID]['parameters']['alphaD']=originalPyramidalDatabase[nomeNeurone]['parameters']['alphaD']*dCoeff
     
    paramRette = ['ValInfLineSup','coeffIsup','costSup','ValSupLineInf','coeffIinf','costInf']
    neuronCopiesDatabase[neuronID]['block_line_params']={}
    for parametro in paramRette:
        val = originalPyramidalDatabase[nomeNeurone]['block_line_params'][parametro]
        if val != 'inf' and val !='-inf':
            print('****')
            print(parametro,val)
            neuronCopiesDatabase[neuronID]['block_line_params'][parametro]=np.float64(originalPyramidalDatabase[nomeNeurone]['block_line_params'][parametro])+R1coeff
        else:
            neuronCopiesDatabase[neuronID]['block_line_params'][parametro]=originalPyramidalDatabase[nomeNeurone]['block_line_params'][parametro]
    
    # parametri da sistemare nel DB
    neuronCopiesDatabase[neuronID]['parameters']['Iadap']=originalPyramidalDatabase[nomeNeurone]['parameters']['Iadap']*iadapCoeff
    neuronCopiesDatabase[neuronID]['parameters']['p']=pCoeff
    
        
    return(neuronCopiesDatabase)
