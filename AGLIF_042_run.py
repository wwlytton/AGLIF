# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 15:10:58 2024

@author: Caterina

richiede: 
    il nome della copia da simulare: Es: copyName = '95817003_iadap_0.5_Idep_2_Idep0_1_c_1_d_1_R_0_-50_0_0_0_0_p_1'    
        NB: un neurone originale ha un nome della forma: '95817003_iadap_0_Idep_1_Idep0_1_c_1_d_1_R_0_0_0_0_0_0_p_1' 
    
    expNeuronDB_V006small_piramidali_dati_2023_06_15.json
    expNeuronDB_V006small_interneuroni_dati_2023_06_15.json
    
restituisce i file
    copyName+'_'+str(Istim)+'_t_spk_simulated.txt'    
"""
import AGLIF_042
import matplotlib.pyplot as plt
from AGLIF_042 import AGLIFsynaptic
import retreive_parameters_from_copy_name
from retreive_parameters_from_copy_name import retrive_parameters_from_copy_name
import numpy as np
import json

'''
MAIN
'''
# user input -------------------------------------------------------------------

copyName = '95817003_iadap_0_Idep_1_Idep0_1_c_1_d_1_R_0_0_0_0_0_0_p_1'
print('########### %s #########'%copyName)

piramidali = True
interneuroni = False

# corrente costante
Istim = 1000
# durata della simulaizone in ms
sim_lenght = 1000

tSpikeOutputFileName = copyName+'_'+str(Istim)+'_t_spk_simulated_.txt'
voltageOutputFileName = copyName+'_'+str(Istim)+'_voltage_simulated_.txt'

# END user input -------------------------------------------------------------------



'''
elaborazione di dati forniti in input
'''
change_cur=1
campionamento=40
d_dt=0.005*campionamento
corr_list=np.ones(int(sim_lenght/d_dt))*0 # lista di zeri
corr_list[int(change_cur/d_dt):int(sim_lenght/d_dt)+1] = np.ones(len(corr_list[int(change_cur/d_dt):int(sim_lenght/d_dt)+1]))*Istim
corr_list[int(sim_lenght*0.8/d_dt)+1 : int(sim_lenght/d_dt)+1] = 0
#[0,0,0,0,800,800,800,800,...]

selCurr = Istim

time_step = d_dt

corr_time = np.arange(0,sim_lenght,d_dt)

if piramidali:
    # dove leggere il fil JSON del DB 
    JSONFilePath=""
    JSONfileName = 'expNeuronDB_V006small_piramidali_dati_2023_06_15'

if interneuroni:
    # dove leggere il fil JSON del DB 
    JSONFilePath=""
    JSONfileName = 'expNeuronDB_V006small_interneuroni_dati_2023_06_15'

f = open(JSONFilePath+JSONfileName+'.json',)
# returns JSON object as a dictionary
originalPyramidalDatabase = json.load(f)
# Closing file
f.close()

nomeNeurone=copyName.split('_')[0]

# recupera i dati del neurone
print(originalPyramidalDatabase[nomeNeurone])
params =retrive_parameters_from_copy_name(originalPyramidalDatabase,copyName)


print('params',nomeNeurone, params)

retteOrig = [float(x) for x in list(params[nomeNeurone]['block_line_params'].values())]

# equilibrium parameters
v_min = -90
minCurr = -185
# Neuron
zeta = 3.5e-3
eta = 2.5e-3
rho = 1e-3
csi = 3.5e-3

equilibriumParameters = [v_min,minCurr,zeta,eta,rho,csi]


neuronParameters = [
    params[nomeNeurone]['parameters']['EL'],
    params[nomeNeurone]['parameters']['Vres'],
    params[nomeNeurone]['parameters']['VTM'],
    params[nomeNeurone]['parameters']['Cm']*params[nomeNeurone]['parameters']['p'],
    params[nomeNeurone]['parameters']['Ith']*params[nomeNeurone]['parameters']['p'],
    params[nomeNeurone]['parameters']['tao'],
    params[nomeNeurone]['parameters']['sc']*params[nomeNeurone]['parameters']['p'],
    params[nomeNeurone]['parameters']['bet'],
    params[nomeNeurone]['parameters']['delta1'],
    params[nomeNeurone]['parameters']['Idep_ini'],#Idep_start
    params[nomeNeurone]['parameters']['Idep_ini_vr'],#Idep0
    params[nomeNeurone]['parameters']['psi'],
    params[nomeNeurone]['parameters']['A'],
    params[nomeNeurone]['parameters']['B'],
    params[nomeNeurone]['parameters']['C'],
    params[nomeNeurone]['parameters']['alphaD'],
    200,#istim_min_spikinig_exp
    1000,#istim_max_spikinig_exp
    sim_lenght,#sim_lenght
    retteOrig,
    time_step
    ]

print(neuronParameters)


'''
# chiamata alla funzione AGLIF
'''
AGLIFsynaptic(neuronParameters,equilibriumParameters,corr_list,tSpikeOutputFileName,voltageOutputFileName, Istim,Istim)


'''
plot
'''
plt.figure()
voltage = np.loadtxt(voltageOutputFileName, usecols=(1),dtype=np.float64, unpack=True)#,delimiter=',',)
time = np.loadtxt(voltageOutputFileName, usecols=(0),dtype=np.float64, unpack=True)#,delimiter=',',)
plt.plot(time,voltage,'o-')
plt.axhline(y = params[nomeNeurone]['parameters']['VTM'],color='r',linestyle='-',linewidth=1)
plt.title('Voltage @ '+str(selCurr)+'pA input')           
#plt.ylim([-75.0,-43.0])
#plt.xlim([95,200])

        
plt.figure()
plt.plot(corr_time,corr_list,'o-')
ith=params[nomeNeurone]['parameters']['Ith']*params[nomeNeurone]['parameters']['p']
plt.axhline(y = ith,color='r',linestyle='-',linewidth=1)
plt.grid(which='both')
plt.title('corrente')
#plt.xlim([14,16])

