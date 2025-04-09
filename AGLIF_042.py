# 002: testi in inglese
import numpy as np
#import sys
import matplotlib.pyplot as plt
import os.path
import re
import time
#from ipdb import set_trace as st


def AGLIFsynaptic(neuronParameters,equilibriumParameters,corr_list,tSpikeOutputFileName,voltageOutputFileName, Istim, corr_time):
    
    EL = neuronParameters[0]
    vres = neuronParameters[1]
    vtm = neuronParameters[2]
    Cm = neuronParameters[3]
    ith = neuronParameters[4]
    tao_m = neuronParameters[5]
    sc = neuronParameters[6]
    bet = neuronParameters[7]
    delta1 = neuronParameters[8]
    cost_idep_ini = neuronParameters[9]
    Idep_ini_vr = neuronParameters[10]
    psi1 = neuronParameters[11]
    a=neuronParameters[12]
    b=neuronParameters[13]
    c=neuronParameters[14]
    alp=neuronParameters[15]
    istim_min_spikinig_exp=neuronParameters[16]
    istim_max_spikinig_exp=neuronParameters[17]
    time_scale = 1 / (-sc / (Cm * EL))
    print(time_scale)
    H = (90+EL)*sc*(bet-delta1)/(EL*(-200))
    sim_lenght = neuronParameters[18]
    retteParParsed = neuronParameters[19]
    time_step = neuronParameters[20]
    
    # equilibrium params
    v_min = equilibriumParameters[0]
    minCurr = equilibriumParameters[1]
    zeta = equilibriumParameters[2]
    eta = equilibriumParameters[3]
    rho = equilibriumParameters[4]
    csi = equilibriumParameters[5]
         

    #sampling = 2
    #d_dt = 0.05*sampling
    # campionamento=40
    # d_dt=0.005*campionamento
    
    d_dt= time_step #sim_lenght/len(corr_list) # 
    # print('REsolution =', d_dt)
    # reads current
    cor = np.array(corr_list)
    #cor = cor[0:len(cor):1]
    
    
    def tagliorette(corr,retteParParsed):
        vinc_sup = retteParParsed[0]
        coeffSup = retteParParsed[1]
        constSup = retteParParsed[2]
        vinc_inf = retteParParsed[3]
        coeffInf = retteParParsed[4]
        constInf = retteParParsed[5]
        
        dur_sign=np.inf
    
        if corr<vinc_inf and corr>0:
            dur_sign = coeffInf*corr + constInf
        
        if corr>vinc_sup:
            dur_sign = coeffSup*corr + constSup
        return dur_sign    
    
    def V(t, delta, Psi, alpha, beta, IaA0, IdA0, t0, V0):
        return (1 / 2) * (beta + (-1) * delta) ** (-1) * (beta ** 2 + ((-1) + beta) * delta) ** (-1) * (4 * beta + (- 1) * (1 + delta) ** 2) ** (-1) * Psi * (2 * np.exp(((-1) * t + t0) * beta) * IdA0 * ((-1) + beta) * beta * (beta + (-1) * delta) * Psi + (-2) * (alpha + (-1) * beta + delta) * (beta ** 2 + ((- 1) + beta) * delta) * Psi + np.exp((1 / 2) * (t + (-1) * t0) * ((-1) + delta + (-1) * Psi)) * (IdA0 * beta * (beta + (-1) * delta) * ((-1) + (-1) * delta + beta * (3 + delta + (-1) * Psi) + Psi) + (-1) * (beta ** 2 + (-1) * delta + beta * delta) * (alpha * (1 + (-2) * beta + delta + (-1) * Psi) + (beta + (-1) * delta) * ((-1) + 2 * IaA0 * beta + (-1) * delta + Psi + V0 * ((-1) + (-1) * delta + Psi)))) + np.exp((1 / 2) * (t + (-1) * t0) * ((-1) + delta + Psi)) * ((-1) * IdA0 * beta * (beta+(-1) * delta) * ((-1) + (-1) * delta + (-1) * Psi + beta * (3 + delta + Psi)) + (beta ** 2 + (-1) * delta+beta * delta) * (alpha * (1 + (-2) * beta + delta + Psi) + (beta + (-1) * delta) * ((-1) + 2 * IaA0 * beta+(-1) * delta + (-1) * Psi + (-1) * V0 * (1 + delta + Psi)))))
    
    
    def Iadap(t, delta, Psi, alpha, beta, IaA0, IdA0, t0, V0):
        return (-2*alpha*(-4*beta**3+beta**2*(-1+delta)**2-delta*(1+delta)**2+beta*delta*(5+2*delta+delta**2))+2*np.exp(((-1)*t + t0) * beta)*IdA0*beta*(4*beta**2+delta*(1+delta)**2-beta*(1+6*delta+delta**2))+np.exp((1 / 2)*(t-t0)*(-1+delta+Psi))*(-IdA0*beta*(beta-delta)*(-1+(-2)*delta-delta**2-Psi+delta*Psi+2*beta*(2+Psi))+(beta**2-delta+beta*delta)*(alpha*(1+(-4)*beta+2*delta+delta**2+Psi-delta*Psi)+(beta-delta)*(4*IaA0*beta-2*(1+V0)*Psi+IaA0*(1+delta)*(-1-delta+Psi))))+np.exp((-1)*(1 / 2) * (t-t0) * (1-delta+Psi))*(IdA0*beta*(beta-delta)*(1+2*delta+delta**2-Psi+delta*Psi+2*beta*(-2+Psi))+(beta**2-delta+beta*delta)*(alpha*(1-4*beta+2*delta+delta**2-Psi+delta*Psi)-(beta-delta)*(-4*IaA0*beta-2*(1+V0)*Psi+IaA0*(1+delta)*(1+delta+Psi)))))/(2*(beta-delta)*(beta**2+(-1+beta)*delta)*(4*beta-(1+delta)**2))
    
    
    def Idep(t, beta, IdA0, t0):
        return np.exp(((-1) * t + t0) * beta) * IdA0
    
    
    def exp_cum(x, a, b):
        return a * (1 - np.exp(-b * x))
    
    
    def monod(x, a, b, c, alp):
        return c + (a * np.exp(b) * x) / (alp + x)
    
    
    tic = time.perf_counter()
    
    Vconvfact = -EL
    vth = vtm/Vconvfact
    vrm = vres/Vconvfact
    
    t0_val = 0
    vini_neg = EL
    
    ts = np.inf
    
    dt = d_dt/time_scale
    # print('Time step = ', d_dt, ' Normalized time step dt= ', dt)
    init_sign = 0
    ref_t = 2
    
    t0_val = 0
    psi1 = ((-4)*bet+((1+delta1)**2))**(0.5)
    
    Idep_ini = 0
    Iadap_ini = 0
    out = []
    t_out = []
    
    t_final = t0_val+dt
    v_ini = -1
    vini_prec = v_ini
    
    mul = 15
    
    v_star_min = -v_min/EL
    alpha_neg = minCurr/sc
    
    f = open(tSpikeOutputFileName, 'w')
    i = 0
    
    soglia_sign = 10
    Ide = []
    Iada = []
    Ide2 = []
    Iada2 = []
    tetalist = []
    
    t_spk = -3*d_dt
    blockActive = False
    afirst = 0
    meancorlastis = 0
    stdcorlastis = 0
    sis = 0
    
    firstSpikeFlag = False
    counter = 0


    # print('Sim Lenght ', sim_lenght)
    
    while(t_final*time_scale < sim_lenght):
        #print(t_final*time_scale)
        # if t0_val*time_scale < 15:
            # print("PYTHON Blockactive ", counter, t0_val*time_scale , blockActive)
            # counter += 1
        # if i< 200:
        #     print('controll blocco ',i,(t_final-init_sign)*time_scale,tagliorette(cor[i],retteParParsed),cor[i])
        if ((t_final-init_sign)*time_scale >= tagliorette(cor[i],retteParParsed) and blockActive == True):
            # print('blocco ',(t_final-init_sign)*time_scale,tagliorette(cor[i],retteParParsed),cor[i])
            if cor[i] > ith:
                if cor[i-1] < ith or i == 0:
                    
                    blockActive = False
                    
                    init_sign = t_final
                    Idep_ini = max(Idep_ini_vr,cost_idep_ini*(cor[i]-ith))
                    Iadap_ini = 0
                    
                    v_ini = ((EL + (1 - np.exp(-(zeta*1000*cor[i] - rho*1000*ith)/1000) )*(vtm - EL))/(-EL))
                    
                    v_ini = V(t_final, delta1, psi1,
                             cor[i]/sc, bet, Iadap_ini, Idep_ini, t0_val, v_ini)
                    Iadap_ini = Iadap(
                        t_final, delta1, psi1, cor[i] / sc, bet, Iadap_ini, Idep_ini, t0_val, v_ini) 
                    Idep_ini = Idep(t_final, bet, Idep_ini, t0_val)
           
            if cor[i-1] == 0:
                v_ini = vini_prec
            else:
                
                if cor[i]<ith and cor[i]>0:                
                  v_ini = ((EL + (1 - np.exp(-(eta*1000*cor[i])/1000) )*(vtm - EL))/(-EL))
                elif cor[i]<=0:                  
                  v_ini = ((EL + (1 - np.exp(-(csi*1000*cor[i])/1000) )*(vtm - EL))/(-EL))
                else:
                  v_ini = ((EL + (1 - np.exp(-(zeta*1000*cor[i] - rho*1000*ith)/1000) )*(vtm - EL))/(-EL))
                
                
            vini_prec = v_ini
            out.append(v_ini)
            t_out.append(t_final*time_scale)
            Iada.append(Iadap_ini)
            Ide.append(Idep_ini)
        else:
            vini_prec = v_ini
            
            if (cor[i] < ith and cor[i] >= 0) or i == 0:

                # if t0_val*time_scale < 15:
                #     print('0 Corrente positiva crescente time',t0_val * time_scale, " to ", t_final * time_scale, i, cor[i])
                v_ini = ((EL + (1 - np.exp(-(eta*1000*cor[i])/1000) )*(vtm - EL))/(-EL))

            else:
                if cor[i] < cor[i-1] and cor[i] > 0 and (t_spk+2*d_dt) < t_final*time_scale:
                    #print('teta',t_final*time_scale)
                    teta = (out[i-1]/(cor[i-1] / sc))*(1/dt-delta1) - \
                        (out[i-2]/((cor[i-1] / sc)*dt))-delta1/(cor[i-1] / sc)-1
                    if teta < 0:
                        teta = 0
                    Idep_ini = Iadap_ini + teta * (cor[i] / sc) / bet
                    tetalist.append(teta)
                    v_ini = V(t=t_final, delta=delta1, Psi=psi1,
                              alpha=cor[i]/sc, beta=bet, IaA0=Iadap_ini, IdA0=Idep_ini, t0=t0_val, V0=v_ini)
                    Iadap_ini = Iadap(t=t_final, delta=delta1, Psi=psi1,
                                      alpha=cor[i]/sc, beta=bet, IaA0=Iadap_ini, IdA0=Idep_ini, t0=t0_val, V0=v_ini)
                    Idep_ini = Idep(t=t_final, beta=bet,
                                    IdA0=Idep_ini, t0=t0_val)
    
                else:
                    if cor[i] > 0:
                        #print('1 time',t0_val * time_scale, v_ini, 'Iadap',Iadap_ini,'Idep_ini',Idep_ini,cor[i])
                        
                        # print(' t0_val ', t0_val*time_scale, ' delta ', delta1, ' Psi ', psi1,
                        #           'alpha ', cor[i]/sc, ' beta ' , bet, ' IaA0 ', Iadap_ini, ' IdA0 ' , Idep_ini, ' t0 ', t0_val, ' V0 ', v_ini)
                        v_ini = V(t=t_final, delta=delta1, Psi=psi1,
                                  alpha=cor[i]/sc, beta=bet, IaA0=Iadap_ini, IdA0=Idep_ini, t0=t0_val, V0=v_ini)
                        # print("v_ini ", v_ini)
                        # print("")
                        Iadap_ini = Iadap(t=t_final, delta=delta1, Psi=psi1, alpha=cor[i]/sc, beta=bet, IaA0=Iadap_ini, IdA0=Idep_ini, t0=t0_val, V0=v_ini)
                        Idep_ini = Idep(t=t_final, beta=bet, IdA0=Idep_ini, t0=t0_val) 
                                         
                        #print('2 time', t_final * time_scale, v_ini, 'Iadap',Iadap_ini,'Idep_ini',Idep_ini,cor[i])
                                                

                if cor[i-1] != cor[i] and (cor[i] < 0 and cor[i] > minCurr):

                    v_ini = vini_prec
                                   
                if cor[i] < 0 and cor[i] > minCurr:

                    v_ini = ((EL + (1 - np.exp(-(csi*1000*cor[i])/1000) )*(vtm - EL))/(-EL))                                    
    
                if cor[i-1] != cor[i] and cor[i] <= minCurr:
                    Iadap_ini = -v_min/EL + 1
                    Idep_ini = 0
    
                    v_ini = ((EL + (1 - np.exp(-(csi*1000*cor[i])/1000) )*(vtm - EL))/(-EL))                   
    
                if cor[i] <= minCurr:
                    v_ini=v_star_min
    
            if v_ini*Vconvfact < v_min:
                v_ini = v_min/Vconvfact
                Iadap_ini = 0
            
            if cor[i] > ith:

                if cor[i-1] < ith:
                    
                    blockActive = False # UPGRADE
                    
                    init_sign = t_final
                    Idep_ini = max(Idep_ini_vr,cost_idep_ini*(cor[i]-ith))
                    Iadap_ini = 0                    
                    
                    #print('da sotto a sopra 1',t0_val * time_scale, " to ", t_final * time_scale, v_ini, cor[i],i)
                    v_ini =  ((EL + (1 - np.exp(-(zeta*1000*cor[i] - rho*1000*ith)/1000) )*(vtm - EL))/(-EL))
                    
                    #print('da sotto a sopra 2',t0_val * time_scale, " to ", t_final * time_scale, v_ini, cor[i],i)
                    #

                    if cor[i-1]<1e-11: 
                        v_ini = -1
                        print('<0')
            
            out.append(v_ini)
            t_out.append(t_final*time_scale)
            Iada.append(Iadap_ini)
            Ide.append(Idep_ini)
            
            if v_ini > vth:
                     
                
                t_spk = t_final*time_scale
                f.write(str(round(t_spk, 3)) + ' \n')
                v_ini = vrm
                
                blockActive = True
        
                print('***spike***')
                print('t ', t_final*time_scale, 'val_ist V', v_ini * Vconvfact, 'adap',
                      Iadap_ini, 'idep', Idep_ini, 't_ini', init_sign)
                print('************')
                
                if cor[i] < istim_min_spikinig_exp or cor[i] > istim_max_spikinig_exp:
                    
                    c_aux = c
                    Iadap_ini = monod((t_final-init_sign) *
                          time_scale, a, b*cor[i]/1000, c_aux, alp)
                    
                else:
                    
                    Iadap_ini = monod((t_final-init_sign) * time_scale, a, b*cor[i]/1000, c, alp)                    
                    
                    if Iadap_ini<0:
                        print('monod negativa')

                        paramL = Iadap_ini
                        if a > 0:
                            c_aux = c - paramL
                        else:
                            c_aux = -a*np.exp(b*cor[i]/1000)
                        Iadap_ini = monod((t_final-init_sign) * time_scale, a, b*cor[i]/1000, c_aux, alp)
                            
                if cor[i] < ith:
 
                    Idep_ini = 0
                    Iadap_ini = 0
                else:
                    Idep_ini = Idep_ini_vr
                
                for k in range(int(ref_t / d_dt)):
                    out.append(v_ini)
                    t_final = t_final + dt
                    t_out.append(t_final*time_scale)
                    Iada.append(Iadap_ini)
                    Ide.append(Idep_ini)
                    i = i + 1                
    
            vini_prec = v_ini
    
        i = i + 1
        t0_val = t_final
        t_final = t0_val+dt
    

    
    file = open(voltageOutputFileName, "w")
    for i in range(len(t_out)):
            file.write(str(t_out[i]) + " " + str(out[i]*Vconvfact) + "\n")
    file.close()
    
    
    print(t_final)
    
    toc = time.perf_counter()
    print(f"time: {toc - tic:0.4f} seconds")
    #plt.plot(np.array(t_out)*time_scale, np.array(out)*Vconvfact, label=Istim)
    #plt.show()
    # plt.figure()
    # plt.plot(np.array(t_out)*time_scale, np.array(out)*Vconvfact, label='Voltage')
    
    f.close()

    return np.array(t_out), np.array(out)*Vconvfact, cor


            
     
    
    









