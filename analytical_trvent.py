# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 12:27:40 2015

@author: kdarnell
"""
import numpy as np
import matplotlib.pyplot as plt
import Darnelletal as D
import os


#-------------------------Main Body--------------------------------------------
#Standard values to be used in calculations
wd = 550.0 #water depth in m
SMTZ = 20 #depth of sulfate-methane transition zone in m
T_sf = 3.0 #seafloor temperature in C
T_grad = 40.0 #temperature gradient in C/km
del_T = 2.7 #increase in seafloor temperature in C
Sh = 0.05 #hydrate pore volume percent at start (this will exist between B_i and SMTZ)
seawater = 0.035 # salinity in kg/kg

#Call both methods for easy comparison, then pick one for entire calculation
wrkng_dir = os.getcwd() + '/' #or some other directory
Liu_mthd = D.Liu(wrkng_dir,T_unit='C',S_unit='kg/kg')
Tish_mthd = D.Tishchenko(T_unit='C',S_unit='kg/kg')
Eq_method = Liu_mthd    

#Calculate lambda the relevant parameter found in the body of the manuscript (EQ-4)
lam = D.find_lambda(T_sf,T_grad,SMTZ,wd,del_T,Sh,seawater,Eq_method).flatten()

#Do you want to look at some plots? (True/False)
make_plots = True
do_sensitivity = False 

#--------------------Print some output-----------------------------------------
print('Lambda = ',lam)
if lam>1 and lam != 999.:
    print('Transient venting occurs!!')
elif lam<1 and lam!=0.:
    print('No Venting occurs!!')
elif lam==0.:
    print('Hydrate not stable to start!!')
else:
    print('Complete Venting occurs!!')
    
#------------------------End of simple output----------------------------------


if make_plots or do_sensitivity:
    plt.style.use('ggplot')

    # Sample calculation
    eta = 1.0
    z = D.z
    Sh_abv = np.zeros(np.shape(z))
    Sh_abv[z>=SMTZ] = Sh
    T_profile = T_sf + (T_grad/1e3)*z
    T_warm = T_profile + del_T
    Hyd_stat = D.Hydstat_pressure(wd,z)
    sal_init = np.array([Eq_method.S_eq(T_profile[i],Hyd_stat[i]) for i in range(len(z))]).flatten()
    sal_warm = np.array([Eq_method.S_eq(T_warm[i],Hyd_stat[i]) for i in range(len(z))]).flatten()
    if any(sal_init>seawater):
        B_i = max(z[sal_init>seawater])
        if any(z[sal_warm>seawater]):
            B_f = max(z[sal_warm>=seawater])
            if B_f<=B_i:
                beta = np.trapz(Sh_abv[(z<B_i)&(z>=B_f)],x=z[(z<B_i)&(z>=B_f)])
                gamma = np.trapz(1.0 - (seawater/sal_warm[z<B_f])*(1.0-Sh_abv[z<B_f]) - \
                                Sh_abv[z<B_f],x=z[z<B_f])
                lam = eta*beta/gamma
            else:
                lam = 999.
        else:
            lam = 999.
    else:
        lam = 0.0
    print(lam)
    


#------------------------Make optional plots-----------------------------------
if make_plots:
    #------------------------Compare Equilibrium methods-----------------------
    Temps =  np.arange(0,30.1,0.1,dtype='double')#Temperatures in Celsius
    salinity = np.array([0, 0.035,0.070,0.14],dtype='double')#Salinities in 'kg/kg'      
    P_dis1 = Liu_mthd.P_dis(salinity,Temps)
    P_dis2 = Tish_mthd.P_dis(salinity,Temps)
    
    
    #-------------------------Plots--------------------------------------------
#    with plt.style.context('fivethirtyeight'):
    
    # Figure 1
    plt.figure()
    for i in range(len(salinity)):
        plt.plot(Temps,P_dis1[:,i], linewidth=2, label ='S =' + str(salinity[i]))
        plt.plot(Temps,P_dis2[:,i], '--', linewidth=2, label ='S =' + str(salinity[i]))
    plt.ylim([0, 40])
    plt.xlim([0, 25])
    plt.legend(loc='upper left')
    plt.xlabel('Temperature, (C)')
    plt.ylabel('Pressure, (MPa)')
    plt.title('Hydrate Phase Diagram \n (Liu is dashed, Tishchenko is solid)')
    #    plt.grid()
    plt.show()
    
    # Figure 2
    plt.figure()
    plt.plot(T_profile,z,label = 'initial profile')
    plt.plot(T_warm,z,label = 'warmed profile')
    plt.plot(T_profile,Eq_method.P_dis([0.035],T_profile)*1e6/D.rho_w/D.g - wd,'--',label='stability profile')
    plt.ylim([min(z), 1.5*B_i])
    plt.xlim([0, 1.2*max(T_profile[sal_init>=seawater])])
    plt.legend(loc='lower left')
    plt.xlabel('Temperature, (C)')
    plt.ylabel('Depth, (mbsf)')
    plt.title('Example profile at water depth = ' + str(wd) + 'mbsl \n with ' + \
    str(Eq_method.name) + ' method')
    plt.gca().invert_yaxis()
    #    plt.grid()
    plt.show()
    
    # Figure 3
    plt.figure()
    plt.plot(100*sal_init,z,label='initial')
    plt.plot(100*sal_warm,z,label='warmed')
    plt.plot(100*np.array([0.035,0.035]),np.array([z.min(), z.max()]),label='seawater for ref.',color='black',linewidth=2)
    plt.ylim([min(z), 1.5*B_i])
    plt.xlabel('Equilibrium salinity, (wt. %)')
    plt.ylabel('Depth, (mbsf)')
    plt.legend(loc='lower right')
    plt.gca().invert_yaxis()
    #    plt.grid()
    plt.show()

if do_sensitivity:
    pt_chng = 0.9
    sens_pts = 100
    
    Temp_test = np.linspace(T_sf*(1 - pt_chng),T_sf*(1 + pt_chng),sens_pts)
    Sens1 = np.array([D.find_lambda(Ts,T_grad,SMTZ,wd,del_T,Sh,seawater,Eq_method,eta=0.4) for Ts in Temp_test]).flatten()
    
    delTemp_test = np.linspace(del_T*(1 - pt_chng),del_T*(1 + pt_chng),sens_pts)
    Sens2 = np.array([D.find_lambda(T_sf,T_grad,SMTZ,wd,delTs,Sh,seawater,Eq_method,eta=0.4) for delTs in delTemp_test]).flatten()
    
    wd_test = np.linspace(wd*(1 - pt_chng),wd*(1 + pt_chng),sens_pts)
    Sens3 = np.array([D.find_lambda(T_sf,T_grad,SMTZ,wdepths,del_T,Sh,seawater,Eq_method,eta=0.4) for wdepths in wd_test]).flatten()
    
    grad_test = np.linspace(T_grad*(1 - pt_chng),T_grad*(1 + pt_chng),sens_pts)
    Sens4 = np.array([D.find_lambda(T_sf,grads,SMTZ,wd,del_T,Sh,seawater,Eq_method,eta=0.4) for grads in grad_test]).flatten()
    
    SM_test = np.linspace(SMTZ*(1 - pt_chng),SMTZ*(1 + pt_chng),sens_pts)
    Sens5 = np.array([D.find_lambda(T_sf,T_grad,SMTZ_s,wd,del_T,Sh,seawater,Eq_method,eta=0.4) for SMTZ_s in SM_test]).flatten()
    
    Sh_test = np.linspace(Sh*(1 - pt_chng),Sh*(1 + pt_chng),sens_pts)
    Sens6 = np.array([D.find_lambda(T_sf,T_grad,SMTZ,wd,del_T,Sh_s,seawater,Eq_method,eta=0.4) for Sh_s in Sh_test]).flatten()
    
    
    plt.figure()
    plt.plot(100.0*(Temp_test - T_sf)/T_sf,Sens1,label='$T_{sf}$',linewidth=2)
    plt.plot(100.0*(delTemp_test - del_T)/del_T,Sens2,label='$\Delta T$',linewidth=2)
    plt.plot(100.0*(wd_test[(Sens3!=0.0)&(Sens3!=999.)] - wd)/wd,Sens3[(Sens3!=0.0)&(Sens3!=999.)],label='water depth',linewidth=2)
    plt.plot(100.0*(grad_test - T_grad)/T_grad,Sens4,label='$\\nabla T$',linewidth=2)
    plt.plot(100.0*(SM_test - SMTZ)/SMTZ,Sens5,label='SMTZ',linewidth=2)
    plt.plot(100.0*(Sh_test - Sh)/Sh,Sens6,label='$S_h$',linewidth=2)
    plt.plot([-100,100],[1,1],'k',linewidth=4)
    

    plt.title('Sensitivity plot \n with ' + Eq_method.name + ' method')
    plt.ylim([0, 2.0*lam])
    plt.legend(bbox_to_anchor=(0., -0.4, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0.)
    plt.xlabel('Percent change in parameter')
    plt.ylabel(' $\Lambda$')
    plt.savefig('/Users/kdarnell/Desktop/correctedsens',format='eps')




