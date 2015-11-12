# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 14:35:30 2015

@author: kdarnell
"""
import numpy as np
import scipy as sp
from scipy import interpolate


#This code is the implementation of calculations for a GRL manuscript in review.
#The calculations rely on an equilibrium pressure, which is a funciton of temperature and salinity.
#We provide two types of equilibirum pressures (computed in slightly different ways).
#The method of Tischenko uses a semi-analytical equation (with empirical constants)
#The method of Liu solves Gibbs Energy Minimization at a Presssure, Temperature, Salinity (P,T,S)
#Liu relys on reading in .csv files.
#Each method should provide identical output.
#Additional methods calculate the resulting $\Lambda$ value.


#Constants:
rho_w = 1030 #seawater density in kg/m^3
z = np.arange(0,300,5) #vertical grid in m (in very deep water, change the z.max() to ensure you capture B_i)
g = 9.81 #gravity in m^2/s
rho_h = 912 #hydrate density in kg/m^3
theta_g = 0.14 #methane mass fraction in hydrate
rho_g = 50 #methane gas density (approx) in kg/m^3
c_g = 0.0014 #methane solubility in water kg/kg
s_gr_ir = 0.02 #irreducible water saturation m^3/m^3

#Two separate classes for handling different approaches to solving equilibrium equations
class Tishchenko:
    # Methane Hydrate Dissociation Pressure from Tishchenko et. al., 2005
    #--------------------
    def __init__(self,T_unit='K',S_unit='ppt'):
        self.name = 'Tishchenko'
        self.T_unit = T_unit
        self.S_unit = S_unit
        #Simple dictionary for converting units of temperature and salinity
        #Pressure will always be in MPa
        self.T_add = {'K': 0.0, 'C': 273.15, 'F': 459.67}#Equation takes K
        self.T_multiply = {'K': 1.0, 'C': 1.0, 'F': 5.0/9.0}#Equation takes K
        self.S_multiply = {'kg/kg': 1000.0, 'ppt': 1.0, 'wt. %': 10.0}#Equation takes ppt
       
    def c_factors(self,T):
        T = T*self.T_multiply[self.T_unit] + self.T_add[self.T_unit]
        c0 = -1.6444866e3 - 0.1374178*T + (5.4979866e4)/T + (2.64118188e2)*np.log(T)
        c1 = (1.1178266e4 + 7.67420344*T - (4.515213e-3)*(T**2) - (2.04872879e5)/T - 
            (2.17246046e3)*np.log(T))
        c2 = (1.70484431e2 + 0.118594073*T - (7.0581304e-5)*(T**2) - (3.09796169e3)/T - 
            33.2031996*np.log(T))
        return c0,c1,c2
    
    def P_func(self,S,T):
        c0,c1,c2 = self.c_factors(T)
        S = S*self.S_multiply[self.S_unit]
        P = np.exp(c0 + c1*S + c2*(S)**2)
        return P
        
    def P_dis(self,sals,Temps):
        P = np.array([ [self.P_func(S,T) for S in sals] for T in Temps])
        return P 
        
    # Solving Tishchenko Pressure equation to find three-phase equilibrium salinity 
    # at given temperature and pressure
    def S_eq(self,T,P):
        c0,c1,c2 = self.c_factors(T)
        coeff = [c2, c1, c0-np.log(P)]
        S = np.real(sp.roots(coeff))
        S_temp = S[S>=0.0]
        S_out = S_temp[S_temp<200]
        if len(S_out)==0:
            S_out = 0.0
        return S_out/self.S_multiply[self.S_unit]

class Liu:
    # Methane Hydrate Dissociation Pressure originally used in Liu and Flemings, 2006
    # This is dissociation pressures used in the simulation of the paper corresponding to this code
    #--------------------
    # Download two csv files, called "Liu_method_Smat.csv" and "Liu_method_Pmat.csv"
    # Place both files into the same folder and pass folder name to instance of Liu!     
    def __init__(self,folder,T_unit='C',S_unit = 'kg/kg'):
        self.name = 'Liu'
        self.folder = folder
        self.T_unit = T_unit
        self.S_unit = S_unit
        #Simple dictionary for converting units of temperature and salinity
        #Pressure will always be in MPa
        self.S_multiply = {'kg/kg': 1.0, 'ppt': 0.001, 'wt. %': 0.01}#Liu calculation is in kg/kg
        self.T_add = {'K': -273.15, 'C': 0.0, 'F': 32}#Liu calculation is in C
        self.T_multiply = {'K': 1.0, 'C': 1.0, 'F': 5.0/9.0}#Liu calculation is in C
        #Load calculated dissociation pressure from file
        P_file = self.folder + 'Liu_method_Pmat.csv'
        A = np.loadtxt(P_file,delimiter=',')
        Temp = A[1:,0]#Celsius
        Sal = A[0,1:]#kg/kg
        Peq = A[1:,1:]#MPa
        self.P_func = interpolate.interp2d(Sal,Temp,Peq,kind='cubic')
        #Load calculated dissociation salinity from file
        S_file = self.folder + 'Liu_method_Smat.csv'
        B = np.loadtxt(S_file,delimiter=',')
        Pres = B[1:,0]#Celsius
        Temp2 = B[0,1:]#MPa
        Seq = B[1:,1:]*self.S_multiply[self.S_unit]#kg/kg
        Seq[Seq<1e-3]=0
        self.S_func = interpolate.interp2d(Temp2,Pres,Seq,kind='cubic')
        
    def P_dis(self,S,T):
        P_out = self.P_func(S,T)
        return P_out
        
    def S_eq(self,T,P):
        S_out = self.S_func(T,P)
        return S_out

#Simple function for hydrostatic pressure
def Hydstat_pressure(wd,z):
    Hyd_stat = (rho_w*g*(wd + z))/1e6
    return Hyd_stat
    
#Parameterized calculation of the sample for calculating $\Lambda$
# "eta" is an efficienty factor that is set to 1.0, but can be modified.
def find_lambda(T_sf,T_grad,SMTZ,wd,del_T,Sh,seawater,Eq_method,eta=1.0):
    
    # Hydrate saturation (Sh_abv)
    #"Sh_abv" is modified to be zero where z<SMTZ 
    Sh_abv = np.zeros(np.shape(z))
    Sh_abv[z>=SMTZ] = Sh
    
    # Temperature
    # "T_profile" is a linear function with slope "T_grad" and intercept of "T_sf"
    T_profile = T_sf + (T_grad/1e3)*z
    #Apply increase to temperature
    T_warm = T_profile + del_T
    
    #Pressure
    Hyd_stat = Hydstat_pressure(wd,z)
    
    #Salinity calculated from equilibrium type
    sal_init = np.array([Eq_method.S_eq(T_profile[i],Hyd_stat[i]) for i in range(len(z))]).flatten()
    sal_warm = np.array([Eq_method.S_eq(T_warm[i],Hyd_stat[i]) for i in range(len(z))]).flatten()
    
    #Determine $\Lambda$
    if any(sal_init>seawater):
        B_i = max(z[sal_init>seawater]) #Original base
        if any(z[sal_warm>seawater]):
            B_f = max(z[sal_warm>=seawater]) #Warmed base
            if B_f<=B_i:
                #Main calculation!!
            
                #Integrate "Sh_abv" from "B_i" to "B_f"
                beta = np.trapz(Sh_abv[(z<B_i)&(z>=B_f)],x=z[(z<B_i)&(z>=B_f)])
                
                #Integrate "Sh_eq", which comes fom "sal_warm" from "0" to "B_f"
                gamma = np.trapz(1.0 - (seawater/(sal_warm[z<B_f] + 1.0e-6))*(1.0-Sh_abv[z<B_f]) - \
                                Sh_abv[z<B_f],x=z[z<B_f])
                                
                #Find ratio                
                lam = eta*beta/gamma
            else:
                #Provide exit flag for a cooling (should be a warming)
                lam = np.array([999.])
        else:
            #Provide exit flag for complete venting
            lam = np.array([999.])
    else:
        #Provide exit flag for being outside of hydrate stability before warming
        lam = np.array([0.0])
    return lam
