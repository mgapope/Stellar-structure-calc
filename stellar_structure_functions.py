#--Packages--
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import pandas as pd

#--Constants-- ALL IN CGS
M_sun = 1.989e33 #Mass of sun
R_sun = 6.96e10 #Radius of sun cgs
L_sun = 3.826e33 #Luminosity of sun cgs

a = 7.5646e-15 #Blackbody energy constant
Na = 6.022e23 #Avagadro's number
kB = 1.38e-16 #Boltzmann constant
m_H = 1.6726231e-24 #Mass of hydrogen 
G = 6.67259e-8 #Graviatational constant 
c = 2.99792458e10 #Speed of light 
sigma_sb = 5.67051e-5 #stefan boltzmann constant

#--Functions--
def density(P,T,X):
    #Give P and T in standard non-log units
    #Variables
    mmu = 4/(3+5*X)
    rho = (P * mmu * m_H) / (kB * T)
    gas_press = P - (1/3) * a * T**4 * (Na*kB/mmu)**-1
    #if np.isnan(rho):
    #    print(rho,P,T)
    #elif rho < 0:
    #    print(rho,P,T)
    return rho, gas_press #Returns log density and gas pressure

def opacity_interp(table, T, rho):
    #Give T and rho in standard units
    
    # Extract y values (temperatures)
    y_vals = table.iloc[:, 0].values  # First column is y-values
    #print(y_vals)

    # Extract x values (R)
    x_vals = table.columns[1:].astype(float) 
    #print(x_vals)

    # Extract z values (everything past the 0th column)
    z_vals = table.iloc[:, 1:].values  # Convert to numpy array
    #print(z_vals)

    # Create linear interpolation object
    interpolator = RegularGridInterpolator((y_vals, x_vals), z_vals, method='linear', bounds_error=False, fill_value=None)

    log_T = np.log10(T)
    
    # Convert input density and temp to R
    T6 = T*1e-6
    R = rho/T6**3
    log_R = np.log10(R)

    log_kappa = interpolator((log_T, log_R))
    
    return 10**log_kappa #Returns unlogged opacity

def energy_gen(P,T,composition):
    #Given pressure temperature in standard cgs units
    T9 = T*1e-9
    T7 = T*1e-7
    X = composition[0]
    Y = composition[1]
    Z = 1 - X - Y
    
    
    rho = density(P,T,composition[0])[0]
    if rho < 0:
        rho = 0 
        
    f11 = np.exp(5.92e-3 * (rho * T7**-3)**(1/2))
    g11 = 1 + 3.82*T9 + 1.51*T9**2 + 0.144*T9**3 - 0.0114*T9**4
    g14 = 1 - 2*T9 + 3.41*T9**2 - 2.43*T9**3
    psi = 1.5

    #if f11.any() > 10**20:
    #    print(f11,rho,T7,P)
    pp_rate = 2.57e4 * g11 * psi * f11 * X**2 * rho * T9**-(2/3) * np.exp(-3.381*T9**-(1/3))

    cno_rate = 8.24e25 * g14 * X * Z * rho * T9**-(2/3) * np.exp(-15.231*T9**-(1/3) - (T9/0.8)**2)

#    print(f'The pp-chain energy generation rate for a region with log(T/K) = {Temp}, log(p/gcm⁻³) = {pres}, and composition X,Y ={X,Y} is {pp_rate} erg g⁻¹ s⁻¹')
#    print(f'The CNO-cycle energy generation rate for a region with log(T/K) = {Temp}, log(p/gcm⁻³) = {pres}, and composition X,Y ={X,Y} is {cno_rate} erg g⁻¹ s⁻¹')

    return [pp_rate, cno_rate]

#--Calculation--
def help_guess(Mass,composition):
    #A helper function to make a guess if you don't know the values, uses homology relations
    #All these are from chapter 2 of the textbook
    R = Mass**0.5 * R_sun
    L = Mass**3.5 * L_sun
    #Uses constant density model
    mmu = 4/(3+5*composition[0])
    Pc = 110.381 * 1.34e15 * (Mass)**2 * (R/R_sun)**-4 #Core pressure from constant density, scaled to match values of sun
    Tc = 2.0376 * 1.15e7 * mmu * Mass * (R/R_sun)**-1 #Core temperature from constant density, scaled to match values of sun
    
    return [Pc,R,L,Tc]
    
def load1(guess,Mass,composition):
    opacity = 'X_' + str(composition[0]) + '_Y_' + str(composition[1]) + '.txt'
    opacity_table = pd.read_csv(opacity, delimiter=",",na_values=[""])

    Pc, R, L, Tc = guess
    
    m_small = 1e-5 * Mass * M_sun
    
    rho_c = density(Pc,Tc,composition[0])[0] #Density
    
    epsilon_c = np.sum(energy_gen(Pc,Tc,composition)) #Sum of all energy generation
    
    kappa = opacity_interp(opacity_table,Tc,rho_c) #Find from interpolated opacity table (OPAL)

    r = (3 * m_small / (4 * np.pi * rho_c))**(1/3)
    
    P = Pc - (2 * np.pi / 3) * G * rho_c**2 * r**2
        
    l = epsilon_c * m_small #Luminosity of small volume
    
    rad_gradient =  (3/(16 * np.pi * a * c)) * (Pc * kappa / Tc**4) * (l/(G * m_small))
    
    adiabat_gradient = 0.4
    
    if rad_gradient > adiabat_gradient: #Checks for convection by Schwarzchild criteria
        #print('This region is radiative, using the radiative temperature gradient')
        T = Tc - (epsilon_c**2 * rho_c**2 * kappa * r**2) / (8 * a * c * (Tc)**3) #Without convection
    else:
        #print('This region is convective, using the adiabatic temperature gradient')
        T = Tc - (2 * np.pi / 3) * (G * adiabat_gradient * rho_c**2 * Tc * r**2) / (Pc) #With convection
    
    y0 = [P,r,l,T]
    
    return y0

def load2(guess,Mass,composition):
    #Using the tau=2/3 surface, where temperature equals blackbody temperature
    opacity = 'X_' + str(composition[0]) + '_Y_' + str(composition[1]) + '.txt'
    opacity_table = pd.read_csv(opacity, delimiter=",",na_values=[""])

    Pc, R, L, Tc = guess
    
    mmu = 4/(3+5*composition[0])
    
    l = L #Total luminosity
    
    r = R #Total radius
    
    T = (L / (4 * np.pi * R**2 * sigma_sb))**(1/4) #Effective temperature blackbody
    
    kappa = 0.6
    
    P = (2/3) * G * (Mass * M_sun) / R**2 * (1/kappa)
    #P = 0
    
    yf = [P,r,l,T]
    
    return yf

def derivs(mass,y,composition):
    if np.isnan(mass):
        print("WARNING: NaN mass value passed to derivs")
        #return [0, 0, 0, 0]
    if np.isinf(mass):
        print("WARNING: inf mass value passed to derivs")
    X = composition[0]
    Y = composition[1]
    composition = [composition[0],composition[1]]
    opacity = 'X_' + str(X) + '_Y_' + str(Y) + '.txt'
    opacity_table = pd.read_csv(opacity, delimiter=",",na_values=[""])

    P, r, l, T = y #Putting all the variables in one list ALL IN CGS UNITS
    
    m = mass  #So we can work in cgs
    #print(m)
    
    rho = density(P,T,X)[0]
    
    kappa = opacity_interp(opacity_table,T,rho) #Find from interpolated opacity table (OPAL)
    
    rad_gradient =  (3/(16 * np.pi * a * c)) * (P * kappa / T**4) * (l/(G * m))
    
    adiabat_gradient = 0.4
    
    #COUPLED DIF EQS
    dP_dm = (- G * m) / (4 * np.pi * r**4) #Hydrostatic equilibrium
    
    dr_dm = 1 / (4 * np.pi * r**2 * rho) #Mass continuity
    
    dl_dm = np.sum(energy_gen(P,T,composition)) #Energy generation #Takes the sum of pp and CNO energy generation

    if rad_gradient > adiabat_gradient: #Checks for convection by Schwarzchild criteria
        #print(f'This region {-np.log10(1 - mass/M_sun)} is convective, using the adiabatic temperature gradient')
        delt = adiabat_gradient
    else:
        #print(f'This region {-np.log10(1 - mass/M_sun)} is radiative, using the radiative temperature gradient {rad_gradient}')
        delt = rad_gradient 
        
    dT_dm = delt * dP_dm * (T/P) #Temperature gradient

    #print(f'The pressure gradient at mass {m} is {dP_dm}')
    
    return [dP_dm, dr_dm, dl_dm, dT_dm]

##--Solving integrator--     
def shootf(guess, Mass, composition, steps=1000, M_meeting_point = 0.5, M_start=1e-5,direction='inwards'):
    
    P, R, L, T = guess
    M = Mass * M_sun
    
    #steps = 1000 if not dense else 100000
    m_to_surface = np.linspace(M_start, M_meeting_point, steps) * M #Array of masses from center to desired midpoint, use load1
    m_to_center = np.linspace(1, M_meeting_point, steps) * M #Array of masses from surface to desired midpoint, use load2
    #print(m_to_surface)
    #print(m_to_center)
    if direction == 'outwards':
        #print("integrating outwards from core")
        sol_to_surface = solve_ivp(derivs, (m_to_surface[0], m_to_surface[-1]), load1(guess,Mass,composition), args=(composition,), method='RK45', t_eval=m_to_surface)
        return {'Solution':sol_to_surface,'Masses': m_to_surface} 
    elif direction == 'inwards':
        #print("integrating inwards from surface")
        #print(load2(guess,Mass,composition))
        sol_to_center = solve_ivp(derivs, (m_to_center[0], m_to_center[-1]), load2(guess,Mass,composition), args=(composition,), method='RK45', t_eval=m_to_center,rtol=1e-2)
        return {'Solution':sol_to_center,'Masses': m_to_center}

def resids(params_scale, guess, Mass, composition):
    parameters = guess * params_scale
    sol_to_center = shootf(parameters, Mass, composition,direction='inwards')
    sol_to_surface = shootf(parameters, Mass, composition,direction='outwards')
    scale = (sol_to_center['Solution'].y[:, 0] - sol_to_surface['Solution'].y[:, 0])
    diff_frac = (sol_to_center['Solution'].y[:, -1] - sol_to_surface['Solution'].y[:, -1]) / scale
    chi2 = np.sum(diff_frac**2)
    return np.sum(chi2)

def minimizer(resids):
    bounds = np.array([[0.5]*4, [1.5]*4]).T
    params_scale = [1,1,1,1]
    fit = minimize(resids, x0=param_scale, args=(guess,Mass,composition), bounds=bounds, method='L-BFGS-B')

def get_difference(MESA, inwards):
    return np.abs(inwards-MESA)/(MESA)

def convective_test(rad,ad):
    return rad > ad
        
def save_table(masses,data,composition):
    opacity = 'X_' + str(composition[0]) + '_Y_' + str(composition[1]) + '.txt'
    opacity_table = pd.read_csv(opacity, delimiter=",",na_values=[""])
    bigT = pd.DataFrame()
    bigT['Mass'] = masses
    bigT['Pressure'] = data[0]
    bigT['Radius'] = data[1]
    bigT['Luminosity'] = data[2]
    bigT['Temperature'] = data[3]
    bigT['Density'] = density(data[0],data[3],composition[0])[0]
    vec_energy = np.vectorize(lambda p, t: sum(energy_gen(p, t, composition)))
    bigT['Energy'] = vec_energy(data[0], data[3]) #energy_gen(data[0],data[3],composition)[0] + energy_gen(data[0],data[3],composition)[1]
    bigT['Opacity'] = opacity_interp(opacity_table, data[3], bigT['Density'])
    bigT['rad_gradient'] = (3/(16 * np.pi * a * c)) * (bigT['Pressure'] * bigT['Opacity'] / bigT['Temperature']**4) * (bigT['Luminosity']/(G * bigT['Mass']))
    bigT['adiabat_gradient'] = 0.4 * np.ones(len(bigT['Mass']))
    bigT['convection?'] = convective_test(bigT['rad_gradient'],bigT['adiabat_gradient'])
    return bigT