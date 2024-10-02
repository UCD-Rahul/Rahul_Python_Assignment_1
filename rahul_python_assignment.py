import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import re
import argparse

#Reference code:

#https://stackoverflow.com/questions/1706198/python-how-to-ignore-comment-lines-when-reading-in-a-file    [1]

#https://stackoverflow.com/questions/27805919/how-to-only-read-lines-in-a-text-file-after-a-certain-string    [2]

#https://stackoverflow.com/questions/3939361/remove-specific-characters-from-a-string-in-python   (Replacing characters in string)   [3]

#https://www.youtube.com/watch?v=peBOquJ3fDo (Used video to get some help with getting errors from covariance matrix) [4]


def file_info_reading(filename):
    """
    opens the file and puts the detector specifications into a dictionary
    """
    specs = {}   # empty dictionary to store the information
    with open(filename,'r') as file:    
        for line in file:               # iterates through each line in the file
            if line.startswith('#'):    # checks if the line starts with a comment     Inspired by code from   [1]
                line = re.sub('[#]','', line)    # removes the comment                                                    [3]
                line = re.sub('\n','', line)     # removes the '\n' that kept getting printed with each line              [3]
                line.strip()                     # removes any leading or trailing whitespaces from the string
                next_line = next(file)           #gets the line underneath the commented line
                next_line = re.sub('\n','', next_line)    # removes the '\n' from the new line
                specs[line] = next_line                 # makes the commented line the key and the line underneath it the value in the dictionary   

    return specs


def file_data_reading(filename):
    """
    opens the file and checks for the 'WAVELENGTH, FLUX' line and appends the data underneath it to a list
    """
    data = []   # empty list for storing the data
    with open(filename,'r') as file:
        for line in file:    # iterates through each line in the file
            if line.startswith('WAVELENGTH,FLUX'):    # checks for this specific string in the file
                for line in file:        # iterates through each line underneath that string              [2]
                    wavelength,flux = line.strip().split(',')   # strips the lines of whitespaces and splits it into 'wavelength' and 'flux'
                    data.append((float(wavelength), float(flux)))  # adds each wavelength and flux value into the data list as floats
    return data


def file_data_splitting(data):
    """
    takes the data from the file and makes arrays for the wavelength and flux
    """
    wavelength = []
    flux = []
    for i in data:    # iterates through each entry in the input data list
        wavelength.append(i[0])         # appends each wavelength entry to the empty wavelength list
        flux.append(i[1])               # appends each flux entry to the empty flux list
    wavelength = np.array(wavelength)   # turns the wavelength list into a numpy array
    flux = np.array(flux)               # turns the flux list into a numpy array

    return wavelength, flux             # returns the wavelength and flux numpy arrays


def gaussian_function(x,A,mu,s):
    """
    function that returns a gaussian equation to be used to fit the emission line
    """
    #gaussian function from assignment document. A is the amplitude, mu is the centroid wavelength(mean), s is the standard deviation
    gauss_func = A * (1/(s * np.sqrt(2*np.pi))) * np.exp((-1 * (x - mu ) ** 2)/(2 * s ** 2 ))
    return gauss_func


def quadratic_function(x, a, b, c):
    """
    quadratic function to be used to fit the continuum
    """
    # a affects the reflection and stretching, h is the horizontal translation, k is the vertical translation
    #quad_func = -a*((x-h)**2)+k
    quad_func = a*x**2+(b*x)+c
    return quad_func

def combined_function(x,a,b,c,A,mu,s):
    """
    function that combines the previous gaussian and quadratic equations. Used to fit the entire spectrum.
    """
    combined_func = a*x**2+(b*x)+c + (A * (1/(s * np.sqrt(2*np.pi))) * np.exp((-1 * (x - mu ) ** 2)/(2 * s ** 2 )))
    return combined_func



initial_quad = [0.5,1,1]     # initial guess parameters for the quadratic fit
initial_combined = [0.5,1,1,50,6685,2] # initial guess parameters for the combined fit


def parameter_function(x,y,p0_1,p0_2):
    """
    takes the wavelength and flux values and the intial guess parameters and uses them to return the optimised parameters for 
    the continuum and the whole spectrum.
    """

    # optimal parameters and covariance matrix for the continuum fit
    quad_popt, quad_pcov = curve_fit(quadratic_function,x,y,p0 = p0_1)   
    # optimal parameters for the full spectrum fit
    combine_popt, combine_pcov = curve_fit(combined_function,x,y,p0 = p0_2,maxfev = 5000)
    
    return  quad_popt, quad_pcov, combine_popt, combine_pcov


def fit_calculator(p1,p2,x):
    """
    function that calculates each fit using the optimal parameters from curve_fit
    """
    continuum = quadratic_function(x,*p1)  # fit of the continuum using the optimal parameters
    combined_fit = combined_function(x,*p2) # fit of the whole spectrum using the optimal parameters
            
    return continuum, combined_fit


def plot_data(wav,flx):
    """
    function to plot the data. Takes the wavelength and flux as inputs. 
    """
    plt.figure(figsize=(9,5))
    plt.scatter(wav,flx,color = 'springgreen', alpha = 0.8,label = "Data")
    plt.xlabel("Wavelength: (Å)")
    plt.ylabel('Flux: ADU')
    plt.title('Spectrum')
    plt.grid(alpha  = 0.3)
    plt.legend()


def plot_continuum(wav,flx,contin):
    """
    function to plot the data and the continuum fit overtop. Takes the wavelength, flux and continuum fit values as inputs.
    """
    plt.figure(figsize=(9,5))
    plt.scatter(wav,flx,color = 'springgreen', alpha = 0.8,label = "Data")
    plt.plot(wav,contin,color = 'blue',label = "Continuum Fit")
    plt.xlabel("Wavelength: (Å)")
    plt.ylabel('Flux: ADU')
    plt.title('Spectrum with Continuum Fit Overplotted')
    plt.grid(alpha  = 0.3)
    plt.legend()


def plot_combined(wav,flx,combined):
    """
    function to plot the data and the combined fit overtop. Takes the wavelength, flux and combined fit values as inputs. 
    """
    plt.figure(figsize=(9,5))
    plt.scatter(wav,flx,color = 'springgreen', alpha = 0.8,label = "Data")
    plt.plot(wav,combined,color = 'red',label = "Combined Continuum and Emission Peak Fit")
    plt.xlabel("Wavelength: (Å)")
    plt.ylabel('Flux: ADU')
    plt.title('Spectrum with Continuum+Emission Fit Overplotted ')
    plt.grid(alpha  = 0.3)
    plt.legend()
    plt.show()


def parameter_print(par1,cov1):
    """
    Function that prints the optimal parameters of the gaussian part of the combined fit with uncertainty values. Also calculates and prints
    the FWHM value. Takes the optimal parameters and covariance matrix of the combined fit as inputs.
    """
    error_emiss = np.sqrt(np.diag(cov1))    # gets the parameter uncertainties from the square root of the diagonal of the covariance matrix [4]

    Amplitude = par1[3]*(1/(par1[5] * np.sqrt(2*np.pi)))  # Gets the actual amplitude of the fit by dividing by the first part of the gaussian equation.
    Amplitude = round(Amplitude,2)  # Rounds the amplitude to 2 decimal places
    Amplitude_err = round(error_emiss[3],2) # Uncertainty of the amplitude

    Centroid_Wavelength = round(par1[4],2)  # Centroid Wavelength 
    Centroid_Wavelength_err = round(error_emiss[4],2)  # Centroid Wavelength uncertainty

    Standard_Deviation = round(par1[5],2)  # Standard Deviation
    Standard_Deviation_err = round(error_emiss[5],2) # Standard Deviation Uncertainty
    
    #prints each parameter with its associated uncertainty value, rounded to 2 decimal places)
    print("Gaussian Emission Fit Parameters:")
    print("Amplitude = ",Amplitude,u" \u00B1 ", Amplitude_err, " ADU")       
    print("Centroid Wavelength = ",Centroid_Wavelength,u" \u00B1 ", Centroid_Wavelength_err, " Å" ) 
    print("Standard Deviation = ", Standard_Deviation,u" \u00B1 ", Standard_Deviation_err, " Å")   
    
    FWHM = 2 * np.sqrt(2 * np.log(2)) * par1[5]   #Calculates the FWHM using the standard deviation and the equation from the assignment pdf
    error_FWHM = 2 * np.sqrt(2 * np.log(2)) * error_emiss[5]   #Calculates the uncertainty of the FWHM
    print("FWHM = ",round(FWHM,2),u" \u00B1 ", round(error_FWHM,2), " Å")   # prints the FWHM with its uncertainty


def main(file):
    """
    main function that takes the file as an input and outputs the required plots and information
    """
    file_info = file_info_reading(file)   # Reads the detector information from the file
    print("Detector Information: ")   # prints this string
    print(" ")  # prints a space between the string and the detector information dictionary
    print(file_info)    # prints the detector information dictionary
    print("  ")  # prints a space between the printed detector information and printed parameter information
    
    split_file_data = file_data_splitting(file_data_reading(file))    # separates the wavelength and flux values in the file
    wave = split_file_data[0]  #separate array containing only the wavelength values
    flux = split_file_data[1]  #separate array containing only the flux values
    
    parameters = parameter_function(wave,flux,initial_quad,initial_combined)   #calcualtes the optimal parameters for the quadratic and combined functions
    fits = fit_calculator(parameters[0],parameters[2],wave)         # uses the optimal parameters to calculate the continuum and combined fits

    parameter_print(parameters[2],parameters[3])    # prints the combined fit parameters with uncertainties along with the FWHM

    plot_data(wave,flux)   # plots the data on its own

    plot_continuum(wave,flux,fits[0])    # plots the data with the continuum fit overtop

    plot_combined(wave,flux,fits[1])   # plots the data with the combined continuum and gaussian fits overtop

    


parser = argparse.ArgumentParser()

parser.add_argument("filename", help = "file one")

args = parser.parse_args()

file = args.filename   # plain filename


main(file)