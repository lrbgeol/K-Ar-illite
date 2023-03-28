#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 21:17:45 2023

@author: lydiabailey
"""
# --- ILLITE AGE ANALYSIS ---

# This code performs Illite-Age-Analysis, and produces a weighted least squares
# fit of a straight line to K-Ar illite dates of sample size separates with 
# error in both coordinates.The code calculates the authigenic (1md) end-member
# illite age. It can handle bivariate regression where the errors in both 
# coordinates are correlated and is capable of performing force-fit regression.
#
# For details on the algorithm used in the code, see _Unified Equations 
# for the slope, intercept, and standard errors of the best straight line_ 
# (<http://is.gd/jVA5fE York et al., 2004>)
#
# This code was written by Lydia Bailey (lydiabailey@fas.harvard.edu) and 
# the regression was adapted from a MATLAB code written by Kaustubh Thirumalai 
# of the University of Arizona. 
# Citation: Thirumalai, K., A. Singh, and R. Ramesh (2011), _A MATLAB code to 
# perform weighted linear regression with (correlated or uncorrelated) errors 
# in bivariate data_, Journal of the Geological Society of India, 
# 77(4), 377380, doi: <http://is.gd/sk1hMu 10.1007/s12594-011-0044-1>.

# --- INPUT ---
# Excel file/array with the following format: 
# Column 1|Column 2 |Column 3 |Column 4  | Column 5  |  Column 6  | Column 7 |
# Size    |Age data |sigAge   |Et-1 data | sig(Et-1) | 2m1% data  | sig2m1%  |
# Size    |Y data   |Y error  |Y2 data   | Y2 error  | X data     | X error  |

# You can input an excel file with multiple sheets, with different samples on
# each sheet. Sheet name should be sample name. The code will perform
# regression & make plots for all sheets/samples.

# --- OUTPUT ---
# Graph of Age/e(lamdat)-1 vs. 2m1 % (with errorbars)
# Comparison of Weighted Linear Regression vs. Simple Linear Regression
# Errors on slope and intercept of the line of best-fit
# Intercept & intercept error = 1md end-member age +- error

# NOTE: Correlation coefficient between errors in X(i) and Y(i) has been
# taken as zero in this program. If there is any correlation between the
# errors, then you can include it directly in the worksheet in an additional
# column 8 and you can de-comment it in the code below. By default it will be
# taken as zero. For force-fit regression, place a sufficiently low error (i.e.
# ~infinite weight) on the coordinates of the point at which the line is 
# to be forced.


# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sigfig 
import math
from scipy.stats import t
from scipy.stats import t as t_dist

# Make default font Arial
plt.rcParams['font.family'] = 'Arial'

# --- INPUT ---
# Load the Excel file into a Pandas dataframe
excel_file = pd.ExcelFile('data.xlsx')
xl = pd.read_excel('data.xlsx',header=0)

tol = 1e-8;    #Tolerance

# Extract column names (top row)
sheet_names = excel_file.sheet_names

# Decay constant (lamda) for radioactive decay K40 -> Ar40
lamda = 5.543e-10;

# Loop through each sheet name and read the data into a Pandas dataframe
for sheet_name in sheet_names:
    df = excel_file.parse(sheet_name)
    
    # Access the data without the first column using .iloc and a slice
    data = df.iloc[:, 1:]
    print(sheet_name)
    print(data)
    
    # Regressions calculated using e(lamda*t)-1 values (Y), not Age (Y2)
    
# --- PERFORM SIMPLE LINEAR REGRESSION (SLR) FOR COMPARISON ---
    X = df['2m1']
    sigX = df['2m1_1s']
    Y = df['et-1']
    sigY = df['et-1_1s']
    
    R = np.corrcoef(X, Y) #correlation coefficient, linear regression
    n = len(X)
    
    ri = 0
    X1 = np.array([[n, np.sum(X)], [np.sum(X), np.sum(X*X)]]) #LSE Value of 'b'
    Y1 = np.array([np.sum(Y), np.sum(X*Y)])  # Polyfit can also be used
    Z1 = np.linalg.solve(X1, Y1)
    a1, b1 = Z1[0], Z1[1]
    sigres = np.sum((Y - a1 - b1*X)**2)/(n-2)  # Sigma Residual
    delta = np.linalg.det(X1)  # Determinant
    varx = np.var(X)
    sigb1sq = (n**2)*(n-1)*varx*sigres/(delta**2)  # Sigma(b) without weights
    sigb1 = np.sqrt(sigb1sq)
    Xbar = np.mean(X)
    siga1sq = (sigres/(varx*n))*(varx + Xbar**2)
    siga1 = np.sqrt(siga1sq)
    A_SLR = [a1, siga1]  # SLR Intercept [e(lamda*t)-1]
    B_SLR = [b1, sigb1]  # SLR Slope
    
    AgeSLR = np.log(a1+1)/lamda/1000000 # SLR Intercept [in Age (Ma)]
    AgeErrorS = np.log(siga1+1)/lamda/1000000 
    
    #SLR p-value
    
    B1 = 0
    t1 = (b1 - B1)/sigb1
    Pval1 = 2*(1 - t.cdf(abs(t1), 1))
    
# --- MAXIMUM LIKELIHOOD METHOD (WEIGHTED LINEAR REGRESSION) ---
    # Implements the _York et al._ [2004] algorithm to perform weighted linear
    # regression using the maximum likelihood method (MLM)

    #---- Weighting Errors ----
    
    wX = np.abs(1/(sigX**2))
    wY = np.abs(1/(sigY**2))
    alpha = np.sqrt(wX*wY)
    #ri = mat[:,8];         % CorrCoef between sig(X) and sig(Y)
    
    b = b1
    d = tol
    i = 0
    
    #---- York et al. (2004) Algorithm ----
    
    while (d > tol or d == tol):         # Tolerance check loop
        i = i+1
        b2 = b
        W = wX*wY/(wX + b**2*wY - 2*b*alpha*ri)
        meanX = np.sum(W*X)/np.sum(W)
        meanY =  np.sum(W*Y)/np.sum(W)
        U = X.values - meanX
        V = Y.values - meanY
        Beta = W*((U/wY)+(b*V/wX) - (b*U + V)*(ri/alpha))
        meanBeta = np.sum(W*Beta)/np.sum(W)
        b = np.sum(W*Beta*V)/np.sum(W*Beta*U)
        dif = b - b2
        d = np.abs(dif)
    
    U2 = U**2
    V2 = V**2
    a = meanY - b*meanX
    x = meanX + Beta
    meanx = np.sum(W*x)/np.sum(W)
    u = x - meanx
    sigbsq = 1/(np.sum(W*(u*u)))
    sigb = np.sqrt(sigbsq)
    sigasq = 1/np.sum(W) + meanx**2*(sigbsq)
    siga = np.sqrt(sigasq)
    S = np.sum(W*((Y - b*X - a))**2)
    WR = sum(U*V)/math.sqrt((sum(U2)*sum(V2))); #correlation coefficient, weighted regression
    
    A_WLR = [a, siga]  # WLR Intercept [e(lamda*t)-1]
    B_WLR = [b, sigb]  # WLR slope [e(lamda*t)-1]
    AgeWLR = np.log(a+1)/lamda/1000000 # WLR Intercept [in Age (Ma)]
    AgeErrorW = np.log(siga+1)/lamda/1000000
    
    DetritalWLR=np.log((b*100+a)+1)/lamda/1000000
    DetErrorWLR=np.log((sigb*100+siga)+1)/lamda/1000000
    
    #---- MLE p-value ----
    B = 0
    t_val=(b-B)/sigb
    Pval=2*(1-t_dist.cdf(np.abs(t_val),len(X)-2))
    
# --- OUTPUT ---
    
    vectorized_round = np.vectorize(sigfig.round)
    
    AgeSimple = np.array([AgeSLR, AgeErrorS]) # [Age, Error] in Ma for simple regression
    AgeSimple = vectorized_round(AgeSimple,2) # Rounds age to 2 sig fig
    print('1md End-Member Age (Simple Regression) (Ma) =', AgeSimple)
    AgeWeighted = np.array([AgeWLR, AgeErrorW]) # [Age, Error] in Ma for weighted regression
    AgeWeighted = vectorized_round(AgeWeighted,2) # Rounds age to 2 sig fig
    print('1md End-Member Age (Weighted Regression) (Ma) =', AgeWeighted)
    AgeDetrital = np.array([DetritalWLR, DetErrorWLR]) #2m1 end member age
    AgeDetrital = vectorized_round(AgeDetrital,2)
    print('2m1 End-Member Age (Weighted Regression) (Ma) =', AgeDetrital)
    
# --- PLOT ---
    # Plot data in a IAA figure
    # Compute the axis limits
    y_max = max(Y)+(0.1*max(Y)) # Y axis limit is 10% greater than highest Y value
    y2_max = np.log(y_max+1)/lamda/1000000; # Same as above but converted to Ma 
    x_max = 10*math.ceil((max(X)+0.1*max(X))/10) # Rounds max x limit (10% greater than highest X val) to nearest 10
    
    fig, ax1 = plt.subplots()
    ax1.errorbar(X,Y,xerr=sigX,yerr=sigY,marker='o', linestyle = 'None',markersize=8, 
                 markeredgecolor='k', markerfacecolor='w', ecolor='k', capsize=4)
    ax1.set_xlabel('$2M_{1}$ (detrital) illite [%]', fontsize=10)
    ax1.set_ylabel('$e^{\lambda t}-1$', fontsize=10)
    ax1.set_ylim(0, y_max)
    ax1.set_xlim(0,x_max)
    ax1.set_title(sheet_name)
    ax1.grid(True)
    
    #plot regression lines
    p = np.linspace(0-(max(X)-min(X))/10, 100, num=100)
    q = a + b*p
    h2 = plt.plot(p,q,'r',linewidth=2)
    q1 = a1 + b1*p
    h3 = plt.plot(p,q1,'--k')
    
    #create second y-axis
    ax2 = ax1.twinx() # create a twin axes
    ax2.set_ylabel('Age [Ma]', fontsize=10)
    ax2.set_ylim(0, y2_max)
    
    ax1.legend(['Weighted Line (MLM)','Simple Line (SLR)','Data'], 
               loc='upper left',fontsize=10)
    
    ax1.text(0.95,0.05,f'100% authigenic illite age = {vectorized_round(AgeWLR,2)}\
 \u00B1 {vectorized_round(AgeErrorW,2)} Ma', ha='right',va='bottom',\
            transform=plt.gca().transAxes, fontsize=10)
        
   #trying to plot shaded area behind regression line
    x_mean_sq = np.mean(X**2)
    reg_line_err = 2*np.sqrt((sigb**2 * x_mean_sq)+(siga**2))
                          
    ax1.fill_between(p,q - reg_line_err, q + reg_line_err, alpha=0.2,color='gray')
    
    # Add r2 values for weighted regression. Change WR to R for linear regression
    ax1.text(0.95,0.10,f'$r^2$ (weighted) = {vectorized_round(WR,3)}'\
             ,ha='right',va='bottom',transform=plt.gca().transAxes, fontsize=10)
    # save the figure as an .ai or .svg file
    fig.savefig(f"{sheet_name}.pdf")
    # ax1.savefig(f"{sheet_name}.svg") # for Scalable Vector Graphics
    
    plt.show()
