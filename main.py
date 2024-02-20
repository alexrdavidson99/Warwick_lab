import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.signal import chirp, find_peaks, peak_widths

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit


def gaus(x, a, x0, sigma):
    '''
     1D gaussian
    '''
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


# Read in the data
data = pd.read_csv("hists/hist data 10k.csv", skiprows= 100, names = ["volts_seconds", "counts"], sep=",")
time_data = pd.read_csv("hists/time_hist_data_gen.csv", names=["time_bins","counts"], sep=",")
time_data_laser = pd.read_csv("hists/time_hist_data_zaster.csv", names=["time_bins","counts"], sep=",")

#time_data = time_data[time_data["counts"] > 0]
#time_data_laser = time_data_laser[time_data_laser["counts"] > 0]
print(time_data)
print(sum(time_data["counts"]))
print(sum(time_data_laser["counts"]))

popt, pcov = curve_fit(gaus, time_data["time_bins"], time_data["counts"], method='lm', maxfev=10000, p0=[3095, -1.376000e-08, 3e-09])
popt_laser, pcov_laser = curve_fit(gaus, time_data_laser["time_bins"], time_data_laser["counts"], method='lm', maxfev=10000, p0=[1160, -1.326000e-08, 3e-09])
print(popt)

x_gauss = np.linspace(-14500e-12, -13500e-12, 2000)
x_gauss_laser = np.linspace(-13500e-12, -12500e-12, 2000)
y_gauss = gaus(x_gauss, popt[0], popt[1], popt[2])
y_gauss_laser = gaus(x_gauss_laser, popt_laser[0], popt_laser[1], popt_laser[2])

# Find peaks and calculate FWHM for the original data
peaks, _ = find_peaks(time_data["counts"], height=540)
fwhm_data = peak_widths(time_data["counts"], peaks, rel_height=0.5)
print(f'FWHM for original data: {fwhm_data[0][0]*40:.2e} ps')

# Find peaks and calculate FWHM for the laser data
peaks_laser, _ = find_peaks(time_data_laser["counts"], height=540)
fwhm_laser = peak_widths(time_data_laser["counts"], peaks_laser, rel_height=0.5)
print(f'FWHM for laser data: {fwhm_laser[0][0]*40:.2e} ps')


plt.plot(x_gauss, y_gauss, 'r--', linewidth=2, label = f"FWHM Gen: {fwhm_data[0][0]*40:.2f}ps ")
plt.plot(x_gauss_laser , y_gauss_laser, 'g--', linewidth=2, label = f"FWHM Laser: {fwhm_laser[0][0]*40:.2f}ps ")

# Add labels and a title
plt.legend()
plt.xlabel('time (s)')
plt.ylabel('Frequency')
plt.title('Gaussian Distribution of oscilloscope data at looking timing jitter of the laser and the pulse generator')



plt.scatter(time_data["time_bins"],time_data["counts"], color='blue', label = f"counts")
plt.scatter(time_data_laser["time_bins"],time_data_laser["counts"], color='red', label = f"counts_laser")

plt.xlim(-14500e-12,-12500e-12)

plt.show()
test = 1
