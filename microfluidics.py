# Import native libraries.
from decimal import Decimal


# Import extension libraries.
import numpy as np
import scipy as sp

# Define global constants.
pi = np.pi


def resistance_sum_term(n, w, h):

    return (1 / n**5) * np.tan(h) * ( (n*pi*w) / (2*h) )


def resistance_sum(w, h, num_terms=5):
    
    current_r_sum = 0

    for n in range(1, num_terms+1):

        current_r_sum += resistance_sum_term(n, w, h)

    return current_r_sum


def convert_um_to_m(udist):
    
    return udist*10.**(-6)


def calculate_resistance(w, h, L, mu=0.6913*10**(-3), scientific=True, num_digits=4):
    
    """
    Calculates the fluidic resistance of a microfludic channel whose cross-
    sectional area is rectangular.

    Parameters
    ----------
    w: float
        The channel's width, usually in microns.
    h: float
        The channel's height, usuallyin microns.
    mu: float, default (0.6913*10^(-3)) Pa.sec.
        The viscosity of the in-chip fluid, assumed to be water at 37 C.
    num_digits: int, default 4
        The number of digits to display when
    """
    
    # Convert to distances from microns to meters to make compatible
    # with the units of viscosity.
    w = convert_um_to_m(w)
    h = convert_um_to_m(h)
    L = convert_um_to_m(L)
    
    a = (12.*mu*L)/(w*h**3)
    
    b = (
        1. - (h / w) * ( (192. / pi**5) * resistance_sum(w, h)  )
    )**(-1)
    
    resistance = a * b
    
    if scientific:
        return '%.{num_digits}E'.format(num_digits=num_digits) %  Decimal(str(resistance))
    else:
        return resistance
