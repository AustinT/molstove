import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pkg_resources
import scipy.integrate
from scipy.constants import Planck  # J * s
from scipy.constants import elementary_charge  # Coulomb per electron
from scipy.constants import speed_of_light  # m / s

RESOURCES_FOLDER = 'resources'

# Reference Solar Spectral Irradiance: ASTM G-173
# from https://rredc.nrel.gov/solar//spectra/am1.5/ASTMG173/ASTMG173.html
ASTMG173_FILE_NAME = 'ASTMG173.csv'

INPUT_POWER = 100.037065557  # mW cm^-2


@dataclass
class ScharberResults:
    pce: float  # Power conversion efficiency (percent)
    voc: float  # Open circuit voltage (V)
    jsc: float  # Short circuit current density (mA cm^-2)

    ff: float  # Fill factor (dimensionless)
    eqe: float  # External quantum efficiency (dimensionless)
    lumo_acceptor: float  # Energy of the acceptor's LUMO (eV)
    e_charge_sep: float  # Charge separation energy (eV)
    e_empirical_loss: float  # Empirical loss energy (eV)


def get_integrals(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    assert (x.shape == y.shape)
    assert (len(y.shape) == 1)

    integrals = np.zeros(shape=y.shape)
    for i in range(y.shape[0]):
        integrals[i] = scipy.integrate.simps(y=y[:i + 1], x=x[:i + 1])

    return integrals


def load_astm_data() -> pd.DataFrame:
    """
    We are using the "Global Tilt" = spectral radiation from solar disk plus sky diffuse and diffuse reflected from ground
    on south facing surface tilted 37 deg from horizontal. Johannes Hachmann did the same.

    :return: pandas DataFrame with wavelength (nm) and J_sc (mA cm^-2)
    """
    m_per_cm = 0.01  # m / cm
    m_per_nm = 1E-9  # m / nm
    mA_per_A = 1000  # mA / A

    path = pkg_resources.resource_filename(__package__, os.path.join(RESOURCES_FOLDER, ASTMG173_FILE_NAME))

    # Wavelength (nm) and spectral radiations (W * m^-2 * nm^-1)
    astm = pd.read_csv(path, header=0, skiprows=1, names=['wavelength', 'ETR', 'global_tilt', 'direct'])

    astm = astm.drop(columns=['ETR', 'direct'])
    astm = astm.rename(columns={'global_tilt': 'spectral_irradiance'})

    # Calculate energy of photon
    astm['e_photon'] = astm['wavelength'].apply(lambda l: Planck * speed_of_light / (l * m_per_nm))  # in J

    # Calculate number of photons per second
    astm['photon_rate'] = astm['spectral_irradiance'] / astm['e_photon']  # s^-1 m^-2 nm^-1

    # Calculate max Jsc by integration
    astm['jsc'] = (
        get_integrals(y=astm['photon_rate'], x=astm['wavelength'])  # s^-1 m^-2
        * elementary_charge * m_per_cm**2 * mA_per_A  # mA cm^-2
    )

    return astm


astm_data = load_astm_data()


def max_jsc(e_gap: float) -> float:
    """
    Return max JSC value for gap.

    :param e_gap: HOMO LUMO gap (eV)
    :return: maximum Jsc in (mA cm^-2)
    """

    joule_per_ev = scipy.constants.physical_constants['electron volt'][0]
    nm_per_m = 1E9

    e_gap_nm = speed_of_light * nm_per_m * Planck / (e_gap * joule_per_ev)

    for index, row in astm_data.iterrows():
        if row['wavelength'] > e_gap_nm:
            return row['jsc']

    return astm_data.iloc[-1]['jsc']


def calculate_scharber_props(
        homo: float,  # eV
        lumo: float,  # eV
        lumo_acceptor=-4.3,  # eV
        e_charge_sep=0.3,  # eV
        e_empirical_loss=0.3,  # eV
        eqe=0.65,  # dimensionless
        ff=0.65,  # dimensionless
) -> ScharberResults:
    """
    This function takes HOMO and LUMO values and returns PCE, Jsc, and Voc according to the Scharber model.

    :param homo: energy of HOMO (eV)
    :param lumo: energy of LUMO (eV)
    :param lumo_acceptor: energy of LUMO of acceptor (eV)
    :param e_charge_sep: energy difference required for charge separation (eV)
    :param e_empirical_loss: empirical loss (eV)
    :param eqe: external quantum efficiency (dimensionless)
    :param ff: fill factor (dimensionless)
    :return: Scharber results in a ScharberResults container
    """
    gap = lumo - homo

    # Required for charge separation
    if lumo - lumo_acceptor < e_charge_sep:
        raise RuntimeError('Energy difference between the LUMO of the donor and the one of the acceptor is too small')

    # Scharber definitions
    v_oc = lumo_acceptor - homo - e_empirical_loss  # V (since per electron)
    j_sc = eqe * max_jsc(gap)  # mA cm^-2
    pce = v_oc * j_sc * ff / INPUT_POWER * 100  # percent
    return ScharberResults(
        pce=pce,
        voc=v_oc,
        jsc=j_sc,
        ff=ff,
        eqe=eqe,
        lumo_acceptor=lumo_acceptor,
        e_charge_sep=e_charge_sep,
        e_empirical_loss=e_empirical_loss,
    )
