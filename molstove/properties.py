import dataclasses
import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import scipy.integrate
from scipy.constants import Planck  # J * s
from scipy.constants import elementary_charge  # Coulomb per electron
from scipy.constants import speed_of_light  # m / s

from molstove import tools, calculator, conformers

# Reference Solar Spectral Irradiance: ASTM G-173
# from https://rredc.nrel.gov/solar//spectra/am1.5/ASTMG173/ASTMG173.html
ASTMG173_FILE_NAME = 'resources/ASTMG173.csv'

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

    path = os.path.abspath(os.path.join(os.path.dirname(__file__), ASTMG173_FILE_NAME))

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


def compute_pv_props(
        smiles: str,
        seed: int,
        max_num_conformers: int,
        rmsd_threshold: float,
        delta_e_threshold: float,
        energy_window: float,
        max_num_opt_conformers: int,
        basis_set: str,
        xc_functional: str,
) -> List[dict]:
    """
    Compute photovoltaic properties.

    :param smiles: SMILES representation of molecule
    :param seed: random seed for conformer generation
    :param max_num_conformers: maximum number of conformers generated
    :param rmsd_threshold: minimum RMSD between two conformers to be considered different (in Angstrom)
    :param delta_e_threshold: minimum energy difference between to be considered different (in kcal/mol)
    :param energy_window: maximum energy difference the most stable conformer and any other conformer (in kcal/mol)
    :param max_num_opt_conformers: maximum number of optimized conformers
    :param basis_set: basis set used for QC calculations
    :param xc_functional: XC functional used for QC calculations
    :return: list of reports, one for each conformer
    """
    # Generate molecule from SMILES
    mol = tools.mol_from_smiles(smiles)

    # Generate conformers
    conformers.generate_conformers(mol, max_num_conformers=max_num_conformers, seed=seed)
    energies = conformers.minimize_conformers(mol)
    conformer_list = conformers.collect_clusters(
        mol=mol,
        energies=energies,
        rmsd_threshold=rmsd_threshold,
        delta_e_threshold=delta_e_threshold,
        energy_window=energy_window,
        max_num_conformers=max_num_opt_conformers,
    )

    # Convert conformers to PySCF format
    atoms_list = [tools.conformer_to_atoms(mol=mol, conformer=conformer) for conformer in conformer_list]
    charge = tools.get_molecular_charge(mol)

    reports = []

    # Run QC calculations and get predictions based on Scharber model
    for i, atoms in enumerate(atoms_list):
        try:
            c = calculator.Calculator(atoms=atoms, charge=charge, basis=basis_set, xc=xc_functional)
            atoms_opt = calculator.optimize(c)

            c2 = calculator.Calculator(atoms=atoms_opt, charge=charge, basis=basis_set, xc=xc_functional)
            c2.run()

            homo, lumo = c2.get_homo_lumo()
            scharber = calculate_scharber_props(homo=homo, lumo=lumo)

            report = calculator.generate_report(c2)
            report.update(dataclasses.asdict(scharber))
            report['smiles'] = smiles

            reports.append(report)

        except RuntimeError as e:
            print(f'Calculations for conformer {i} failed: {e}')

    return reports
