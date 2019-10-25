import abc
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np

from molstove import process, parser
from molstove.parser import Orbital
from molstove.tools import Atoms


class Calculator(abc.ABC):
    def __init__(self,
                 atoms: Atoms,
                 charge: int,
                 spin_multiplicity: int,
                 basis: str,
                 method: str,
                 num_processes: int = 1,
                 base_dir: Optional[str] = None):
        """
        Construct Orca calculator

        :param atoms: atoms (in Angstrom)
        :param charge: charge of system
        :param spin_multiplicity: spin multiplicity of system
        :param basis: orbital basis
        :param method: method to be employed (e.g., exchange correlation functional for DFT)
        :param num_processes: number of mpi processes
        :param base_dir: base directory of calculation
        """
        self.atoms = atoms
        self.charge = charge
        self.spin_multiplicity = spin_multiplicity

        self.basis = basis
        self.method = method

        if base_dir is None:
            base_dir = os.getcwd()

        self.base_dir = base_dir
        self.calculation_directory = os.path.join(self.base_dir, self._create_tmp_dir_name())

        self.num_processes = num_processes

        self.executed = False

        self.input_file_name = 'orca.inp'
        self.output_file_name = 'orca.output'
        self.command = f'$(which orca) {self.input_file_name} > {self.output_file_name}'

    @staticmethod
    def _create_tmp_dir_name() -> str:
        return datetime.now().strftime('%Y%m%dT%H%M%S')

    @staticmethod
    def _create_directory(path: str) -> None:
        os.makedirs(name=path)

    @staticmethod
    @abc.abstractmethod
    def get_job_string() -> str:
        """
        Return string specifying the type of calculation
        """
        raise NotImplementedError

    @staticmethod
    def _render_input(atoms: List, charge: int, spin_multiplicity: int, method: str, basis: str, job: str,
                      num_processes: int) -> str:
        header = f'! {method} {basis} {job}\n'
        processes_settings = f'%pal\nnprocs {num_processes}\nend\n'
        scf_settings = '%scf\nMaxIter 1200\nend\n'
        geom_settings = '%geom\nMaxIter 1200\nend\n'
        coords_header = f'*xyz {charge} {spin_multiplicity}'
        coords = '\n'.join([f'{atom.element} {atom.x} {atom.y} {atom.z}' for atom in atoms])
        coords_footer = '*'

        return '\n'.join(
            [header, processes_settings, scf_settings, geom_settings, coords_header, coords, coords_footer])

    def _write_input_file(self, string: str):
        with open(os.path.join(self.calculation_directory, self.input_file_name), mode='w') as f:
            f.write(string)

    def run(self) -> process.ExecutionResult:
        """
        Run ORCA calculation

        :return: ExecutionResult
        """
        self._create_directory(self.calculation_directory)

        input_string = self._render_input(atoms=self.atoms,
                                          charge=self.charge,
                                          spin_multiplicity=self.spin_multiplicity,
                                          method=self.method,
                                          basis=self.basis,
                                          job=self.get_job_string(),
                                          num_processes=self.num_processes)

        self._write_input_file(string=input_string)

        result = process.execute_command(command=self.command, directory=self.calculation_directory, strict=True)

        self.executed = True

        return result

    @abc.abstractmethod
    def parse_results(self):
        raise NotImplementedError


@dataclass
class SCFResult:
    energy: float  # Hartree
    homo: float  # Hartree
    lumo: float  # Hartree


class SinglePointCalculator(Calculator):
    @staticmethod
    def get_job_string() -> str:
        return 'SP'

    def parse_results(self) -> SCFResult:
        p = parser.OrcaParser(directory=self.calculation_directory, output_file_name=self.output_file_name)
        p.sanity_check()

        orbitals = p.get_last_orbitals()
        homo, lumo = self._get_homo_lumo_energies(orbitals)

        return SCFResult(
            energy=p.get_last_final_single_point_energies(),
            homo=homo,
            lumo=lumo,
        )

    @staticmethod
    def _get_homo_lumo_energies(orbitals: List[Orbital]) -> Tuple[float, float]:
        homo, lumo = 0.0, 0.0
        for i, orbital in enumerate(orbitals):
            if np.isclose(orbital.occupation, 2.0):
                homo = orbital.energy
            elif np.isclose(orbital.occupation, 0.0):
                lumo = orbital.energy
                break
            else:
                raise RuntimeError(f'Cannot determine if HOMO or LUMO ({orbital.energy})')

        return homo, lumo


@dataclass
class StructureOptResult:
    atoms: Atoms


class StructureOptCalculator(Calculator):
    @staticmethod
    def get_job_string() -> str:
        return 'Opt'

    def parse_results(self) -> StructureOptResult:
        p = parser.OrcaParser(directory=self.calculation_directory, output_file_name=self.output_file_name)
        p.sanity_check()

        return StructureOptResult(atoms=p.get_last_coordinates())
