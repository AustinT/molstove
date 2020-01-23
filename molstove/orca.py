import abc
import pathlib
from dataclasses import dataclass
from shutil import copyfile
from typing import Optional

import pkg_resources

from molstove import process, parser, tools
from molstove.tools import Atoms


class Calculator(abc.ABC):
    basis_file_dict = {
        'def2-SVP': 'def2-svp.bas',
        'def2-TZVP': 'def2-tzvp.bas',
        'STO-6G': 'sto-6g.bas',
        'def2/J': 'def2-universal-jfit.bas',
    }

    BASIS_SETS_FOLDER = pathlib.Path('resources') / 'basis_sets'

    def __init__(
            self,
            atoms: Atoms,
            charge: int,
            spin_multiplicity: int,
            basis: str,
            method: str,
            aux_basis: Optional[str] = None,
            open_shell: bool = False,
            num_processes: int = 1,
            directory: Optional[pathlib.Path] = None,
    ):
        """
        Construct Orca calculator

        :param atoms: atoms (in Angstrom)
        :param charge: charge of system
        :param spin_multiplicity: spin multiplicity of system
        :param basis: orbital basis
        :param method: method to be employed (e.g., exchange correlation functional for DFT)
        :param aux_basis: auxiliary orbital basis
        :param open_shell: boolean indicating if the calculation is open shell
        :param num_processes: number of mpi processes
        :param directory: base directory of calculation
        """
        self.atoms = atoms
        self.charge = charge
        self.spin_multiplicity = spin_multiplicity

        self.basis = basis
        self.aux_basis = aux_basis
        self.method = method
        self.open_shell = open_shell

        self.directory = directory if directory else pathlib.Path.cwd()

        self.num_processes = num_processes

        self.input_file_name = 'orca.inp'
        self.output_file_name = 'orca.output'
        self.command = f'$(which orca) {self.input_file_name} > {self.output_file_name}'

    @staticmethod
    @abc.abstractmethod
    def get_job_string() -> str:
        """
        Return string specifying the type of calculation
        """
        raise NotImplementedError

    def _render_input(self) -> str:
        job = self.get_job_string()
        header = f'! {self.method} {job}\n'

        if self.aux_basis:
            basis_settings = '%basis\n' \
                             f'GTOName="{self.basis_file_dict[self.basis]}";\n' \
                             f'GTOAuxName="{self.basis_file_dict[self.aux_basis]}";\n' \
                             'end\n'
        else:
            basis_settings = '%basis\n' \
                             f'GTOName="{self.basis_file_dict[self.basis]}";\n' \
                             'end\n'

        processes_settings = f'%pal\nnprocs {self.num_processes}\nend\n'
        scf_settings = '%scf\nMaxIter 1200\nend\n'
        geom_settings = '%geom\nMaxIter 1200\nend\n'
        coords_header = f'*xyz {self.charge} {self.spin_multiplicity}'
        coords = '\n'.join([f'{atom.element} {atom.x} {atom.y} {atom.z}' for atom in self.atoms])
        coords_footer = '*'

        return '\n'.join([
            header, basis_settings, processes_settings, scf_settings, geom_settings, coords_header, coords,
            coords_footer
        ])

    def _write_input_file(self, string: str):
        with open(self.directory / self.input_file_name, mode="w") as f:
            f.write(string)

    def copy_basis_set_files(self) -> None:
        for basis in [self.basis, self.aux_basis]:
            if not basis:
                continue

            basis_set_file = self.basis_file_dict[basis]
            basis_set_file_path = self.BASIS_SETS_FOLDER / basis_set_file
            src = pkg_resources.resource_filename(__package__, str(basis_set_file_path))
            dst = self.directory / basis_set_file
            copyfile(src, dst)

    def run(self) -> process.ExecutionResult:
        """
        Run ORCA calculation

        :return: ExecutionResult
        """
        self.directory.mkdir(parents=True, exist_ok=False)

        input_string = self._render_input()
        self._write_input_file(string=input_string)

        self.copy_basis_set_files()

        result = process.execute_command(command=self.command, directory=self.directory, strict=True)

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
        p = parser.OrcaParser(directory=self.directory, output_file_name=self.output_file_name)
        p.sanity_check()

        if not self.open_shell:
            orbitals = p.get_last_orbitals()
        else:
            orbitals = p.get_last_open_shell_orbitals()

        homo, lumo = tools.get_homo_lumo_energies(orbitals, is_open_shell=self.open_shell)

        return SCFResult(
            energy=p.get_last_final_single_point_energies(),
            homo=homo,
            lumo=lumo,
        )


@dataclass
class StructureOptResult:
    atoms: Atoms


class StructureOptCalculator(Calculator):
    @staticmethod
    def get_job_string() -> str:
        return 'Opt'

    def parse_results(self) -> StructureOptResult:
        p = parser.OrcaParser(directory=self.directory, output_file_name=self.output_file_name)
        p.sanity_check()

        return StructureOptResult(atoms=p.get_last_coordinates())
