import os
import re
from typing import List

from molstove.tools import Atoms, Atom, Orbital


class ParserError(Exception):
    """Error raised when an error occurs while parsing results"""


class OrcaParser:
    """Parser for ORCA calculations"""
    def __init__(self, directory: str, output_file_name, xyz_file_name: str = 'orca.xyz') -> None:
        self.directory = directory
        self.output_path = os.path.join(self.directory, output_file_name)
        self.xyz_path = os.path.join(self.directory, xyz_file_name)

    @staticmethod
    def find(file_path: str, pattern, exactly_one=True):
        with open(file_path) as f:
            content = f.read()

            matches = pattern.findall(content)
            if exactly_one and len(matches) > 1:
                raise ParserError('Found multiple matches in file: ' + str(file_path) + '; pattern: "' +
                                  str(pattern.pattern) + '"')
            else:
                return pattern.search(content)

    @staticmethod
    def find_all(file_path: str, pattern):
        with open(file_path) as f:
            content = f.read()
            return pattern.finditer(content)

    def sanity_check(self) -> None:
        done_re = re.compile(r'(\*){4}ORCA TERMINATED NORMALLY(\*){4}')
        if not self.find(self.output_path, done_re):
            raise ParserError('Calculation did not pass ORCA sanity check')

    def get_all_final_single_point_energies(self) -> List[float]:
        sp_energy_re = re.compile(r'FINAL SINGLE POINT ENERGY\s+(?P<energy>-?\d+\.\d+)')
        try:
            return [
                float(energy_match.group('energy'))
                for energy_match in self.find_all(self.output_path, pattern=sp_energy_re)
            ]
        except (AttributeError, ValueError):
            raise ParserError('Cannot find all final single point energies')

    def get_last_final_single_point_energies(self) -> float:
        return self.get_all_final_single_point_energies()[-1]

    def get_last_coordinates(self) -> Atoms:
        # XYZ in Angstrom, Atoms in Angstrom, too.
        coord_re = re.compile(r'\s*(?P<element>\w+)\s*(?P<x>-?\d+\.\d+)\s*(?P<y>-?\d+\.\d+)\s*(?P<z>-?\d+\.\d+)\s*')
        atoms = []
        for coord_match in self.find_all(self.xyz_path, coord_re):
            atom = Atom(
                element=coord_match.group('element'),
                x=float(coord_match.group('x')),
                y=float(coord_match.group('y')),
                z=float(coord_match.group('z')),
            )
            atoms.append(atom)

        return atoms

    def get_all_orbitals(self) -> List[List[Orbital]]:
        orbitals_list = []

        orbital_line = r'\s*\d+\s+(?P<occupation>\d+\.\d+)\s+(?P<energy_ha>-?\d+\.\d+)\s+(?P<energy_ev>-?\d+\.\d+)'
        orbital_re = re.compile(orbital_line)

        block_re = re.compile(r'-{16}\nORBITAL ENERGIES\n-{16}\n\n'
                              r'\s+NO\s+OCC\s+E\(Eh\)\s+E\(eV\)\s+\n'
                              r'(?P<orbitals>(' + orbital_line + r')+)')

        try:
            for block in self.find_all(self.output_path, block_re):
                orbitals = []
                for orbital_match in orbital_re.finditer(block.group('orbitals')):
                    orbitals.append(
                        Orbital(
                            occupation=float(orbital_match.group('occupation')),
                            energy=float(orbital_match.group('energy_ha')),
                        ))
                orbitals_list.append(orbitals)

        except (AttributeError, ValueError):
            raise ParserError('Cannot parse orbital energies')

        return orbitals_list

    def get_last_orbitals(self) -> List[Orbital]:
        return self.get_all_orbitals()[-1]

    def get_open_shell_orbitals(self) -> List[List[Orbital]]:
        orbitals_list = []

        orbital_line = r'\s*\d+\s+(?P<occupation>\d+\.\d+)\s+(?P<energy_ha>-?\d+\.\d+)\s+(?P<energy_ev>-?\d+\.\d+)'
        orbital_re = re.compile(orbital_line)

        block_re = re.compile(r'-{16}\nORBITAL ENERGIES\n-{16}\n'
                              r'\s+SPIN UP ORBITALS'
                              r'\s+NO\s+OCC\s+E\(Eh\)\s+E\(eV\)\s+\n'
                              r'(?P<spin_up>(\s*\d+\s+(\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+))+)'
                              r'\s+SPIN DOWN ORBITALS'
                              r'\s+NO\s+OCC\s+E\(Eh\)\s+E\(eV\)\s+\n'
                              r'(?P<spin_down>(\s*\d+\s+(\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+))+)')

        try:
            for block in self.find_all(self.output_path, block_re):
                orbitals = []
                for orbital_match in orbital_re.finditer(block.group('spin_up')):
                    orbitals.append(
                        Orbital(
                            occupation=float(orbital_match.group('occupation')),
                            energy=float(orbital_match.group('energy_ha')),
                        ))
                for orbital_match in orbital_re.finditer(block.group('spin_down')):
                    orbitals.append(
                        Orbital(
                            occupation=float(orbital_match.group('occupation')),
                            energy=float(orbital_match.group('energy_ha')),
                        ))

                orbitals_list.append(sorted(orbitals, key=lambda x: x.energy))

        except (AttributeError, ValueError):
            raise ParserError('Cannot parse orbital energies')

        return orbitals_list

    def get_last_open_shell_orbitals(self) -> List[Orbital]:
        return self.get_open_shell_orbitals()[-1]
