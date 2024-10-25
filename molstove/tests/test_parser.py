from unittest import TestCase
from pathlib import Path

import pkg_resources

from molstove.parser import OrcaParser, ParserError

RESOURCES_FOLDER = 'resources'


class TestOrcaParser(TestCase):
    RESOURCES = Path(
        pkg_resources.resource_filename(__package__, RESOURCES_FOLDER))

    def test_sanity_check(self):
        p = OrcaParser(directory=self.RESOURCES,
                       output_file_name='orca.output')
        self.assertIsNone(p.sanity_check())  # type: ignore

    def test_sanity_check_fail(self):
        p = OrcaParser(directory=self.RESOURCES, output_file_name='fail.output')
        with self.assertRaises(ParserError):
            p.sanity_check()

    def test_all_final_single_point_energies(self):
        p = OrcaParser(directory=self.RESOURCES, output_file_name='orca.output')
        energies = p.get_all_final_single_point_energies()
        self.assertEqual(len(energies), 1)
        self.assertAlmostEqual(energies[0], -1.10605458839)

    def test_orbital_parsers_1(self):
        p = OrcaParser(directory=self.RESOURCES, output_file_name='orca.output')
        orbitals_list = p.get_all_orbitals()

        self.assertEqual(len(orbitals_list), 1)

        orbitals = orbitals_list[0]
        self.assertEqual(len(orbitals), 10)
        self.assertAlmostEqual(orbitals[0].occupation, 2.0)
        self.assertAlmostEqual(orbitals[1].occupation, 0.0)

    def test_orbital_parsers_2(self):
        p = OrcaParser(directory=self.RESOURCES, output_file_name='orbitals.output')
        orbitals_list = p.get_all_orbitals()

        self.assertEqual(len(orbitals_list), 2)

        orbitals = orbitals_list[0]
        self.assertEqual(len(orbitals), 34)
        self.assertAlmostEqual(orbitals[0].energy, -9.897528)
        self.assertAlmostEqual(orbitals[33].energy, 2.828269)

    def test_open_shell_parsers(self):
        p = OrcaParser(directory=self.RESOURCES, output_file_name='open_orbitals.output')
        orbitals_list = p.get_open_shell_orbitals()

        self.assertEqual(len(orbitals_list), 1)

        orbitals = orbitals_list[0]
        self.assertEqual(len(orbitals), 20)
        self.assertAlmostEqual(orbitals[0].occupation, 1.0)
        self.assertAlmostEqual(orbitals[3].occupation, 0.0)
