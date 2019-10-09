from unittest import TestCase

from molstove import properties


class TestScharber(TestCase):
    def test_small_band_gap(self):
        self.assertAlmostEqual(properties.max_jsc(0.001), 68.9825175)

    def test_band_gap(self):
        """
        From https://www.sciencedirect.com/topics/engineering/short-circuit-current-density:
        The incoming light, i.e., that part of the solar spectrum with photon energy hv larger than the band-gap energy
        Eg=1.155 eV of the specific absorber would correspond to a (maximum possible) Jsc of 41.7 mA cm–2.
        """
        self.assertAlmostEqual(properties.max_jsc(1.155), 42.0, places=0)

    def test_experiment_1(self) -> None:
        # Experiment: https://onlinelibrary.wiley.com/doi/epdf/10.1002/adfm.200600171
        # PCE: 0.4 %
        # VoC: 0.85 V
        # Jsc: 0.89 mA cm^-2
        homo = -5.61  # eV
        lumo = -2.83  # eV

        self.assertAlmostEqual(properties.max_jsc(lumo - homo), 3.2613212)

        scharber = properties.calculate_scharber_props(homo=homo, lumo=lumo)
        self.assertAlmostEqual(scharber.pce, 1.39117166)
        self.assertAlmostEqual(scharber.voc, 1.01)
        self.assertAlmostEqual(scharber.jsc, 2.1198588)

    def test_experiment_2(self) -> None:
        # Experiment: https://pubs.rsc.org/en/content/articlelanding/2011/ee/c1ee01072g
        # PCE: 5.5 %
        # Voc: 0.8 V
        # Jsc: 10.3 mA cm^-2
        homo = -5.35  # eV
        lumo = -3.53  # eV

        scharber = properties.calculate_scharber_props(homo=homo, lumo=lumo)
        self.assertAlmostEqual(scharber.pce, 6.0723036)
        self.assertAlmostEqual(scharber.voc, 0.75)
        self.assertAlmostEqual(scharber.jsc, 12.4606242)
