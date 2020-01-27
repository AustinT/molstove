from typing import Tuple, List

from rdkit import Chem
from rdkit.Chem import Mol, AllChem
from rdkit.Chem.rdMolAlign import AlignMol


def generate_conformers(mol: Mol, max_num_conformers: int, seed: int = 42) -> int:
    """
    Generate conformers for Molecule mol and return number of conformers generated.

    :param mol: molecule for which conformers are generated
    :param max_num_conformers: maximum number of conformers to be generated
    :param seed: random seed of EDG algorithm
    :return: number of conformers generated
    """
    ids = AllChem.EmbedMultipleConfs(
        mol,
        numConfs=max_num_conformers,
        maxAttempts=1000,
        pruneRmsThresh=0.1,
        clearConfs=True,
        randomSeed=seed,
        numThreads=0,
    )
    return len(ids)


def minimize_conformers(mol: Mol, max_iters: int = 10000,
                        fix_se_error: bool = True) -> List[float]:
    """
    Minimize conformers in molecule and return their energies (in kcal/mol).

    :param mol: molecule containing conformers to be optimized
    :return: list of energies in kcal/mol
    """

    # Determine whether MMFF potentials can be used,
    # or whether one must fall back to UFF potentials
    if AllChem.MMFFHasAllMoleculeParams(mol):
        optimization_method = AllChem.MMFFOptimizeMoleculeConfs
    else:
        optimization_method = AllChem.UFFOptimizeMoleculeConfs

        # Change the hybridization of selenium to account for a bug
        # in the UFF potential behaviour
        if fix_se_error:
            print("Warning: changing Se geometries to SP3")
            for a in mol.GetAtoms():
                if a.GetSymbol() == "Se":
                    a.SetHybridization(Chem.rdchem.HybridizationType.SP3)

    # Run the optimizations for a really long time
    opt_results = optimization_method(mol, maxIters=max_iters)

    # Discard any conformers that don't work
    energies = []
    for idx, (not_converged, energy) in enumerate(opt_results):
        if not_converged:
            mol.RemoveConformer(idx)
        else:
            energies.append(energy)
    assert len(energies) == mol.GetNumConformers(), \
        (len(energies), mol.GetNumConformers())

    # Re-index the conformers, to account for the ones removed
    conf_list = sorted([(c.GetId(), c) for c in mol.GetConformers()])
    for idx, (old_id, c) in enumerate(conf_list):
        c.SetId(idx)

    return energies


def collect_clusters(mol: Mol,
                     energies: List[float],
                     rmsd_threshold=0.2,
                     delta_e_threshold=0.5,
                     energy_window=5,
                     max_num_conformers=10) -> List[AllChem.Conformer]:
    """
    Collect unique conformers of molecule.

    :param mol: molecule containing conformers
    :param energies: energies of the conformers
    :param rmsd_threshold: minimum RMSD between two conformers (in Angstrom)
    :param delta_e_threshold: minimum energy difference between two conformers (in kcal/mol)
    :param energy_window: maximum energy difference between a conformer and the most stable conformer (in kcal/mol)
    :param max_num_conformers: maximum number of conformers
    :return: list of conformers
    """
    assert (mol.GetNumConformers() == len(energies))

    mol_no_h = AllChem.RemoveHs(mol)

    conf_energy_list = sorted([(conf_id, energy) for conf_id, energy in enumerate(energies)], key=lambda t: t[1])
    min_energy = min(energy for conf_id, energy in conf_energy_list)

    accepted_list: List[Tuple[int, float]] = []

    for conf_id, energy in conf_energy_list:
        duplicated = False

        if len(accepted_list) >= max_num_conformers:
            break

        # If conformer too high in energy, break
        if abs(energy - min_energy) > energy_window:
            break

        for accepted_id, accepted_energy in accepted_list:
            # If energy is too similar, structures are likely to be similar as well
            if abs(energy - accepted_energy) < delta_e_threshold:
                duplicated = True
                break

            rmsd = AlignMol(mol_no_h, mol_no_h, prbCid=conf_id, refCid=accepted_id)
            if rmsd < rmsd_threshold:
                duplicated = True
                break

        if not duplicated:
            accepted_list.append((conf_id, energy))

    return [mol.GetConformer(conf_id) for conf_id, energy in accepted_list]
