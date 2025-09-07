from typing import Optional
from padelpy import from_smiles
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize

# Suppress RDKit info/debug logging to reduce console verbosity
RDLogger.DisableLog('rdApp.*')


def standardize_smiles(smiles: str) -> Optional[str]:
    """
    Standardize a SMILES string by:
    - Removing common salts using RDKit's SaltRemover
    - Filtering out biologics (heavy atom > 50), inorganic molecules,
      and chemical mixtures (multiple fragments after salt stripping)
    - Normalizing, uncharging, selecting the largest fragment,
      and canonicalizing tautomers (with fallback on error)

    Args:
        smiles: Input SMILES string.

    Returns:
        A standardized canonical SMILES string, or None if the input
        is a biologic, inorganic, or mixture, or if the input is invalid.
    """
    # Validate input
    if not isinstance(smiles, str) or not smiles:
        return None

    # Parse molecule
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return None
    if mol is None:
        return None

    # Strip salts
    remover = SaltRemover.SaltRemover()
    mol_clean = remover.StripMol(mol)

    # Reject mixtures
    frags = Chem.GetMolFrags(mol_clean)
    if len(frags) > 1:
        return None

    # Reject inorganics: require at least one carbon and only allowed elements
    allowed_elements = {'C', 'H', 'O', 'N', 'P', 'S', 'F', 'Cl', 'Br', 'I'}
    atom_symbols = [atom.GetSymbol() for atom in mol_clean.GetAtoms()]
    # Must contain at least one carbon atom
    if 'C' not in atom_symbols:
        return None
    # All atoms must be from allowed set
    for sym in atom_symbols:
        if sym not in allowed_elements:
            return None

    # Reject biologics
    if mol_clean.GetNumHeavyAtoms() > 50:
        return None

    # Normalize and uncharge
    normalizer = rdMolStandardize.Normalizer()
    mol_norm = normalizer.normalize(mol_clean)
    uncharger = rdMolStandardize.Uncharger()
    mol_uncharge = uncharger.uncharge(mol_norm)

    # Select largest fragment
    chooser = rdMolStandardize.LargestFragmentChooser()
    mol_largest = chooser.choose(mol_uncharge)

    # Tautomer canonicalization with error fallback
    tauto_enum = rdMolStandardize.TautomerEnumerator()
    try:
        mol_final = tauto_enum.Canonicalize(mol_largest)
    except Exception:
        mol_final = mol_largest

    # Return canonical SMILES
    return Chem.MolToSmiles(mol_final, isomericSmiles=True)


def generate_PubchemFP(smiles: str) -> Optional[np.ndarray]:
    """
    Compute the 881-bit PubChem fingerprint for a given SMILES string.
    Returns a numpy array of 0/1 bits, or None if computation fails.

    Args:
        smiles: Input SMILES string.

    Returns:
        A numpy array of shape (881,) with 0/1 values, or None on failure.
    """
    try:
        # Optionally standardize first
        std_smiles = standardize_smiles(smiles)
        if std_smiles is None:
            return None

        # Use PaDELPy to calculate only fingerprints (no other descriptors)
        result = from_smiles(std_smiles, descriptors=False, fingerprints=True)
        # 'result' is an OrderedDict with PubChem fingerprint bits as values
        bit_values = [int(val) for val in result.values()]  # convert '0'/'1' strings to ints
        fingerprint_array = np.array(bit_values, dtype=int)
        # Verify length
        if fingerprint_array.size == 881:
            return fingerprint_array
        else:
            # Unexpected size, treat as failure
            return None
    except Exception:
        # Handle any errors (e.g., PaDEL not installed, invalid SMILES)
        return None