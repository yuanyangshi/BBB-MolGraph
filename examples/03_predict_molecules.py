"""
Example 03: Predicting Blood-Brain Barrier (BBB) permeability for custom SMILES strings.
"""

from bbb_molgraph import load_pretrained_model, predict_smiles
from bbb_molgraph.utils.logging import logger


def main() -> None:
    # Example pharmaceutical candidate compounds
    candidates = [
        ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
        ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
        ("Diazepam", "CN1C(=O)CN=C(C2=C1C=CC(=C2)Cl)C3=CC=CC=C3"),
        ("Dopamine", "C1=CC(=C(C=C1CCN)O)O"),
    ]

    model = load_pretrained_model("default")

    logger.info("=== BBB Permeability Predictions ===")
    for name, smiles in candidates:
        prob, explanation = predict_smiles(smiles, model=model, explain=True)
        status = "BBB+ (Permeable)" if prob >= 0.5 else "BBB- (Non-permeable)"
        logger.info(f"Compound: {name:<12} | SMILES: {smiles}")
        logger.info(f"  -> Predicted Probability: {prob:.4f} | Classification: {status}")
        logger.info(f"  -> Analyzed {explanation['num_atoms']} heavy atoms for attention attribution.")


if __name__ == "__main__":
    main()
