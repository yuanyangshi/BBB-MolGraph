"""
Example 04: Publication-ready 2D pharmacophore attribution heatmap generation.
"""

from pathlib import Path
from bbb_molgraph import load_pretrained_model, predict_smiles
from bbb_molgraph.explainability.visualizer import PharmacophoreVisualizer
from bbb_molgraph.utils.config_parser import get_project_root
from bbb_molgraph.utils.logging import logger


def main() -> None:
    # Target molecule: Diazepam (well-known central nervous system permeable drug)
    smiles = "CN1C(=O)CN=C(C2=C1C=CC(=C2)Cl)C3=CC=CC=C3"
    compound_name = "Diazepam"

    model = load_pretrained_model("default")
    prob, explanation = predict_smiles(smiles, model=model, explain=True)

    logger.info(f"Target Compound: {compound_name}")
    logger.info(f"Predicted BBB+ Probability: {prob:.4f}")

    visualizer = PharmacophoreVisualizer(colormap="Reds", image_size=(600, 600))
    output_dir = get_project_root() / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    svg_path = output_dir / f"{compound_name}_heatmap.svg"
    png_path = output_dir / f"{compound_name}_heatmap.png"

    # Render publication-grade vector SVG
    visualizer.render_to_svg(
        smiles=smiles,
        atom_weights=explanation["atom_attentions"],
        output_path=svg_path,
    )

    # Render high-resolution PNG
    try:
        visualizer.render_to_png(
            smiles=smiles,
            atom_weights=explanation["atom_attentions"],
            output_path=png_path,
        )
    except Exception as e:
        logger.warning(f"Cairo PNG renderer unavailable in current environment ({e}). SVG generated.")

    logger.info(f"Visualization completed. Vector file generated at: {svg_path}")


if __name__ == "__main__":
    main()
