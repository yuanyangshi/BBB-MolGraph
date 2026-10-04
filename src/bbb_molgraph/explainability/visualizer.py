"""
Publication-ready 2D molecular structure heatmap visualizer with vector output.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from bbb_molgraph.utils.logging import logger


class PharmacophoreVisualizer:
    """
    Renders 2D molecular structures with atom-level attention heatmap coloring.
    Supports SVG, PDF, and high-resolution publication-quality PNG exports.
    """

    def __init__(self, colormap: str = "Reds", image_size: Tuple[int, int] = (600, 600)) -> None:
        try:
            import matplotlib as mpl
            self.cmap = mpl.colormaps[colormap]
        except (AttributeError, KeyError):
            self.cmap = cm.get_cmap(colormap)
        self.image_size = image_size

    def render_to_svg(
        self,
        smiles: str,
        atom_weights: np.ndarray,
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        Generate SVG representation of molecule with atom highlight colors.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string for rendering: {smiles}")

        rdMolDraw2D.PrepareMolForDrawing(mol)
        drawer = rdMolDraw2D.MolDraw2DSVG(self.image_size[0], self.image_size[1])

        # Normalize weights to [0, 1]
        w_min, w_max = float(np.min(atom_weights)), float(np.max(atom_weights))
        if w_max > w_min:
            norm_weights = (atom_weights - w_min) / (w_max - w_min)
        else:
            norm_weights = np.ones_like(atom_weights) * 0.5

        atom_colors: Dict[int, Tuple[float, float, float]] = {}
        atom_radii: Dict[int, float] = {}

        num_atoms = min(mol.GetNumAtoms(), len(norm_weights))
        for i in range(num_atoms):
            rgba = self.cmap(float(norm_weights[i]))
            atom_colors[i] = (rgba[0], rgba[1], rgba[2])
            atom_radii[i] = 0.35 + 0.25 * float(norm_weights[i])

        drawing_options = drawer.drawOptions()
        drawing_options.clearBackground = True
        drawing_options.circleAtoms = False

        drawer.DrawMolecule(
            mol,
            highlightAtoms=list(atom_colors.keys()),
            highlightAtomColors=atom_colors,
            highlightAtomRadii=atom_radii,
            highlightBonds=[],
        )
        drawer.FinishDrawing()
        svg_content = drawer.GetDrawingText()

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(svg_content)
            logger.info(f"Saved publication SVG heatmap to {output_path}")

        return svg_content

    def render_to_png(
        self,
        smiles: str,
        atom_weights: np.ndarray,
        output_path: Union[str, Path],
    ) -> None:
        """
        Render molecule to a high-resolution PNG file.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string for rendering: {smiles}")

        rdMolDraw2D.PrepareMolForDrawing(mol)
        drawer = rdMolDraw2D.MolDraw2DCairo(self.image_size[0], self.image_size[1])

        w_min, w_max = float(np.min(atom_weights)), float(np.max(atom_weights))
        if w_max > w_min:
            norm_weights = (atom_weights - w_min) / (w_max - w_min)
        else:
            norm_weights = np.ones_like(atom_weights) * 0.5

        atom_colors = {}
        atom_radii = {}
        num_atoms = min(mol.GetNumAtoms(), len(norm_weights))
        for i in range(num_atoms):
            rgba = self.cmap(float(norm_weights[i]))
            atom_colors[i] = (rgba[0], rgba[1], rgba[2])
            atom_radii[i] = 0.35 + 0.25 * float(norm_weights[i])

        drawer.drawOptions().clearBackground = True
        drawer.DrawMolecule(
            mol,
            highlightAtoms=list(atom_colors.keys()),
            highlightAtomColors=atom_colors,
            highlightAtomRadii=atom_radii,
            highlightBonds=[],
        )
        drawer.FinishDrawing()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        drawer.WriteDrawingText(str(output_path))
        logger.info(f"Saved publication PNG heatmap to {output_path}")
