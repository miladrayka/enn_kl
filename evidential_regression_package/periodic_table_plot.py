"""Draw periodic table heat map."""

import os
from collections import Counter
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from pymatgen.core import Composition
from matplotlib import font_manager

font_manager.findfont("Helvetica Light")
plt.rc("font", family="Helvetica Light")
plt.rc("font", serif="Helvetica Light", size=28)
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["xtick.major.size"] = 8
plt.rcParams["xtick.major.width"] = 1.5
plt.rcParams["ytick.major.size"] = 8
plt.rcParams["ytick.major.width"] = 1.5
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["legend.markerscale"] = 2

plt.rcParams["mathtext.it"] = "Helvetica Light:italic"
plt.rcParams["mathtext.rm"] = "Helvetica Light"
plt.rcParams["mathtext.default"] = "regular"


def get_classic_coordinates() -> dict[str, tuple[float, float]]:
    """Return classic periodic table coordinates for each element.

    Returns
    -------
    dict[str, tuple[float, float]]
        Element symbols mapped to (x, y) grid coordinates.
    """
    element_coords = {
        "H": (1, 9),
        "He": (18, 9),
        "Li": (1, 8),
        "Be": (2, 8),
        "B": (13, 8),
        "C": (14, 8),
        "N": (15, 8),
        "O": (16, 8),
        "F": (17, 8),
        "Ne": (18, 8),
        "Na": (1, 7),
        "Mg": (2, 7),
        "Al": (13, 7),
        "Si": (14, 7),
        "P": (15, 7),
        "S": (16, 7),
        "Cl": (17, 7),
        "Ar": (18, 7),
        "K": (1, 6),
        "Ca": (2, 6),
        "Sc": (3, 6),
        "Ti": (4, 6),
        "V": (5, 6),
        "Cr": (6, 6),
        "Mn": (7, 6),
        "Fe": (8, 6),
        "Co": (9, 6),
        "Ni": (10, 6),
        "Cu": (11, 6),
        "Zn": (12, 6),
        "Ga": (13, 6),
        "Ge": (14, 6),
        "As": (15, 6),
        "Se": (16, 6),
        "Br": (17, 6),
        "Kr": (18, 6),
        "Rb": (1, 5),
        "Sr": (2, 5),
        "Y": (3, 5),
        "Zr": (4, 5),
        "Nb": (5, 5),
        "Mo": (6, 5),
        "Tc": (7, 5),
        "Ru": (8, 5),
        "Rh": (9, 5),
        "Pd": (10, 5),
        "Ag": (11, 5),
        "Cd": (12, 5),
        "In": (13, 5),
        "Sn": (14, 5),
        "Sb": (15, 5),
        "Te": (16, 5),
        "I": (17, 5),
        "Xe": (18, 5),
        "Cs": (1, 4),
        "Ba": (2, 4),
        "La": (3, 2.5),
        "Ce": (4, 2.5),
        "Pr": (5, 2.5),
        "Nd": (6, 2.5),
        "Pm": (7, 2.5),
        "Sm": (8, 2.5),
        "Eu": (9, 2.5),
        "Gd": (10, 2.5),
        "Tb": (11, 2.5),
        "Dy": (12, 2.5),
        "Ho": (13, 2.5),
        "Er": (14, 2.5),
        "Tm": (15, 2.5),
        "Yb": (16, 2.5),
        "Lu": (17, 2.5),
        "Hf": (4, 4),
        "Ta": (5, 4),
        "W": (6, 4),
        "Re": (7, 4),
        "Os": (8, 4),
        "Ir": (9, 4),
        "Pt": (10, 4),
        "Au": (11, 4),
        "Hg": (12, 4),
        "Tl": (13, 4),
        "Pb": (14, 4),
        "Bi": (15, 4),
        "Po": (16, 4),
        "At": (17, 4),
        "Rn": (18, 4),
        "Ac": (3, 1.5),
        "Th": (4, 1.5),
        "Pa": (5, 1.5),
        "U": (6, 1.5),
        "Np": (7, 1.5),
        "Pu": (8, 1.5),
        "Am": (9, 1.5),
        "Cm": (10, 1.5),
        "Bk": (11, 1.5),
        "Cf": (12, 1.5),
        "Es": (13, 1.5),
        "Fm": (14, 1.5),
        "Md": (15, 1.5),
        "No": (16, 1.5),
        "Lr": (17, 1.5),
    }
    return element_coords


def get_special_coordinates() -> dict[str, tuple[float, float]]:
    """Return special coordinates for lanthanide and actinide placeholders.

    Returns
    -------
    dict[str, tuple[float, float]]
        Placeholder strings mapped to (x, y) positions.
    """
    special_coords = {
        " * ": (3, 4.1),
        " ** ": (3, 3.7),
        "*": (2, 2.4),
        "**": (2, 1.4),
    }
    return special_coords


def make_table_fig() -> tuple[plt.Figure, plt.Axes]:
    """Create a base periodic table layout with empty element grids.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        The initialized matplotlib figure and axes.
    """
    special_coords = get_special_coordinates()
    coords = get_classic_coordinates()
    x_vals = [x for x, _ in coords.values()]
    y_vals = [y for _, y in coords.values()]

    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)

    fig, ax = plt.subplots(figsize=(x_max - x_min + 2, y_max - y_min + 2))

    for _, (x, y) in coords.items():
        rect = patches.Rectangle(
            (x - 0.5, y - 0.5),
            1,
            1,
            edgecolor="black",
            linewidth=1.3,
            facecolor="none",
        )
        ax.add_patch(rect)

    for label, (x, y) in special_coords.items():
        ax.text(x, y, label, ha="center", va="center", fontsize=22)

    ax.set_xlim(x_min - 1, x_max + 1)
    ax.set_ylim(y_min - 1, y_max + 1)
    ax.set_aspect("equal")
    ax.axis("off")

    return fig, ax


def make_heatmap(
    ax: plt.Axes,
    values: dict[str, float],
    heatmap_colors: list[str],
    cmap_name: str = "custom",
) -> None:
    """Color elements dynamically and generate a horizontal colorbar.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to receive the heatmap rendering.
    values : dict[str, float]
        Chemical element symbols mapped to target numeric intensities.
    heatmap_colors : list of str
        List of hex color strings determining the custom gradient range.
    cmap_name : str, default "custom"
        Identifier name for the generated linear colormap pipeline.
    """
    coords = get_classic_coordinates()
    cmap = LinearSegmentedColormap.from_list(cmap_name, heatmap_colors, N=256)

    vmin = min(values.values()) if values else 0
    vmax = max(values.values()) if values else 1
    norm = Normalize(vmin=vmin, vmax=vmax)

    for symbol, val in values.items():
        if symbol not in coords:
            continue
        x, y = coords[symbol]
        if val == 0:
            rect = patches.Rectangle(
                (x - 0.5, y - 0.5),
                1,
                1,
                facecolor="none",
                edgecolor="none",
                zorder=0,
            )
            ax.add_patch(rect)
            ax.text(
                x,
                y,
                symbol,
                ha="center",
                va="center",
                fontsize=25,
                color="k",
                alpha=0.5,
            )
            continue

        rgba = cmap(norm(val))
        rect = patches.Rectangle(
            (x - 0.5, y - 0.5),
            1,
            1,
            facecolor=rgba,
            edgecolor="none",
            zorder=0,
        )
        ax.add_patch(rect)

        txt_color = "w" if norm(val) > 0.74 else "k"
        ax.text(
            x,
            y,
            symbol,
            ha="center",
            va="center",
            fontsize=25,
            color=txt_color,
            alpha=1.0,
        )

    ticks = sorted(set([int(i) for i in np.linspace(vmin, vmax, 6)]))
    dummy = np.linspace(vmin, vmax, 100).reshape(1, -1)
    im = ax.imshow(dummy, extent=[0, 1, 0, 0.1], cmap=cmap, visible=False)
    cax = ax.inset_axes((0.19, 0.77, 0.4, 0.02))
    cbar = plt.colorbar(im, cax=cax, orientation="horizontal", ticks=ticks)
    cbar.ax.tick_params(labelsize=22)
    cbar.set_label("Element Count", fontsize=25, loc="center")


def get_element_list() -> list[str]:
    """Return an ordered sequence of standard IUPAC chemical element symbols.

    Returns
    -------
    list[str]
        Ordered collection of element strings up to Og.
    """
    return [
        "H",
        "He",
        "Li",
        "Be",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
        "Na",
        "Mg",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
        "K",
        "Ca",
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Ga",
        "Ge",
        "As",
        "Se",
        "Br",
        "Kr",
        "Rb",
        "Sr",
        "Y",
        "Zr",
        "Nb",
        "Mo",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "In",
        "Sn",
        "Sb",
        "Te",
        "I",
        "Xe",
        "Cs",
        "Ba",
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
        "Hf",
        "Ta",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
        "Tl",
        "Pb",
        "Bi",
        "Po",
        "At",
        "Rn",
        "Fr",
        "Ra",
        "Ac",
        "Th",
        "Pa",
        "U",
        "Np",
        "Pu",
        "Am",
        "Cm",
        "Bk",
        "Cf",
        "Es",
        "Fm",
        "Md",
        "No",
        "Lr",
        "Rf",
        "Db",
        "Sg",
        "Bh",
        "Hs",
        "Mt",
        "Ds",
        "Rg",
        "Cn",
        "Nh",
        "Fl",
        "Mc",
        "Lv",
        "Ts",
        "Og",
    ]


def generate_periodic_table_heatmap(
    input_xlsx: str,
    cache_xlsx: str = "elements_sorted_element_count.xlsx",
    output_png: str = "new.png",
    heatmap_colors: list[str] | None = None,
) -> None:
    """Extract formula counts from Excel and generate a periodic table heatmap image.

    Parameters
    ----------
    input_xlsx : str
        Source file pathway pointing to raw compositional Excel tables.
    cache_xlsx : str, default "elements_sorted_element_count.xlsx"
        Storage path target for parsed index tracking datasets.
    output_png : str, default "new.png"
        Target output file path for the completed heatmap image structure.
    heatmap_colors : list of str, optional
        List of hex color strings mapping low-to-high colors. Defaults to your original teal theme.
    """
    skip_elements = {
        "Fr",
        "Ra",
        "Rf",
        "Db",
        "Sg",
        "Bh",
        "Hs",
        "Mt",
        "Ds",
        "Rg",
        "Cn",
        "Nh",
        "Fl",
        "Mc",
        "Lv",
        "Ts",
        "Og",
    }

    if heatmap_colors is None:
        heatmap_colors = ["#00A5CF", "#004E64"]

    if not os.path.exists(cache_xlsx):
        df = pd.read_excel(input_xlsx)

        elem_counter = Counter()
        for formula in df["Formula"]:
            elem_counter.update(Composition(formula).as_dict().keys())

        elem_tracker = pd.DataFrame.from_dict(
            elem_counter, orient="index", columns=["# Element"]
        ).sort_values("# Element", ascending=False)
        elem_tracker.index.name = "Element"
        elem_tracker.to_excel(cache_xlsx)

    elem_tracker = pd.read_excel(cache_xlsx, index_col=0)

    _, ax = make_table_fig()

    tracker_dict = elem_tracker["# Element"].to_dict()

    values = {
        symbol: tracker_dict.get(symbol, 0)
        for symbol in get_element_list()
        if symbol not in skip_elements
    }

    make_heatmap(ax, values, heatmap_colors=heatmap_colors)
    plt.savefig(output_png, format="png", bbox_inches="tight", dpi=300)
