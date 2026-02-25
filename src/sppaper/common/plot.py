def setup_matplotlib_params():
    import matplotlib

    matplotlib.rcParams.update(
        {
            # Font type
            "font.family": "Arial",
            "pdf.fonttype": 42,
            # Font sizes
            "font.size": 6,  # default text size
            "axes.labelsize": 6,  # x/y labels
            "xtick.labelsize": 5,  # x tick labels
            "ytick.labelsize": 5,  # y tick labels
            "legend.fontsize": 5,  # legend text
            "axes.titlesize": 6,  # axes title
            "figure.titlesize": 7,  # suptitle
            # Line dimensions
            "lines.linewidth": 0.75,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2,
            "ytick.major.size": 2,
            "patch.linewidth": 0.5,  # used for border of the legend frame
        }
    )


palette = ["#546a76", "#79a63d"]


def find_font_path(family, weight="normal", style="normal"):
    """
    Find the file path of a font given its family, weight, and style.

    Args:
        family: Font family name (e.g., "Arial").
        weight: Font weight (e.g., "normal", "bold").
        style:  Font style (e.g., "normal", "italic").

    Returns:
        The file path of the matching font, or None if not found.
    """
    import matplotlib.font_manager as fm

    font_props = fm.FontProperties(family=family, weight=weight, style=style)
    font_path = fm.findfont(font_props)
    return font_path
