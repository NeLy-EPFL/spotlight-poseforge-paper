def setup_matplotlib_params():
    import matplotlib

    matplotlib.rcParams.update({
        # Font type
        "font.family": "Arial",
        "pdf.fonttype": 42,
        # Font sizes
        "font.size": 6,          # default text size
        "axes.labelsize": 6,     # x/y labels
        "xtick.labelsize": 5,    # x tick labels
        "ytick.labelsize": 5,    # y tick labels
        "legend.fontsize": 5,    # legend text
        "axes.titlesize": 6,     # axes title
        "figure.titlesize": 7,   # suptitle
        # Line dimensions
        "lines.linewidth": 0.75,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "patch.linewidth": 0.5,  # used for border of the legend frame
    })

palette = ["#546a76", "#79a63d"]