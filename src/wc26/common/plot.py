def setup_matplotlib_params():
    import matplotlib

    matplotlib.rcParams.update({
        "font.family": "Arial",
        "pdf.fonttype": 42,
        # "font.size": 5,          # default text size
        # "axes.labelsize": 5,     # x/y labels
        # "xtick.labelsize": 5,    # x tick labels
        # "ytick.labelsize": 5,    # y tick labels
        # "legend.fontsize": 5,    # legend text
        # "axes.titlesize": 6,     # axes title
        # "figure.titlesize": 6,   # suptitle
    })

palette = ["#546a76", "#79a63d"]