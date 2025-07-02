"""
Plotting utilities for ElasticNet interpretation in sk-neuro
"""
import matplotlib.pyplot as plt
import seaborn as sns



def plot_betas_heatmap(
    data2plot: dict,
    fname2save: str,
    dpi: int,
    regions: list[str],
    conditions: list[str],
    condition_names: list[str]
) -> None:
    """
    Plot heatmaps of interpreted betas by condition and region.
    Args:
        data2plot: Dict from interpret_betas_by_region.
        fname2save: Path to save the figure.
        dpi: DPI for saving.
        regions: List of region names.
        conditions: List of condition codes.
        condition_names: List of subplot titles for each condition.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, axn = plt.subplots(len(conditions), 1, figsize=(27, 7), sharex=True, sharey=True)
    cbar_ax = fig.add_axes([.9, .3, .03, .4])
    for i, ax in enumerate(axn.flat):
        sns.heatmap(data2plot[f'con_{conditions[i]}'].T, linewidths=0.2, square=True,
                    vmax=0.1, vmin=-0.1, cmap='RdBu_r',
                    yticklabels=regions, ax=ax,
                    cbar=i == 0,
                    cbar_ax=None if i else cbar_ax)
        ax.set_title(condition_names[i])
    fig.tight_layout(rect=[0, 0, .9, 1])
    plt.savefig(fname2save, dpi=dpi)
