"""Figure-construction helpers shared across the plotting modules.

Lifted out of the former top-level ``utils.py``, which was a catch-all holding
four live functions for three different packages and three dead ones. These two
are the figure half; the config half moved into ``config/esn_config.py``, and
`interpolate` (a duplicate of `dynamodels.utils.interpolate`) and
`save_figs_to_pdf` (unused) were dropped.

Deliberately imports nothing from `models` or `datasets`: `plotting.pod` reads
`models.data_driven.autoencoders`, so anything here that reached back the other
way would close an import cycle.

`add_pdf_page` and `get_figsize_based_on_domain` originate from A. Nóvoa's
real-time bias-aware DA repository.
"""

import matplotlib.backends.backend_pdf as plt_pdf
import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plt_pdf", "add_pdf_page", "get_figsize_based_on_domain"]


def add_pdf_page(pdf, fig_to_add, close_figs=True):
    """Append one figure to an open `plt_pdf.PdfPages` and close it."""
    pdf.savefig(fig_to_add)
    if close_figs:
        plt.close(fig_to_add)


def get_figsize_based_on_domain(domain, total_subplots, max_cols=5, total_width=6):
    """
    Returns ((fig_width, fig_height), ncols, nrows) where figsize respects the domain
    aspect ratio and total_width constraint across the subplot grid.

    Parameters:
    - domain: [xmin, xmax, ymin, ymax]
    - total_subplots: number of subplots to arrange
    - max_cols: maximum columns (prevents excessively wide layouts)
    - total_width: desired figure width in inches

    Returns:
    - Tuple: ((fig_width, fig_height), ncols, nrows)
    """
    x_span = abs(domain[1] - domain[0])
    y_span = abs(domain[3] - domain[2])
    aspect_ratio = y_span / x_span if x_span != 0 else 1

    ncols = min(max_cols, total_subplots)
    nrows = int(np.ceil(total_subplots / ncols))

    per_subplot_width = total_width / ncols
    per_subplot_height = per_subplot_width * aspect_ratio

    fig_width = total_width
    fig_height = per_subplot_height * nrows

    return (fig_width, fig_height), ncols, nrows
