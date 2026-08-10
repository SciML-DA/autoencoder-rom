# -*- coding: utf-8 -*-
"""
Created on Wed May 11 09:45:48 2022

@author: Andrea Nóvoa @andrea_novoa
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as plt_pdf

from scipy.interpolate import interp1d

from typing import Optional


def find_first_ascending_folder(start_dir, target_names):
    """
    Ascends from start_dir looking for any of the target_names folders.
    Returns (parent_path, found_folder), or (None, None) if not found.

    # Usage Example:
    parent, found = find_first_ascending_folder('.', ['src', 'dev'])
    if parent:
        print(f"Found {found} in {parent}")
    """
    dir_path = os.path.abspath(start_dir)
    while True:
        existing = [name for name in target_names if name in os.listdir(dir_path)]
        if existing:
            return dir_path, existing[0]
        parent = os.path.dirname(dir_path)
        if parent == dir_path:
            return None, False
        dir_path = parent


def get_project_root( root='.'):
    """Return the project root directory."""

    project_root, found = find_first_ascending_folder(root, ['src', 'dev']) 
    if found == 'dev':
        
        project_root = f'{project_root}/real_public'
        print('On dev folder, root=' , project_root)
        
    elif not found:
        raise FileNotFoundError("Project root directory not found. Ensure you are in the correct directory structure.")
    
    return project_root


def convert_to_python_type(obj, *, float_ndigits=12):
    """Convert numpy types to native Python types, with canonical float rounding."""
    if obj is None:
        return "none"

    if isinstance(obj, np.generic):
        if np.issubdtype(type(obj), np.integer):
            return int(obj)
        elif np.issubdtype(type(obj), np.floating):
            return round(float(obj), float_ndigits)
        elif np.issubdtype(type(obj), np.bool_):
            return bool(obj)
        elif np.issubdtype(type(obj), np.complexfloating):
            c = complex(obj)
            return (round(c.real, float_ndigits), round(c.imag, float_ndigits))
        else:
            return obj.item()

    elif isinstance(obj, float):
        return round(obj, float_ndigits)

    elif isinstance(obj, np.ndarray):
        return [convert_to_python_type(x, float_ndigits=float_ndigits) for x in obj.tolist()]
    elif isinstance(obj, tuple):
        return [convert_to_python_type(item, float_ndigits=float_ndigits) for item in obj]
    elif isinstance(obj, list):
        return [convert_to_python_type(item, float_ndigits=float_ndigits) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_python_type(v, float_ndigits=float_ndigits) for k, v in obj.items()}

    # if Path, change to sttring
    elif isinstance(obj, os.PathLike):
        return str(obj)

    return obj


def interpolate(t_y, y, t_eval, fill_values: Optional[tuple[float, float]] = None):
    # interpolator = PchipInterpolator(t_y, y)

    if fill_values is None:
        fill_values = (y[0], y[-1])

    interpolator = interp1d(t_y, y,
                            axis=0,  # interpolate along columns
                            bounds_error=False,
                            kind='linear',
                            fill_value=fill_values # type: ignore #tuple[float, float]
                            )
    return interpolator(t_eval)


def save_figs_to_pdf(pdf_name, figs=None):

    pdf_file = plt_pdf.PdfPages(pdf_name)
    if figs is None:
        figs = [plt.figure(ii) for ii in plt.get_fignums()]
    elif not isinstance(figs, list):
        figs = [figs]

    for fig in figs:
        pdf_file.savefig(fig, dpi=300)  # Save figure to PDF
        plt.close(fig)

    pdf_file.close()  # Close results pdf


def add_pdf_page(pdf, fig_to_add, close_figs=True):
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
