import anndata as ad
import numpy as np
import pandas as pd

from ._getRegionComposition import getRegionComposition


def getSlidingWindowsComposition(
    adata,
    window_size,
    step_size,
    phenotype_col,
    region_col="region",
    region_subset=None,
    min_cells=0,
):
    """
    Get Sliding window cell composition for cells in region subset.
    :param adata: Anndata object
    :param window_size: Size of the sliding window
    :param step_size: Size of the step
    :param phenotype_col: list of columns containing the cell type markers,
        for cell type composition
    :param region_col: Column containing the region information
    :param region_subset: List of regions to consider. If None, consider all cells.
    :param min_cells: Minimum number of cells in a window to consider it
    :return: A dataframe containing the cell type composition of the region in each window
    """
    if region_subset is None:
        cells_roi = adata.copy()
    else:
        cells_roi = adata[adata.obs[region_col].isin(region_subset)].copy()
    cells_roi_maxx = int(cells_roi.obs["X_centroid"].max())
    cells_roi_maxy = int(cells_roi.obs["Y_centroid"].max())
    cells_roi_minx = int(cells_roi.obs["X_centroid"].min())
    cells_roi_miny = int(cells_roi.obs["Y_centroid"].min())
    all_windows_comp_df = []
    # speed up when window size == step size
    if window_size == step_size:
        cells_roi.obs["X_start"] = (
            cells_roi.obs["X_centroid"] - 
            (cells_roi.obs["X_centroid"] - cells_roi_minx) % step_size
        ).astype(int)
        cells_roi.obs["Y_start"] = (
            cells_roi.obs["Y_centroid"] - 
            (cells_roi.obs["Y_centroid"] - cells_roi_miny) % step_size
        ).astype(int)
        all_windows_comp_df = cells_roi.obs.groupby(
            ["X_start", "Y_start"]
        ).filter(
            lambda x: x.shape[0] > min_cells
        ).groupby(
            ["X_start", "Y_start"]
        ).apply(
            lambda obs: getRegionComposition(
                ad.AnnData(None, obs), phenotype_col, region_subset, region_col
            ), include_groups=False
        ).reset_index()
        all_windows_comp_df["window_size"] = window_size
        all_windows_comp_df["step_size"] = step_size
        return all_windows_comp_df

    for x in range(cells_roi_minx, cells_roi_maxx + window_size, step_size):
        for y in range(cells_roi_miny, cells_roi_maxy + window_size, step_size):
            cells_roi_window = cells_roi[
                (cells_roi.obs["X_centroid"] >= x)
                & (cells_roi.obs["X_centroid"] < x + window_size)
                & (cells_roi.obs["Y_centroid"] >= y)
                & (cells_roi.obs["Y_centroid"] < y + window_size)
            ]
            if cells_roi_window.shape[0] > min_cells:
                cells_roi_window_composition = getRegionComposition(
                    cells_roi_window, phenotype_col, region_subset, region_col
                )
                cells_roi_window_composition["X_start"] = x
                cells_roi_window_composition["Y_start"] = y
                cells_roi_window_composition["window_size"] = window_size
                cells_roi_window_composition["step_size"] = step_size
                all_windows_comp_df.append(cells_roi_window_composition)
    all_windows_comp_df = pd.concat(all_windows_comp_df)
    return all_windows_comp_df


def get_comp_mask(df, pheno_col, pheno_vals, step_size):
    """
    Get a mask of the composition of the region in each window

    :param df: A dataframe containing the cell type composition of pheno_vals in each window
    :param pheno_col: Column containing the cell type information
    :param pheno_vals: List of cell types to consider
    :param step_size: Size of the step
    :return: A np array mask of the composition of the region in each window
    """
    maxx, maxy = df["X_start"].max() + step_size, df["Y_start"].max() + step_size
    mask = np.zeros((maxy + 2000, maxx + 2000))
    df1 = df[df[pheno_col].isin(pheno_vals)]
    for i in range(len(df1)):
        x = int(df1.iloc[i]["X_start"])
        y = int(df1.iloc[i]["Y_start"])
        mask[y : y + step_size, x : x + step_size] = df1.iloc[i]["composition"]
    return mask
