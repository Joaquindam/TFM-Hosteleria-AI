from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure


TimeAggregation = Literal[
    'count',
    'nunique',
    'sum',
    'mean',
    'median',
]

ValueAggregation = Literal[
    'sum',
    'mean',
    'median',
]


__all__ = [
    'plot_time_series',
    'plot_distribution',
    'plot_category_counts',
    'plot_aggregated_bars',
    'plot_scatter',
    'plot_missing_values',
]


def _validate_dataframe(
    dataframe: pd.DataFrame,
) -> None:
    '''
    Validate that the input object is a non-empty pandas DataFrame.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame to validate.

    Raises
    ------
    TypeError
        If dataframe is not a pandas DataFrame.
    ValueError
        If dataframe is empty.
    '''
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            'dataframe must be a pandas DataFrame.'
        )

    if dataframe.empty:
        raise ValueError(
            'The DataFrame is empty.'
        )


def _validate_columns(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> None:
    '''
    Validate that all required columns exist in a DataFrame.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame to validate.
    columns : list[str]
        Required column names.

    Raises
    ------
    ValueError
        If one or more required columns are missing.
    '''
    missing_columns = [
        column
        for column in columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            'Missing required columns: '
            + ', '.join(missing_columns)
        )


def _validate_top_n(
    top_n: int | None,
) -> None:
    '''
    Validate a top_n plotting parameter.

    Parameters
    ----------
    top_n : int or None
        Maximum number of categories to display.

    Raises
    ------
    ValueError
        If top_n is not greater than zero.
    '''
    if top_n is not None and top_n <= 0:
        raise ValueError(
            'top_n must be greater than zero.'
        )


def _column_label(
    column: str,
) -> str:
    '''
    Convert a snake_case column name to a readable plot label.

    Parameters
    ----------
    column : str
        Column name.

    Returns
    -------
    str
        Human-readable column label.
    '''
    return (
        column
        .replace('_', ' ')
        .strip()
        .title()
    )


def _finalize_plot(
    fig: Figure,
    save_path: str | Path | None,
) -> None:
    '''
    Apply the final layout and optionally save a figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to finalize.
    save_path : str or pathlib.Path or None
        Path where the figure will be saved.
    '''
    fig.tight_layout()

    if save_path is None:
        return

    path = Path(save_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches='tight',
    )


def plot_time_series(
    dataframe: pd.DataFrame,
    date_column: str,
    value_column: str | None = None,
    aggregation: TimeAggregation = 'count',
    frequency: str = 'D',
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    marker: bool = False,
    figsize: tuple[float, float] = (12, 6),
    save_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    '''
    Plot an aggregated time series from a standardized DataFrame.

    The function groups observations using a pandas time frequency and
    applies the selected aggregation.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input dataset.
    date_column : str
        Column containing dates or datetimes.
    value_column : str, optional
        Numeric or identifier column to aggregate. It is required for all
        aggregations except 'count'.
    aggregation : {'count', 'nunique', 'sum', 'mean', 'median'},
        default='count'
        Aggregation applied inside each time period.
    frequency : str, default='D'
        Pandas time frequency used to group the data.
    title : str, optional
        Plot title.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    marker : bool, default=False
        If True, draw a marker at every time-series observation.
    figsize : tuple[float, float], default=(12, 6)
        Figure size.
    save_path : str or pathlib.Path, optional
        Path where the figure will be saved.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Matplotlib figure and axes.
    '''
    _validate_dataframe(
        dataframe
    )

    if aggregation not in {
        'count',
        'nunique',
        'sum',
        'mean',
        'median',
    }:
        raise ValueError(
            f'Unsupported aggregation: {aggregation}'
        )

    if (
        aggregation != 'count'
        and value_column is None
    ):
        raise ValueError(
            'value_column is required when aggregation '
            'is not count.'
        )

    required_columns = [
        date_column,
    ]

    if value_column is not None:
        required_columns.append(
            value_column
        )

    _validate_columns(
        dataframe,
        required_columns,
    )

    data = dataframe[
        required_columns
    ].copy()

    data[date_column] = pd.to_datetime(
        data[date_column],
        errors='coerce',
    )

    data = data.dropna(
        subset=[
            date_column,
        ],
    )

    if data.empty:
        raise ValueError(
            f'No valid dates found in column: '
            f'{date_column}'
        )

    grouper = pd.Grouper(
        key=date_column,
        freq=frequency,
    )

    if aggregation == 'count':
        series = (
            data
            .groupby(grouper)
            .size()
        )

    elif aggregation == 'nunique':
        series = (
            data
            .groupby(grouper)[value_column]
            .nunique()
        )

    else:
        data[value_column] = pd.to_numeric(
            data[value_column],
            errors='coerce',
        )

        data = data.dropna(
            subset=[
                value_column,
            ],
        )

        if data.empty:
            raise ValueError(
                f'No valid numeric values found in column: '
                f'{value_column}'
            )

        series = (
            data
            .groupby(grouper)[value_column]
            .agg(aggregation)
        )

    series = series.sort_index()

    fig, ax = plt.subplots(
        figsize=figsize,
    )

    ax.plot(
        series.index,
        series.values,
        marker='o' if marker else None,
    )

    ax.set_title(
        title or 'Time series',
    )

    ax.set_xlabel(
        xlabel or _column_label(
            date_column
        ),
    )

    if aggregation == 'count':
        default_ylabel = 'Count'

    else:
        default_ylabel = _column_label(
            value_column
        )

    ax.set_ylabel(
        ylabel or default_ylabel,
    )

    ax.grid(
        axis='y',
        alpha=0.25,
    )

    _finalize_plot(
        fig,
        save_path,
    )

    return fig, ax


def plot_distribution(
    dataframe: pd.DataFrame,
    column: str,
    bins: int = 30,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str = 'Frequency',
    figsize: tuple[float, float] = (10, 6),
    save_path: str | Path | None = None,
    yscale: Literal['linear', 'log'] = 'log',
) -> tuple[Figure, Axes]:
    '''
    Plot the distribution of a numeric column as a histogram.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input dataset.
    column : str
        Numeric column to plot.
    bins : int, default=30
        Number of histogram bins.
    title : str, optional
        Plot title.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, default='Frequency'
        Label for the y-axis.
    figsize : tuple[float, float], default=(10, 6)
        Figure size.
    save_path : str or pathlib.Path, optional
        Path where the figure will be saved.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Matplotlib figure and axes.
    '''
    _validate_dataframe(
        dataframe
    )

    _validate_columns(
        dataframe,
        [
            column,
        ],
    )

    if bins <= 0:
        raise ValueError(
            'bins must be greater than zero.'
        )

    values = pd.to_numeric(
        dataframe[column],
        errors='coerce',
    ).dropna()

    if values.empty:
        raise ValueError(
            f'No valid numeric values found in column: '
            f'{column}'
        )

    fig, ax = plt.subplots(
        figsize=figsize,
    )

    ax.hist(
        values,
        bins=bins,
    )

    ax.set_title(
        title
        or (
            f'Distribution of '
            f'{_column_label(column)}'
        ),
    )

    ax.set_xlabel(
        xlabel or _column_label(
            column
        ),
    )

    ax.set_ylabel(
        ylabel,
    )

    ax.set_yscale(yscale)

    ax.grid(
        axis='y',
        alpha=0.25,
    )

    _finalize_plot(
        fig,
        save_path,
    )

    return fig, ax


def plot_category_counts(
    dataframe: pd.DataFrame,
    category_column: str,
    top_n: int | None = None,
    include_missing: bool = False,
    horizontal: bool = True,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] = (10, 6),
    save_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    '''
    Plot the number of observations in each category.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input dataset.
    category_column : str
        Categorical column to count.
    top_n : int, optional
        Maximum number of categories to display.
    include_missing : bool, default=False
        If True, include missing values as a separate category.
    horizontal : bool, default=True
        If True, create a horizontal bar chart.
    title : str, optional
        Plot title.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    figsize : tuple[float, float], default=(10, 6)
        Figure size.
    save_path : str or pathlib.Path, optional
        Path where the figure will be saved.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Matplotlib figure and axes.
    '''
    _validate_dataframe(
        dataframe
    )

    _validate_columns(
        dataframe,
        [
            category_column,
        ],
    )

    _validate_top_n(
        top_n
    )

    categories = dataframe[
        category_column
    ].astype('string')

    if include_missing:
        categories = categories.fillna(
            'Missing'
        )

    else:
        categories = categories.dropna()

    categories = (
        categories
        .str.strip()
    )

    categories = categories[
        categories != ''
    ]

    counts = categories.value_counts()

    if top_n is not None:
        counts = counts.head(
            top_n
        )

    if counts.empty:
        raise ValueError(
            f'No valid categories found in column: '
            f'{category_column}'
        )

    fig, ax = plt.subplots(
        figsize=figsize,
    )

    if horizontal:
        counts = counts.sort_values()

        ax.barh(
            counts.index.astype(str),
            counts.values,
        )

        ax.set_xlabel(
            xlabel or 'Count',
        )

        ax.set_ylabel(
            ylabel
            or _column_label(
                category_column
            ),
        )

        ax.grid(
            axis='x',
            alpha=0.25,
        )

    else:
        ax.bar(
            counts.index.astype(str),
            counts.values,
        )

        ax.set_xlabel(
            xlabel
            or _column_label(
                category_column
            ),
        )

        ax.set_ylabel(
            ylabel or 'Count',
        )

        ax.tick_params(
            axis='x',
            rotation=45,
        )

        ax.grid(
            axis='y',
            alpha=0.25,
        )

    ax.set_title(
        title
        or (
            f'Counts by '
            f'{_column_label(category_column)}'
        ),
    )

    _finalize_plot(
        fig,
        save_path,
    )

    return fig, ax


def plot_aggregated_bars(
    dataframe: pd.DataFrame,
    category_column: str,
    value_column: str,
    aggregation: ValueAggregation = 'sum',
    top_n: int | None = 10,
    horizontal: bool = True,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] = (10, 6),
    save_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    '''
    Aggregate a numeric variable by category and plot the result as bars.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input dataset.
    category_column : str
        Categorical column used to group the data.
    value_column : str
        Numeric column to aggregate.
    aggregation : {'sum', 'mean', 'median'}, default='sum'
        Aggregation applied to each category.
    top_n : int, optional, default=10
        Maximum number of categories to display. If None, display all.
    horizontal : bool, default=True
        If True, create a horizontal bar chart.
    title : str, optional
        Plot title.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    figsize : tuple[float, float], default=(10, 6)
        Figure size.
    save_path : str or pathlib.Path, optional
        Path where the figure will be saved.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Matplotlib figure and axes.
    '''
    _validate_dataframe(
        dataframe
    )

    _validate_columns(
        dataframe,
        [
            category_column,
            value_column,
        ],
    )

    _validate_top_n(
        top_n
    )

    if aggregation not in {
        'sum',
        'mean',
        'median',
    }:
        raise ValueError(
            f'Unsupported aggregation: {aggregation}'
        )

    data = dataframe[
        [
            category_column,
            value_column,
        ]
    ].copy()

    data[category_column] = (
        data[category_column]
        .astype('string')
        .str.strip()
    )

    data[value_column] = pd.to_numeric(
        data[value_column],
        errors='coerce',
    )

    data = data.dropna(
        subset=[
            category_column,
            value_column,
        ],
    )

    data = data[
        data[category_column] != ''
    ]

    if data.empty:
        raise ValueError(
            'No valid category and numeric value '
            'pairs found.'
        )

    values = (
        data
        .groupby(
            category_column
        )[value_column]
        .agg(
            aggregation
        )
        .sort_values(
            ascending=False
        )
    )

    if top_n is not None:
        values = values.head(
            top_n
        )

    fig, ax = plt.subplots(
        figsize=figsize,
    )

    if horizontal:
        values = values.sort_values()

        ax.barh(
            values.index.astype(str),
            values.values,
        )

        ax.set_xlabel(
            xlabel
            or _column_label(
                value_column
            ),
        )

        ax.set_ylabel(
            ylabel
            or _column_label(
                category_column
            ),
        )

        ax.grid(
            axis='x',
            alpha=0.25,
        )

    else:
        ax.bar(
            values.index.astype(str),
            values.values,
        )

        ax.set_xlabel(
            xlabel
            or _column_label(
                category_column
            ),
        )

        ax.set_ylabel(
            ylabel
            or _column_label(
                value_column
            ),
        )

        ax.tick_params(
            axis='x',
            rotation=45,
        )

        ax.grid(
            axis='y',
            alpha=0.25,
        )

    ax.set_title(
        title
        or (
            f'{aggregation.title()} of '
            f'{_column_label(value_column)} by '
            f'{_column_label(category_column)}'
        ),
    )

    _finalize_plot(
        fig,
        save_path,
    )

    return fig, ax


def plot_scatter(
    dataframe: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    alpha: float = 0.6,
    figsize: tuple[float, float] = (10, 6),
    save_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    '''
    Plot the relationship between two numeric variables.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input dataset.
    x_column : str
        Numeric column displayed on the x-axis.
    y_column : str
        Numeric column displayed on the y-axis.
    title : str, optional
        Plot title.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    alpha : float, default=0.6
        Point transparency between 0 and 1.
    figsize : tuple[float, float], default=(10, 6)
        Figure size.
    save_path : str or pathlib.Path, optional
        Path where the figure will be saved.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Matplotlib figure and axes.
    '''
    _validate_dataframe(
        dataframe
    )

    _validate_columns(
        dataframe,
        [
            x_column,
            y_column,
        ],
    )

    if not 0 <= alpha <= 1:
        raise ValueError(
            'alpha must be between 0 and 1.'
        )

    data = dataframe[
        [
            x_column,
            y_column,
        ]
    ].copy()

    data[x_column] = pd.to_numeric(
        data[x_column],
        errors='coerce',
    )

    data[y_column] = pd.to_numeric(
        data[y_column],
        errors='coerce',
    )

    data = data.dropna()

    if data.empty:
        raise ValueError(
            'No valid numeric pairs found for '
            'the scatter plot.'
        )

    fig, ax = plt.subplots(
        figsize=figsize,
    )

    ax.scatter(
        data[x_column],
        data[y_column],
        alpha=alpha,
    )

    ax.set_title(
        title
        or (
            f'{_column_label(y_column)} vs '
            f'{_column_label(x_column)}'
        ),
    )

    ax.set_xlabel(
        xlabel
        or _column_label(
            x_column
        ),
    )

    ax.set_ylabel(
        ylabel
        or _column_label(
            y_column
        ),
    )

    ax.grid(
        alpha=0.25,
    )

    _finalize_plot(
        fig,
        save_path,
    )

    return fig, ax


def plot_missing_values(
    dataframe: pd.DataFrame,
    top_n: int | None = None,
    include_zero: bool = False,
    title: str = 'Missing values by column',
    figsize: tuple[float, float] = (10, 6),
    save_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    '''
    Plot the percentage of missing values in each DataFrame column.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input dataset.
    top_n : int, optional
        Maximum number of columns to display.
    include_zero : bool, default=False
        If True, include columns without missing values.
    title : str, default='Missing values by column'
        Plot title.
    figsize : tuple[float, float], default=(10, 6)
        Figure size.
    save_path : str or pathlib.Path, optional
        Path where the figure will be saved.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Matplotlib figure and axes.
    '''
    _validate_dataframe(
        dataframe
    )

    _validate_top_n(
        top_n
    )

    missing = (
        dataframe
        .isna()
        .mean()
        .mul(100)
        .sort_values(
            ascending=False
        )
    )

    if not include_zero:
        missing = missing[
            missing > 0
        ]

    if top_n is not None:
        missing = missing.head(
            top_n
        )

    if missing.empty:
        raise ValueError(
            'No missing values available to plot.'
        )

    missing = missing.sort_values()

    fig, ax = plt.subplots(
        figsize=figsize,
    )

    bars = ax.barh(
        missing.index.astype(str),
        missing.values,
    )

    ax.bar_label(
        bars,
        fmt='%.1f%%',
        padding=3,
    )

    ax.set_title(
        title,
    )

    ax.set_xlabel(
        'Missing values (%)',
    )

    ax.set_ylabel(
        'Column',
    )

    ax.grid(
        axis='x',
        alpha=0.25,
    )

    _finalize_plot(
        fig,
        save_path,
    )

    return fig, ax