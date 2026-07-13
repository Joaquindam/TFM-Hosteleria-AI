from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd


SUPPORTED_EXCEL_EXTENSIONS = {'.xls', '.xlsx'}

REPORT_DATE_PATTERN = re.compile(r'(\d{2}/\d{2}/\d{4})')
ARTICLE_SALES_SUMMARY_STEMS = {'total_articulos'}


def _normalize_text(value: Any) -> str:
    '''
    Normalize a value for robust text comparisons.

    Parameters
    ----------
    value : Any
        Value to normalize.

    Returns
    -------
    str
        Lowercase text without accents or repeated spaces.
    '''
    if pd.isna(value):
        return ''

    text = unicodedata.normalize('NFKD', str(value))
    text = ''.join(
        character
        for character in text
        if not unicodedata.combining(character)
    )
    text = re.sub(r'\s+', ' ', text)

    return text.strip().lower()


def _normalize_column_name(value: Any) -> str:
    '''
    Convert an Excel column name to lowercase snake_case.

    Parameters
    ----------
    value : Any
        Original column name.

    Returns
    -------
    str
        Normalized column name.
    '''
    text = _normalize_text(value)
    text = re.sub(r'[^a-z0-9]+', '_', text)

    return text.strip('_')


def _parse_report_date(value: Any) -> pd.Timestamp:
    '''
    Extract a DD/MM/YYYY date from a report text cell.

    Parameters
    ----------
    value : Any
        Cell value containing a date.

    Returns
    -------
    pandas.Timestamp or pandas.NaT
        Parsed date.
    '''
    match = REPORT_DATE_PATTERN.search(str(value))

    if match is None:
        return pd.NaT

    return pd.Timestamp(
        datetime.strptime(match.group(1), '%d/%m/%Y')
    )


def _parse_excel_date(value: Any) -> pd.Timestamp:
    '''
    Convert a scalar Excel date value to pandas Timestamp.

    Parameters
    ----------
    value : Any
        Excel date value.

    Returns
    -------
    pandas.Timestamp or pandas.NaT
        Parsed date.
    '''
    if pd.isna(value):
        return pd.NaT

    if isinstance(value, pd.Timestamp):
        return value.normalize()

    if isinstance(value, datetime):
        return pd.Timestamp(value).normalize()

    if isinstance(value, date):
        return pd.Timestamp(value)

    text = str(value).strip()

    for date_format in (
        '%d/%m/%Y',
        '%Y-%m-%d',
        '%Y-%m-%d %H:%M:%S',
    ):
        try:
            return pd.Timestamp(
                datetime.strptime(text, date_format)
            ).normalize()
        except ValueError:
            continue

    return pd.NaT


def _extract_value_after_colon(value: Any) -> str | None:
    '''
    Return the text located after the first colon in a cell.

    Parameters
    ----------
    value : Any
        Cell value.

    Returns
    -------
    str or None
        Text after the colon.
    '''
    text = str(value)

    if ':' not in text:
        return None

    result = text.split(':', maxsplit=1)[1].strip()

    return result or None


def _extract_report_metadata(
    raw: pd.DataFrame,
) -> dict[str, Any]:
    '''
    Extract common metadata from the exported POS reports.

    Parameters
    ----------
    raw : pandas.DataFrame
        Excel sheet read without interpreting any row as a header.

    Returns
    -------
    dict[str, Any]
        Report dates, terminal information and shift.
    '''
    metadata: dict[str, Any] = {
        'report_start': pd.NaT,
        'report_end': pd.NaT,
        'report_generated_on': pd.NaT,
        'terminal_start': None,
        'terminal_end': None,
        'turn': None,
    }

    for value in raw.to_numpy().ravel():
        if pd.isna(value):
            continue

        normalized = _normalize_text(value)

        if normalized.startswith('terminal inicial'):
            metadata['terminal_start'] = (
                _extract_value_after_colon(value)
            )

        elif normalized.startswith('terminal final'):
            metadata['terminal_end'] = (
                _extract_value_after_colon(value)
            )

        elif normalized.startswith('fecha inicial'):
            metadata['report_start'] = (
                _parse_report_date(value)
            )

        elif normalized.startswith('fecha final'):
            metadata['report_end'] = (
                _parse_report_date(value)
            )

        elif re.match(r'^fecha\s*:', normalized):
            metadata['report_generated_on'] = (
                _parse_report_date(value)
            )

        elif normalized.startswith('turno'):
            metadata['turn'] = (
                _extract_value_after_colon(value)
            )

    return metadata


def _read_excel_raw(
    file_path: str | Path,
) -> pd.DataFrame:
    '''
    Read an Excel file without assuming where the header is located.

    Old-style .xls files are read with the calamine engine because some
    source files contain OLE workbook inconsistencies that xlrd cannot
    parse. Modern .xlsx files are read with openpyxl.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to an .xls or .xlsx file.

    Returns
    -------
    pandas.DataFrame
        Raw Excel contents.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file extension is not supported.
    ImportError
        If the required Excel engine is not installed.
    '''
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f'File not found: {path}'
        )

    if path.suffix.lower() not in SUPPORTED_EXCEL_EXTENSIONS:
        raise ValueError(
            f'Unsupported file extension: {path.suffix}. '
            f'Expected one of '
            f'{sorted(SUPPORTED_EXCEL_EXTENSIONS)}.'
        )

    if path.suffix.lower() == '.xls':
        try:
            return pd.read_excel(
                path,
                header=None,
                dtype=object,
                engine='calamine',
            )

        except ImportError as exc:
            raise ImportError(
                'Reading .xls files requires python-calamine. '
                'Install it with: '
                'python -m pip install python-calamine'
            ) from exc

    return pd.read_excel(
        path,
        header=None,
        dtype=object,
        engine='openpyxl',
    )


def _detect_file_type(
    raw: pd.DataFrame,
) -> str:
    '''
    Detect the dataset type from the contents of an Excel file.

    Parameters
    ----------
    raw : pandas.DataFrame
        Excel sheet read without a header.

    Returns
    -------
    str
        Detected file type.

    Raises
    ------
    ValueError
        If the file structure is not recognized.
    '''
    preview = raw.iloc[:30]

    normalized_cells = {
        _normalize_text(value)
        for value in preview.to_numpy().ravel()
        if pd.notna(value)
    }

    if 'articulos x departamentos venta' in normalized_cells:
        return 'article_sales'

    if 'documentos con emision de comprobante' in normalized_cells:
        return 'tickets'

    if 'resumen propinas' in normalized_cells:
        return 'tips'

    first_row = {
        _normalize_column_name(value)
        for value in raw.iloc[0].tolist()
        if pd.notna(value)
    }

    reservation_columns = {
        'fecha',
        'hora',
        'estado',
        'turno',
        'personas',
    }

    article_columns = {
        'articulo',
        'descripcion',
        'descripcion_abreviada',
        'cod_departamento_venta',
    }

    department_columns = {
        'codigo',
        'descripcion',
        'descripcion_abreviada',
    }

    menu_columns = {
        'articulo',
        'descripcion',
    }

    if reservation_columns.issubset(first_row):
        return 'reservations'

    if article_columns.issubset(first_row):
        return 'articles'

    if department_columns.issubset(first_row):
        return 'departments'

    if first_row == menu_columns:
        return 'menu'

    raise ValueError(
        'Excel format not recognized.'
    )


def _add_report_metadata(
    dataframe: pd.DataFrame,
    file_path: str | Path,
    metadata: dict[str, Any],
) -> pd.DataFrame:
    '''
    Add source and report metadata columns to a parsed dataset.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Parsed dataset.
    file_path : str or pathlib.Path
        Original file path.
    metadata : dict[str, Any]
        Report metadata.

    Returns
    -------
    pandas.DataFrame
        Dataset with source metadata.
    '''
    result = dataframe.copy()
    path = Path(file_path)

    result.insert(
        0,
        'source_file',
        path.name,
    )

    result.insert(
        1,
        'report_start',
        metadata['report_start'],
    )

    result.insert(
        2,
        'report_end',
        metadata['report_end'],
    )

    result.insert(
        3,
        'report_generated_on',
        metadata['report_generated_on'],
    )

    result.insert(
        4,
        'terminal_start',
        metadata['terminal_start'],
    )

    result.insert(
        5,
        'terminal_end',
        metadata['terminal_end'],
    )

    result.insert(
        6,
        'turn',
        metadata['turn'],
    )

    return result


def _table_from_first_row(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    '''
    Build a table using the first Excel row as normalized column names.

    Parameters
    ----------
    raw : pandas.DataFrame
        Raw Excel sheet.

    Returns
    -------
    pandas.DataFrame
        Table with normalized headers and without the original header row.
    '''
    columns = [
        _normalize_column_name(value)
        for value in raw.iloc[0].tolist()
    ]

    data = raw.iloc[1:].copy()
    data.columns = columns

    return data.reset_index(drop=True)


def _clean_string_series(
    series: pd.Series,
) -> pd.Series:
    '''
    Convert a pandas Series to stripped nullable strings.

    Parameters
    ----------
    series : pandas.Series
        Series to convert.

    Returns
    -------
    pandas.Series
        Nullable string Series without leading or trailing spaces.
    '''
    return (
        series
        .astype('string')
        .str.strip()
    )


def read_article_sales(
    file_path: str | Path,
    raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    '''
    Read an 'Artículos x Departamentos Venta' report.

    Department headers, repeated column headers, subtotal rows, blank rows
    and the final total are removed.

    One output row represents one article inside one sales department for
    the period covered by the report.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the Excel report.
    raw : pandas.DataFrame, optional
        Previously loaded raw sheet.

    Returns
    -------
    pandas.DataFrame
        Standardized article sales table.
    '''
    path = Path(file_path)

    raw = (
        _read_excel_raw(path)
        if raw is None
        else raw
    )

    metadata = _extract_report_metadata(raw)

    rows: list[dict[str, Any]] = []

    department_code: Any = pd.NA
    department_name: Any = pd.NA

    for _, row in raw.iterrows():
        code = (
            row.iloc[0]
            if len(row) > 0
            else pd.NA
        )

        description = (
            row.iloc[2]
            if len(row) > 2
            else pd.NA
        )

        units = (
            row.iloc[4]
            if len(row) > 4
            else pd.NA
        )

        amount = (
            row.iloc[6]
            if len(row) > 6
            else pd.NA
        )

        numeric_code = pd.to_numeric(
            pd.Series([code]),
            errors='coerce',
        ).iloc[0]

        numeric_units = pd.to_numeric(
            pd.Series([units]),
            errors='coerce',
        ).iloc[0]

        numeric_amount = pd.to_numeric(
            pd.Series([amount]),
            errors='coerce',
        ).iloc[0]

        has_description = (
            pd.notna(description)
            and str(description).strip() != ''
        )

        is_department = (
            pd.notna(numeric_code)
            and has_description
            and pd.isna(numeric_units)
            and pd.isna(numeric_amount)
        )

        if is_department:
            department_code = int(numeric_code)
            department_name = str(description).strip()

            continue

        is_article = (
            pd.notna(numeric_code)
            and has_description
            and pd.notna(numeric_units)
            and pd.notna(numeric_amount)
        )

        if not is_article:
            continue

        rows.append(
            {
                'department_code': department_code,
                'department_name': department_name,
                'article_code': int(numeric_code),
                'article_name': str(description).strip(),
                'units': float(numeric_units),
                'amount': float(numeric_amount),
            }
        )

    data = pd.DataFrame(
        rows,
        columns=[
            'department_code',
            'department_name',
            'article_code',
            'article_name',
            'units',
            'amount',
        ],
    )

    data['department_code'] = (
        data['department_code']
        .astype('Int64')
    )

    data['article_code'] = (
        data['article_code']
        .astype('Int64')
    )

    return _add_report_metadata(
        data,
        path,
        metadata,
    )


def read_articles(
    file_path: str | Path,
    raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    '''
    Read the article master table.

    One output row represents one article registered in the POS system.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to ARTICULOS.xls or an equivalent article master file.
    raw : pandas.DataFrame, optional
        Previously loaded raw sheet.

    Returns
    -------
    pandas.DataFrame
        Standardized article master table.

    Raises
    ------
    ValueError
        If the expected article columns are missing.
    '''
    path = Path(file_path)

    raw = (
        _read_excel_raw(path)
        if raw is None
        else raw
    )

    data = _table_from_first_row(raw)

    column_mapping = {
        'articulo': 'article_code',
        'descripcion': 'article_name',
        'descripcion_abreviada': 'article_short_name',
        'cod_departamento_venta': 'department_code',
    }

    data = data.rename(
        columns=column_mapping,
    )

    required_columns = {
        'article_code',
        'article_name',
        'department_code',
    }

    missing_columns = sorted(
        required_columns.difference(data.columns)
    )

    if missing_columns:
        raise ValueError(
            'The article file is missing required columns: '
            + ', '.join(missing_columns)
        )

    data['article_code'] = pd.to_numeric(
        data['article_code'],
        errors='coerce',
    ).astype('Int64')

    data['department_code'] = pd.to_numeric(
        data['department_code'],
        errors='coerce',
    ).astype('Int64')

    data['article_name'] = _clean_string_series(
        data['article_name']
    )

    if 'article_short_name' in data.columns:
        data['article_short_name'] = _clean_string_series(
            data['article_short_name']
        )

    data = data[
        data['article_code'].notna()
        & data['article_name'].notna()
    ].copy()

    data.insert(
        0,
        'source_file',
        path.name,
    )

    preferred_columns = [
        'source_file',
        'article_code',
        'article_name',
        'article_short_name',
        'department_code',
    ]

    ordered_columns = [
        column
        for column in preferred_columns
        if column in data.columns
    ]

    remaining_columns = [
        column
        for column in data.columns
        if column not in ordered_columns
    ]

    return data[
        ordered_columns + remaining_columns
    ].reset_index(drop=True)


def read_menu(
    file_path: str | Path,
    raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    '''
    Read the menu or CARTA article table.

    One output row represents one article included in the exported menu
    list.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to CARTA.xls or an equivalent menu file.
    raw : pandas.DataFrame, optional
        Previously loaded raw sheet.

    Returns
    -------
    pandas.DataFrame
        Standardized menu table.

    Raises
    ------
    ValueError
        If the expected menu columns are missing.
    '''
    path = Path(file_path)

    raw = (
        _read_excel_raw(path)
        if raw is None
        else raw
    )

    data = _table_from_first_row(raw)

    column_mapping = {
        'articulo': 'article_code',
        'descripcion': 'article_name',
    }

    data = data.rename(
        columns=column_mapping,
    )

    required_columns = {
        'article_code',
        'article_name',
    }

    missing_columns = sorted(
        required_columns.difference(data.columns)
    )

    if missing_columns:
        raise ValueError(
            'The menu file is missing required columns: '
            + ', '.join(missing_columns)
        )

    data['article_code'] = pd.to_numeric(
        data['article_code'],
        errors='coerce',
    ).astype('Int64')

    data['article_name'] = _clean_string_series(
        data['article_name']
    )

    data = data[
        data['article_code'].notna()
        & data['article_name'].notna()
    ].copy()

    data.insert(
        0,
        'source_file',
        path.name,
    )

    return data[
        [
            'source_file',
            'article_code',
            'article_name',
        ]
    ].reset_index(drop=True)


def read_departments(
    file_path: str | Path,
    raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    '''
    Read the sales department master table.

    One output row represents one sales department registered in the POS
    system.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to DEPARTAMENTOS.xls or an equivalent department master file.
    raw : pandas.DataFrame, optional
        Previously loaded raw sheet.

    Returns
    -------
    pandas.DataFrame
        Standardized department master table.

    Raises
    ------
    ValueError
        If the expected department columns are missing.
    '''
    path = Path(file_path)

    raw = (
        _read_excel_raw(path)
        if raw is None
        else raw
    )

    data = _table_from_first_row(raw)

    column_mapping = {
        'codigo': 'department_code',
        'descripcion': 'department_name',
        'descripcion_abreviada': 'department_short_name',
    }

    data = data.rename(
        columns=column_mapping,
    )

    required_columns = {
        'department_code',
        'department_name',
    }

    missing_columns = sorted(
        required_columns.difference(data.columns)
    )

    if missing_columns:
        raise ValueError(
            'The department file is missing required columns: '
            + ', '.join(missing_columns)
        )

    data['department_code'] = pd.to_numeric(
        data['department_code'],
        errors='coerce',
    ).astype('Int64')

    data['department_name'] = _clean_string_series(
        data['department_name']
    )

    if 'department_short_name' in data.columns:
        data['department_short_name'] = _clean_string_series(
            data['department_short_name']
        )

    data = data[
        data['department_code'].notna()
        & data['department_name'].notna()
    ].copy()

    data.insert(
        0,
        'source_file',
        path.name,
    )

    preferred_columns = [
        'source_file',
        'department_code',
        'department_name',
        'department_short_name',
    ]

    ordered_columns = [
        column
        for column in preferred_columns
        if column in data.columns
    ]

    remaining_columns = [
        column
        for column in data.columns
        if column not in ordered_columns
    ]

    return data[
        ordered_columns + remaining_columns
    ].reset_index(drop=True)


def read_tickets(
    file_path: str | Path,
    raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    '''
    Read the ticket list exported by the POS system.

    Report headers and pagination information are ignored.

    One output row represents one ticket or document.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the ticket report.
    raw : pandas.DataFrame, optional
        Previously loaded raw sheet.

    Returns
    -------
    pandas.DataFrame
        Standardized ticket table.
    '''
    path = Path(file_path)

    raw = (
        _read_excel_raw(path)
        if raw is None
        else raw
    )

    metadata = _extract_report_metadata(raw)

    dates = (
        raw.iloc[:, 1]
        .map(_parse_excel_date)
    )

    document_total = pd.to_numeric(
        raw.iloc[:, 5],
        errors='coerce',
    )

    receipt_count = pd.to_numeric(
        raw.iloc[:, 7],
        errors='coerce',
    )

    mask = (
        dates.notna()
        & raw.iloc[:, 3].notna()
        & document_total.notna()
        & receipt_count.notna()
    )

    data = pd.DataFrame(
        {
            'date': dates.loc[mask],
            'document_id': (
                raw.loc[
                    mask,
                    raw.columns[3],
                ]
                .astype('string')
                .str.strip()
            ),
            'document_total': (
                document_total
                .loc[mask]
                .astype(float)
            ),
            'receipt_count': (
                receipt_count
                .loc[mask]
                .astype('Int64')
            ),
        }
    ).reset_index(drop=True)

    return _add_report_metadata(
        data,
        path,
        metadata,
    )


def read_tips(
    file_path: str | Path,
    raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    '''
    Read the tip summary exported by the POS system.

    Report headers, total rows and the manual-tip footer are ignored.

    One output row represents one ticket with a registered tip.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the tip report.
    raw : pandas.DataFrame, optional
        Previously loaded raw sheet.

    Returns
    -------
    pandas.DataFrame
        Standardized tip table.
    '''
    path = Path(file_path)

    raw = (
        _read_excel_raw(path)
        if raw is None
        else raw
    )

    metadata = _extract_report_metadata(raw)

    document_id = (
        raw.iloc[:, 0]
        .astype('string')
        .str.strip()
    )

    document_amount = pd.to_numeric(
        raw.iloc[:, 2],
        errors='coerce',
    )

    tip = pd.to_numeric(
        raw.iloc[:, 4],
        errors='coerce',
    )

    document_total = pd.to_numeric(
        raw.iloc[:, 6],
        errors='coerce',
    )

    mask = (
        document_id.notna()
        & document_amount.notna()
        & tip.notna()
        & document_total.notna()
        & ~document_id.str.upper().str.startswith(
            'TOTAL',
            na=False,
        )
    )

    data = pd.DataFrame(
        {
            'document_id': document_id.loc[mask],
            'document_amount': (
                document_amount
                .loc[mask]
                .astype(float)
            ),
            'tip': (
                tip
                .loc[mask]
                .astype(float)
            ),
            'document_total': (
                document_total
                .loc[mask]
                .astype(float)
            ),
        }
    ).reset_index(drop=True)

    return _add_report_metadata(
        data,
        path,
        metadata,
    )


def _combine_date_and_time(
    date_values: pd.Series,
    time_values: pd.Series,
) -> pd.Series:
    '''
    Combine separate Excel date and time columns into one datetime.

    Parameters
    ----------
    date_values : pandas.Series
        Date values.
    time_values : pandas.Series
        Time values.

    Returns
    -------
    pandas.Series
        Combined datetime values.
    '''
    dates = pd.to_datetime(
        date_values,
        errors='coerce',
    ).dt.normalize()

    time_text = (
        time_values
        .astype('string')
        .str.strip()
    )

    time_delta = pd.to_timedelta(
        time_text,
        errors='coerce',
    )

    return dates + time_delta


def read_reservations(
    file_path: str | Path,
    raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    '''
    Read and standardize the reservation export.

    Column names are converted to a common English snake_case convention.

    Reservation and creation dates and times are combined into datetime
    columns.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the reservation Excel file.
    raw : pandas.DataFrame, optional
        Previously loaded raw sheet.

    Returns
    -------
    pandas.DataFrame
        Standardized reservation table.

    Raises
    ------
    ValueError
        If the minimum required reservation columns are missing.
    '''
    path = Path(file_path)

    raw = (
        _read_excel_raw(path)
        if raw is None
        else raw
    )

    data = _table_from_first_row(raw)

    column_mapping = {
        'fecha': 'reservation_date',
        'hora': 'reservation_time',
        'estado': 'status',
        'turno': 'shift',
        'personas': 'people',
        'origen': 'origin',
        'prescriptor': 'referrer',
        'fecha_anadida': 'created_date',
        'hora_anadida': 'created_time',
        'restaurante': 'restaurant',
        'tipo': 'reservation_type',
        'mesa': 'table',
        'zona': 'zone',
        'anotado_por': 'entered_by',
        'grupo': 'group',
        'referencia': 'reference',
        'codigo_de_referencia': 'reference_code',
    }

    data = data.rename(
        columns=column_mapping,
    )

    required_columns = {
        'reservation_date',
        'reservation_time',
        'status',
        'shift',
        'people',
    }

    missing_columns = sorted(
        required_columns.difference(data.columns)
    )

    if missing_columns:
        raise ValueError(
            'The reservations file is missing required columns: '
            + ', '.join(missing_columns)
        )

    data['reservation_date'] = pd.to_datetime(
        data['reservation_date'],
        errors='coerce',
    ).dt.normalize()

    if 'created_date' in data.columns:
        data['created_date'] = pd.to_datetime(
            data['created_date'],
            errors='coerce',
        ).dt.normalize()

    data['people'] = pd.to_numeric(
        data['people'],
        errors='coerce',
    ).astype('Int64')

    data['reservation_datetime'] = (
        _combine_date_and_time(
            data['reservation_date'],
            data['reservation_time'],
        )
    )

    if {
        'created_date',
        'created_time',
    }.issubset(data.columns):
        data['created_datetime'] = (
            _combine_date_and_time(
                data['created_date'],
                data['created_time'],
            )
        )

    if 'table' in data.columns:
        data['table'] = (
            data['table']
            .astype('string')
        )

    data.insert(
        0,
        'source_file',
        path.name,
    )

    preferred_columns = [
        'source_file',
        'reservation_datetime',
        'created_datetime',
        'reservation_date',
        'reservation_time',
        'status',
        'shift',
        'people',
        'origin',
        'referrer',
        'created_date',
        'created_time',
        'restaurant',
        'reservation_type',
        'table',
        'zone',
        'entered_by',
        'group',
        'reference',
        'reference_code',
    ]

    ordered_columns = [
        column
        for column in preferred_columns
        if column in data.columns
    ]

    remaining_columns = [
        column
        for column in data.columns
        if column not in ordered_columns
    ]

    return data[
        ordered_columns + remaining_columns
    ].reset_index(drop=True)


def read_data_file(
    file_path: str | Path,
) -> tuple[str, pd.DataFrame]:
    '''
    Detect and read one supported Excel file.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to an .xls or .xlsx file.

    Returns
    -------
    tuple[str, pandas.DataFrame]
        Dataset name and standardized DataFrame.

    Raises
    ------
    ValueError
        If the file structure is not recognized.
    '''
    path = Path(file_path)

    raw = _read_excel_raw(path)
    file_type = _detect_file_type(raw)

    if file_type == 'article_sales':
        normalized_stem = _normalize_column_name(
            path.stem
        )

        dataset_name = (
            'article_sales_summary'
            if normalized_stem
            in ARTICLE_SALES_SUMMARY_STEMS
            else 'article_sales_periodic'
        )

        return (
            dataset_name,
            read_article_sales(
                path,
                raw=raw,
            ),
        )

    if file_type == 'articles':
        return (
            'articles',
            read_articles(
                path,
                raw=raw,
            ),
        )

    if file_type == 'menu':
        return (
            'menu',
            read_menu(
                path,
                raw=raw,
            ),
        )

    if file_type == 'departments':
        return (
            'departments',
            read_departments(
                path,
                raw=raw,
            ),
        )

    if file_type == 'tickets':
        return (
            'tickets',
            read_tickets(
                path,
                raw=raw,
            ),
        )

    if file_type == 'tips':
        return (
            'tips',
            read_tips(
                path,
                raw=raw,
            ),
        )

    if file_type == 'reservations':
        return (
            'reservations',
            read_reservations(
                path,
                raw=raw,
            ),
        )

    raise ValueError(
        f'No reader implemented for file type: '
        f'{file_type}'
    )


def load_data_directory(
    data_dir: str | Path,
    recursive: bool = True,
    strict: bool = False,
) -> tuple[
    dict[str, pd.DataFrame],
    pd.DataFrame,
]:
    '''
    Load all supported Excel files from a data directory.

    Files are detected from their contents, parsed with the corresponding
    reader and grouped into standardized DataFrames.

    Parameters
    ----------
    data_dir : str or pathlib.Path
        Directory containing the raw data files.
    recursive : bool, default=True
        If True, search inside subdirectories as well.
    strict : bool, default=False
        If True, stop at the first file that cannot be read. If False,
        keep loading the remaining files and return the errors separately.

    Returns
    -------
    tuple[dict[str, pandas.DataFrame], pandas.DataFrame]
        Dictionary of standardized datasets and DataFrame containing
        loading errors.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist or contains no Excel files.
    NotADirectoryError
        If data_dir is not a directory.
    '''
    directory = Path(data_dir)

    if not directory.exists():
        raise FileNotFoundError(
            f'Data directory not found: {directory}'
        )

    if not directory.is_dir():
        raise NotADirectoryError(
            f'Expected a directory: {directory}'
        )

    candidates = (
        directory.rglob('*')
        if recursive
        else directory.glob('*')
    )

    excel_files = sorted(
        path
        for path in candidates
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_EXCEL_EXTENSIONS
            and not path.name.startswith('~$')
        )
    )

    if not excel_files:
        raise FileNotFoundError(
            f'No .xls or .xlsx files found in: '
            f'{directory}'
        )

    grouped_data: dict[
        str,
        list[pd.DataFrame],
    ] = {}

    errors: list[
        dict[str, str]
    ] = []

    for file_path in excel_files:
        try:
            dataset_name, data = read_data_file(
                file_path
            )

            grouped_data.setdefault(
                dataset_name,
                [],
            ).append(data)

        except Exception as exc:
            if strict:
                raise

            errors.append(
                {
                    'source_file': file_path.name,
                    'error_type': type(exc).__name__,
                    'error_message': str(exc),
                }
            )

    datasets = {
        dataset_name: pd.concat(
            frames,
            ignore_index=True,
            sort=False,
        )
        for dataset_name, frames
        in grouped_data.items()
    }

    error_dataframe = pd.DataFrame(
        errors,
        columns=[
            'source_file',
            'error_type',
            'error_message',
        ],
    )

    return datasets, error_dataframe


def summarize_datasets(
    datasets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    '''
    Create a compact summary of the loaded datasets.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Dictionary returned by load_data_directory.

    Returns
    -------
    pandas.DataFrame
        Dataset name, number of rows and number of columns.
    '''
    rows = [
        {
            'dataset': dataset_name,
            'rows': len(dataframe),
            'columns': len(dataframe.columns),
        }
        for dataset_name, dataframe
        in sorted(datasets.items())
    ]

    return pd.DataFrame(rows)