import re
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
from pandas import DataFrame


class DataCleaner:
    def __init__(self):
        self.cleaning_report = {}

    def clean_dataframe(self, df: DataFrame) -> DataFrame:
        df_clean = df.copy()

        for col in df_clean.columns:
            df_clean[col] = self._clean_column(df_clean[col], col)

        return df_clean

    def _clean_column(self, series: pd.Series, col_name: str) -> pd.Series:

        if series.isna().all():
            return series

        series_clean = series.copy()
        str_series = series.dropna().astype(str)

        if len(str_series) == 0:
            return series

        if self._has_currency_format(str_series):
            series_clean = series_clean.apply(self._clean_currency)

        if self._has_whitespace_issues(str_series):
            series_clean = series_clean.apply(
                lambda x: x.strip() if isinstance(x, str) else x
            )

        categorical_map = self._get_categorical_mapping(str_series, col_name)
        if categorical_map:
            series_clean = series_clean.apply(
                lambda x: str(x).lower().strip() if pd.notna(x) else x
            )

            series_clean = series_clean.map(
                lambda x: categorical_map.get(x, x) if pd.notna(x) else x
            )

        return series_clean

    def _clean_currency(self, value):

        if pd.isna(value):
            return value

        if isinstance(value, (int, float)):
            return value

        cleaned = str(value).replace('$', '').replace(',', '').replace('"', '').strip()

        try:
            return float(cleaned)
        except ValueError:
            return value

    def _get_categorical_mapping(self, series: pd.Series, col_name: str) -> Dict[str, str]:

        if series.nunique() > 50 or series.nunique() < 2:
            return {}

        unique_values = [str(v).lower().strip() for v in series.unique()]
        return self._generate_standardization_map(unique_values, col_name)

    def analyze_csv(self, file_path: str) -> Dict[str, Any]:
        df = pd.read_csv(file_path)
        report = {
            'file': file_path,
            'columns': {},
            'total_rows': len(df),
            'total_columns': len(df.columns)
        }

        for col in df.columns:
            column_report = self._analyze_column(df[col], col)
            if column_report['needs_cleaning']:
                report['columns'][col] = column_report

        return report

    def _analyze_column(self, series: pd.Series, col_name: str) -> Dict[str, Any]:
        report = {
            'column_name': col_name,
            'needs_cleaning': False,
            'cleaning_operations': [],
            'dtype': str(series.dtype),
            'null_count': series.isna().sum(),
            'unique_values': series.nunique(),
            'sample_values': series.dropna().head(10).tolist()
        }

        if series.isna().all():
            return report

        str_series = series.dropna().astype(str)

        if len(str_series) == 0:
            return report

        if self._has_currency_format(str_series):
            report['needs_cleaning'] = True
            report['cleaning_operations'].append({
                'type': 'remove_currency',
                'description': 'Remove $ and commas from numeric values',
                'pattern': r'[\$,]',
                'target_dtype': 'float'
            })

        categorical_issues = self._detect_categorical_inconsistencies(str_series, col_name)
        if categorical_issues:
            report['needs_cleaning'] = True
            report['cleaning_operations'].append(categorical_issues)

        if self._has_whitespace_issues(str_series):
            report['needs_cleaning'] = True
            report['cleaning_operations'].append({
                'type': 'strip_whitespace',
                'description': 'Remove leading/trailing whitespace'
            })

        if self._has_case_inconsistencies(str_series):
            report['needs_cleaning'] = True
            report['cleaning_operations'].append({
                'type': 'standardize_case',
                'description': 'Standardize text case (lowercase)',
                'recommended': 'lowercase'
            })

        return report

    def _has_currency_format(self, series: pd.Series) -> bool:

        sample = series.head(100)
        pattern = r'[\$,]'
        return any(re.search(pattern, str(val)) for val in sample)

    def _has_whitespace_issues(self, series: pd.Series) -> bool:

        sample = series.head(100)
        return any(str(val) != str(val).strip() for val in sample)

    def _has_case_inconsistencies(self, series: pd.Series) -> bool:

        if series.nunique() > 50:
            return False

        unique_values = series.unique()
        lowercase_map = {}

        for val in unique_values:
            lower_val = str(val).lower()
            if lower_val in lowercase_map:
                return True
            lowercase_map[lower_val] = val

        return False

    def _detect_categorical_inconsistencies(self, series: pd.Series, col_name: str) -> Dict[str, Any]:

        if series.nunique() > 50 or series.nunique() < 2:
            return None

        unique_values = [str(v).lower().strip() for v in series.unique()]
        standardization_maps = self._generate_standardization_map(unique_values, col_name)

        if standardization_maps and len(standardization_maps) < series.nunique():
            return {
                'type': 'standardize_categorical',
                'description': f'Merge similar categorical values (found {series.nunique()} unique, can reduce to {len(standardization_maps)})',
                'mapping': standardization_maps,
                'unique_values_sample': unique_values[:20]
            }

        return None

    def _generate_standardization_map(self, values: List[str], col_name: str) -> Dict[str, str]:

        standardization = {}

        if 'employment' in col_name.lower():
            employment_patterns = {
                'full_time': ['ft', 'full_time', 'full-time', 'fulltime', 'full time'],
                'part_time': ['pt', 'part_time', 'part-time', 'parttime', 'part time'],
                'self_employed': ['self emp', 'self_employed', 'self-employed', 'self employed', 'self-emp'],
                'contract': ['contractor', 'contract']
            }

            for val in values:
                val_clean = val.lower().strip()
                for standard, patterns in employment_patterns.items():
                    if val_clean in patterns:
                        standardization[val] = standard
                        break


        elif 'status' in col_name.lower() or 'account' in col_name.lower():
            status_patterns = {
                'active': ['active', 'act-1', 'act-2', 'act-3', 'a01', 'a02', 'a03'],
            }

            for val in values:
                val_clean = val.lower().strip()
                for standard, patterns in status_patterns.items():
                    if any(pattern in val_clean for pattern in patterns):
                        standardization[val] = standard
                        break


        elif 'education' in col_name.lower():
            education_patterns = {
                'high_school': ['high school', 'hs', 'highschool'],
                'some_college': ['some college', 'college'],
                'bachelor': ['bachelor', 'bachelors', 'ba', 'bs'],
                'graduate': ['graduate', 'master', 'masters', 'ma', 'ms'],
                'advanced': ['advanced', 'phd', 'doctorate']
            }

            for val in values:
                val_clean = val.lower().strip()
                for standard, patterns in education_patterns.items():
                    if any(pattern in val_clean for pattern in patterns):
                        standardization[val] = standard
                        break

        return standardization if standardization else {}

    def analyze_directory(self, directory_path: str) -> Dict[str, Any]:

        dir_path = Path(directory_path)
        csv_files = list(dir_path.glob('*.csv'))

        full_report = {
            'directory': directory_path,
            'total_files': len(csv_files),
            'files_needing_cleaning': 0,
            'files': {}
        }

        for csv_file in csv_files:
            file_report = self.analyze_csv(str(csv_file))

            if file_report['columns']:
                full_report['files_needing_cleaning'] += 1
                full_report['files'][csv_file.name] = file_report

        return full_report