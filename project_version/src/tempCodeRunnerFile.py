import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
from pandas import DataFrame

from data_cleaner import DataCleaner


class DataLoader:
    def __init__(self):
        self.cleaner = DataCleaner()

        self.id_column_aliases = {
            'cust_id', 'customer_id', 'cust_num', 'customer_num',
            'customer_number', 'customer_ref', 'id', 'customer_reference',
            'custid', 'customerid', 'client_id', 'clientid'
        }

    def _output_path_from_input_path(self, input_path: str):
        return str(Path(input_path).with_suffix('.csv'))

    def _xlsx_to_csv(self, input_path) -> str:
        df = pd.read_excel(input_path)
        output_path = self._output_path_from_input_path(input_path)
        df.to_csv(output_path, index=False)
        return output_path

    def _jsonl_to_csv(self, input_file: str) -> str:
        data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        df = pd.DataFrame(data)
        output_path = self._output_path_from_input_path(input_file)
        df.to_csv(output_path, index=False)
        return output_path

    def _parquet_to_csv(self, input_file: str) -> str:
        df = pd.read_parquet(input_file)
        output_path = self._output_path_from_input_path(input_file)
        df.to_csv(output_path, index=False)
        return output_path

    def _xml_to_csv(self, input_file: str) -> str:
        df = pd.read_xml(input_file)
        output_path = self._output_path_from_input_path(input_file)
        df.to_csv(output_path, index=False)
        return output_path

    def load_df(self, input_file: str, clean: bool = False) -> DataFrame:
        input_file_suffix = Path(input_file).suffix

        if input_file_suffix == '.xlsx':
            df = pd.read_csv(self._xlsx_to_csv(input_file))
        elif input_file_suffix == '.jsonl':
            df = pd.read_csv(self._jsonl_to_csv(input_file))
        elif input_file_suffix == '.parquet':
            df = pd.read_csv(self._parquet_to_csv(input_file))
        elif input_file_suffix == '.xml':
            df = pd.read_csv(self._xml_to_csv(input_file))
        else:
            df = pd.read_csv(input_file)

        if clean:
            df = self.cleaner.clean_dataframe(df)

        return df

    def _detect_id_column(self, df: DataFrame) -> Optional[str]:
        for col in df.columns:
            col_lower = col.lower().strip()
            if col_lower in self.id_column_aliases:
                return col
        return None

    def _discover_files_in_directory(
            self,
            directory: str,
            exclude_files: Optional[List[str]] = None
    ) -> List[str]:
        dir_path = Path(directory)

        if not dir_path.exists():
            raise ValueError(f"Directory does not exist: {directory}")

        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")

        supported_extensions = ['.csv', '.xlsx', '.parquet', '.xml', '.jsonl']

        default_excludes = {'__init__.py', 'merged_clean_data.csv', 'results.csv'}

        if exclude_files:
            default_excludes.update(exclude_files)

        discovered_files = []

        for ext in supported_extensions:
            files = list(dir_path.glob(f'*{ext}'))
            for file in files:
                if file.name not in default_excludes:
                    discovered_files.append(str(file))

        discovered_files.sort()

        return discovered_files

    def load_and_merge_datasets(
            self,
            source: str | List[str],
            output_path: Optional[str] = None,
            clean: bool = True,
            merge_on: str = 'customer_id',
            exclude_files: Optional[List[str]] = None
    ) -> DataFrame:
        if isinstance(source, str):
            file_paths = self._discover_files_in_directory(source, exclude_files)

        elif isinstance(source, list):

            file_paths = source
        else:
            raise ValueError("source must be either a directory path (str) or list of file paths")

        if not file_paths:
            raise ValueError("No files found to process")

        merged_df = None

        for i, file_path in enumerate(file_paths):

            try:

                df = self.load_df(file_path, clean=False)

                id_col = self._detect_id_column(df)
                if not id_col:
                    continue

                if clean:
                    df = self.cleaner.clean_dataframe(df)

                if id_col != merge_on:
                    df = df.rename(columns={id_col: merge_on})

                if merged_df is None:
                    merged_df = df

                else:
                    before_cols = len(merged_df.columns)
                    merged_df = merged_df.merge(df, on=merge_on, how='outer', suffixes=('', '_dup'))
                    after_cols = len(merged_df.columns)

            except Exception as e:

                continue

        if merged_df is None:
            raise ValueError("No data was successfully loaded and merged")

        dup_cols = [col for col in merged_df.columns if col.endswith('_dup')]
        if dup_cols:
            merged_df = merged_df.drop(columns=dup_cols)

        if output_path:
            merged_df.to_csv(output_path, index=False)

        return merged_df