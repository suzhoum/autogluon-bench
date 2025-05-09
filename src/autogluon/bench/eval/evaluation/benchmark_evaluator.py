from __future__ import annotations

import os
import warnings
from collections import defaultdict
from typing import List, Optional

import pandas as pd

from autogluon.common.loaders import load_pd
from autogluon.common.utils.s3_utils import is_s3_url

from .constants import DATASET, FOLD, FRAMEWORK, METRIC, METRIC_ERROR, PROBLEM_TYPE, TIME_INFER_S, TIME_TRAIN_S
from .metadata.metadata_loader import load_task_metadata
from .preprocess.preprocess_utils import (
    assert_unique_dataset_tid_pairs,
    convert_folds_into_separate_datasets,
    fill_missing_results_with_default,
    fill_missing_results_with_worst,
)


class BenchmarkEvaluator:
    def __init__(
        self,
        results_dir: str = "data/results/",
        results_dir_input: str = None,
        results_dir_output: str = None,
        output_suffix: str = "ag_full_v5/1h8c",
        use_tid_as_dataset_name: bool = False,
        filter_errors: bool = False,
        framework_nan_fill: Optional[str] = None,
        worst_nan_fill: bool = False,
        task_metadata: str | pd.DataFrame = "task_metadata.csv",
        filter_columns: bool = True,
        convert_infer_time_to_per_row: bool = True,
        columns_to_keep: Optional[List[str]] = None,
        columns_to_keep_extra: Optional[List[str]] = None,
        round_error_decimals: int | None = 5,
    ):
        """
        # TODO: Describe purpose of class
        # TODO: Add docstring for `load_data`
        # TODO: rename `results_raw` to be more descriptive

        Parameters
        ----------
        results_dir: str, default = 'data/results/'
            If results_dir_input or results_dir_output is absent,
                results_dir_input = results_dir + "input/prepared/openml/"
                results_dir_output = results_dir + f"output/openml/{output_suffix}/"
        results_dir_input: str, default = None
        results_dir_output: str, default = None
        output_suffix: str, default = 'ag_full_v5/1h8c'
            # TODO: Add docstring
        use_tid_as_dataset_name : bool, default = False
            If True, replaces `dataset` value with the `tid` value.
            Only valid when the results come from OpenML tasks.
            This is useful if `dataset` value is not globally unique but `tid` is,
            to avoid datasets with the same name being considered the same task.
        filter_errors : bool, default = False
            If True, any dataset that has a failure in any of the listed frameworks will be filtered out of all results.
                This means that there will be 0 errors for any framework in the returned results (dense representation).
            If False, datasets will not be filtered as long as they have at least 1 framework that did not fail on them.
        framework_nan_fill : Optional[str], default = None
            If specified, the value should refer to a framework used as the fill value for any other framework errors.
            If a framework had an error on a dataset,
            its result will be set to the result of the `framework_nan_fill` framework for that dataset.
            For example, if `framework_nan_fill='constantpredictor'`,
            framework dataset errors will be set to the `constantpredictor` framework result.
            This is aligned with how results were computed in the 2022 AMLB paper.
            This value is ignored if `filter_errors=True`, as the errors will already be filtered prior to this logic.
        worst_nan_fill : bool, default = False
            If True, will fill missing results with the worst result observed on a given task.
            Can be used as an alternative to `framework_nan_fill`.
            This value is ignored if `filter_errors=True`, as the errors will already be filtered prior to this logic.
        task_metadata: str | pd.DataFrame, default = 'task_metadata_289.csv'
            The path to task metadata file, or loaded pd.DataFrame of task metadata.
            This is only used when `clean_data=True` when calling `self.load_data`.
            This is used to filter to `tid` in `task_metadata` and join to get additional columns.
            This ensures that only datasets present in `task_metadata` will be used for analysis.
        filter_columns : bool, default = True
            If True, uses `columns_to_keep` and `extra_columns_to_keep` to filter columns in the output of `load_data`.
        columns_to_keep : Optional[List[str]], default = None
            Columns to filter to if `filter_columns=True`.
            If None, will use the default columns:
                DATASET
                FOLD
                FRAMEWORK
                METRIC_ERROR
                METRIC
                PROBLEM_TYPE
                TIME_TRAIN_S
                TIME_INFER_S
        columns_to_keep_extra : Optional[List[str]], default = None
            Extra columns to filter to if `filter_columns=True`.
            Will be concatenated with `columns_to_keep` if specified.
            This is used as a convenience argument to avoid having to respecify all
            standard columns when adding a new output column.
        round_error_decimals : int | None, default = 5
            The number of decimal places to round the metric error to.
            For example, if round_error_decimals=5, then a metric error of 0.03485612 becomes 0.03486.
            This is useful for detecting ties among frameworks by avoiding issues with floating point precision.
        """
        if columns_to_keep is None:
            columns_to_keep = [
                DATASET,
                FOLD,
                FRAMEWORK,
                METRIC_ERROR,
                METRIC,
                PROBLEM_TYPE,
                TIME_TRAIN_S,
                TIME_INFER_S,
            ]
        if columns_to_keep_extra is None:
            columns_to_keep_extra = []

        columns_to_keep = columns_to_keep + columns_to_keep_extra
        columns_to_keep_unique = set(columns_to_keep)
        if len(columns_to_keep_unique) != len(columns_to_keep):
            col_count_dict = defaultdict(int)
            for c in columns_to_keep:
                col_count_dict[c] += 1
            raise ValueError(
                f"Columns cannot be listed multiple times across "
                f"columns_to_keep and extra_columns_to_keep!"
                f"\n\tcolumn counts:\n\t\t{col_count_dict}"
            )

        self.convert_infer_time_to_per_row = convert_infer_time_to_per_row
        self.results_dir = results_dir
        self.results_dir_input = (
            results_dir + "input/prepared/openml/" if results_dir_input is None else results_dir_input
        )
        self.results_dir_output = (
            results_dir + f"output/openml/{output_suffix}/" if results_dir_output is None else results_dir_output
        )
        self._use_tid_as_dataset_name = use_tid_as_dataset_name
        self._filter_errors = filter_errors
        self._task_metadata = task_metadata
        if self._filter_errors:
            framework_nan_fill = None
        self._framework_nan_fill = framework_nan_fill
        self._worst_nan_fill = worst_nan_fill
        if self._worst_nan_fill and self._framework_nan_fill:
            raise AssertionError(
                f"Only one of framework_nan_fill and worst_nan_fill can be enabled: "
                f"(framework_nan_fill={self._framework_nan_fill}, worst_nan_fill={self._worst_nan_fill})"
            )
        if filter_columns:
            self._columns_to_keep = columns_to_keep
        else:
            self._columns_to_keep = None
        self.round_error_decimals = round_error_decimals

    def _load_results(
        self, paths: list | pd.DataFrame, clean_data: bool = False, banned_datasets: list = None
    ) -> pd.DataFrame:
        """paths can be either a list of file paths or a pandas DataFrame"""
        if isinstance(paths, list):
            results_raw = self.load_results_raw(paths=paths)
        else:
            assert isinstance(paths, pd.DataFrame)
            results_raw = paths
        if TIME_INFER_S in results_raw:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                results_raw.loc[results_raw[TIME_INFER_S] == 0, TIME_INFER_S] = 0.001
        if clean_data:
            # FIXME: This doesn't work on new tasks due to not comprehensive metadata
            results_raw = self._clean_data(results_raw=results_raw)
        if banned_datasets is not None:
            results_raw = results_raw[~results_raw[DATASET].isin(banned_datasets)]
        if self._use_tid_as_dataset_name:
            assert_unique_dataset_tid_pairs(results_raw=results_raw)
            results_raw[DATASET] = results_raw["tid"].astype(int).astype(str)
            if banned_datasets is not None:
                results_raw = results_raw[~results_raw[DATASET].isin(banned_datasets)]
        len_results = len(results_raw)
        results_raw = results_raw.drop_duplicates(subset=[FRAMEWORK, DATASET, FOLD, METRIC])
        len_results_post_drop = len(results_raw)
        if len_results != len_results_post_drop:
            print(
                f"Note: Duplicate results detected! Dropping {len_results - len_results_post_drop} duplicate rows"
                f" ({len_results} -> {len_results_post_drop})...",
            )
        self._check_results_valid(results_raw=results_raw)
        return results_raw

    def load_results_raw(self, paths: list) -> pd.DataFrame:
        paths = [path if is_s3_url(path) else self.results_dir_input + path for path in paths]
        return pd.concat([load_pd.load(path) for path in paths], ignore_index=True, sort=True)

    def _check_results_valid(self, results_raw: pd.DataFrame):
        self._assert_unique_metric(results_raw=results_raw)

        if results_raw[METRIC_ERROR].min() < 0:
            eps = -1 / 1e8
            num_negative = len(results_raw[results_raw[METRIC_ERROR] < 0])
            if results_raw[METRIC_ERROR].min() < eps:
                raise AssertionError(
                    f"{METRIC_ERROR} cannot be negative! There may be a bug. "
                    f"Found min value: {results_raw[METRIC_ERROR].min()}"
                )
            else:
                print(
                    f"WARNING: min {METRIC_ERROR} was found to be negative, but was higher than epsilon {eps}! "
                    f"({results_raw[METRIC_ERROR].min()}) {num_negative} rows had negative values! "
                    f"Setting all negative values to 0."
                )
                results_raw.loc[results_raw[METRIC_ERROR] < 0, METRIC_ERROR] = 0

    def load_data(
        self,
        paths: list,
        frameworks: list = None,
        folds: list = None,
        clean_data: bool = False,
        problem_type=None,
        valid_datasets: list = None,
        banned_datasets: list = None,
        infer_batch_size: int = None,
        treat_folds_as_datasets: bool = False,
    ) -> pd.DataFrame:
        results_raw = self._load_results(paths=paths, clean_data=clean_data, banned_datasets=banned_datasets)
        if folds is not None:
            results_raw = results_raw[results_raw[FOLD].isin(folds)]
        if problem_type is not None:
            if isinstance(problem_type, list):
                results_raw = results_raw[results_raw[PROBLEM_TYPE].isin(problem_type)]
            else:
                results_raw = results_raw[results_raw[PROBLEM_TYPE] == problem_type]
            print(f"Filtering to the following problem_type: {problem_type}")
        if banned_datasets is not None:
            results_raw = results_raw[~results_raw[DATASET].isin(banned_datasets)]
        if valid_datasets is not None:
            results_raw = results_raw[results_raw[DATASET].isin(valid_datasets)]
        if infer_batch_size is not None:
            results_raw = self._update_infer_batch_size(results_raw=results_raw, infer_batch_size=infer_batch_size)
        if self._framework_nan_fill is not None:
            results_raw = fill_missing_results_with_default(
                framework_nan_fill=self._framework_nan_fill, frameworks_to_fill=frameworks, results_raw=results_raw
            )
        elif self._worst_nan_fill:
            results_raw = fill_missing_results_with_worst(
                frameworks_to_consider=frameworks, frameworks_to_fill=frameworks, results_raw=results_raw
            )

        if frameworks is not None:
            results_raw = self._filter_frameworks(results_raw=results_raw, frameworks=frameworks)
        if treat_folds_as_datasets:
            results_raw = convert_folds_into_separate_datasets(results_raw=results_raw)
            folds = [0]
        if self._filter_errors:
            results_raw = self.filter_errors(results_raw=results_raw, folds=folds)
        if frameworks is not None:
            frameworks_present = list(results_raw[FRAMEWORK].unique())
            if set(frameworks) != set(frameworks_present):
                diff = list(set(frameworks).symmetric_difference(set(frameworks_present)))
                diff = sorted(diff)
                raise AssertionError(f"Difference in expected frameworks present: {diff}")
        # Round error
        if self.round_error_decimals is not None:
            results_raw.loc[:, METRIC_ERROR] = results_raw[METRIC_ERROR].round(decimals=self.round_error_decimals)

        if self._columns_to_keep:
            results_raw = results_raw[self._columns_to_keep]
        return results_raw

    def task_metadata(self) -> pd.DataFrame | None:
        if isinstance(self._task_metadata, pd.DataFrame):
            return self._task_metadata
        elif self._task_metadata is not None:
            return load_task_metadata(path=self._task_metadata)
        else:
            return None

    def _clean_data(self, results_raw):
        assert self._task_metadata is not None, (
            f"Cannot clean data if task_metadata is None! "
            f"Either set `clean_data=False` or specify `task_metadata` during init."
        )
        task_metadata = self.task_metadata()
        task_metadata[DATASET] = task_metadata["name"]
        # FIXME: TEMP
        results_raw = results_raw.drop(columns=[DATASET])
        results_raw["tid"] = results_raw["tid"].astype(int)
        pre_unique_tid = len(results_raw["tid"].unique())
        # results_raw['dataset'] = results_raw['dataset'].map({'numerai28_6': 'numerai28.6'}).fillna(results_raw['dataset'])
        results_raw = results_raw.merge(task_metadata[["NumberOfInstances", DATASET, "tid"]], on="tid")

        post_unique_tid = len(results_raw["tid"].unique())

        print(f"Joined with task_metadata, filtered task IDs: {pre_unique_tid} -> {post_unique_tid}")

        # FIXME: TEMP
        if self.convert_infer_time_to_per_row:
            results_raw[TIME_INFER_S] = results_raw[TIME_INFER_S] / results_raw["NumberOfInstances"] * 10
        return results_raw

    def _update_infer_batch_size(self, results_raw: pd.DataFrame, infer_batch_size: int):
        # Update infer time
        if f"pred_time_test_with_transform_batch_size_{infer_batch_size}" in results_raw.columns:
            # AutoGluon specific infer time column
            results_raw["time_infer_s"] = results_raw[
                f"pred_time_test_with_transform_batch_size_{infer_batch_size}"
            ].fillna(results_raw["time_infer_s"])
        if f"pred_time_test_with_transform_{infer_batch_size}" in results_raw.columns:
            # AutoGluon specific infer time column
            results_raw["time_infer_s"] = results_raw[f"pred_time_test_with_transform_{infer_batch_size}"].fillna(
                results_raw["time_infer_s"]
            )
        if f"infer_batch_size_df_{infer_batch_size}" in results_raw.columns:
            # The new infer time column in AMLB Nov 2023 results
            if f"infer_batch_size_file_{infer_batch_size}" in results_raw.columns:
                results_raw[f"infer_batch_size_df_{infer_batch_size}"] = results_raw[
                    f"infer_batch_size_df_{infer_batch_size}"
                ].fillna(results_raw[f"infer_batch_size_file_{infer_batch_size}"])
            results_raw["time_infer_s"] = results_raw[f"infer_batch_size_df_{infer_batch_size}"].fillna(
                results_raw["time_infer_s"]
            )
            # FIXME: Divide by infer_batch_size?
            if infer_batch_size != 1:
                raise AssertionError(
                    f"Debug this logic before working with batch_sizes other than 1 to make sure it is working as intended."
                )
        elif f"infer_batch_size_file_{infer_batch_size}" in results_raw.columns:
            # The new infer time column in AMLB Nov 2023 results
            tmp = results_raw[f"infer_batch_size_file_{infer_batch_size}"] / infer_batch_size
            results_raw["time_infer_s"] = tmp.fillna(results_raw["time_infer_s"])
        return results_raw

    def filter_errors(self, results_raw: pd.DataFrame, folds, frameworks: list = None):
        """
        For each framework in frameworks, ensure that only datasets without failures from this framework are kept.
        """
        # FIXME: Ensure correct folds, not just count
        if frameworks is None:
            frameworks = list(results_raw["framework"].unique())
        for f in frameworks:
            datasets_keep = results_raw[results_raw["framework"] == f]["dataset"].value_counts()
            datasets_keep = list(datasets_keep[datasets_keep == len(folds)].index)
            results_raw = results_raw[results_raw["dataset"].isin(datasets_keep)]
        return results_raw

    def _filter_frameworks(self, results_raw: pd.DataFrame, frameworks: list):
        return results_raw[results_raw["framework"].isin(frameworks)]

    def filter_datasets(
        self, *, max_rows=None, min_rows=None, max_rows_missing_val=None, max_features_categorical=None
    ):
        # TODO: Consider having task_metadata be its own class
        task_metadata = self.task_metadata()

        if max_rows is not None:
            task_metadata = task_metadata[task_metadata["NumberOfInstances"] <= max_rows]
        if min_rows is not None:
            task_metadata = task_metadata[task_metadata["NumberOfInstances"] >= min_rows]
        if max_rows_missing_val is not None:
            task_metadata = task_metadata[task_metadata["NumberOfInstancesWithMissingValues"] <= max_rows_missing_val]
        if max_features_categorical is not None:
            task_metadata = task_metadata[task_metadata["NumberOfSymbolicFeatures"] <= max_features_categorical]

        return list(task_metadata["name"])

    @staticmethod
    def _assert_unique_metric(results_raw: pd.DataFrame):
        """
        Assert that each dataset uses exactly one unique metric.

        Parameters
        ----------
        results_raw

        """
        dataset_metric_counts = results_raw[[DATASET, METRIC]].value_counts().reset_index(drop=False)
        num_metrics_per_dataset = dataset_metric_counts[DATASET].value_counts()
        dataset_metric_counts["metrics_in_dataset"] = dataset_metric_counts[DATASET].map(num_metrics_per_dataset)
        dataset_metric_counts = dataset_metric_counts[dataset_metric_counts["metrics_in_dataset"] > 1]
        if len(dataset_metric_counts) != 0:
            num_invalid_datasets = len(dataset_metric_counts[DATASET].unique())
            raise AssertionError(
                f"Found {num_invalid_datasets} datasets using multiple different metrics. This is not yet supported in BenchmarkEvaluator. "
                f"Each dataset must map to a single metric. "
                f"One way to avoid this error is to append the metric name to the dataset name."
                f"\nList of invalid datasets:\n"
                f"{dataset_metric_counts}"
            )
