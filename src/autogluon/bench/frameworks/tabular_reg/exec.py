import argparse
import csv
import importlib
import json
import logging
import os
import pandas as pd
import time
from datetime import datetime
from typing import Optional, Union
import zipfile
import itertools

from autogluon.bench.datasets.dataset_registry import multimodal_dataset_registry
from autogluon.core.metrics import make_scorer
from autogluon.core.utils.savers import save_pkl
from autogluon.tabular import TabularPredictor
from autogluon.tabular import __version__ as ag_version

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def normalize_path(path):
    return os.path.realpath(os.path.expanduser(path))

def walk_apply(dir_path, apply, topdown=True, max_depth=-1, filter_=None):
    dir_path = normalize_path(dir_path)
    for dir, subdirs, files in os.walk(dir_path, topdown=topdown):
        if max_depth >= 0:
            depth = 0 if dir == dir_path else len(str.split(os.path.relpath(dir, dir_path), os.sep))
            if depth > max_depth:
                continue
        for p in itertools.chain(files, subdirs):
            path = os.path.join(dir, p)
            if filter_ is None or filter_(path):
                apply(path, isdir=(p in subdirs))

def zip_path(path, dest_archive, compression=zipfile.ZIP_DEFLATED, arc_path_format='short', filter_=None):
    path = normalize_path(path)
    if not os.path.exists(path): return
    with zipfile.ZipFile(dest_archive, 'w', compression) as zf:
        if os.path.isfile(path):
            in_archive = os.path.basename(path)
            zf.write(path, in_archive)
        elif os.path.isdir(path):
            def add_to_archive(file, isdir):
                if isdir: return
                in_archive = (os.path.relpath(file, path) if arc_path_format == 'short'
                              else os.path.relpath(file, os.path.dirname(path)) if arc_path_format == 'long'
                              else os.path.basename(file) is arc_path_format == 'flat'
                              )
                zf.write(file, in_archive)
            walk_apply(path, add_to_archive,
                       filter_=lambda p: (filter_ is None or filter_(p)) and not os.path.samefile(dest_archive, p))



def _flatten_dict(data):
    flattened = {}
    for key, value in data.items():
        if isinstance(value, dict):
            flattened.update(_flatten_dict(value))
        else:
            flattened[key] = value
    return flattened


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Dataset that has been registered with multimodal_dataset_registry.",
    )
    parser.add_argument("--framework", type=str, help="Framework (and) branch/version.")
    parser.add_argument("--benchmark_dir", type=str, help="Directory to save benchmarking run.")
    parser.add_argument("--metrics_dir", type=str, help="Directory to save benchmarking metrics.")
    parser.add_argument("--constraint", type=str, default=None, help="AWS resources constraint setting.")
    parser.add_argument("--params", type=str, default=None, help="AWS resources constraint setting.")
    parser.add_argument(
        "--custom_dataloader", type=str, default=None, help="Custom dataloader to use in the benchmark."
    )
    parser.add_argument("--custom_metrics", type=str, default=None, help="Custom metrics to use in the benchmark.")
    parser.add_argument("--time_limit", type=int, default=None, help="Time limit used to fit the predictor.")

    args = parser.parse_args()
    return args


def load_dataset(dataset_name: str, custom_dataloader: dict = None):  # dataset name
    """Loads and preprocesses a dataset.

    Args:
        dataset_name (str): The name of the dataset to load.
        custom_dataloader (dict): A dictionary containing information about a custom dataloader to use. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and test datasets.
    """
    splits = ["train", "val", "test"]
    data = {}
    if dataset_name in multimodal_dataset_registry.list_keys():
        logger.info(f"Loading dataset {dataset_name} from multimodal_dataset_registry")
        for split in splits:
            data[split] = multimodal_dataset_registry.create(dataset_name, split)
    elif custom_dataloader is not None:
        logger.info(f"Loading dataset {dataset_name} from custom dataloader {custom_dataloader}.")
        custom_dataloader_file = custom_dataloader.pop("dataloader_file")
        class_name = custom_dataloader.pop("class_name")
        spec = importlib.util.spec_from_file_location(class_name, custom_dataloader_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        custom_class = getattr(module, class_name)
        for split in splits:
            data[split] = custom_class(dataset_name=dataset_name, split=split, **custom_dataloader)
    else:
        raise ModuleNotFoundError(f"Dataset Loader for dataset {dataset_name} is not available.")

    return data.values()


def load_custom_metrics(custom_metrics: dict):
    """Loads a custom metrics and convert it to AutoGluon Scorer.

    Args:
        custom_metrics (dict): A dictionary containing information about a custom metrics to use. Defaults to None.

    Returns:
        scorer (Scorer)
            scorer: An AutoGluon Scorer object to pass to MultimodalPredictor.
    """

    try:
        custom_metrics_path = custom_metrics.pop("metrics_path")
        func_name = custom_metrics.pop("function_name")
        spec = importlib.util.spec_from_file_location(func_name, custom_metrics_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        score_func = getattr(module, func_name)

        scorer = make_scorer(
            name=func_name,
            score_func=score_func,
            **custom_metrics,  # https://auto.gluon.ai/stable/tutorials/tabular/advanced/tabular-custom-metric.html
        )
    except:
        raise ModuleNotFoundError(f"Unable to load custom metrics function {func_name} from {custom_metrics_path}.")

    return scorer


def save_metrics(metrics_path: str, metrics: dict):
    """Saves evaluation metrics to a JSON file.

    Args:
        metrics_path (str): The path to the directory where the metrics should be saved.
        metrics: The evaluation metrics to save.

    Returns:
        None
    """
    if metrics is None:
        logger.warning("No metrics were created.")
        return

    if not os.path.exists(metrics_path):
        os.makedirs(metrics_path)
    file = os.path.join(metrics_path, "results.csv")
    flat_metrics = _flatten_dict(metrics)
    field_names = flat_metrics.keys()

    with open(file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        writer.writerow(flat_metrics)
    logger.info("Metrics saved to %s.", file)
    f.close()


def run(
    dataset_name: Union[str, dict],
    framework: str,
    benchmark_dir: str,
    metrics_dir: str,
    constraint: Optional[str] = None,
    params: Optional[dict] = None,
    custom_dataloader: Optional[dict] = None,
    custom_metrics: Optional[dict] = None,
    time_limit: Optional[int] = None,
):
    """Runs the AutoGluon multimodal benchmark on a given dataset.

    Args:
        dataset_name (Union[str, dict]): Dataset that has been registered with multimodal_dataset_registry.

                            To get a list of datasets:

                            from autogluon.bench.datasets.dataset_registry import multimodal_dataset_registry
                            multimodal_dataset_registry.list_keys()

        benchmark_dir (str): The path to the directory where benchmarking artifacts should be saved.
        constraint (str): The resource constraint used by benchmarking during AWS mode, default: None.
        params (str): The multimodal params, default: {}.
        custom_dataloader (dict): A dictionary containing information about a custom dataloader to use. Defaults to None.
                                To define a custom dataloader in the config file:

                                custom_dataloader:
                                    dataloader_file: path_to/dataloader.py   # relative path to WORKDIR
                                    class_name: DataLoaderClass
                                    dataset_config_file: path_to/dataset_config.yaml
                                    **kwargs (of DataLoaderClass)
        custom_metrics (dict): A dictionary containing information about a custom metrics to use. Defaults to None.
                                To define a custom metrics in the config file:

                                custom_metrics:
                                    metrics_path: path_to/metrics.py   # relative path to WORKDIR
                                    function_name: custom_metrics_function
                                    **kwargs (of autogluon.core.metrics.make_scorer)
    Returns:
        None
    """
    train_data, val_data, test_data = load_dataset(dataset_name=dataset_name, custom_dataloader=custom_dataloader)
    try:
        label_column = train_data.label_columns[0]
    except (AttributeError, IndexError):  # Object Detection does not have label columns
        label_column = None
    if params is None:
        params = {}
    predictor_args = {
        "label": label_column,
        "problem_type": train_data.problem_type,
        "path": os.path.join(benchmark_dir, "models"),
        "eval_metric": train_data.metric
    }

    if time_limit is not None:
        params["time_limit"] = time_limit
        logger.warning(
            f'params["time_limit"] is being overriden by time_limit specified in constraints.yaml. params["time_limit"] = {time_limit}'
        )

    metrics_func = None
    if custom_metrics is not None and custom_metrics["function_name"] == train_data.metric:
        metrics_func = load_custom_metrics(custom_metrics=custom_metrics)

    predictor = TabularPredictor(**predictor_args)

    ### leaderboard
    _leaderboard_extra_info = params.pop('_leaderboard_extra_info', False)  # whether to get extra model info (very verbose)
    _leaderboard_test = params.pop('_leaderboard_test', False)  # whether to compute test scores in leaderboard (expensive)
    artifacts = params.pop('_save_artifacts', ['leaderboard'])
    leaderboard_kwargs = dict(extra_info=_leaderboard_extra_info)

    fit_args = {"train_data": train_data.data, "tuning_data": val_data.data, **params}

    utc_time = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    start_time = time.time()
    predictor.fit(**fit_args)
    end_time = time.time()
    training_duration = round(end_time - start_time, 1)

    if isinstance(test_data.data, dict):  # multiple test datasets
        test_data_dict = test_data.data
    else:
        test_data_dict = {dataset_name: test_data}

    for dataset_name, test_data in test_data_dict.items():
        evaluate_args = {
            "data": test_data.data,
            "auxiliary_metrics": False
        }

        start_time = time.time()
        scores = predictor.evaluate(**evaluate_args)
        end_time = time.time()
        predict_duration = round(end_time - start_time, 1)

        if _leaderboard_test:
            leaderboard_kwargs['data'] = test_data.data
        leaderboard = predictor.leaderboard(**leaderboard_kwargs)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            logger.info(leaderboard)

        if "#" in framework:
            framework, version = framework.split("#")
        else:
            framework, version = framework, ag_version

        metric_name = test_data.metric if metrics_func is None else metrics_func.name
        metrics = {
            "id": "id/0",  # dummy id to make it align with amlb benchmark output
            "task": dataset_name,
            "framework": framework,
            "constraint": constraint,
            "version": version,
            "fold": 0,
            "type": predictor.problem_type,
            "metric": metric_name,
            "utc": utc_time,
            "training_duration": training_duration,
            "predict_duration": predict_duration,
            "scores": scores,
        }
        subdir = f"{framework}.{dataset_name}.{constraint}.local"
        save_metrics(os.path.join(metrics_dir, subdir, "scores"), metrics)

        if 'leaderboard' in artifacts:
            leaderboard_dir = os.path.join(metrics_dir, subdir, "leaderboard")
            os.makedirs(leaderboard_dir, exist_ok=True)
            leaderboard.to_csv(os.path.join(leaderboard_dir, "leaderboard.csv"))

        if 'info' in artifacts:
            info_dir = os.path.join(metrics_dir, subdir, "info")
            os.makedirs(info_dir, exist_ok=True)
            ag_info = predictor.info()
            save_pkl.save(path=os.path.join(info_dir, "info.pkl"), object=ag_info)

        if 'models' in artifacts:
            models_dir = os.path.join(metrics_dir, subdir, "models")
            os.makedirs(models_dir, exist_ok=True)
            zip_path(predictor.path, os.path.join(models_dir, "models.zip"))


if __name__ == "__main__":
    args = get_args()
    if args.params is not None:
        args.params = json.loads(args.params)
    if args.custom_dataloader is not None:
        args.custom_dataloader = json.loads(args.custom_dataloader)
    if args.custom_metrics is not None:
        args.custom_metrics = json.loads(args.custom_metrics)

    run(
        dataset_name=args.dataset_name,
        framework=args.framework,
        benchmark_dir=args.benchmark_dir,
        metrics_dir=args.metrics_dir,
        constraint=args.constraint,
        params=args.params,
        custom_dataloader=args.custom_dataloader,
        custom_metrics=args.custom_metrics,
        time_limit=args.time_limit,
    )
