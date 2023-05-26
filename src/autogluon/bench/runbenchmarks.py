import argparse
import boto3
import json
import logging
import os
import re
import time
from typing import List, Optional
import yaml
from typing_extensions import Annotated

from autogluon.bench.frameworks.multimodal.multimodal_benchmark import MultiModalBenchmark
from autogluon.bench.frameworks.tabular.tabular_benchmark import TabularBenchmark
app = typer.Typer()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_kwargs(module: str, configs: dict):
    """Returns a dictionary of keyword arguments to be used for setting up and running the benchmark.

    Args:
        module (str): The name of the module to benchmark (either "multimodal" or "tabular").
        configs (dict): A dictionary of configuration options for the benchmark.

    Returns:
        A dictionary containing the keyword arguments to be used for setting up and running the benchmark.
    """

    if module == "multimodal":
        git_uri, git_branch = configs["git_uri#branch"].split("#")
        return {
            "setup_kwargs": {
                "git_uri": git_uri,
                "git_branch": git_branch,
            },
            "run_kwargs": {
                "dataset_name": configs["dataset_name"],
                "presets": configs.get("presets"),
                "hyperparameters": configs.get("hyperparameters"),
                "time_limit": configs.get("time_limit"),
            },
        }
    elif module == "tabular":
        return {
            "setup_kwargs": {},
            "run_kwargs": {
                "framework": f'{configs["framework"]}:{configs["label"]}',
                "benchmark": configs["amlb_benchmark"],
                "constraint": configs["amlb_constraint"],
                "task": configs.get("amlb_task"),
                "custom_branch": configs.get("amlb_custom_branch"),
            },
        }


def run_benchmark(configs: dict, split_id: Optional[str] = None):
    """Runs a benchmark based on the provided configuration options.

    Args:
        configs (dict): A dictionary of configuration options for the benchmark.
    """

    module_to_benchmark = {
        "multimodal": MultiModalBenchmark,
        "tabular": TabularBenchmark,
    }
    module_name = configs["module"]
    default_benchmark_name = "ag_bench"
    benchmark_name = configs.get("benchmark_name", default_benchmark_name)
    if benchmark_name is None:
        benchmark_name = default_benchmark_name
    if split_id is not None:
        benchmark_name=f"{benchmark_name}_{split_id}"
        
    benchmark_class = module_to_benchmark.get(module_name, None)
    if benchmark_class is None:
        raise NotImplementedError
    benchmark = benchmark_class(benchmark_name=benchmark_name)
    module_kwargs = get_kwargs(module=module_name, configs=configs)
    benchmark.setup(**module_kwargs.get("setup_kwargs", {}))
    benchmark.run(**module_kwargs.get("run_kwargs", {}))
    benchmark.save_configs(configs=configs)

    if configs.get("metrics_bucket", None):
        benchmark.upload_metrics(s3_bucket=configs["metrics_bucket"], s3_dir=f'{module_name}/{benchmark.benchmark_name}')
    

def upload_config(bucket: str, file: str):
    """Uploads a configuration file to an S3 bucket.

    Args:
        bucket (str): The name of the S3 bucket to upload the file to.
        file (str): The path to the local file to upload.

    Returns:
        The S3 path of the uploaded file.
    """

    s3 = boto3.client("s3")
    file_name = f'{file.split("/")[-1].split(".")[0]}_{time.strftime("%Y%m%dT%H%M%S", time.localtime())}.yaml'
    s3_path = f"configs/{file_name}"
    s3.upload_file(file, bucket, s3_path)
    return f"s3://{bucket}/{s3_path}"


def download_config(s3_path: str, dir: str="/tmp"):
    """Downloads a configuration file from an S3 bucket.

    Args:
        s3_path (str): The S3 path of the file to download.
        dir (str): The local directory to download the file to (default: "/tmp").

    Returns:
        The local path of the downloaded file.
    """

    s3 = boto3.client("s3")
    file_path = os.path.join(dir, s3_path.split("/")[-1])
    bucket = s3_path.strip("s3://").split("/")[0]
    s3_path = s3_path.split(bucket)[-1].lstrip("/")
    s3.download_file(bucket, s3_path, file_path)
    return file_path


def invoke_lambda(configs: dict, config_file: str):
    """Invokes an AWS Lambda function to run benchmarks based on the provided configuration options.

    Args:
        configs (dict): A dictionary of configuration options for the AWS infrastructure.
        config_file (str): The path of the configuration file to use for running the benchmarks.
    """

    lambda_client = boto3.client("lambda", configs["CDK_DEPLOY_REGION"])
    payload = {
        "config_file": config_file
    }
    response = lambda_client.invoke(
        FunctionName=configs["LAMBDA_FUNCTION_NAME"],
        InvocationType='RequestResponse',
        Payload=json.dumps(payload)
    )
    response = json.loads(response['Payload'].read().decode('utf-8'))
    logger.info("AWS Batch jobs submitted by %s.", configs["LAMBDA_FUNCTION_NAME"])
    
    return response


@app.command()
def get_job_status(
    job_ids: Optional[List[str]] = typer.Option(None, "--job-ids", help="List of job ids, separated by space."),
    cdk_deploy_region: Optional[str] = typer.Option(None, "--cdk_deploy_region", help="AWS region that the Batch jobs run in."),
    config_file: Optional[str] = typer.Option(None, "--config-file", help="Path to YAML config file containing job ids.")
):
    """
    Query the status of AWS Batch job ids. 
get_job_status
    The job ids can either be passed in directly or read from a YAML configuration file.
    
    Args:
        job_ids (list[str], optional):
            A list of job ids to query the status for. 
        cdk_deploy_region (str, optional):
            AWS region that the Batch jobs run in.
        config_file (str, optional):
            A path to a YAML config file containing job ids. The YAML file should have the structure:
                job_configs:
                    <job_id>: <job_config>
                    <job_id>: <job_config>
                    ...

    Returns:
        dict
            A dictionary containing the status of the queried job ids.
    """
    if config_file is not None:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            job_ids = list(config.get('job_configs', {}).keys())
            cdk_deploy_region = config.get('CDK_DEPLOY_REGION', cdk_deploy_region)
    
    if job_ids is None or cdk_deploy_region is None:
        raise ValueError("Either job_ids or cdk_deploy_region must be provided or configured in the config_file.")
    
    batch_client = boto3.client('batch', region_name=cdk_deploy_region)

    status_dict = {}

    for job_id in job_ids:
        response = batch_client.describe_jobs(jobs=[job_id])
        job = response['jobs'][0]
        status_dict[job_id] = job['status']
    
    logger.info(status_dict)
    return status_dict

    while True:
        all_jobs_completed = True
        failed_jobs = []

        for job_id in job_ids:
            response = batch_client.describe_jobs(jobs=[job_id])
            job = response["jobs"][0]
            job_status = job["status"]

            if job_status == "FAILED":
                failed_jobs.append(job_id)
            elif job_status not in ["SUCCEEDED", "FAILED"]:
                all_jobs_completed = False

        if all_jobs_completed:
            break
        else:
            time.sleep(60)  # Poll job statuses every 60 seconds

    return failed_jobs

def _get_split_id(file_name: str):
    if "split" in file_name:
        file_name = os.path.basename(file_name)
        match = re.search(r'([a-f0-9]{32})', file_name)
        if match:
            return match.group(1)
        else:
            return None
    
    return None


@app.command()
def run(
    config_file: Annotated[str, typer.Argument(help="Path to custom config file.")],
    remove_resources: Annotated[bool, typer.Option("--remove_resources", help="Remove resources after run.")] = False,
):
    """Main function that runs the benchmark based on the provided configuration options."""

    configs = {}
    if config_file.startswith("s3"):
        config_file = download_config(s3_path=config_file)
    with open(config_file, "r") as f:
        configs = yaml.safe_load(f)

    if configs["mode"] == "aws":
        infra_configs = deploy_stack(configs=configs.get("cdk_context", {}))
        config_s3_path = upload_config(bucket=configs["metrics_bucket"], file=args.config_file)
        lambda_response = invoke_lambda(configs=infra_configs, config_file=config_s3_path)
        batch_client = boto3.client("batch", infra_configs["CDK_DEPLOY_REGION"])
        
        if args.remove_resources:
            failed_jobs = wait_for_jobs_to_complete(batch_client=batch_client, job_ids=lambda_response["job_ids"])
            if failed_jobs:
                logger.warning("Warning: Some jobs have failed: %s. Resources are not being removed.", failed_jobs)
            else:
                destroy_stack(configs=infra_configs)
    elif configs["mode"] == "local":
        split_id = _get_split_id(config_file)
    else:
        raise NotImplementedError
