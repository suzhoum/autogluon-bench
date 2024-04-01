import random
n_experiments = 5
seeds = []
for i in range(n_experiments):
    seeds.append(random.randint(0, 100))

seeds = [22, 92, 54, 86, 41]

config_paths = [
    "sample_configs/canvas_tabular_cloud_configs.yaml",
]
frameworks = [
    "AutoGluon_bq_ds_auto:example",
    "AutoGluon_gq_ds_auto:example",
    "AutoGluon_hq_ds_auto:example",
    "AutoGluon_bq_ds:example",
    "AutoGluon_hq_ds:example",
    "AutoGluon_gq_ds:example",
    "AutoGluon_bq:example",
    "AutoGluon_hq:example",
    "AutoGluon_gq:example",
    "AutoGluon_gq_refit:example",
    "AutoGluon_gq_refit_ds:example",
    "AutoGluon_mq:example",

]

amlb_constraints = [
     "48c120m",
     "48c45m",
     "48c60m",
     "48c90m",
     "48c15m",
     "48c30m",
     "48c720m",
     "48c1440m",
]


module = "tabular_reg"

import yaml
import os
import subprocess

config_root = "./temp_configs"
os.makedirs(config_root, exist_ok=True)

for framework in frameworks:
    os.makedirs(f"{config_root}/{framework}", exist_ok=True)
    for amlb_constraint in amlb_constraints:
        config_dir = f"{config_root}/{framework}/{amlb_constraint}"
        os.makedirs(config_dir, exist_ok=True)

        for config_path in config_paths:
            with open(config_path, "r") as f:
                configs = yaml.safe_load(f)
                configs["amlb_constraint"] = amlb_constraint
                configs["framework"] = framework
                configs["module"] = module
                # configs["custom_dataloader"]["shot"] = shot
                job_name = f"{configs['benchmark_name']}-{framework}-{amlb_constraint}"
                job_name = job_name.replace("_", "-")
                job_name = job_name.replace(":", "-")
                configs["benchmark_name"] = job_name
                new_config_path = os.path.join(config_dir, os.path.basename(config_path))
                with open(new_config_path, "w") as new_f:
                    yaml.dump(configs, new_f)
                print("Running config: ", new_config_path)
                command = ["agbench", "run", new_config_path]
                subprocess.run(command)
