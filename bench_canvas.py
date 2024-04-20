import random
n_experiments = 5
seeds = []
for i in range(n_experiments):
    seeds.append(random.randint(0, 100))

module = "tabular_reg"
seeds = [22, 92, 54, 86, 41]
seeds = [22]
config_paths = [
    #"sample_configs/canvas_tabular_cloud_configs.yaml",
    "sample_configs/canvas_tabular_5g_cloud_configs.yaml",
]
frameworks = [
    # 'AutoGluon_mq',
    # 'AutoGluon_bq', 
    'AutoGluon_bq_ds_auto',
    'AutoGluon_bq_ds', 
    #'AutoGluon_hq', 
    'AutoGluon_hq_ds_auto',
    'AutoGluon_hq_ds', 
    # 'AutoGluon_gq', 
    #'AutoGluon_gq_ds_auto',
    #'AutoGluon_gq_ds', 
    # 'AutoGluon_gq_refit', 
    # 'AutoGluon_gq_refit_ds', 
    'AutoGluon_lightgbm',
    'AutoGluon_xgboost',
    'AutoGluon_catboost',
    'AutoGluon_randomforest',
    'AutoGluon_constantpredictor',
]

constraints = [
    # '48c15m', 
    # '48c30m', 
    # '48c45m', 
    # '48c60m', 
    '48c90m', 
    '48c120m',
    '48c720m',
    # '48c1440m'
]



module = "tabular_reg"

import yaml
import os
import subprocess

config_root = "./temp_configs"
os.makedirs(config_root, exist_ok=True)

for framework in frameworks:
    config_dir = f"{config_root}/{framework}"
    os.makedirs(config_dir, exist_ok=True)
    for constraint in constraints:
        config_dir = f"{config_root}/{framework}/{constraint}"
        os.makedirs(config_dir, exist_ok=True)
    
        for config_path in config_paths:
            with open(config_path, "r") as f:
                configs = yaml.safe_load(f)
                configs["constraint"] = constraint
                configs["framework"] = framework
                configs["module"] = module
                configs["benchmark_name"] = f"{configs['benchmark_name']}"
                new_config_path = os.path.join(config_dir, os.path.basename(config_path))
                with open(new_config_path, "w") as new_f:
                    print("saving config: ", new_config_path)
                    yaml.dump(configs, new_f)
                print("Running config: ", new_config_path)
                command = ["agbench", "run", new_config_path]
                subprocess.run(command)
