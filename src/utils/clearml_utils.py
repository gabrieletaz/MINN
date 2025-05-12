import json
from typing import Union
import clearml
from omegaconf import DictConfig, OmegaConf
from os.path import join


def connect_confiuration(clearml_task: clearml.Task, configuration: DictConfig) -> DictConfig:
    return OmegaConf.create(str(clearml_task.connect(OmegaConf.to_object(configuration), name="hydra_config"))) # type: ignore

def setting_up_task(cfg: DictConfig) -> clearml.Task:
    task: clearml.Task = clearml.Task.init(project_name="e-muse/FBA_ML/different_input_output", task_name=cfg.exp_name, reuse_last_task_id=False,
                                           auto_connect_frameworks={'hydra': False})
    
    with open("/clearml_conf/server_credentials.json", 'r') as file:
        bot_credentials = json.load(file)
    
    task.set_base_docker(
        docker_image="amnfba:latest",
        docker_arguments="--env CLEARML_AGENT_SKIP_PIP_VENV_INSTALL=1 \
        --env CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL=1 \
        --env CLEARML_AGENT_GIT_USER={access_key} \
        --env CLEARML_AGENT_GIT_PASS={secret_key} \
        --mount type=bind,source=/srv/nfs-data/tazza/server_credentials.json,target=/clearml_conf/server_credentials.json \
        --runtime=nvidia --ipc=host".format(**bot_credentials)
    )

    return task