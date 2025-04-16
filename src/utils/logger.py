import wandb
from pathlib import Path
 
def setup_run(logger_cfg):

    api = wandb.Api()

    projects = api.projects(entity=str(logger_cfg.entity))
    if all(False for _ in projects):
        api.create_project(logger_cfg.project, entity=logger_cfg.entity)
    elif any(True for _ in projects):
        project_names = []
        for project in projects:
            project_names.append(project.name)
        if logger_cfg.project not in project_names:
            api.create_project(logger_cfg.project, entity=logger_cfg.entity)

    project_path = str(logger_cfg.entity + "/" + logger_cfg.project)
    if len(api.runs(path=project_path, filters={"group": logger_cfg.group})) == 0:
        run_name = "0_" + logger_cfg.group   
    else:
        last_run_group = api.runs(path=project_path, filters={"group": logger_cfg.group})[0]
        run_number_group = int(last_run_group.name.split("_")[0]) + 1
        run_name = str(run_number_group) + "_" + logger_cfg.group

    # setup save directory for wandb logs if it no exist already
    save_path = Path(logger_cfg.save_dir) / logger_cfg.project / logger_cfg.group
    logger_cfg.save_dir = str(save_path)
    save_path.mkdir(parents=True, exist_ok=True)


    logger_cfg.name = run_name    
