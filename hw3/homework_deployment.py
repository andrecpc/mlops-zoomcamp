from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner

DeploymentSpec(
    flow_location='homework.py',
    name="model_training-dev",
    schedule=CronSchedule(cron="0 9 15 * *"),
    flow_runner=SubprocessFlowRunner(),
    tags=["dev"],
)