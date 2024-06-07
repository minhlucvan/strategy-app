import json
from datetime import datetime, timedelta

from airflow.decorators import dag, task # DAG and task decorators for interfacing with the TaskFlow API
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator

@dag(
    # This defines how often your DAG will run, or the schedule by which your DAG runs. In this case, this DAG
    # will run every 30 mins
    # schedule_interval=timedelta(minutes=30),
    # This DAG is set to run for the first time on January 1, 2021. Best practice is to use a static
    # start_date. Subsequent DAG runs are instantiated based on scheduler_interval
    # start_date=datetime(2021, 1, 1),
    # When catchup=False, your DAG will only run for the latest schedule_interval. In this case, this means
    # that tasks will not be run between January 1, 2021 and 30 mins ago. When turned on, this DAG's first
    # run will be for the next 30 mins, per the schedule_interval
    catchup=False,
    tags=['xxx']) # If set, this tag is shown in the DAG view of the Airflow UI
def config_env_dag():
    """
    ### Config Env Dag
    """

    @task.bash()
    def pip_install():
        """
        #### Pip install task
        A simple "pip install" task to install required packages.
        """
        # run pip install in /opt/airflow/dags/app/sync
        return "pip install -r /opt/airflow/dags/app/sync/requirements.txt"

    order_data = pip_install()


    
example_dag_basic = config_env_dag()