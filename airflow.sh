# airflow db upgrade && sleep 5 && airflow users create --username admin --password admin --firstname Anonymous --lastname Admin --role Admin --email admin@example.org
# cp ./airflow/dags/* ~/airflow/dags/
# airflow webserver --port 8080 
# kill $(ps -o ppid= -p $(cat ~/airflow/airflow-webserver.pid))
