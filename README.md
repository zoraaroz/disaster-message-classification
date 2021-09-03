## Distaster Message Classification

### Date created
July 2021

### Description
This project employs an ETL pipeline and an ML pipeline to classify messages sent during disaster events into different categories. The results are deployed through a web-app.

### Usage
To run the webapp, run the web-app/app/run.py

### Files
- ETL_pipeline_notebook/ETL Pipeline Preparation.ipynb: Jupyter notebook used as preparation for the ETL pipeline
- ML_pipeline_notebook/ML Pipeline Preparation.ipynb: Jupyter notebook used as preparation for the ML pipeline
- webapp/app/run.py: script for starting the webapp (uses the files in the /templates folder)
- webapp/data: Folder containing csv-files, database (DisasterResponse.db) and ETL script (process_data.py)
- webapp/models: Folder containing pickle file with ML-pipeline (pipeline_SVC.pkl) and ML script (train_classifier.py)

### Credits
The data used was provided by [Figure Eight](https://appen.com/).

The basic file structure for the web-app was provided by the [Udacity Data Scientist
Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025) course.

### License
The contents of this repository are covered under the MIT License.
