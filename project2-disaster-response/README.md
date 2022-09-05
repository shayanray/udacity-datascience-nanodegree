## Project Title
udacity-datascience-nanodegree Disaster Response Pipeline Project for Shayan Ray

## Summary of the project

The goal of this project was to:
- build a ETL pipeline to extract, transform and load the data in a SQLLite DB
- build a NLP/ML pipeline to use the messages from people in disaster struck areas and classify them into 36 categories
- integrate them into a web-app with the option to enter a new message for classification and create visualizations for exploratory data analysis.


## Files in the repository

app

| - template
| |- master.html        # web app landing page 
| |- go.html            # classification results
|- run.py               # Flask file to run web-app

data

|- disaster_categories.csv  # input data
|- disaster_messages.csv    # input data
|- process_data.py          # etl pipeline
|- InsertDatabaseName.db    # sqllite db

models

|- train_classifier.py # machine learning pipeline
|- classifier.pkl # saved model

README.md
## Build Status

## Code Style
Python style guide followed

## Screenshots
None

## Tech Framework
### Built with 
Jupyter
Python
SQLLite DB
Flask



## Installation Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage


## API Reference
Not applicable

## Tests
Not applicable

## How to use
As a reference for understanding the implementation

## Contribute
Not applicable

## Credits
Thanks to udacity for such interesting assignments as part of the datascience nanodegree program

## Acknowledgements
Datasets: https://www.kaggle.com/datasets/airbnb/seattle


## License
MIT © Shayan Ray
