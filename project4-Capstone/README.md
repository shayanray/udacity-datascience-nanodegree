## Project Title
udacity-datascience-capstone Sparkify project  for Shayan Ray

## Summary of the project

The goal of this project was to:
- perform Churn prediction for users using a music app . The data used was a subset (128MB) of the full dataset available (12GB).

Medium blog: https://shayan-ray.medium.com/to-churn-or-not-to-churn-b2d39d36a750

## Files in the repository

- README.md - this file describes the project details
- Sparkify.ipynb - this file is the project notebook  
- Sparkify.html - this file is the static representation of the project notebook  

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
Pyspark
Matplotlib



## Installation Instructions
These are ipython notebooks and corresponding HTML snapshots. Install anaconda to open and use these .ipynb files


## API Reference
Not applicable

## Tests
Not applicable

## summary of the results of the analysis
Based on exploratory data analysis I have used the following features to come up with a model for churn prediction

Useful features for model building

#### Categorical:
- - gender
- - level

#### Numerical:
- - number of unique songs played per userId
- - number of total songs played per userId
- - number of unique artists per userId
- - number of Ads action per userId
- - number of thumb down action per userId
- - number of thumbs up action per userId
- - number of friends added per userId
- - number of days after initial registration per userId

Based on the size of the dataset used and features generated and limited experiments performed, RandomForestsClassifier appears to be a decent choice for churn prediction with a F1 score of 0.5121 on the validation dataset. (7 churned users and 13 not-churned users)


## How to use
As a reference for understanding the implementation

## Contribute
Not applicable

## Credits
Thanks to udacity for such interesting assignments as part of the datascience nanodegree program

## Acknowledgements
the mini-dataset file is mini_sparkify_event_data.json

## License
MIT © Shayan Ray
