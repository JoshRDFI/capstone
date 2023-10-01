# capstone
WGU Capstone project: A recommender system using machine learning written in python

# Overview
In this project I am creating an anime recommender system. The files I used are found on Kaggle at [https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset](MyAnimeList Dataset). machine learning portion runs on PyTorch and uses a Neural Collaborative Filtering (NCF) approach. This allows the model to predict how a user might rate other anime based on the ratings they provided for the top 10. Those predictions can then be used to rank and recommend anime to the user.

# Files
There are 6 files needed to run this project (in order of operation). 
1. Pull_Images.ipynb -- Must be run first, but only once. Scrapes images and collects them into /anime_img. Then it updates the 'anime-dataset-2023' dataframe to the domain for hosting and changes the name of each file to the 'English name' in the dataframe. It then saves an updated CSV for use in the next step.
2. AnimeRecommender.ipynb -- This file handles all the cleaning and visualizations of the 3 main datasets found on the Kaggle page.
3. model.py -- This is the implementation of the PyTorch model.
4. flask1.py -- This file provides the framework to run the model against user input and provide results.
5. ratings.html -- The page displayed by flask1.py to allow user provided scores. Found in the /templates folder
6. watchnext.html -- The results page to give the user a list of 5 anime to watch based on the ratings of other anime they gave a score to in ratings.html. Found in the /templates folder

# Notes
This was all run locally using Anaconda for the Jupyter Notebooks and MS Visual Studio 2022 for the Python and HTML files. System to run / create this project has a Nvidia 2070 Super and 32 GB ram. 
