# WGU Capstone project
A recommender system using machine learning written in python

# Overview
In this project I created an anime recommender system. The files I used are found on Kaggle at [MyAnimeList Dataset](https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset). The machine learning portion runs on PyTorch and uses a Neural Collaborative Filtering (NCF) approach. This allows the model to predict how a user might rate other anime based on the ratings they provided for the top 10 anime shown to them. Those predictions can then be used to rank and recommend anime to the user.

I'm including in this repository the 2 papers I wrote as a part of this capstone project. 

The first paper, [Task2Paper](Task2Paper.pdf), explains the project proposal and organizational need. It's designed to be presented to executives and stakeholders who would approve the project and its budget. 

The second paper, [Task3Paper](Task3Paper.pdf), is a report of the concluded project. While it does share much from the Task2Paper, its focus is on project execution and the results. 

# Files
These are the files needed to run this project (in order of operation). 
1. Pull_Images.ipynb -- Must be run first, but only once. Scrapes images and collects them into /anime_img. Then it updates the 'anime-dataset-2023' dataframe to the domain for hosting and changes the name of each file to the 'English name' in the dataframe. It then saves an updated CSV for use in the next step.
2. AnimeRecommender.ipynb -- This file handles all the cleaning and visualizations of the 3 main datasets found on the Kaggle page.
3. model.py -- This is the implementation of the PyTorch model.
4. flask1.py -- This file provides the framework to run the model against user input and provide results.
5. ratings.html -- The page displayed by flask1.py to allow user provided scores. Found in the /templates folder
6. watchnext.html -- The results page to give the user a list of 5 anime to watch based on the ratings of other anime they gave a score to in ratings.html. Found in the /templates folder
7. style.css -- make ratings.html and watchnext.html look better. Found in the /static folder

# Notes
To run these files on your own system, you will first need to download the CSV files from Kaggle (linked above). 

This was all run locally using Anaconda for the Jupyter Notebooks and MS Visual Studio 2022 for the Python and HTML files. System to run / create this project has a Nvidia 2070 Super and 32 GB ram. To get PyTorch to run on the GPU instead of the CPU, I had to downgrade the Nvidia CUDA toolkit from 12.2 to 11.8.
