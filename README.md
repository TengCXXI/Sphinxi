# Sphinxi - A Word Similarities Game

## Directory Structure

```
.
├── capstone
    ├── code
        ├── data
            ├── hof_all_games.json
            ├── hof_individual.json
            ├── similarities.json
            ├── user_game_data_json.json
            ├── users.txt
        ├── static  
                ├── css
                    ├── styles.css
                ├── img
                    ├── sphinxi.jpg
        ├── templates
            ├── about.html
            ├── base.html
            ├── basic_form.html
            ├── basic_results.html
            ├── contact.html
            ├── game.html
            ├── halloffame.html
            ├── index.html
            ├── register.html
            ├── testing.html
            ├── user.html
        ├── flask_wiki_movies_model.py
        ├── flask_wiki_news_model.py
        ├── google_news_model.ipynb
        ├── run.py
        ├── urban_dict_data_cleaning_eda_and_vector_gen.ipynb
        ├── urban_dictionary_cbow_model.ipynb
        ├── urban_dictionary_skipgram_model.ipynb
        ├── wiki_movies_data_cleaning_eda_and_vector_gen.ipynb
        ├── wiki_movies_skipgram_model.ipynb
        ├── wiki_news_model.ipynb
    ├── data
        ├── bad-words.csv
        ├── top_genres_movies_decade.csv
        ├── top_genres_movies_year.csv
        ├── urban_dictionary_words_most_upvotes.csv
    ├── visuals
        ├── Release Year Distribution.png
        ├── Top 10 Movie Genres 2010's.gif
        ├── Top 10 Movie Genres by Decade 1900-2017.gif
        ├── Top 10 Urban Dictionary Words with Most Up-Votes.png
        ├── Top 10 Urban Dictionary Words with the Most Definitions.png
        ├── Top 15 Casts with the Most Movies.png
        ├── Top 15 Directors with the Most Movies.png
        ├── Top 15 Genres with the Most Movies.png
    ├── DSI8_Capstone_Sphinxi.pdf
    └── README.md
```

## Problem Statement

How do you build a game using natural language processing with a neural net, and why would you want to do so? By using Natural Language Processing (NLP), a model can be created that will have a list of words and their relationships. Without utilizing a team of writers to produce words and their respective similar words, you can leverage the functions found within the Word2Vec model to do the work. This allows for developers to build a game without a large team and save money. The NLP model also enables continuous improvement by being able to incorporate new data as players play the game and submit their answers. Generating new content or categories can be as easy finding the right corpus of data on which to train the model. Instead of relying on a human to drive content, the model will provide. As to how the game will be built, please refer to the [Game Creation](#Game Creation) farther section below.

---

## Description of Data

### Size and Source

- [Google News Vectors](https://code.google.com/archive/p/word2vec/) - 300-dimensional vectors for 3 million words and phrases trained on Google News dataset of about 100 billion words
- [Wikipedia, UMBC, and statmt.org](https://fasttext.cc/docs/en/english-vectors.html) - 300-dimensional vectors for 1 million words trained on Wikipedia 2017, UMBC webbase corpus and statmt.org news dataset (16 billion words)
- [Urban Dictionary](https://www.kaggle.com/therohk/urban-dictionary-words-dataset/) - 2.6 million words (1.6 million unique), authors, up-votes and down-votes, as well as definitions were scraped and put on kaggle for a competition
- [Wikipedia Movie Plots](https://www.kaggle.com/jrobischon/wikipedia-movie-plots/) - 34,886 movies were scraped from Wikipedia and put on kaggle for a competition. Data includes release year, title, origin/ethnicity, director, cast, genre, and plot

### Data Dictionary

#### Urban Dictionary

|Feature|Type|Description|
|---|---|---|
|word_id|float64|Unique id for urban dictionary api| 
|word|object|Word being defined| 
|up_votes|float64|Thumbs up count as of may 2016| 
|down_votes|float64|Thumbs down count as of may 2016| 
|author|object|Hashed screen name of submitter| 
|definition|object|Definition of word | 
|word_no_spaces|object|Created a new column that replaced the spaces with underscores from the "word" column| 
|word_def|object|Created a column that combined the "word_no_spaces" column with the "definition" column| 

#### Wikipedia Movie Plots

|Feature|Type|Description|
|---|---|---|
|Release Year|int64|Year in which the movie was released| 
|Title|object|Movie title| 
|Origin/Ethnicity|object|Origin of movie (i.e. American, Bollywood, Tamil, etc.)| 
|Director|object|Director(s)| 
|Cast|object|Main actors/actresses| 
|Genre|object|Movie genre(s)| 
|Wiki Page|object|URL of Wikipedia page from which the plot description was scraped| 
|Plot|object|Long form description of the movie's plot| 
|cast_no_spaces|object|Created a new column that replaced the spaces with underscores from the "Cast" column and lowercased the names| 
|director_no_spaces|object|Created a new column that replaced the spaces with underscores from the "Director" column and lowercased the names| 
|title_no_spaces|object|Created a new column that replaced the spaces with underscores from the "Title" column and lowercased the titles| 
|movie|object|Created a column that combined the "cast_no_spaces", "director_no_spaces", "title_no_spaces", "Genre", and "Plot" columns| 

### Data Summary

While the model generated with the Google News and Wikipedia, UMBC, and statmt.org datasets are good at showing the relationship of common words, it falls short on the specificity required for categorical prompts. For example, if I wanted movies related to the horror genre, the model would not be able to generate that. In order to generate words and their respective similarities for specific categories, datasets focused on distinct domains would have to be used. The Wikipedia Movie Plots dataset was selected for this purpose as a proof of concept. Considering the popularity of risque games such as Cards Against Humanity, Urban Dictionary is a good source of data to mine for comparable shock value for an adult audience as another category in the game. 

---

## Game Creation

Below is a summary of steps to create the Word2Vec model that generated the word similarities and analogies, as well as deployed the game in a browser:
1. Gather dataset
2. Clean the data
3. Perform exploratory data analysis
4. Generate the list of word prompts for the game
5. Generate the vectors using Word2Vec and save
6. Load the vectors into a model using the Gensim library
7. Pickle the model so that it can be instantiated in Flask
8. Generate the list of similar words for scoring for each of the word prompts
9. Deploy the model in Flask so that the game can be launched in a browser

With the Google News and Wikipedia, UMBC, and statmt.org datasets, the vectors were already generated, so steps 1-4 could be skipped for those models. Vectors were generated for the Urban Dictionary and Wikipedia Movie datasets using both CBOW and SkipGram algorithms. However, the SkipGram vectors generated much more pertinent word similarities and analogies, so the game was built using the SkipGram models. The framework for the Flask code was referenced from [Anthony Bonello's GitHub repo](https://github.com/abonello/riddle-game-app), and then modified for this similarities game. 

---

## Data Visualization

### Urban Dictionary

<img src="./visuals/Top 10 Urban Dictionary Words with Most Up-Votes.png"> 

- A bar chart of the top 10 words with most up-votes, excluding swear words
- Although a high number of up-votes is generally considered a positive attribute, in Urban Dictionary's case, a derogatory or offensive definition of the word, topic, or person may be the source of the up-votes, so ultimately it conveys a negative connotation

<img src="./visuals/Top 10 Urban Dictionary Words with the Most Definitions.png">
- A bar chart of the top 10 words with most definitions, excluding swear words
- This illustrates the engagement certain words/topics generate, as well as the scope of different definitions provided for the top words

### Wikipedia Movie Plots

<img src="./visuals/Release Year Distribution.png">

- A histogram of the movies organized by release year from 1900-2017
- As motion picture film was invented in the late 19th century, the first wide-spread movies would start to appear at the beginning of the 20th century
- The first big jump in the number of movies being made occurs around the end of the 1920's
- There was moderate growth from the 1930's to about the mid-1950's
- The data indicates a dip in the number of movies made per year from the mid-1950's to about mid-1980's
- Since the mid-1980's, the growth in the number of movies has increased significantly, culminating in over 1,500 movies being made around 2017

<img src="./visuals/Top 15 Directors with the Most Movies.png">

- A bar chart of the top 15 directors with the most movies attributed to them

<img src="./visuals/Top 15 Casts with the Most Movies.png">

- A bar chart of the top 15 casts with the most movies attributed to them

<img src="./visuals/Top 15 Genres with the Most Movies.png">

- A bar chart of the top 15 genres with the most movies attributed to them
- Drama led the genres with almost 6,000 movies
- Comedy came in second with over 4,000 movies
- The rest of the top 15 genres ranged from slightly over 1,000 movies to about 500 movies

<img src="./visuals/Top 10 Movie Genres by Decade 1900-2017.gif">

- A bar chart race of the top 10 movie genres by decade from 1900-2017 was created using Flourish
- Drama and comedy dominate at the top of the chart through the decades
- Film noir enjoyed a run of being third in the 1940's
- In the 1950's-1960's Westerns had a turn at being third
- In the 1970's-1980's Horror came on the scene for third
- In the 1990's Action was the genre of choice at 3
- In the 2000's Romance was given a chance at bronze
- So far for the 2010's, Action is back at rounding out the top 3

---

## Conclusion

Natural Language Processing is powerful, but flawed at generating word similarities for a game. The model is dependent on the corpora provided to generate the vectors that quantify the relationship between the words. Specifically, the model utilizes the context of the word in relation to the words surrounding it to calculate that relationship. What subsequently occurs is that word similarities are created simply due to the fact that they were utilized in a sentence together, with no understanding of how the words interact with one another. Yes, the model may tease out certain types of relationships - knowing dog is to puppy as cat is to kitten, or koalas is to Australia as pandas is to China. However, it falls short in other ones - knowing that cars is to road as plane is to... plane, Plane, path, roads, and highway were the top 5 answers the model generated. This is why it's vitally important to build feedback loops into the game so that answers may be confirmed by players, and perhaps even generate better answers that other players are more likely to submit as well. A multiplayer game would be able to provide real-time feedback loops for answers and even new custom word prompts. Despite its flaws, NLP demonstrates its usefulness in creating a foundation on which to build a game, without the need for a team of writers to create the content. While there may be growing pains with using NLP as the basis for a game in the beginning, with time and more data to feed into the model it will ultimately prove to fruitful, and hopefully, fun.

### Next Steps

- Deploy model to AWS
- Finish the game in Flask
- Generate more prompts
- Gather more data for more categories
- Create an analogies option

---

### Outside References:

- https://stackoverflow.com/questions/18039057/python-pandas-error-tokenizing-data
- https://www.machinelearningplus.com/python/python-regex-tutorial-examples/
- https://www.kaggle.com/nicapotato/bad-bad-words
- https://stackoverflow.com/questions/27324292/convert-word2vec-bin-file-to-text
- https://pythonhow.com/measure-execution-time-python-code/
- https://stackoverflow.com/questions/8784396/how-to-delete-the-words-between-two-delimiters
- https://github.com/danielwilentz/Cuisine-Classifier/blob/master/topic_modeling/clustering.ipynb
- https://medium.com/hanman/data-clustering-what-type-of-movies-are-in-the-imdb-top-250-7ef59372a93b
- https://stackoverflow.com/questions/44636370/scikit-learn-gridsearchcv-without-cross-validation-unsupervised-learning
- https://stackoverflow.com/questions/55143629/attributeerror-slice-object-has-no-attribute-flags-error
- https://towardsdatascience.com/step-by-step-tutorial-create-a-bar-chart-race-animation-da7d5fcd7079 
- https://www.numfys.net/howto/animations/
- https://pythonhow.com/measure-execution-time-python-code/
- https://www.shanelynn.ie/word-embeddings-in-python-with-spacy-and-gensim/
- https://github.com/abonello/riddle-game-app