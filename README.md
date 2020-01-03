# DatacampArticleSentimentAnalysis

This project aims to build a spam comments' classification system based on the comments that have already been posted on [DataCamp Community tutorials](http://datacamp.com/community/tutorials/). The comments are scraped using a tool called `UiPath Studio`. 

This work has seen different edge cases for training a deep learning model for real-world problems.
The edge cases are as followed:
1. Manual Data Collection:
    As the topic here is Spam Comment detection of Datacamp articles, to make it the portal specific the data has been scrapped from the Datacamp website itself. For this purpose, we have UiPath Studio(an RPA tool), due to some limitation of BeatifulSoup and other popular Python data scrapping libraries to deal with dynamic content.

2. Dealing with imbalanced data distribution:
    As the topic itself mentions, it is more likely that the number of spam comments received from the well-educated users of a popular website like Datacamp will be very less compared to its counterpart. 

    Because of that, we have seen average result while preparing our initial baseline model 0.89(f1-score).
    The work can be found here https://github.com/Desmond00/DatacampArticleSentimentAnalysis/blob/master/notebooks/BaselineModelPreparation.ipynb

    So we initiated to deal with the class imbalance. We have selected the data generation approach out of different approaches available to deal with the same. We have generated more data for SPAM comments by training an LSTM for the next word prediction. 
    The work can be found here https://github.com/Desmond00/DatacampArticleSentimentAnalysis/blob/master/notebooks/BaselineModelPreparation.ipynb

    After including the generated SPAM comments, we achieved considerably improved results 0.95(f1-score).
    The work can be found here https://github.com/Desmond00/DatacampArticleSentimentAnalysis/blob/master/notebooks/Extended-BaselineModelPreparation.ipynb


* `DatacampArtcieScraping` folder contains the working files resulted from Uipath Studio scripts.
* `commentsDataDetailed` is the repository containing different article comments along with some descriptions about the comment.
* `goodData` contains the master data regarding different articles.
* `notebooks` contains all the IPYNBs
* `backup` contains the backup of model and other Python object's pickle files
* `web` contains all the files to expose the model as API
