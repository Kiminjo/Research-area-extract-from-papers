# Research area extract from papers

The major research areas are derived using the paper data of the researchers at Seoul National University of Science and Technology. This project was carried out as part of "Data and Business Innovation Lab."'s project.
<br></br>

## Extract research topics from papers

In this project, I used topic modeling and spherical K-means clustering to classify research area. The data used in the experiment are all papers by researchers at Seoul National University of Science and Technology published from 2016 to 2021.

Before proceeding with this project, various text analysis techniques such as topic modeling and doc2vec were applied. As a result of qualitatively analyzing the experimental results, it is judged that LDA and clustering show the best interpretation, and the main research fields are explained using them.

This study focused on visualization to clearly explain this rather than text analysis. Visualization techniques such as wordcloud and networks were actively used to clearly explain the research field derived from the analysis results.
<br></br>

## Dataset

For the data used in the project, all papers published by researchers at Seoul National University of Science and Technology from 2016 to May 2021 were used. Only the 'title' and 'introductions' were used among the paper texts.

Words 'paper', 'research', etc., which are commonly used in the paper, were designated as stopwords and removed from the text.
<br></br>

## Software Requirements

- python >= 3.5
- gensim
- scikit-learn
- pandas 
- numpy 
- nltk 
