# Sentiment Analysis

## Using XGBoost in SageMaker

_Deep Learning Nanodegree Program | Deployment_

---

As our first example of using Amazon's SageMaker service we will construct a random tree model to predict the sentiment of a movie review. You may have seen a version of this example in a pervious lesson although it would have been done using the sklearn package. Instead, we will be using the XGBoost package as it is provided to us by Amazon.

## Instructions

Some template code has already been provided for you, and you will need to implement additional functionality to successfully complete this notebook. You will not need to modify the included code beyond what is requested. Sections that begin with '**TODO**' in the header indicate that you need to complete or implement some portion within them. Instructions will be provided for each section and the specifics of the implementation are marked in the code block with a `# TODO: ...` comment. Please be sure to read the instructions carefully!

In addition to implementing code, there may be questions for you to answer which relate to the task and your implementation. Each section where you will answer a question is preceded by a '**Question:**' header. Carefully read each question and provide your answer below the '**Answer:**' header by editing the Markdown cell.

> **Note**: Code and Markdown cells can be executed using the **Shift+Enter** keyboard shortcut. In addition, a cell can be edited by typically clicking it (double-click for Markdown cells) or by pressing **Enter** while it is highlighted.

## Step 1: Downloading the data

The dataset we are going to use is very popular among researchers in Natural Language Processing, usually referred to as the [IMDb dataset](http://ai.stanford.edu/~amaas/data/sentiment/). It consists of movie reviews from the website [imdb.com](http://www.imdb.com/), each labeled as either '**pos**itive', if the reviewer enjoyed the film, or '**neg**ative' otherwise.

> Maas, Andrew L., et al. [Learning Word Vectors for Sentiment Analysis](http://ai.stanford.edu/~amaas/data/sentiment/). In _Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies_. Association for Computational Linguistics, 2011.

We begin by using some Jupyter Notebook magic to download and extract the dataset.


```python
%mkdir ../data
!wget -O ../data/aclImdb_v1.tar.gz http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
!tar -zxf ../data/aclImdb_v1.tar.gz -C ../data
```

    mkdir: cannot create directory ‘../data’: File exists
    --2020-04-19 23:56:23--  http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
    Resolving ai.stanford.edu (ai.stanford.edu)... 171.64.68.10
    Connecting to ai.stanford.edu (ai.stanford.edu)|171.64.68.10|:80... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 84125825 (80M) [application/x-gzip]
    Saving to: ‘../data/aclImdb_v1.tar.gz’
    
    ../data/aclImdb_v1. 100%[===================>]  80.23M  23.2MB/s    in 5.1s    
    
    2020-04-19 23:56:28 (15.9 MB/s) - ‘../data/aclImdb_v1.tar.gz’ saved [84125825/84125825]
    


## Step 2: Preparing the data

The data we have downloaded is split into various files, each of which contains a single review. It will be much easier going forward if we combine these individual files into two large files, one for training and one for testing.


```python
import os
import glob

def read_imdb_data(data_dir='../data/aclImdb'):
    data = {}
    labels = {}
    
    for data_type in ['train', 'test']:
        data[data_type] = {}
        labels[data_type] = {}
        
        for sentiment in ['pos', 'neg']:
            data[data_type][sentiment] = []
            labels[data_type][sentiment] = []
            
            path = os.path.join(data_dir, data_type, sentiment, '*.txt')
            files = glob.glob(path)
            
            for f in files:
                with open(f) as review:
                    data[data_type][sentiment].append(review.read())
                    # Here we represent a positive review by '1' and a negative review by '0'
                    labels[data_type][sentiment].append(1 if sentiment == 'pos' else 0)
                    
            assert len(data[data_type][sentiment]) == len(labels[data_type][sentiment]), \
                    "{}/{} data size does not match labels size".format(data_type, sentiment)
                
    return data, labels
```


```python
data, labels = read_imdb_data()
print("IMDB reviews: train = {} pos / {} neg, test = {} pos / {} neg".format(
            len(data['train']['pos']), len(data['train']['neg']),
            len(data['test']['pos']), len(data['test']['neg'])))
```

    IMDB reviews: train = 12500 pos / 12500 neg, test = 12500 pos / 12500 neg



```python
from sklearn.utils import shuffle

def prepare_imdb_data(data, labels):
    """Prepare training and test sets from IMDb movie reviews."""
    
    #Combine positive and negative reviews and labels
    data_train = data['train']['pos'] + data['train']['neg']
    data_test = data['test']['pos'] + data['test']['neg']
    labels_train = labels['train']['pos'] + labels['train']['neg']
    labels_test = labels['test']['pos'] + labels['test']['neg']
    
    #Shuffle reviews and corresponding labels within training and test sets
    data_train, labels_train = shuffle(data_train, labels_train)
    data_test, labels_test = shuffle(data_test, labels_test)
    
    # Return a unified training data, test data, training labels, test labets
    return data_train, data_test, labels_train, labels_test
```


```python
train_X, test_X, train_y, test_y = prepare_imdb_data(data, labels)
print("IMDb reviews (combined): train = {}, test = {}".format(len(train_X), len(test_X)))
```

    IMDb reviews (combined): train = 25000, test = 25000



```python
train_X[100]
```




    'Before I start...let me say that I fully believe in God. I believe in Heaven and in Hell. Kay now that thats out of the way, I just wanna say that What in the world do these morons that call themselves "hosts" think they are doing?? The last time I checked a host doesn\'t discriminate, spew hatred filled rants on TV, or try to shove their own beliefs down every unfortunate soul that ventures onto the channel. ALl of these that crazy, idiotic, conservitive, bible thumping, Fred Phelps lover Pat Robertson does daily. I am all for free speech, but since when does that cover a guy who pretty much says that if you venture off his ideal way of life you are right away sent to hell? This is just a perfect example of why religion is the cause of SOOOO many problems. One day in my class room we had a substitute teacher in so we decided to watch some TV since the teach didn\'t give us any work. And we (against many of us\'s will) watched 700\'s Club, and of course that jerk Pat was on ranting and raving about the bible, and he said Simon along the lines of "God says Homosexuality is a sin" and I actually heard a kid go "Hmm I guess he\'s right." WTF??? Seriously, if the host is trying to make people think that someone else\'s sexual orientation is a huge sin, then they seriously need to take that host, duct tape them, and throw them off of a cruise liner in the middle of the arctic.'



## Step 3: Processing the data

Now that we have our training and testing datasets merged and ready to use, we need to start processing the raw data into something that will be useable by our machine learning algorithm. To begin with, we remove any html formatting that may appear in the reviews and perform some standard natural language processing in order to homogenize the data.


```python
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import *
stemmer = PorterStemmer()
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     /home/ec2-user/nltk_data...
    [nltk_data]   Unzipping corpora/stopwords.zip.



```python
import re
from bs4 import BeautifulSoup

def review_to_words(review):
    text = BeautifulSoup(review, "html.parser").get_text() # Remove HTML tags
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) # Convert to lower case
    words = text.split() # Split string into words
    words = [w for w in words if w not in stopwords.words("english")] # Remove stopwords
    words = [PorterStemmer().stem(w) for w in words] # stem
    
    return words
```


```python
import pickle

cache_dir = os.path.join("../cache", "sentiment_analysis")  # where to store cache files
os.makedirs(cache_dir, exist_ok=True)  # ensure cache directory exists

def preprocess_data(data_train, data_test, labels_train, labels_test,
                    cache_dir=cache_dir, cache_file="preprocessed_data.pkl"):
    """Convert each review to words; read from cache if available."""

    # If cache_file is not None, try to read from it first
    cache_data = None
    if cache_file is not None:
        try:
            with open(os.path.join(cache_dir, cache_file), "rb") as f:
                cache_data = pickle.load(f)
            print("Read preprocessed data from cache file:", cache_file)
        except:
            pass  # unable to read from cache, but that's okay
    
    # If cache is missing, then do the heavy lifting
    if cache_data is None:
        # Preprocess training and test data to obtain words for each review
        #words_train = list(map(review_to_words, data_train))
        #words_test = list(map(review_to_words, data_test))
        words_train = [review_to_words(review) for review in data_train]
        words_test = [review_to_words(review) for review in data_test]
        
        # Write to cache file for future runs
        if cache_file is not None:
            cache_data = dict(words_train=words_train, words_test=words_test,
                              labels_train=labels_train, labels_test=labels_test)
            with open(os.path.join(cache_dir, cache_file), "wb") as f:
                pickle.dump(cache_data, f)
            print("Wrote preprocessed data to cache file:", cache_file)
    else:
        # Unpack data loaded from cache file
        words_train, words_test, labels_train, labels_test = (cache_data['words_train'],
                cache_data['words_test'], cache_data['labels_train'], cache_data['labels_test'])
    
    return words_train, words_test, labels_train, labels_test
```


```python
# Preprocess data
train_X, test_X, train_y, test_y = preprocess_data(train_X, test_X, train_y, test_y)
```

    Read preprocessed data from cache file: preprocessed_data.pkl


### Extract Bag-of-Words features

For the model we will be implementing, rather than using the reviews directly, we are going to transform each review into a Bag-of-Words feature representation. Keep in mind that 'in the wild' we will only have access to the training set so our transformer can only use the training set to construct a representation.


```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
# joblib is an enhanced version of pickle that is more efficient for storing NumPy arrays

def extract_BoW_features(words_train, words_test, vocabulary_size=5000,
                         cache_dir=cache_dir, cache_file="bow_features.pkl"):
    """Extract Bag-of-Words for a given set of documents, already preprocessed into words."""
    
    # If cache_file is not None, try to read from it first
    cache_data = None
    if cache_file is not None:
        try:
            with open(os.path.join(cache_dir, cache_file), "rb") as f:
                cache_data = joblib.load(f)
            print("Read features from cache file:", cache_file)
        except:
            pass  # unable to read from cache, but that's okay
    
    # If cache is missing, then do the heavy lifting
    if cache_data is None:
        # Fit a vectorizer to training documents and use it to transform them
        # NOTE: Training documents have already been preprocessed and tokenized into words;
        #       pass in dummy functions to skip those steps, e.g. preprocessor=lambda x: x
        vectorizer = CountVectorizer(max_features=vocabulary_size,
                preprocessor=lambda x: x, tokenizer=lambda x: x)  # already preprocessed
        features_train = vectorizer.fit_transform(words_train).toarray()

        # Apply the same vectorizer to transform the test documents (ignore unknown words)
        features_test = vectorizer.transform(words_test).toarray()
        
        # NOTE: Remember to convert the features using .toarray() for a compact representation
        
        # Write to cache file for future runs (store vocabulary as well)
        if cache_file is not None:
            vocabulary = vectorizer.vocabulary_
            cache_data = dict(features_train=features_train, features_test=features_test,
                             vocabulary=vocabulary)
            with open(os.path.join(cache_dir, cache_file), "wb") as f:
                joblib.dump(cache_data, f)
            print("Wrote features to cache file:", cache_file)
    else:
        # Unpack data loaded from cache file
        features_train, features_test, vocabulary = (cache_data['features_train'],
                cache_data['features_test'], cache_data['vocabulary'])
    
    # Return both the extracted features as well as the vocabulary
    return features_train, features_test, vocabulary
```


```python
# Extract Bag of Words features for both training and test datasets
train_X, test_X, vocabulary = extract_BoW_features(train_X, test_X)
```

    Read features from cache file: bow_features.pkl


## Step 4: Classification using XGBoost

Now that we have created the feature representation of our training (and testing) data, it is time to start setting up and using the XGBoost classifier provided by SageMaker.

### (TODO) Writing the dataset

The XGBoost classifier that we will be using requires the dataset to be written to a file and stored using Amazon S3. To do this, we will start by splitting the training dataset into two parts, the data we will train the model with and a validation set. Then, we will write those datasets to a file and upload the files to S3. In addition, we will write the test set input to a file and upload the file to S3. This is so that we can use SageMakers Batch Transform functionality to test our model once we've fit it.


```python
import pandas as pd

# TODO: Split the train_X and train_y arrays into the DataFrames val_X, train_X and val_y, train_y. Make sure that
#       val_X and val_y contain 10 000 entires while train_X and train_y contain the remaining 15 000 entries.
#val_X = pd.DataFrame(None)
#train_X = pd.DataFrame(None)

#val_y = pd.DataFrame(None)
#train_y = pd.DataFrame(None)

# Solution:
# Earlier we shuffled the training dataset so to make things simple we can just assign
# the first 10 000 reviews to the validation set and use the remaining reviews for training.
val_X = pd.DataFrame(train_X[:10000])
train_X = pd.DataFrame(train_X[10000:])

val_y = pd.DataFrame(train_y[:10000])
train_y = pd.DataFrame(train_y[10000:])
```

The documentation for the XGBoost algorithm in SageMaker requires that the saved datasets should contain no headers or index and that for the training and validation data, the label should occur first for each sample.

For more information about this and other algorithms, the SageMaker developer documentation can be found on __[Amazon's website.](https://docs.aws.amazon.com/sagemaker/latest/dg/)__


```python
# First we make sure that the local directory in which we'd like to store the training and validation csv files exists.
data_dir = '../data/xgboost'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
```


```python
# First, save the test data to test.csv in the data_dir directory. Note that we do not save the associated ground truth
# labels, instead we will use them later to compare with our model output.

pd.DataFrame(test_X).to_csv(os.path.join(data_dir, 'test.csv'), header=False, index=False)

# TODO: Save the training and validation data to train.csv and validation.csv in the data_dir directory.
#       Make sure that the files you create are in the correct format.

# Solution:
pd.concat([val_y, val_X], axis=1).to_csv(os.path.join(data_dir, 'validation.csv'), header=False, index=False)
pd.concat([train_y, train_X], axis=1).to_csv(os.path.join(data_dir, 'train.csv'), header=False, index=False)
```


```python
# To save a bit of memory we can set text_X, train_X, val_X, train_y and val_y to None.

test_X = train_X = val_X = train_y = val_y = None
```

### (TODO) Uploading Training / Validation files to S3

Amazon's S3 service allows us to store files that can be access by both the built-in training models such as the XGBoost model we will be using as well as custom models such as the one we will see a little later.

For this, and most other tasks we will be doing using SageMaker, there are two methods we could use. The first is to use the low level functionality of SageMaker which requires knowing each of the objects involved in the SageMaker environment. The second is to use the high level functionality in which certain choices have been made on the user's behalf. The low level approach benefits from allowing the user a great deal of flexibility while the high level approach makes development much quicker. For our purposes we will opt to use the high level approach although using the low-level approach is certainly an option.

Recall the method `upload_data()` which is a member of object representing our current SageMaker session. What this method does is upload the data to the default bucket (which is created if it does not exist) into the path described by the key_prefix variable. To see this for yourself, once you have uploaded the data files, go to the S3 console and look to see where the files have been uploaded.

For additional resources, see the __[SageMaker API documentation](http://sagemaker.readthedocs.io/en/latest/)__ and in addition the __[SageMaker Developer Guide.](https://docs.aws.amazon.com/sagemaker/latest/dg/)__


```python
import sagemaker

session = sagemaker.Session() # Store the current SageMaker session

# S3 prefix (which folder will we use)
prefix = 'sentiment-xgboost'

# TODO: Upload the test.csv, train.csv and validation.csv files which are contained in data_dir to S3 using sess.upload_data().
test_location = None
val_location = None
train_location = None

# Solution:
test_location = session.upload_data(os.path.join(data_dir, 'test.csv'), key_prefix=prefix)
val_location = session.upload_data(os.path.join(data_dir, 'validation.csv'), key_prefix=prefix)
train_location = session.upload_data(os.path.join(data_dir, 'train.csv'), key_prefix=prefix)
```

### (TODO) Creating the XGBoost model

Now that the data has been uploaded it is time to create the XGBoost model. To begin with, we need to do some setup. At this point it is worth discussing what a model is in SageMaker. It is easiest to think of a model of comprising three different objects in the SageMaker ecosystem, which interact with one another.

- Model Artifacts
- Training Code (Container)
- Inference Code (Container)

The Model Artifacts are what you might think of as the actual model itself. For example, if you were building a neural network, the model artifacts would be the weights of the various layers. In our case, for an XGBoost model, the artifacts are the actual trees that are created during training.

The other two objects, the training code and the inference code are then used the manipulate the training artifacts. More precisely, the training code uses the training data that is provided and creates the model artifacts, while the inference code uses the model artifacts to make predictions on new data.

The way that SageMaker runs the training and inference code is by making use of Docker containers. For now, think of a container as being a way of packaging code up so that dependencies aren't an issue.


```python
from sagemaker import get_execution_role

# Our current execution role is require when creating the model as the training
# and inference code will need to access the model artifacts.
role = get_execution_role()
```


```python
# We need to retrieve the location of the container which is provided by Amazon for using XGBoost.
# As a matter of convenience, the training and inference code both use the same container.
from sagemaker.amazon.amazon_estimator import get_image_uri

container = get_image_uri(session.boto_region_name, 'xgboost')
```

    WARNING:root:There is a more up to date SageMaker XGBoost image. To use the newer image, please set 'repo_version'='0.90-1'. For example:
    	get_image_uri(region, 'xgboost', '0.90-1').



```python
# TODO: Create a SageMaker estimator using the container location determined in the previous cell.
#       It is recommended that you use a single training instance of type ml.m4.xlarge. It is also
#       recommended that you use 's3://{}/{}/output'.format(session.default_bucket(), prefix) as the
#       output path.

xgb = None

# Solution:
xgb = sagemaker.estimator.Estimator(container, # The location of the container we wish to use
                                    role,                                    # What is our current IAM Role
                                    train_instance_count=1,                  # How many compute instances
                                    train_instance_type='ml.m4.xlarge',      # What kind of compute instances
                                    output_path='s3://{}/{}/output'.format(session.default_bucket(), prefix),
                                    sagemaker_session=session)

# TODO: Set the XGBoost hyperparameters in the xgb object. Don't forget that in this case we have a binary
#       label so we should be using the 'binary:logistic' objective.

# Solution:
xgb.set_hyperparameters(max_depth=5,
                        eta=0.2,
                        gamma=4,
                        min_child_weight=6,
                        subsample=0.8,
                        silent=0,
                        objective='binary:logistic',
                        early_stopping_rounds=10,
                        num_round=500)
```

### Fit the XGBoost model

Now that our model has been set up we simply need to attach the training and validation datasets and then ask SageMaker to set up the computation.


```python
s3_input_train = sagemaker.s3_input(s3_data=train_location, content_type='csv')
s3_input_validation = sagemaker.s3_input(s3_data=val_location, content_type='csv')
```


```python
xgb.fit({'train': s3_input_train, 'validation': s3_input_validation})
```

    2020-04-20 00:17:16 Starting - Starting the training job...
    2020-04-20 00:17:17 Starting - Launching requested ML instances...
    2020-04-20 00:18:15 Starting - Preparing the instances for training......
    2020-04-20 00:19:04 Downloading - Downloading input data...
    2020-04-20 00:19:28 Training - Downloading the training image..[34mArguments: train[0m
    [34m[2020-04-20:00:19:51:INFO] Running standalone xgboost training.[0m
    [34m[2020-04-20:00:19:51:INFO] File size need to be processed in the node: 238.47mb. Available memory size in the node: 8493.46mb[0m
    [34m[2020-04-20:00:19:51:INFO] Determined delimiter of CSV input is ','[0m
    [34m[00:19:51] S3DistributionType set as FullyReplicated[0m
    [34m[00:19:53] 15000x5000 matrix with 75000000 entries loaded from /opt/ml/input/data/train?format=csv&label_column=0&delimiter=,[0m
    [34m[2020-04-20:00:19:53:INFO] Determined delimiter of CSV input is ','[0m
    [34m[00:19:53] S3DistributionType set as FullyReplicated[0m
    [34m[00:19:54] 10000x5000 matrix with 50000000 entries loaded from /opt/ml/input/data/validation?format=csv&label_column=0&delimiter=,[0m
    [34m[00:19:57] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 44 extra nodes, 4 pruned nodes, max_depth=5[0m
    [34m[0]#011train-error:0.291933#011validation-error:0.3111[0m
    [34mMultiple eval metrics have been passed: 'validation-error' will be used for early stopping.
    [0m
    [34mWill train until validation-error hasn't improved in 10 rounds.[0m
    [34m[00:19:58] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 36 extra nodes, 8 pruned nodes, max_depth=5[0m
    [34m[1]#011train-error:0.2732#011validation-error:0.2935[0m
    [34m[00:20:00] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 42 extra nodes, 6 pruned nodes, max_depth=5[0m
    [34m[2]#011train-error:0.272533#011validation-error:0.2933[0m
    
    2020-04-20 00:19:50 Training - Training image download completed. Training in progress.[34m[00:20:01] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 26 extra nodes, 4 pruned nodes, max_depth=5[0m
    [34m[3]#011train-error:0.258333#011validation-error:0.2837[0m
    [34m[00:20:03] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 40 extra nodes, 4 pruned nodes, max_depth=5[0m
    [34m[4]#011train-error:0.2596#011validation-error:0.283[0m
    [34m[00:20:04] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 40 extra nodes, 4 pruned nodes, max_depth=5[0m
    [34m[5]#011train-error:0.254#011validation-error:0.28[0m
    [34m[00:20:05] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 32 extra nodes, 14 pruned nodes, max_depth=5[0m
    [34m[6]#011train-error:0.243533#011validation-error:0.2685[0m
    [34m[00:20:07] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 28 extra nodes, 6 pruned nodes, max_depth=5[0m
    [34m[7]#011train-error:0.234867#011validation-error:0.2565[0m
    [34m[00:20:08] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 30 extra nodes, 6 pruned nodes, max_depth=5[0m
    [34m[8]#011train-error:0.2306#011validation-error:0.2532[0m
    [34m[00:20:09] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 26 extra nodes, 6 pruned nodes, max_depth=5[0m
    [34m[9]#011train-error:0.227933#011validation-error:0.2498[0m
    [34m[00:20:10] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 34 extra nodes, 6 pruned nodes, max_depth=5[0m
    [34m[10]#011train-error:0.2196#011validation-error:0.2427[0m
    [34m[00:20:12] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 24 extra nodes, 10 pruned nodes, max_depth=5[0m
    [34m[11]#011train-error:0.2184#011validation-error:0.2415[0m
    [34m[00:20:13] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 30 extra nodes, 14 pruned nodes, max_depth=5[0m
    [34m[12]#011train-error:0.211133#011validation-error:0.2352[0m
    [34m[00:20:14] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 16 extra nodes, 14 pruned nodes, max_depth=5[0m
    [34m[13]#011train-error:0.206#011validation-error:0.2303[0m
    [34m[00:20:15] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 28 extra nodes, 8 pruned nodes, max_depth=5[0m
    [34m[14]#011train-error:0.199867#011validation-error:0.2243[0m
    [34m[00:20:17] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 26 extra nodes, 8 pruned nodes, max_depth=5[0m
    [34m[15]#011train-error:0.195267#011validation-error:0.22[0m
    [34m[00:20:18] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 30 extra nodes, 4 pruned nodes, max_depth=5[0m
    [34m[16]#011train-error:0.194067#011validation-error:0.2192[0m
    [34m[00:20:19] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 22 extra nodes, 10 pruned nodes, max_depth=5[0m
    [34m[17]#011train-error:0.190067#011validation-error:0.2164[0m
    [34m[00:20:20] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 26 extra nodes, 10 pruned nodes, max_depth=5[0m
    [34m[18]#011train-error:0.188#011validation-error:0.2159[0m
    [34m[00:20:22] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 28 extra nodes, 6 pruned nodes, max_depth=5[0m
    [34m[19]#011train-error:0.185533#011validation-error:0.2124[0m
    [34m[00:20:23] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 22 extra nodes, 8 pruned nodes, max_depth=5[0m
    [34m[20]#011train-error:0.1864#011validation-error:0.2113[0m
    [34m[00:20:24] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 28 extra nodes, 6 pruned nodes, max_depth=5[0m
    [34m[21]#011train-error:0.1826#011validation-error:0.2105[0m
    [34m[00:20:26] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 28 extra nodes, 6 pruned nodes, max_depth=5[0m
    [34m[22]#011train-error:0.180867#011validation-error:0.2097[0m
    [34m[00:20:27] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 26 extra nodes, 16 pruned nodes, max_depth=5[0m
    [34m[23]#011train-error:0.178133#011validation-error:0.2069[0m
    [34m[00:20:28] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 42 extra nodes, 4 pruned nodes, max_depth=5[0m
    [34m[24]#011train-error:0.174867#011validation-error:0.2045[0m
    [34m[00:20:29] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 32 extra nodes, 10 pruned nodes, max_depth=5[0m
    [34m[25]#011train-error:0.174067#011validation-error:0.2027[0m
    [34m[00:20:31] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 36 extra nodes, 4 pruned nodes, max_depth=5[0m
    [34m[26]#011train-error:0.171867#011validation-error:0.2014[0m
    [34m[00:20:32] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 30 extra nodes, 4 pruned nodes, max_depth=5[0m
    [34m[27]#011train-error:0.169#011validation-error:0.199[0m
    [34m[00:20:33] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 16 extra nodes, 14 pruned nodes, max_depth=5[0m
    [34m[28]#011train-error:0.1668#011validation-error:0.1975[0m
    [34m[00:20:34] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 24 extra nodes, 10 pruned nodes, max_depth=5[0m
    [34m[29]#011train-error:0.1648#011validation-error:0.1964[0m
    [34m[00:20:36] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 30 extra nodes, 4 pruned nodes, max_depth=5[0m
    [34m[30]#011train-error:0.163267#011validation-error:0.196[0m
    [34m[00:20:37] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 34 extra nodes, 10 pruned nodes, max_depth=5[0m
    [34m[31]#011train-error:0.161133#011validation-error:0.1927[0m
    [34m[00:20:38] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 24 extra nodes, 16 pruned nodes, max_depth=5[0m
    [34m[32]#011train-error:0.160733#011validation-error:0.1914[0m
    [34m[00:20:40] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 28 extra nodes, 8 pruned nodes, max_depth=5[0m
    [34m[33]#011train-error:0.1598#011validation-error:0.1904[0m
    [34m[00:20:41] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 30 extra nodes, 4 pruned nodes, max_depth=5[0m
    [34m[34]#011train-error:0.156533#011validation-error:0.1873[0m
    [34m[00:20:42] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 16 extra nodes, 12 pruned nodes, max_depth=5[0m
    [34m[35]#011train-error:0.1558#011validation-error:0.1862[0m
    [34m[00:20:43] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 20 extra nodes, 14 pruned nodes, max_depth=5[0m
    [34m[36]#011train-error:0.153933#011validation-error:0.1849[0m
    [34m[00:20:45] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 16 extra nodes, 16 pruned nodes, max_depth=5[0m
    [34m[37]#011train-error:0.152467#011validation-error:0.1841[0m
    [34m[00:20:46] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 34 extra nodes, 10 pruned nodes, max_depth=5[0m
    [34m[38]#011train-error:0.150133#011validation-error:0.1837[0m
    [34m[00:20:47] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 16 extra nodes, 16 pruned nodes, max_depth=5[0m
    [34m[39]#011train-error:0.149467#011validation-error:0.1829[0m
    [34m[00:20:48] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 16 extra nodes, 12 pruned nodes, max_depth=5[0m
    [34m[40]#011train-error:0.1496#011validation-error:0.1828[0m
    [34m[00:20:50] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 26 extra nodes, 12 pruned nodes, max_depth=5[0m
    [34m[41]#011train-error:0.148#011validation-error:0.181[0m
    [34m[00:20:51] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 16 extra nodes, 16 pruned nodes, max_depth=5[0m
    [34m[42]#011train-error:0.148267#011validation-error:0.1803[0m
    [34m[00:20:52] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 38 extra nodes, 6 pruned nodes, max_depth=5[0m
    [34m[43]#011train-error:0.145933#011validation-error:0.1802[0m
    [34m[00:20:53] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 22 extra nodes, 2 pruned nodes, max_depth=5[0m
    [34m[44]#011train-error:0.145467#011validation-error:0.1805[0m
    [34m[00:20:55] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 28 extra nodes, 6 pruned nodes, max_depth=5[0m
    [34m[45]#011train-error:0.143867#011validation-error:0.1803[0m
    [34m[00:20:56] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 18 extra nodes, 10 pruned nodes, max_depth=5[0m
    [34m[46]#011train-error:0.142667#011validation-error:0.179[0m
    [34m[00:20:57] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 22 extra nodes, 8 pruned nodes, max_depth=5[0m
    [34m[47]#011train-error:0.1414#011validation-error:0.1775[0m
    [34m[00:20:58] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 24 extra nodes, 10 pruned nodes, max_depth=5[0m
    [34m[48]#011train-error:0.1402#011validation-error:0.1769[0m
    [34m[00:21:00] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 22 extra nodes, 16 pruned nodes, max_depth=5[0m
    [34m[49]#011train-error:0.138733#011validation-error:0.1752[0m
    [34m[00:21:01] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 14 extra nodes, 12 pruned nodes, max_depth=5[0m
    [34m[50]#011train-error:0.1386#011validation-error:0.1743[0m
    [34m[00:21:02] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 26 extra nodes, 10 pruned nodes, max_depth=5[0m
    [34m[51]#011train-error:0.137667#011validation-error:0.1732[0m
    [34m[00:21:03] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 16 extra nodes, 8 pruned nodes, max_depth=5[0m
    [34m[52]#011train-error:0.136067#011validation-error:0.1729[0m
    [34m[00:21:05] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 16 extra nodes, 8 pruned nodes, max_depth=5[0m
    [34m[53]#011train-error:0.135733#011validation-error:0.1709[0m
    [34m[00:21:06] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 28 extra nodes, 14 pruned nodes, max_depth=5[0m
    [34m[54]#011train-error:0.1346#011validation-error:0.1702[0m
    [34m[00:21:07] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 20 extra nodes, 6 pruned nodes, max_depth=5[0m
    [34m[55]#011train-error:0.1352#011validation-error:0.1719[0m
    [34m[00:21:09] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 18 extra nodes, 10 pruned nodes, max_depth=5[0m
    [34m[56]#011train-error:0.1346#011validation-error:0.1705[0m
    [34m[00:21:10] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 24 extra nodes, 8 pruned nodes, max_depth=5[0m
    [34m[57]#011train-error:0.1324#011validation-error:0.1699[0m
    [34m[00:21:11] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 40 extra nodes, 8 pruned nodes, max_depth=5[0m
    [34m[58]#011train-error:0.130667#011validation-error:0.1693[0m
    [34m[00:21:12] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 10 extra nodes, 16 pruned nodes, max_depth=5[0m
    [34m[59]#011train-error:0.1294#011validation-error:0.1685[0m
    [34m[00:21:13] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 20 extra nodes, 10 pruned nodes, max_depth=5[0m
    [34m[60]#011train-error:0.1288#011validation-error:0.1679[0m
    [34m[00:21:15] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 16 extra nodes, 14 pruned nodes, max_depth=5[0m
    [34m[61]#011train-error:0.1276#011validation-error:0.1677[0m
    [34m[00:21:16] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 24 extra nodes, 6 pruned nodes, max_depth=5[0m
    [34m[62]#011train-error:0.127267#011validation-error:0.1668[0m
    [34m[00:21:17] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 22 extra nodes, 6 pruned nodes, max_depth=5[0m
    [34m[63]#011train-error:0.126733#011validation-error:0.1682[0m
    [34m[00:21:19] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 24 extra nodes, 2 pruned nodes, max_depth=5[0m
    [34m[64]#011train-error:0.126#011validation-error:0.1665[0m
    [34m[00:21:20] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 20 extra nodes, 14 pruned nodes, max_depth=5[0m
    [34m[65]#011train-error:0.125667#011validation-error:0.1655[0m
    [34m[00:21:21] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 22 extra nodes, 10 pruned nodes, max_depth=5[0m
    [34m[66]#011train-error:0.1246#011validation-error:0.1651[0m
    [34m[00:21:22] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 16 extra nodes, 14 pruned nodes, max_depth=5[0m
    [34m[67]#011train-error:0.124467#011validation-error:0.1657[0m
    [34m[00:21:24] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 30 extra nodes, 8 pruned nodes, max_depth=5[0m
    [34m[68]#011train-error:0.122333#011validation-error:0.1652[0m
    [34m[00:21:25] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 24 extra nodes, 14 pruned nodes, max_depth=5[0m
    [34m[69]#011train-error:0.121933#011validation-error:0.1653[0m
    [34m[00:21:26] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 12 extra nodes, 6 pruned nodes, max_depth=5[0m
    [34m[70]#011train-error:0.121667#011validation-error:0.1646[0m
    [34m[00:21:27] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 18 extra nodes, 8 pruned nodes, max_depth=5[0m
    [34m[71]#011train-error:0.121#011validation-error:0.1654[0m
    [34m[00:21:29] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 16 extra nodes, 10 pruned nodes, max_depth=5[0m
    [34m[72]#011train-error:0.1206#011validation-error:0.1643[0m
    [34m[00:21:30] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 16 extra nodes, 10 pruned nodes, max_depth=5[0m
    [34m[73]#011train-error:0.1198#011validation-error:0.1647[0m
    [34m[00:21:31] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 14 extra nodes, 6 pruned nodes, max_depth=5[0m
    [34m[74]#011train-error:0.119733#011validation-error:0.1644[0m
    [34m[00:21:32] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 26 extra nodes, 6 pruned nodes, max_depth=5[0m
    [34m[75]#011train-error:0.120133#011validation-error:0.1629[0m
    [34m[00:21:34] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 12 extra nodes, 10 pruned nodes, max_depth=5[0m
    [34m[76]#011train-error:0.118867#011validation-error:0.1634[0m
    [34m[00:21:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 22 extra nodes, 10 pruned nodes, max_depth=5[0m
    [34m[77]#011train-error:0.1186#011validation-error:0.1626[0m
    [34m[00:21:36] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 18 extra nodes, 18 pruned nodes, max_depth=5[0m
    [34m[78]#011train-error:0.118333#011validation-error:0.1618[0m
    [34m[00:21:37] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 12 extra nodes, 4 pruned nodes, max_depth=5[0m
    [34m[79]#011train-error:0.117533#011validation-error:0.1616[0m
    [34m[00:21:39] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 18 extra nodes, 10 pruned nodes, max_depth=5[0m
    [34m[80]#011train-error:0.116667#011validation-error:0.1606[0m
    [34m[00:21:40] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 14 extra nodes, 8 pruned nodes, max_depth=5[0m
    [34m[81]#011train-error:0.116333#011validation-error:0.1612[0m
    [34m[00:21:41] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 16 extra nodes, 10 pruned nodes, max_depth=5[0m
    [34m[82]#011train-error:0.115733#011validation-error:0.1608[0m
    [34m[00:21:42] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 16 extra nodes, 12 pruned nodes, max_depth=5[0m
    [34m[83]#011train-error:0.114267#011validation-error:0.161[0m
    [34m[00:21:44] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 20 extra nodes, 18 pruned nodes, max_depth=5[0m
    [34m[84]#011train-error:0.1134#011validation-error:0.1612[0m
    [34m[00:21:45] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 10 extra nodes, 4 pruned nodes, max_depth=5[0m
    [34m[85]#011train-error:0.113133#011validation-error:0.1608[0m
    [34m[00:21:46] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 26 extra nodes, 10 pruned nodes, max_depth=5[0m
    [34m[86]#011train-error:0.1128#011validation-error:0.1603[0m
    [34m[00:21:47] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 12 extra nodes, 14 pruned nodes, max_depth=5[0m
    [34m[87]#011train-error:0.112#011validation-error:0.1601[0m
    [34m[00:21:49] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 22 extra nodes, 10 pruned nodes, max_depth=5[0m
    [34m[88]#011train-error:0.112067#011validation-error:0.1598[0m
    [34m[00:21:50] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 10 extra nodes, 10 pruned nodes, max_depth=5[0m
    [34m[89]#011train-error:0.112#011validation-error:0.1597[0m
    [34m[00:21:51] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 20 extra nodes, 10 pruned nodes, max_depth=5[0m
    [34m[90]#011train-error:0.112#011validation-error:0.1606[0m
    [34m[00:21:53] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 10 extra nodes, 16 pruned nodes, max_depth=5[0m
    [34m[91]#011train-error:0.111667#011validation-error:0.1603[0m
    [34m[00:21:54] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 20 extra nodes, 4 pruned nodes, max_depth=5[0m
    [34m[92]#011train-error:0.111267#011validation-error:0.1596[0m
    [34m[00:21:55] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 14 extra nodes, 16 pruned nodes, max_depth=5[0m
    [34m[93]#011train-error:0.111533#011validation-error:0.1597[0m
    [34m[00:21:56] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 28 extra nodes, 2 pruned nodes, max_depth=5[0m
    [34m[94]#011train-error:0.1106#011validation-error:0.1593[0m
    [34m[00:21:58] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 20 extra nodes, 8 pruned nodes, max_depth=5[0m
    [34m[95]#011train-error:0.1102#011validation-error:0.159[0m
    [34m[00:21:59] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 18 extra nodes, 12 pruned nodes, max_depth=5[0m
    [34m[96]#011train-error:0.1094#011validation-error:0.1585[0m
    [34m[00:22:00] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 16 extra nodes, 16 pruned nodes, max_depth=5[0m
    [34m[97]#011train-error:0.1084#011validation-error:0.1583[0m
    [34m[00:22:01] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 10 extra nodes, 4 pruned nodes, max_depth=5[0m
    [34m[98]#011train-error:0.108267#011validation-error:0.1568[0m
    [34m[00:22:03] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 10 extra nodes, 12 pruned nodes, max_depth=5[0m
    [34m[99]#011train-error:0.107733#011validation-error:0.1564[0m
    [34m[00:22:04] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 16 extra nodes, 10 pruned nodes, max_depth=5[0m
    [34m[100]#011train-error:0.107467#011validation-error:0.1557[0m
    [34m[00:22:05] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 22 extra nodes, 10 pruned nodes, max_depth=5[0m
    [34m[101]#011train-error:0.106667#011validation-error:0.1558[0m
    [34m[00:22:06] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 10 extra nodes, 10 pruned nodes, max_depth=5[0m
    [34m[102]#011train-error:0.106067#011validation-error:0.1551[0m
    [34m[00:22:08] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 16 extra nodes, 6 pruned nodes, max_depth=5[0m
    [34m[103]#011train-error:0.105667#011validation-error:0.1541[0m
    [34m[00:22:09] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 28 extra nodes, 8 pruned nodes, max_depth=5[0m
    [34m[104]#011train-error:0.1046#011validation-error:0.1543[0m
    [34m[00:22:10] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 12 extra nodes, 8 pruned nodes, max_depth=5[0m
    [34m[105]#011train-error:0.104#011validation-error:0.1539[0m
    [34m[00:22:11] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 22 extra nodes, 8 pruned nodes, max_depth=5[0m
    [34m[106]#011train-error:0.104267#011validation-error:0.1531[0m
    [34m[00:22:13] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 10 extra nodes, 6 pruned nodes, max_depth=5[0m
    [34m[107]#011train-error:0.104133#011validation-error:0.153[0m
    [34m[00:22:14] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 10 extra nodes, 4 pruned nodes, max_depth=5[0m
    [34m[108]#011train-error:0.1044#011validation-error:0.153[0m
    [34m[00:22:15] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 10 extra nodes, 14 pruned nodes, max_depth=5[0m
    [34m[109]#011train-error:0.104333#011validation-error:0.1527[0m
    [34m[00:22:16] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 24 extra nodes, 12 pruned nodes, max_depth=5[0m
    [34m[110]#011train-error:0.103#011validation-error:0.1521[0m
    [34m[00:22:18] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 12 extra nodes, 2 pruned nodes, max_depth=5[0m
    [34m[111]#011train-error:0.102667#011validation-error:0.1516[0m
    [34m[00:22:19] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 18 extra nodes, 8 pruned nodes, max_depth=5[0m
    [34m[112]#011train-error:0.102267#011validation-error:0.1517[0m
    [34m[00:22:20] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 20 extra nodes, 18 pruned nodes, max_depth=5[0m
    [34m[113]#011train-error:0.102667#011validation-error:0.1517[0m
    [34m[00:22:22] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 18 extra nodes, 8 pruned nodes, max_depth=5[0m
    [34m[114]#011train-error:0.103#011validation-error:0.1507[0m
    [34m[00:22:23] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 12 extra nodes, 2 pruned nodes, max_depth=5[0m
    [34m[115]#011train-error:0.102533#011validation-error:0.151[0m
    [34m[00:22:24] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 16 extra nodes, 6 pruned nodes, max_depth=5[0m
    [34m[116]#011train-error:0.101933#011validation-error:0.1507[0m
    [34m[00:22:25] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 14 extra nodes, 10 pruned nodes, max_depth=5[0m
    [34m[117]#011train-error:0.102#011validation-error:0.1509[0m
    [34m[00:22:27] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 14 extra nodes, 6 pruned nodes, max_depth=5[0m
    [34m[118]#011train-error:0.101667#011validation-error:0.1503[0m
    [34m[00:22:28] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 22 extra nodes, 12 pruned nodes, max_depth=5[0m
    [34m[119]#011train-error:0.100733#011validation-error:0.1497[0m
    [34m[00:22:29] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 14 extra nodes, 14 pruned nodes, max_depth=5[0m
    [34m[120]#011train-error:0.100867#011validation-error:0.1498[0m
    [34m[00:22:30] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 18 extra nodes, 22 pruned nodes, max_depth=5[0m
    [34m[121]#011train-error:0.101067#011validation-error:0.1504[0m
    [34m[00:22:32] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 10 extra nodes, 10 pruned nodes, max_depth=5[0m
    [34m[122]#011train-error:0.1016#011validation-error:0.1496[0m
    [34m[00:22:33] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 16 extra nodes, 10 pruned nodes, max_depth=5[0m
    [34m[123]#011train-error:0.101#011validation-error:0.1492[0m
    [34m[00:22:34] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 16 extra nodes, 14 pruned nodes, max_depth=5[0m
    [34m[124]#011train-error:0.100933#011validation-error:0.1485[0m
    [34m[00:22:36] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 10 extra nodes, 6 pruned nodes, max_depth=5[0m
    [34m[125]#011train-error:0.101067#011validation-error:0.1484[0m
    [34m[00:22:37] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 12 extra nodes, 12 pruned nodes, max_depth=5[0m
    [34m[126]#011train-error:0.099733#011validation-error:0.149[0m
    [34m[00:22:38] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 14 extra nodes, 6 pruned nodes, max_depth=5[0m
    [34m[127]#011train-error:0.099267#011validation-error:0.1494[0m
    [34m[00:22:39] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 12 extra nodes, 14 pruned nodes, max_depth=5[0m
    [34m[128]#011train-error:0.099933#011validation-error:0.1486[0m
    [34m[00:22:41] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 14 extra nodes, 2 pruned nodes, max_depth=5[0m
    [34m[129]#011train-error:0.099667#011validation-error:0.1486[0m
    [34m[00:22:42] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 24 extra nodes, 8 pruned nodes, max_depth=5[0m
    [34m[130]#011train-error:0.099133#011validation-error:0.1484[0m
    [34m[00:22:43] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 10 extra nodes, 4 pruned nodes, max_depth=5[0m
    [34m[131]#011train-error:0.0992#011validation-error:0.1482[0m
    [34m[00:22:44] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 14 extra nodes, 10 pruned nodes, max_depth=5[0m
    [34m[132]#011train-error:0.099#011validation-error:0.1479[0m
    [34m[00:22:46] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 14 extra nodes, 6 pruned nodes, max_depth=5[0m
    [34m[133]#011train-error:0.098067#011validation-error:0.1481[0m
    [34m[00:22:47] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 20 extra nodes, 4 pruned nodes, max_depth=5[0m
    [34m[134]#011train-error:0.0974#011validation-error:0.1482[0m
    [34m[00:22:48] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 16 extra nodes, 8 pruned nodes, max_depth=5[0m
    [34m[135]#011train-error:0.0976#011validation-error:0.1488[0m
    [34m[00:22:49] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 24 extra nodes, 8 pruned nodes, max_depth=5[0m
    [34m[136]#011train-error:0.096533#011validation-error:0.1498[0m
    [34m[00:22:51] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 20 extra nodes, 6 pruned nodes, max_depth=5[0m
    [34m[137]#011train-error:0.095733#011validation-error:0.1496[0m
    [34m[00:22:52] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 12 extra nodes, 4 pruned nodes, max_depth=5[0m
    [34m[138]#011train-error:0.095067#011validation-error:0.1496[0m
    [34m[00:22:53] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 16 extra nodes, 10 pruned nodes, max_depth=5[0m
    [34m[139]#011train-error:0.094333#011validation-error:0.1483[0m
    [34m[00:22:54] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 10 extra nodes, 10 pruned nodes, max_depth=5[0m
    [34m[140]#011train-error:0.094#011validation-error:0.1482[0m
    [34m[00:22:56] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 18 extra nodes, 14 pruned nodes, max_depth=5[0m
    [34m[141]#011train-error:0.0932#011validation-error:0.1481[0m
    [34m[00:22:57] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 10 extra nodes, 0 pruned nodes, max_depth=5[0m
    [34m[142]#011train-error:0.092733#011validation-error:0.1479[0m
    [34mStopping. Best iteration:[0m
    [34m[132]#011train-error:0.099#011validation-error:0.1479
    [0m
    
    2020-04-20 00:23:05 Uploading - Uploading generated training model
    2020-04-20 00:23:05 Completed - Training job completed
    Training seconds: 241
    Billable seconds: 241


### (TODO) Testing the model

Now that we've fit our XGBoost model, it's time to see how well it performs. To do this we will use SageMakers Batch Transform functionality. Batch Transform is a convenient way to perform inference on a large dataset in a way that is not realtime. That is, we don't necessarily need to use our model's results immediately and instead we can peform inference on a large number of samples. An example of this in industry might be peforming an end of month report. This method of inference can also be useful to us as it means to can perform inference on our entire test set. 

To perform a Batch Transformation we need to first create a transformer objects from our trained estimator object.


```python
# TODO: Create a transformer object from the trained model. Using an instance count of 1 and an instance type of ml.m4.xlarge
#       should be more than enough.
xgb_transformer = xgb.transformer(instance_count = 1, instance_type = 'ml.m4.xlarge')
```

Next we actually perform the transform job. When doing so we need to make sure to specify the type of data we are sending so that it is serialized correctly in the background. In our case we are providing our model with csv data so we specify `text/csv`. Also, if the test data that we have provided is too large to process all at once then we need to specify how the data file should be split up. Since each line is a single entry in our data set we tell SageMaker that it can split the input on each line.


```python
# TODO: Start the transform job. Make sure to specify the content type and the split type of the test data.
xgb_transformer.transform(test_location, content_type='text/csv', split_type='Line')
```

Currently the transform job is running but it is doing so in the background. Since we wish to wait until the transform job is done and we would like a bit of feedback we can run the `wait()` method.


```python
xgb_transformer.wait()
```

    .....................[34mArguments: serve[0m
    [34m[2020-04-20 00:31:34 +0000] [1] [INFO] Starting gunicorn 19.7.1[0m
    [34m[2020-04-20 00:31:34 +0000] [1] [INFO] Listening at: http://0.0.0.0:8080 (1)[0m
    [34m[2020-04-20 00:31:34 +0000] [1] [INFO] Using worker: gevent[0m
    [34m[2020-04-20 00:31:34 +0000] [38] [INFO] Booting worker with pid: 38[0m
    [34m[2020-04-20 00:31:34 +0000] [39] [INFO] Booting worker with pid: 39[0m
    [34m[2020-04-20 00:31:34 +0000] [40] [INFO] Booting worker with pid: 40[0m
    [34m[2020-04-20:00:31:34:INFO] Model loaded successfully for worker : 38[0m
    [34m[2020-04-20 00:31:34 +0000] [41] [INFO] Booting worker with pid: 41[0m
    [34m[2020-04-20:00:31:34:INFO] Model loaded successfully for worker : 39[0m
    [34m[2020-04-20:00:31:34:INFO] Model loaded successfully for worker : 40[0m
    [34m[2020-04-20:00:31:34:INFO] Model loaded successfully for worker : 41[0m
    [34m[2020-04-20:00:32:07:INFO] Sniff delimiter as ','[0m
    [34m[2020-04-20:00:32:07:INFO] Determined delimiter of CSV input is ','[0m
    [34m[2020-04-20:00:32:07:INFO] Sniff delimiter as ','[0m
    [34m[2020-04-20:00:32:07:INFO] Determined delimiter of CSV input is ','[0m
    [34m[2020-04-20:00:32:08:INFO] Sniff delimiter as ','[0m
    [34m[2020-04-20:00:32:08:INFO] Determined delimiter of CSV input is ','[0m
    [35m[2020-04-20:00:32:07:INFO] Sniff delimiter as ','[0m
    [35m[2020-04-20:00:32:07:INFO] Determined delimiter of CSV input is ','[0m
    [35m[2020-04-20:00:32:07:INFO] Sniff delimiter as ','[0m
    [35m[2020-04-20:00:32:07:INFO] Determined delimiter of CSV input is ','[0m
    [35m[2020-04-20:00:32:08:INFO] Sniff delimiter as ','[0m
    [35m[2020-04-20:00:32:08:INFO] Determined delimiter of CSV input is ','[0m
    [32m2020-04-20T00:32:05.224:[sagemaker logs]: MaxConcurrentTransforms=4, MaxPayloadInMB=6, BatchStrategy=MULTI_RECORD[0m
    [34m[2020-04-20:00:32:08:INFO] Sniff delimiter as ','[0m
    [34m[2020-04-20:00:32:08:INFO] Determined delimiter of CSV input is ','[0m
    [35m[2020-04-20:00:32:08:INFO] Sniff delimiter as ','[0m
    [35m[2020-04-20:00:32:08:INFO] Determined delimiter of CSV input is ','[0m
    [34m[2020-04-20:00:32:10:INFO] Sniff delimiter as ','[0m
    [34m[2020-04-20:00:32:10:INFO] Determined delimiter of CSV input is ','[0m
    [34m[2020-04-20:00:32:10:INFO] Sniff delimiter as ','[0m
    [34m[2020-04-20:00:32:10:INFO] Determined delimiter of CSV input is ','[0m
    [35m[2020-04-20:00:32:10:INFO] Sniff delimiter as ','[0m
    [35m[2020-04-20:00:32:10:INFO] Determined delimiter of CSV input is ','[0m
    [35m[2020-04-20:00:32:10:INFO] Sniff delimiter as ','[0m
    [35m[2020-04-20:00:32:10:INFO] Determined delimiter of CSV input is ','[0m
    [34m[2020-04-20:00:32:10:INFO] Sniff delimiter as ','[0m
    [34m[2020-04-20:00:32:10:INFO] Determined delimiter of CSV input is ','[0m
    [34m[2020-04-20:00:32:11:INFO] Sniff delimiter as ','[0m
    [34m[2020-04-20:00:32:11:INFO] Determined delimiter of CSV input is ','[0m
    [35m[2020-04-20:00:32:10:INFO] Sniff delimiter as ','[0m
    [35m[2020-04-20:00:32:10:INFO] Determined delimiter of CSV input is ','[0m
    [35m[2020-04-20:00:32:11:INFO] Sniff delimiter as ','[0m
    [35m[2020-04-20:00:32:11:INFO] Determined delimiter of CSV input is ','[0m
    [34m[2020-04-20:00:32:13:INFO] Sniff delimiter as ','[0m
    [34m[2020-04-20:00:32:13:INFO] Determined delimiter of CSV input is ','[0m
    [34m[2020-04-20:00:32:13:INFO] Sniff delimiter as ','[0m
    [34m[2020-04-20:00:32:13:INFO] Determined delimiter of CSV input is ','[0m
    [35m[2020-04-20:00:32:13:INFO] Sniff delimiter as ','[0m
    [35m[2020-04-20:00:32:13:INFO] Determined delimiter of CSV input is ','[0m
    [35m[2020-04-20:00:32:13:INFO] Sniff delimiter as ','[0m
    [35m[2020-04-20:00:32:13:INFO] Determined delimiter of CSV input is ','[0m
    [34m[2020-04-20:00:32:15:INFO] Sniff delimiter as ','[0m
    [34m[2020-04-20:00:32:15:INFO] Determined delimiter of CSV input is ','[0m
    [34m[2020-04-20:00:32:15:INFO] Sniff delimiter as ','[0m
    [34m[2020-04-20:00:32:15:INFO] Determined delimiter of CSV input is ','[0m
    [35m[2020-04-20:00:32:15:INFO] Sniff delimiter as ','[0m
    [35m[2020-04-20:00:32:15:INFO] Determined delimiter of CSV input is ','[0m
    [35m[2020-04-20:00:32:15:INFO] Sniff delimiter as ','[0m
    [35m[2020-04-20:00:32:15:INFO] Determined delimiter of CSV input is ','[0m
    [34m[2020-04-20:00:32:15:INFO] Sniff delimiter as ','[0m
    [34m[2020-04-20:00:32:15:INFO] Determined delimiter of CSV input is ','[0m
    [34m[2020-04-20:00:32:15:INFO] Sniff delimiter as ','[0m
    [34m[2020-04-20:00:32:15:INFO] Determined delimiter of CSV input is ','[0m
    [35m[2020-04-20:00:32:15:INFO] Sniff delimiter as ','[0m
    [35m[2020-04-20:00:32:15:INFO] Determined delimiter of CSV input is ','[0m
    [35m[2020-04-20:00:32:15:INFO] Sniff delimiter as ','[0m
    [35m[2020-04-20:00:32:15:INFO] Determined delimiter of CSV input is ','[0m
    [34m[2020-04-20:00:32:17:INFO] Sniff delimiter as ','[0m
    [34m[2020-04-20:00:32:17:INFO] Determined delimiter of CSV input is ','[0m
    [34m[2020-04-20:00:32:17:INFO] Sniff delimiter as ','[0m
    [34m[2020-04-20:00:32:17:INFO] Determined delimiter of CSV input is ','[0m
    [34m[2020-04-20:00:32:18:INFO] Sniff delimiter as ','[0m
    [34m[2020-04-20:00:32:18:INFO] Determined delimiter of CSV input is ','[0m
    [34m[2020-04-20:00:32:18:INFO] Sniff delimiter as ','[0m
    [34m[2020-04-20:00:32:18:INFO] Determined delimiter of CSV input is ','[0m
    [35m[2020-04-20:00:32:17:INFO] Sniff delimiter as ','[0m
    [35m[2020-04-20:00:32:17:INFO] Determined delimiter of CSV input is ','[0m
    [35m[2020-04-20:00:32:17:INFO] Sniff delimiter as ','[0m
    [35m[2020-04-20:00:32:17:INFO] Determined delimiter of CSV input is ','[0m
    [35m[2020-04-20:00:32:18:INFO] Sniff delimiter as ','[0m
    [35m[2020-04-20:00:32:18:INFO] Determined delimiter of CSV input is ','[0m
    [35m[2020-04-20:00:32:18:INFO] Sniff delimiter as ','[0m
    [35m[2020-04-20:00:32:18:INFO] Determined delimiter of CSV input is ','[0m
    [34m[2020-04-20:00:32:20:INFO] Sniff delimiter as ','[0m
    [34m[2020-04-20:00:32:20:INFO] Determined delimiter of CSV input is ','[0m
    [34m[2020-04-20:00:32:20:INFO] Sniff delimiter as ','[0m
    [34m[2020-04-20:00:32:20:INFO] Determined delimiter of CSV input is ','[0m
    [34m[2020-04-20:00:32:20:INFO] Sniff delimiter as ','[0m
    [34m[2020-04-20:00:32:20:INFO] Determined delimiter of CSV input is ','[0m
    [35m[2020-04-20:00:32:20:INFO] Sniff delimiter as ','[0m
    [35m[2020-04-20:00:32:20:INFO] Determined delimiter of CSV input is ','[0m
    [35m[2020-04-20:00:32:20:INFO] Sniff delimiter as ','[0m
    [35m[2020-04-20:00:32:20:INFO] Determined delimiter of CSV input is ','[0m
    [35m[2020-04-20:00:32:20:INFO] Sniff delimiter as ','[0m
    [35m[2020-04-20:00:32:20:INFO] Determined delimiter of CSV input is ','[0m
    [34m[2020-04-20:00:32:20:INFO] Sniff delimiter as ','[0m
    [34m[2020-04-20:00:32:20:INFO] Determined delimiter of CSV input is ','[0m
    [35m[2020-04-20:00:32:20:INFO] Sniff delimiter as ','[0m
    [35m[2020-04-20:00:32:20:INFO] Determined delimiter of CSV input is ','[0m
    [34m[2020-04-20:00:32:22:INFO] Sniff delimiter as ','[0m
    [34m[2020-04-20:00:32:22:INFO] Determined delimiter of CSV input is ','[0m
    [35m[2020-04-20:00:32:22:INFO] Sniff delimiter as ','[0m
    [35m[2020-04-20:00:32:22:INFO] Determined delimiter of CSV input is ','[0m
    [34m[2020-04-20:00:32:22:INFO] Sniff delimiter as ','[0m
    [35m[2020-04-20:00:32:22:INFO] Sniff delimiter as ','[0m
    [34m[2020-04-20:00:32:22:INFO] Determined delimiter of CSV input is ','[0m
    [34m[2020-04-20:00:32:23:INFO] Sniff delimiter as ','[0m
    [34m[2020-04-20:00:32:23:INFO] Determined delimiter of CSV input is ','[0m
    [35m[2020-04-20:00:32:22:INFO] Determined delimiter of CSV input is ','[0m
    [35m[2020-04-20:00:32:23:INFO] Sniff delimiter as ','[0m
    [35m[2020-04-20:00:32:23:INFO] Determined delimiter of CSV input is ','[0m
    [34m[2020-04-20:00:32:25:INFO] Sniff delimiter as ','[0m
    [34m[2020-04-20:00:32:25:INFO] Determined delimiter of CSV input is ','[0m
    [34m[2020-04-20:00:32:25:INFO] Sniff delimiter as ','[0m
    [35m[2020-04-20:00:32:25:INFO] Sniff delimiter as ','[0m
    [35m[2020-04-20:00:32:25:INFO] Determined delimiter of CSV input is ','[0m
    [35m[2020-04-20:00:32:25:INFO] Sniff delimiter as ','[0m
    [34m[2020-04-20:00:32:25:INFO] Determined delimiter of CSV input is ','[0m
    [34m[2020-04-20:00:32:25:INFO] Sniff delimiter as ','[0m
    [34m[2020-04-20:00:32:25:INFO] Determined delimiter of CSV input is ','[0m
    [35m[2020-04-20:00:32:25:INFO] Determined delimiter of CSV input is ','[0m
    [35m[2020-04-20:00:32:25:INFO] Sniff delimiter as ','[0m
    [35m[2020-04-20:00:32:25:INFO] Determined delimiter of CSV input is ','[0m
    [34m[2020-04-20:00:32:25:INFO] Sniff delimiter as ','[0m
    [34m[2020-04-20:00:32:25:INFO] Determined delimiter of CSV input is ','[0m
    [35m[2020-04-20:00:32:25:INFO] Sniff delimiter as ','[0m
    [35m[2020-04-20:00:32:25:INFO] Determined delimiter of CSV input is ','[0m
    [34m[2020-04-20:00:32:27:INFO] Sniff delimiter as ','[0m
    [34m[2020-04-20:00:32:27:INFO] Determined delimiter of CSV input is ','[0m
    [35m[2020-04-20:00:32:27:INFO] Sniff delimiter as ','[0m
    [35m[2020-04-20:00:32:27:INFO] Determined delimiter of CSV input is ','[0m
    [34m[2020-04-20:00:32:27:INFO] Sniff delimiter as ','[0m
    [34m[2020-04-20:00:32:27:INFO] Determined delimiter of CSV input is ','[0m
    [34m[2020-04-20:00:32:27:INFO] Sniff delimiter as ','[0m
    [34m[2020-04-20:00:32:27:INFO] Determined delimiter of CSV input is ','[0m
    [35m[2020-04-20:00:32:27:INFO] Sniff delimiter as ','[0m
    [35m[2020-04-20:00:32:27:INFO] Determined delimiter of CSV input is ','[0m
    [35m[2020-04-20:00:32:27:INFO] Sniff delimiter as ','[0m
    [35m[2020-04-20:00:32:27:INFO] Determined delimiter of CSV input is ','[0m
    [34m[2020-04-20:00:32:28:INFO] Sniff delimiter as ','[0m
    [34m[2020-04-20:00:32:28:INFO] Determined delimiter of CSV input is ','[0m
    [35m[2020-04-20:00:32:28:INFO] Sniff delimiter as ','[0m
    [35m[2020-04-20:00:32:28:INFO] Determined delimiter of CSV input is ','[0m
    [34m[2020-04-20:00:32:29:INFO] Sniff delimiter as ','[0m
    [34m[2020-04-20:00:32:29:INFO] Determined delimiter of CSV input is ','[0m
    [34m[2020-04-20:00:32:30:INFO] Sniff delimiter as ','[0m
    [34m[2020-04-20:00:32:30:INFO] Determined delimiter of CSV input is ','[0m
    [34m[2020-04-20:00:32:30:INFO] Sniff delimiter as ','[0m
    [34m[2020-04-20:00:32:30:INFO] Determined delimiter of CSV input is ','[0m
    [34m[2020-04-20:00:32:30:INFO] Sniff delimiter as ','[0m
    [34m[2020-04-20:00:32:30:INFO] Determined delimiter of CSV input is ','[0m
    [35m[2020-04-20:00:32:29:INFO] Sniff delimiter as ','[0m
    [35m[2020-04-20:00:32:29:INFO] Determined delimiter of CSV input is ','[0m
    [35m[2020-04-20:00:32:30:INFO] Sniff delimiter as ','[0m
    [35m[2020-04-20:00:32:30:INFO] Determined delimiter of CSV input is ','[0m
    [35m[2020-04-20:00:32:30:INFO] Sniff delimiter as ','[0m
    [35m[2020-04-20:00:32:30:INFO] Determined delimiter of CSV input is ','[0m
    [35m[2020-04-20:00:32:30:INFO] Sniff delimiter as ','[0m
    [35m[2020-04-20:00:32:30:INFO] Determined delimiter of CSV input is ','[0m
    


Now the transform job has executed and the result, the estimated sentiment of each review, has been saved on S3. Since we would rather work on this file locally we can perform a bit of notebook magic to copy the file to the `data_dir`.


```python
!aws s3 cp --recursive $xgb_transformer.output_path $data_dir
```

    download: s3://sagemaker-us-east-2-885187242119/xgboost-2020-04-20-00-28-12-384/test.csv.out to ../data/xgboost/test.csv.out


The last step is now to read in the output from our model, convert the output to something a little more usable, in this case we want the sentiment to be either `1` (positive) or `0` (negative), and then compare to the ground truth labels.


```python
predictions = pd.read_csv(os.path.join(data_dir, 'test.csv.out'), header=None)
predictions = [round(num) for num in predictions.squeeze().values]
```


```python
from sklearn.metrics import accuracy_score
accuracy_score(test_y, predictions)
```




    0.85748



## Optional: Clean up

The default notebook instance on SageMaker doesn't have a lot of excess disk space available. As you continue to complete and execute notebooks you will eventually fill up this disk space, leading to errors which can be difficult to diagnose. Once you are completely finished using a notebook it is a good idea to remove the files that you created along the way. Of course, you can do this from the terminal or from the notebook hub if you would like. The cell below contains some commands to clean up the created files from within the notebook.


```python
# First we will remove all of the files contained in the data_dir directory
!rm $data_dir/*

# And then we delete the directory itself
!rmdir $data_dir

# Similarly we will remove the files in the cache_dir directory and the directory itself
!rm $cache_dir/*
!rmdir $cache_dir
```


```python

```
