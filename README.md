![alt text](https://cdn-images-1.medium.com/max/800/0*0ArBGd8Qc76bQmyt)

# ZetaGAN project

### 1. Environment
```
Python  3.7
Pytorch 1.2
```
[ZetaGAN @ Global PyTorch Summer Hackathon](https://devpost.com/software/zetagan-data-generation-service)


## Inspiration ##
1. **_Collecting data takes time:_** AI is powerful when there are numerous labelled data to train the model; however, data collection is time-consuming.
2. **_Imbalanced data:_** Imbalance is common and expected in real world, e.g. Medical diagnosis, Spam filtering, and Fraud detection. 
3. **_Concern of data privacy:_** The concerns of data leak and privacy are increasing. It’s getting harder and harder to collect data due to new regulations and guidelines, e.g. GDPR.

## What it does ##
1. **_Generate Synthetic Samples_**, i.e. pseudo data, for **_structured/tabular data_**.
  * The data’s schema, data distribution, and relationship between columns of the generated data are as close to real data as possible. 
  * The difference of statistical properties between synthetic and real data is slight.
2. **_A Data Augmentation Platform_** to provide the data augmentation service with only small amount of data. 

## How we built it ##
1. **_A Data Generation Model_** built with **_PyTorch_** to generate pseduo data for minority class.
2. Train **_a single-hidden-layer MLP classifier_** built with **_PyTorch_** on pseudo data and part of real data. Testing dataset is the rest part of real data.
3. Calculate the **_performance metric_** of testing dataset with **_PyTorch_**.
4. **_Compare the performances_** between MLP classifiers trained on data with augmentation (by GANs and SMOTE respectively) and trained on data without augmentation

### Structured datasets we used ###
1. **_[Credit Card Fraud Detection Dataset] (https://www.kaggle.com/mlg-ulb/creditcardfraud)_** available on Kaggle:
  * A target variable (0 or 1) with 0.172% are 1
  * 30 independent variables: time, transaction amount, and 28 principal components
2. **_[Pima Indians Diabetes Database] (https://www.kaggle.com/uciml/pima-indians-diabetes-database)_** available on Kaggle: 
  * A target variable (0 or 1) with 34.896% are 1
  * 8 independent variables: demographic attribute and vital signs

## Challenges we ran into ##
+ Among hundreds of GANs model, which one should we choose per the purpose of our project?
+ How should we adjust the network’s parameters to encourage it to produce believable samples?
+ Struggling with the integration of web platform given that we are unfamiliar with web backend and frontend techniques.

## Accomplishments that we're proud of
Developing this project from consolidating idea to building a real web platform in a month!
Every decision and every movement is made within quite short period of time, and **_we are really good at fast-learning!_**

## What we learned ##
+ GANs Models
+ Over-sampling technique
+ How to integrate the machine learning model with the web API into the web platform

## What's next for Data Generation Service ##
+ Modify the platform UI and add more features
+ Try other algorithms for generating data, e.g. Condition GAN model
+ Try to generate synthetic samples for unstructured data, e.g. image of license plate, text to speech
+ Implement our methodology on different domains to help the world, e.g. Defect Inspection, rare disease diagnose
