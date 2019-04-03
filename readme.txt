-----------------------------------------------------------------DESCRIPTION-------------------------------------------------------------------------
The classification of  the gender from the real life images is done using Python programming language on Anaconda software.
We will divide the datasets in two parts i.e Training dataset that will train the model and Testing dataset to check the model in order to obtain a certain accuracy. 
ADIENCE Dataset- It contains around 17500 images which contains images as true as the real world scenarios.This problem lies in Supervised Learning domain(Classification) of Machine Learning as the classes are already defined i.e. MALE and FEMALE.

The steps are: 
-----------------------
STEP 1:DATA PREPARATION
-----------------------
1.The Adience dataset can be downloaded from the:https://talhassner.github.io/home/projects/Adience/Adience-data.html
2.Data set will be stored in a Python list.
3.All the images needs to be resized and converted to grayscale.
4.The features will be stored in a csv file along with their gender.

-------------------------
STEP 2:FEATURE EXTRACTION
-------------------------
1.Feature extraction is done using PCA.The feature vector of the image is created with the help of algorithms like PCA(Principal Component Analysis). 
2.Each image is demonstrated as feature vector with the help of PCA.Scaling is done via scaler that can be imported from sklearn.preprocessing before applying PCA.
3.Different Principal components values like 75,85,100,150,500 etc are tried on the data to understand the effect on various models.

------------------------------------------------------
STEP 3:SPLITTING THE DATASET INTO TRAINING AND TESTING
------------------------------------------------------ 
The data will be split into training and testing sets using sklearn library.The split is done in three ways:
1.The split is 70 - 30% with testing data being 30% of the overall dataset.
2.The split is 80 - 20% with testing data being 20% of the overall dataset.
3.The split is 75 - 25% with testing data being 25% of the overall dataset.

---------------------
STEP 4:CLASSIFICATION
---------------------
Various classification algorithms are used to recognize the gender.All of them were first imported from sklearn python library.
Being a classification problem,the algorithms that can be used are:
1.Support Vector Machines(SVM) using different kernel functions such as Linear,Radial Basis Function(rbf) etc. 
2.Decision Tree(DT)
3.Naive Bayes
4.Logistic Regression
5.K-Nearest Neighbor(KNN)
6.Linear Discriminant Analysis(LDA) 
7.Quadratic Discriminant Analysis(QDA)