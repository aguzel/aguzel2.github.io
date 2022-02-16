
# <A.H Guzel> 

Computer Graphics Researcher  / Artificial Intelligence MSc Student @ University of Leeds / UK 


### Skills 
- C++ 
- Python 
- OpenGL
- Pytorch 

## Course Projects Portfolio

### University of Leeds - AI MSc - Course : Programming For Data Science 
- California Housing Price Prediction
The aim of project analysing the data and creating a model which learns from the data to predict median house pricing for a given new data as input.
Objective of this project can be addressed for imaginary estate agent evaluation company business model. I believe this is a very good use case, since housing price evaluation normally depends on nested calculations which create complex sceneries to specialists should solve. This is time consuming and not scalable. This model should reduce the errors in complex rules of estimation. For example, experts without this model can fail to predict right values due to number of parameters involved in their calculations. So, no estate agents wants to fail to set a descent price on houses for sale.

![cali_final](https://user-images.githubusercontent.com/46696280/154192899-132e5ac7-8103-4d58-9aeb-6b797bafdc9d.png)

#### Summary 
- Achievements
Achievements can be categorised under two parts are data analysis and model selection. Firstly, from raw housing data to learning model ready data process successfully completed without going into detail too much. Main reason for this selected methods to understand and analyse the data well served. Moreover, geographical data visualization is achieved w/o comprising jupyter notebook performance issue. This is completed by researching the high performance library for geographical visualization along with more than 20,000 markers. Furthermore, data pipe-lining successfully completed to prepare data for training. In this part both test and training data enters to pipeline process. Even though a function hasn't been created, it would be good to create a function which allows scalability of data. In terms of model training, two different model successfully used from scikit-learn’s machine learning library. Their performance also compared, and one model is selected as best model to be used in testing data. Model's drawbacks and results are compared in terms of root mean square error which is chosen as performance measure for models.
- Limitations
Regarding to my research for similar projects, this data set is not huge enough to build high accuracy models to predict median house values for any given attribute listed in data set. Moreover, housing median age and median house value data is restricted with selected value. This creates problems to fit the model. Removing them should improve training set but objective of this project will be harmed due to prediction capability of model is limited by median house value prediction and house information.
- Future Work
Possible future work would be using same process for data analysis and improving both models, or adding extra steps to data analysis section while trying more machine learning algorithms. After, other data set for different locations for example London can be gathered and worked to test model.

### University of Leeds - AI MSc - Course : Data Science 
-Fraud Detection / ML Algorithms Comparison
Fraud detection is very important problem for banks. Even percentage of frauds over all time
banking processes, it can be a big problem since fraud costs a lot of money for banks due to
massive transactions in each day. A data scientist should build a very good fraud detection
algorithm to overcome this issue. 

In this project, two different machine learning algorithms will be used to find best model for
fraud detection

Summary of Results; 
kNN Classifier &  Decision Tree Classifier Comparison 

![kNN](https://user-images.githubusercontent.com/46696280/154191684-3a14c0f9-fc22-4a70-9b00-bd67a6cdd111.png),![dT](https://user-images.githubusercontent.com/46696280/154192012-2e44f2f6-af8d-4a5e-a99b-c2ff0ff3ffc2.png)
 


Two techniques which are decision tree and k-nn binary classifier have been selected as machine learning models for fraud detection problem.

Initially, SMOTE over-sampling method is used to overcome class in-balance problem. This method is proved it's power with oversample data without duplicating the target labels. Later, data split statically as test and train from whole data set by using 30% which is recommended in course notes from unit 2 classification part. In the first part of first technique, Decision tree binary classification algorithm was used with static data split. Later, hyperparameter which is n_neighbor is checked by passing a range of values. The similar approach is also used with cross validation method to overcome overfitting problem. Then performance metrics observed to choose the best hyperparameters this time by using cross validation with dynamic splitting. In the second part of study similar approach applied to k-NN binary classifier algorithm. For cross validatio nwith chosen the best parameters, Performance metric values are below;

AUROC --> Decision Tree : 99.79%, k-NN : 99.86%

Average Precision--> Decision Tree : 99.63%, k-NN : 99.86%

in terms of AUROC which is used as proper performance metrics for this study, k-NN is better than decision tree binary classification algorithm. Even its slightly better, for large tests and avoid accumulative costs from fraudulent transactions, k-NN should be chosen as a better option. One drawback of k-NN compared to decision tree algorithm, it takes more time to train due to its technique to handle training. Thanks to low k-number value as optimised hyperparameter, it will not be a big issue. Finally, due to oversampling randomness for each re-runnung the model, a few run has been checked and k-NN showed better results each time.

In static split method, it can be observable that AUROC and other performance metrics for decision tree algorithm is resulted perfect 100%. It clearly shows that there is an overfitting problem withhour cross validation technic.


### University of Leeds - AI MSc - Course : Algorithms 

#### Implementing A* Search Algorithm [C++] 

In this project, A* graph-traversal and path search algorithm is implemented in C++. 
While this course does not provide any requisite for the project, I implement it in C++ to improve my C++ skills. 
So, this project is both my learning from Algorithms course as well as my Udacity C++ developer's course conjuction. 
My implementation was different than Udacity's solution, however there was no difference in the solution for both codes. 

Figure below shows start and goal point along with the A* algorithms' selected shortest path. 

![Astar](https://user-images.githubusercontent.com/46696280/154194127-3dc1e0dc-68fa-4514-8909-a4bd23b1ae73.png)

>180 lines of C++ code written from scratch. 




### Certificates




Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/aguzel/aguzel.github.io/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
