
# <A.H Guzel> 

Computer Graphics Researcher  / Artificial Intelligence MSc Student @ University of Leeds / UK 


#### Programming Skills 
- Python [inc. all ML/Data Analysis libraries] 
- C++ 
- MATLAB 
- 
#### Interests and Skill Gaining Areas 
- Machine Learning & Deep Learning 
- Computer Graphics
- Image Processing 
- Computational Photography
- Real-Time Graphics [virtual reality] 

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

![image](https://user-images.githubusercontent.com/46696280/154197590-64d20a24-4fea-4761-b751-dd7f3b29b751.png)

Figure below shows start and goal point along with the A* algorithms' selected shortest path. 

![image](https://user-images.githubusercontent.com/46696280/154194354-64ab9bf6-ce18-4a70-b30f-0cac450e3b19.png)

More than 180 lines of C++ code have written from scratch. 

### University of Leeds - AI MSc - Course : Deep Learning 
#### Project 1 - Image Classification and Grad-CAM Heat Map Generation 

Through this coursework, I:
-Practiced building, evaluating, and finetuning a convolutional neural network on an image dataset from development to testing.
-Gained a deeper understanding of feature maps and filters by visualizing some from a pre-trained network.

Pytorch is used to complete project. 
3x64x64 resolution >10k image data is used in this project for multiclass classification. 

I developed my own CNN architecture, optimisation method, and training parameter tunning for the best accuracy on both test and training case. 

![image](https://user-images.githubusercontent.com/46696280/154195082-b0120c5b-52aa-4366-bbab-58bb308f0bad.png)

- Analysis of CNN Filters and Max-Pooling Filters 

![image](https://user-images.githubusercontent.com/46696280/154195248-def2420a-5c8c-4404-b915-444692c6d776.png)
![image](https://user-images.githubusercontent.com/46696280/154195321-fc307694-9cfe-4e8e-af6a-428ca9b6d46a.png)


-Grad-CAM 

In this section, I explored using Gradient-weighted Class Activation Mapping (Grad-CAM) to generate coarse localization maps highlighting the important regions in the test images guiding the model's prediction. I used pre-trained AlexNet for a single example image. 

![image](https://user-images.githubusercontent.com/46696280/154195396-631a0307-0913-4a24-b2c5-38d386257828.png)


During the class, a private kaggle competition was opened, and my submission was winner with a notable margin score to the closest student score.
[1st score is the test-case by teaching assistant] 

https://www.kaggle.com/c/leedsimageclassification/leaderboard

#### Project 2 - Image Capture Generation [CNN + RNN] 

Through this coursework, I:
Understood the principles of text pre-processing and vocabulary building.
Gained experience working with an image to text model.
Used and compared two different text similarity metrics for evaluating an image to text model, and understood evaluation challenges.
        
I used the Flickr8k image caption dataset for image caption generation. The dataset consists of 8000 images, each of which has five different descriptions of the salient entities and activities. 

![image](https://user-images.githubusercontent.com/46696280/154197077-a4022ca9-263e-48de-b8ff-de2b09d4481c.png)

During the project, I developed a CNN and RNN models to train a model aiming to generate image captures with the best accuracy. 

Pytorch code lines are written from scratch [there was no boiler-plate code]

Optimisation model, hypertunning the parameters are performed. GPU training is completed in Google Colab-Pro. 

BLEU Scoring 

![image](https://user-images.githubusercontent.com/46696280/154197849-994620c1-c6c5-44bf-802e-9502922201f0.png)

Selected Test Case 

![image](https://user-images.githubusercontent.com/46696280/154197965-cf543de6-29df-484e-a9fd-7bb3119bdf1f.png)

In this project the aim is captioning the image and according the 0.221 value, it can be concluded that it is not a good score since every caption finds the almost 22% meaningful overlap between test case captions. Even though 0.221 is a low number,human judge should be consider. BLEU is largely used in machine translation which compares translated sentence to original sentence. In this project, image captioning is the main aim, and comparing predicted caption with 5 different reference may fail the BLEU as a disadvantage. Before judging BLEU in terms of it's prediction over meaningful sentence or just number of overlapped words, human level judgment should be considered.


### Carnegie Mellon Universuity - Computer Graphics 

This course offered by CMU, and their sources are (inc. assignments) available for public. 
During the course, I completed rasterization assignment in C++. 
In theproject, I implemented a software rasterizer that draws points, lines, triangles, and bitmap images.
Also, a given SVG file is able to be rasterized by my implementation both in hardware and software type implementation. 

#### Hardware Rendering

Initially, I implemented hardware-rendering with OpenGL API support.

![image](https://user-images.githubusercontent.com/46696280/154198540-6771a17e-bb58-46b0-b585-bb1218c7f590.png), ![image](https://user-images.githubusercontent.com/46696280/154198857-ff2f1103-fd13-429b-8ab9-46f291ca7682.png)

#### Rasterization [Software Rendering]

Later, software implementation of rasterization is completed by writing C++ from scratch. 
The challange was, instead of scanning all the pixels, a clever algorithm was asked to improve speed of rasterization. 
My approach was writing an algorithm to find framework of triangles and scanning only space in triangle framework. This improved the speed of software dramatically. Image below is to show triangle - framework capture algorithm's visualisation. 

![image](https://user-images.githubusercontent.com/46696280/154199084-d0f8a4dc-289d-4b26-8882-b182426024c5.png)


#### Rasterization [Software Rendering] - Anti-Aliasing Using Supersampling

In this task, I extended my rasterizer to anti-alias triangle edges via supersampling.
Supersampling implementation is completed by using right memory management approach since cost of supersampling is high. 
Results below shows 2 different sampling rate's (1x, 4x) performance over my implementation. 

![image](https://user-images.githubusercontent.com/46696280/154200375-661967f5-2e70-43e5-8b7f-a2c8a03784a4.png)


### Personal Game Development Project - Twin Interaction eVTOL 

I developed a game in Unity VS C# environment. Game interacts with hobby motors and communicate over serial communication port. Gamer's
control on aircraft speed reacts the real hardware and sensors on hardware de-rates the performance of aircraft. Idea was building a real hardware
based physics engine for propulsion system. Game consist more than 700 code lines in C#.

![image](https://user-images.githubusercontent.com/46696280/154200602-44411813-58cf-44db-b4cb-e67a548c0e8b.png)


### Certificates

Arizona State University - Data Structure and Algorithms

Arizona State University - Computer Organization and Assembly Language processing 

Arizona State University - Operating Systems 

Arizona State University - Deep Learning in Visual Computing Systems

UIUC - OOP Data Structures in C++ 

Michigan State University - Game Development 

Udacity C++ [ongoing]



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
