




### Ahmet Hamdi Guzel
![image](https://user-images.githubusercontent.com/46696280/172728916-70a6977a-9fe5-4952-b43d-eaaa426dbd75.png)




#### Artificial Intelligent MSc Student @ University of Leeds / UK 

Hello, I am Artificial Intelligence MSc student/researcher at University of Leeds pursuing a career change to Artificial Intelligence research from motor-sports simulation engineer with several years of experience developing FEA simulation methods to improve modelling of F1/FE power-train systems.
I am also a master's student researcher in the [Computational Light Laboratory at University College London](https://complightlab.com/). 

This is my AI related research/portfolio page. If you want to look at my previous engineering career(publications, patents etc.) please find me on LinkedIn. I am also the active member of Mensa UK where I meet with great friends. 

#### Research Interest : _**Human-like Machine Vision**_

_"Intelligence is not pattern recognition."_
**Josh Tenenbaum** 

The way machines and humans perceive their worlds is central to their intelligence. However, the differences in their perception could cause a lack of empathy and vulnerability in their future actions. Though vision is not the only difference in their perception, I aim to address the challenge of creating robust machine systems that perceive visuals like humans. 

![image](https://user-images.githubusercontent.com/46696280/208502523-ebd5774e-65ef-4665-9bf4-b8c767189e38.png)


## Publications 
**1- ChromaCorrect: Prescription Correction in Virtual Reality Headsets through Perceptual Guidance**

(https://arxiv.org/abs/2212.04264)

![image](https://user-images.githubusercontent.com/46696280/208499704-8c69395d-eead-49f3-a4b2-bb50b4f49b13.png)



## Gradute Coursework/Projects Portfolio

### University of Leeds - AI MSc - Course : Deep Learning 
#### Project 1 - Image Classification and Grad-CAM Heat Map Generation 

Through this coursework, I:

-Practiced building, evaluating, and finetuning a convolutional neural network on an image dataset from development to testing.
-Gained a deeper understanding of feature maps and filters by visualizing some from a pre-trained network.

Pytorch is used to complete project. 
3x64x64 resolution >10k image data is used in this project for multi-class classification. 

I developed my own CNN architecture, optimisation method, and training parameter tunning for the best accuracy on both test and training case. 

- Analysis of CNN Filters and Max-Pooling Filters 

![image](https://user-images.githubusercontent.com/46696280/154195248-def2420a-5c8c-4404-b915-444692c6d776.png)
![image](https://user-images.githubusercontent.com/46696280/154195321-fc307694-9cfe-4e8e-af6a-428ca9b6d46a.png)


#### Grad-CAM Generation

In this section, I explored using Gradient-weighted Class Activation Mapping (Grad-CAM) to generate coarse localization maps highlighting the important regions in the test images guiding the model's prediction. I used pre-trained AlexNet for a single example image. 

![image](https://user-images.githubusercontent.com/46696280/154195396-631a0307-0913-4a24-b2c5-38d386257828.png)


During the class, a private kaggle competition was opened, and my submission was winner with a notable margin score to the closest student score.
[1st score is the test-case by teaching assistant] 

<https://www.kaggle.com/c/leedsimageclassification/leaderboard>

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

#### BLEU Scoring 

![image](https://user-images.githubusercontent.com/46696280/154197849-994620c1-c6c5-44bf-802e-9502922201f0.png)

#### Selected Test Case for Model's Prediction Performance 

![image](https://user-images.githubusercontent.com/46696280/154197965-cf543de6-29df-484e-a9fd-7bb3119bdf1f.png)

In this project the aim is captioning the image and according the 0.221 value, it can be concluded that it is not a good score since every caption finds the almost 22% meaningful overlap between test case captions. Even though 0.221 is a low number,human judge should be consider. BLEU is largely used in machine translation which compares translated sentence to original sentence. In this project, image captioning is the main aim, and comparing predicted caption with 5 different reference may fail the BLEU as a disadvantage. Before judging BLEU in terms of it's prediction over meaningful sentence or just number of overlapped words, human level judgment should be considered.

### University of Leeds - AI MSc - Course : Robotics / Reinforcement Learning 
his projects explores the challenge of autonomous mobile robot navigation in complex environments. The objective of autonomous mobile robot navigation is to reach goal position and return back to original position, without colliding with obstacles. Between two objectives, there is another task which is robot should manipulate the object on the goal position, before returning the starting position of task one.

Instead of path planning and SLAM algorithms, deep reinforcement learning is applied in this project. There are two different reinforcement learning methods are applied during start position to goal and goal to start position. Both methods are designed with neural network architecture (deep reinforcement learning).
1st Method : Deep Reinforcement Learning 1 (Policy Gradient) for navigation 1

2nd Method : Deep Reinforcement Learning 2 (DQN) for navigation 2
In this project Gazebo is used as simulator, since it best supports the requirements for training the navigation robot agent. It provides physics engine and sensors simulation at faster than real world physical robot. OpenAI Robot Operating System (OpenAI ROS) is also used in this project. OpenAI ROS interfaces directly with Gazebo, without necessitating any changes to software in order to run in simulation as opposed to the physical world, and it provides wide range reinforcement Learning libraries that allow to train turtlebot on tasks. Since creating this simulation ecosystem for the project is time consuming, a virtual machine is provided by the instructor is used instead of creating from sctrach.Instead of VM ready world,a new world is created for navigation task.

Hyper parameter tunning and reward function design are studied,and results are compared in terms of total reward after each step.

![image](https://user-images.githubusercontent.com/46696280/167741257-c7dce585-7744-48de-bf11-887c286f877e.png)

![image](https://user-images.githubusercontent.com/46696280/167741324-dc0cb8ef-e2b2-4c0f-a596-ded22c169c09.png)

![image](https://user-images.githubusercontent.com/46696280/167741368-be0650f4-1a21-4f45-86fe-0ec085b8ce67.png)


### University of Leeds - AI MSc - Course : Programming For Data Science 
-California Housing Price Prediction

The aim of project analysing the data and creating a model which learns from the data to predict median house pricing for a given new data as input.
Objective of this project can be addressed for imaginary estate agent evaluation company business model. I believe this is a very good use case, since housing price evaluation normally depends on nested calculations which create complex sceneries to specialists should solve. This is time consuming and not scalable. This model should reduce the errors in complex rules of estimation. For example, experts without this model can fail to predict right values due to number of parameters involved in their calculations. So, no estate agents wants to fail to set a descent price on houses for sale.

![cali_final](https://user-images.githubusercontent.com/46696280/154192899-132e5ac7-8103-4d58-9aeb-6b797bafdc9d.png)

#### Summary 
-Achievements

Achievements can be categorised under two parts are data analysis and model selection. Firstly, from raw housing data to learning model ready data process successfully completed without going into detail too much. Main reason for this selected methods to understand and analyse the data well served. Moreover, geographical data visualization is achieved w/o comprising jupyter notebook performance issue. This is completed by researching the high performance library for geographical visualization along with more than 20,000 markers. Furthermore, data pipe-lining successfully completed to prepare data for training. In this part both test and training data enters to pipeline process. Even though a function hasn't been created, it would be good to create a function which allows scalability of data. In terms of model training, two different model successfully used from scikit-learn’s machine learning library. Their performance also compared, and one model is selected as best model to be used in testing data. Model's drawbacks and results are compared in terms of root mean square error which is chosen as performance measure for models.

-Limitations

Regarding to my research for similar projects, this data set is not huge enough to build high accuracy models to predict median house values for any given attribute listed in data set. Moreover, housing median age and median house value data is restricted with selected value. This creates problems to fit the model. Removing them should improve training set but objective of this project will be harmed due to prediction capability of model is limited by median house value prediction and house information.

-Future Work

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

Initially, SMOTE over-sampling method is used to overcome class in-balance problem. This method is proved it's power with oversample data without duplicating the target labels. Later, data split statically as test and train from whole data set by using 30% which is recommended in course notes from unit 2 classification part. In the first part of first technique, Decision tree binary classification algorithm was used with static data split. Later, hyperparameter which is n_neighbour is checked by passing a range of values. The similar approach is also used with cross validation method to overcome overfitting problem. Then performance metrics observed to choose the best hyperparameters this time by using cross validation with dynamic splitting. In the second part of study similar approach applied to k-NN binary classifier algorithm. For cross validatio nwith chosen the best parameters, Performance metric values are below;

AUROC --> Decision Tree : 99.79%, k-NN : 99.86%

Average Precision--> Decision Tree : 99.63%, k-NN : 99.86%

in terms of AUROC which is used as proper performance metrics for this study, k-NN is better than decision tree binary classification algorithm. Even its slightly better, for large tests and avoid accumulative costs from fraudulent transactions, k-NN should be chosen as a better option. One drawback of k-NN compared to decision tree algorithm, it takes more time to train due to its technique to handle training. Thanks to low k-number value as optimised hyperparameter, it will not be a big issue. Finally, due to oversampling randomness for each re-runnung the model, a few run has been checked and k-NN showed better results each time.

In static split method, it can be observable that AUROC and other performance metrics for decision tree algorithm is resulted perfect 100%. It clearly shows that there is an overfitting problem withhour cross validation technic.


### University of Leeds - AI MSc - Course : Algorithms 

#### Implementing A* Search Algorithm [C++] 

In this project, A* graph-traversal and path search algorithm is implemented in C++. 
While this course does not provide any requisite for the project, I implement it in C++ to improve my C++ skills. 
So, this project is both my learning from Algorithms course as well as my Udacity C++ developer's course conjunction. 
My implementation was different than Udacity's solution, however there was no difference in the solution for both codes. 

![image](https://user-images.githubusercontent.com/46696280/154197590-64d20a24-4fea-4761-b751-dd7f3b29b751.png)

Figure below shows start and goal point along with the A* algorithms' selected shortest path. 

![image](https://user-images.githubusercontent.com/46696280/154194354-64ab9bf6-ce18-4a70-b30f-0cac450e3b19.png)

More than 180 lines of C++ code have written from scratch. 


### Carnegie Mellon University - Computer Graphics 

This course offered by CMU, and their sources are (inc. assignments) available for public. 
During the course, I completed rasterization assignment in C++. 
In the-project, I implemented a software rasterizer that draws points, lines, triangles, and bitmap images.
Also, a given SVG file is able to be rasterized by my implementation both in hardware and software type implementation. 

#### Hardware Rendering

Initially, I implemented hardware-rendering with OpenGL API support.

![image](https://user-images.githubusercontent.com/46696280/154198540-6771a17e-bb58-46b0-b585-bb1218c7f590.png), ![image](https://user-images.githubusercontent.com/46696280/154198857-ff2f1103-fd13-429b-8ab9-46f291ca7682.png)

#### Rasterization [Software Rendering]

Later, software implementation of rasterization is completed by writing C++ from scratch. 
The challenge was, instead of scanning all the pixels, a clever algorithm was asked to improve speed of rasterization. 
My approach was writing an algorithm to find framework of triangles and scanning only space in triangle framework. This improved the speed of software dramatically. Image below is to show triangle - framework capture algorithm's visualisation. 

![image](https://user-images.githubusercontent.com/46696280/154199084-d0f8a4dc-289d-4b26-8882-b182426024c5.png)


#### Rasterization [Software Rendering] - Anti-Aliasing Using Super-sampling

In this task, I extended my rasterizer to anti-alias triangle edges via super-sampling.
Super-sampling implementation is completed by using right memory management approach since cost of super-sampling is high. 
Results below shows 2 different sampling rate's (1x, 4x) performance over my implementation. 

![image](https://user-images.githubusercontent.com/46696280/154200375-661967f5-2e70-43e5-8b7f-a2c8a03784a4.png)

#### C++ Developer Nanodegree Project - A* Search Route Planner  

![image](https://user-images.githubusercontent.com/46696280/158911661-e2adb85e-c194-4299-8217-abd95894ce05.png)


In this project, I have created a route planner that plots a path between two points on a map using real map data from the OpenStreeMap project.
I02D Library for rendering was used to visualize the algorithm. 

The project was written in C++ using real map data and A* search to find a path between two points,similar to mobile path planning application. OpenStreetMap project is used for map data. The OpenStreetMap project is an open-source, collaborative endeavor to create free, user-generated maps of every part of the world. These maps are similar to the maps you might use in Google Maps or the Apple Maps app on your phone, but they are completely generated by individuals who volunteer to perform ground surveys of their local environment.

The code was written by using OOP techniques and basic software design principals.

#### C++ Developer Nanodegree Project - Linux System Monitor 
![image](https://user-images.githubusercontent.com/46696280/158911718-0d3a0533-9b86-44e4-a406-6937c275634e.png)

In this project, I developed system monitor by using advanced OOP techniques in C++. The developed program is light version of htop-system viewer application. 

Linux OS keeps real-time operating system information by using file system. In this project, developed C++ application reads the files from the folders,  collects the data and structures it, then data is processed and formatted for outputing to Linux terminal. 

The project is using ncurses which is a library that facilitates text-based graphical output in the terminal. 

#### C++ Developer Nanodegree Project - Chatbot Memory Management Project

The ChatBot code creates a dialogue where users can ask questions about some aspects of memory management in C++. After the knowledge base of the chatbot has been loaded from a text file, a knowledge graph representation is created in computer memory, where chatbot answers represent the graph nodes and user queries represent the graph edges. After a user query has been sent to the chatbot, the Levenshtein distance is used to identify the most probable answer. The code is fully functional as-is and uses raw pointers to represent the knowledge graph and interconnections between objects throughout the project.

In this project I analyzed and modified the program. Although the program can be executed and works as intended, I have added advanced concepts which are smart pointers, move semantics, ownership and memory allocation. 
![image](https://user-images.githubusercontent.com/46696280/167740562-d8b750d9-25ff-4b6b-a401-75d03d218d72.png)

#### C++ Developer Nanodegree Project - Concurrent Traffic Simulation 

This is the project for the fourth course in the Udacity C++ Nanodegree Program: Concurrency.
Throughout the Concurrency course, I developed a traffic simulation in which vehicles are moving along streets and are crossing intersections. However, with increasing traffic in the city, traffic lights are needed for road safety. Each intersection will therefore be equipped with a traffic light. In this project, I built a suitable and thread-safe communication protocol between vehicles and intersections to complete the simulation. I used my knowledge of concurrent programming (such as mutexes, locks and message queues) to implement the traffic lights and integrate them properly in the code base.
![image](https://user-images.githubusercontent.com/46696280/167740743-d2dc9173-e4c0-47ca-977a-48a0eaf2c328.png)
 
 #### C++ Developer Nanodegree Project - Snake Game with Diffuculty Level Setting
 
 This C++ project is the capstone project (final) of the Udacity C++ Nanodegree. The source code has been mostly adapted from the provided starter code located at (Udacity's repo)[https://github.com/udacity/CppND-Capstone-Snake-Game]. The code base can be divided architecturally and functionally into four distinct class-based components:
Renderer component is responsible for rendering the state of the game using the popular SDL library
Game component constructs and maintains the game board and placement of the game elements like the snake and food.
Snake component constructs and maintains the snake object as it moves across the board gaining points and checking if it ran into itself.
Controller component receives input from the user in order to control movement of the snake.
Once the game starts and creates the Game, Controller, and Snake objects, the game continues to loop through each component as it grabs input from the user, Controller, updates the state of the Game, and graphically renderers the state of the game, Render.

![image](https://user-images.githubusercontent.com/46696280/167740912-a574d934-4e86-4345-bc16-2a913d999f12.png)
![image](https://user-images.githubusercontent.com/46696280/167740879-82d19771-9ffe-4e9a-965a-5232f24a96a9.png)


### Personal Game Development Project - Twin Interaction eVTOL 

I developed a game in Unity VS C# environment. Game interacts with hobby motors and communicate over serial communication port. Gamer's
control on aircraft speed reacts the real hardware and sensors on hardware de-rates the performance of aircraft. Idea was building a real hardware
based physics engine for propulsion system. Game consist more than 700 code lines in C#.

![image](https://user-images.githubusercontent.com/46696280/154200602-44411813-58cf-44db-b4cb-e67a548c0e8b.png)


### Certificates
Udacity Nanodegree - C++ Developer 

Arizona State University - Deep Learning in Visual Computing Systems [Pytorch]

Duke University - Introduction to Machine Learning [Pytorch]

UIUC - OOP Data Structures in C++ 

Arizona State University - Data Structure and Algorithms

Arizona State University - Computer Organization and Assembly Language processing 

Arizona State University - Operating Systems 

Michigan State University - Game Development [Unity]


- Contact : od20ahg@leeds.ac.uk 

