# Speech Image Description Using YOLO Technology
This application Identifies images in real-time using YOLO technology and provides the output in speech format using gtts. 
An application named "Aider" is created to access this process easily and it'll help blind people who struggle with their daily activities.
 

## ACKNOWLEDGEMENT  
I would like to express my deep and sincere gratitude to my guide Dr. Amol
Bhopale (https://www.linkedin.com/in/dr-amol-bhopale-45b45126/), Professor of Computer Science and Engineering Department, RCOEM, for giving
me the opportunity to work on this project and providing valuable guidance throughout the
project. It was a great privilege and honor to work under his guidance. I'm extremely
grateful for the experience I had in this project with him.  
I express my sincere gratitude to Dr. Avinash Agrawal (https://www.linkedin.com/in/avinash-agrawal-28a748102/), Head of the Department
of Computer Science Department, RCOEM for his guidance. Talent wins games, but
teamwork and intelligence win championships. I would like to take this opportunity to
express my deep gratitude to all those who extended their support and guided me to
complete this project.  


## Table of contents  
List of Figures   
List of Table   
1. Chapter 1: Introduction    
1.1 Abstract   
1.2 Problem Statement    
1.3 Introduction   
1.4 Vision and Objectives    
2. Chapter 2 : Literature Survey    
2.1 SeeingAI    
2.2 TapTapsee   
2.2 Proposed Research paper on Object Detection and Captioning    
3. Chapter 3 : Image Detection and Captioning    
3.1 Theories and Algorithm    
3.1.1 Convolutional Neural Network    
3.2 Technologies   
3.2.1 Image Captioning  
3.2.2 Text to Speech  
3.2.3 Application Development  
3.2.4 Libraries Used   
3.3 Image detection and captioning process  
3.3.1 Object Detection using You Look Only Once(YOLO)   
3.3.2 Feature Extraction From CNN  
3.2.2.1 Feature Extraction using VGG-16  
3.3.3 Preprocessing the training captions  
3.3.5 Generate Model for Training  
3.3.5.1 Recurrent Neural Network(RNN)  
3.3.5.2 Long Short Term Memory(LSTM)  
3.3.5.3 Model Training   
3.4 Proposed Methodology                              
4. Chapter 4: Dataset Details and Evaluation Matrix 
4.1 Dataset Details  
4.2 Evaluation Matrix   
5. Chapter 5:Implementation Results And Observation of Result    
5.1. Object Detection Using YOLO   
5.2. Captions generated    
6. Chapter 6:Project Scope   


### List of Figure  
Fig No. Figure Name                  
Figure 3.1 General Flow Diagram   
Figure 3.2 VGG16 Architecture   
Figure 3.3 RNN Architecture   
Figure 3.4 LSTM cell                     
Figure 3.5 Image Captioning Model   
Figure 3.6 Proposed Methodology                                               
Figure 5.1. Object Detection results using YOLO implemented on Flickr8K   
Figure 5.2. Captions generated as CNN on Flickr8K Dataset   

## Chapter 1 : Introduction      
### 1.1 Abstract:                                                                   
Image captioning is a task in the field of deep learning that involves generating a
textual description of an image. It has a wide range of applications, including helping people
with low or no vision to better understand the visual world around them. NVIDIA is one
company that has developed an application using image captioning technology for this
purpose.                    

The process of image captioning involves feeding an image into a deep learning
model, which then generates a corresponding textual description of the image. The model is
trained on a large dataset of images and their corresponding captions, which allows it to
learn the relationship between the visual features of an image and the words used to describe
them. 

One of the challenges of image captioning is that it requires a model to not only
recognize the objects and scenes depicted in an image, but also to understand the context
and relationships between them. This requires a deep understanding of language and
semantics, which is something that deep learning models are particularly well-suited for.
In addition to helping people with low or no vision, image captioning has a number
of other potential applications. For example, it could be used to generate captions for social
media posts or to automatically generate alt text for images on the web, making them more
accessible to people with disabilities. It could also be used in a variety of other contexts
where a textual description of an image would be useful, such as in image search engines or
for image annotation and organization.             

Image captioning is a significant task in the field of deep learning with a wide range
of potential applications. It is being actively researched and developed by companies and
researchers around the world, and it is likely to have an increasingly important role in the
coming years as deep learning technology continues to advance.  

### 1.2 Problem Statement:
It is true that the technology of artificial intelligence and machine learning has made
significant progress in recent years, and it has the potential to greatly benefit people with
disabilities. One way that this technology could be used to help the visually impaired is
through the development of applications like the one you described, which would allow a
user to scan or input an image and receive a textual description of the scene depicted in the
image.   

However, developing such an application is not without its challenges. As you
mentioned, one of the main challenges is to ensure that the application provides accurate
results and is able to accurately describe the scene depicted in the image. This requires a
deep learning model that is trained on a large and diverse dataset of images and their
corresponding captions, so that it can learn to recognize a wide variety of objects and scenes
and understand the context and relationships between them.   

In addition to this technical challenge, there is also the challenge of making the
environment more accommodating for the visually impaired. This could involve designing
infrastructure that is more accessible and user-friendly for people who are blind or have low
vision, such as by incorporating tactile elements or audio cues. It could also involve
addressing social challenges, such as promoting greater understanding and awareness of the
needs and abilities of the visually impaired within society.   

Overall, it is important to continue researching and developing technologies like
image captioning that can help improve the lives of the visually impaired and other groups
of people with disabilities. By working together and using our collective knowledge and
resources, we can make significant progress in this area and create a more inclusive and
accessible world for everyone.   

### 1.3 Vision and Objectives:  
It is true that people with visual impairments often face challenges in their daily lives
and can struggle with anxiety and depression due to their reliance on others for assistance.
These challenges can be particularly difficult for people who are completely blind, as they may have difficulty identifying and understanding their surroundings and may feel isolated
or disconnected from the world around them.   

However, there are many ways in which technology and assistive devices can help
people with visual impairments to lead more independent and fulfilling lives. For example,
devices such as screen readers and Braille displays can help people who are blind or have
low vision to access information and communicate with others. There are also a range of
adaptive technologies and devices designed to help people with visual impairments navigate
their environment and interact with the world around them, such as cane or guide dog
systems, or GPS-based devices that provide verbal directions.  

Overall, it is important to continue researching and developing technologies and
assistive devices that can help people with visual impairments to live more independent and
fulfilling lives, and to work towards creating a more inclusive and accessible society for
everyone.

Gaining independence is often a major goal for people with disabilities, as it allows
them to have greater control over their lives and to more fully participate in society. People
with visual impairments may face particular challenges in this regard, as they may need to
rely on others for assistance with tasks and activities that involve vision.

However, there are many ways in which technology and assistive devices can help
people with visual impairments to lead more independent lives. For example, as mentioned
earlier, devices such as screen readers and Braille displays can help people who are blind or
have low vision to access information and communicate with others. There are also a range
of adaptive technologies and devices designed to help people with visual impairments
navigate their environment and interact with the world around them, such as cane or guide
dog systems, or GPS-based devices that provide verbal directions.

By using these and other assistive technologies, people with visual impairments can
often lead more independent lives and have greater control over their daily activities and
interactions with the world around them. This can be a valuable way of improving their
quality of life and helping them to feel more connected and engaged with society.

This actually is exactly where our main objective lies, to aid these people with the
use of the currently trending technology which will actually let them actually be able to
definitely learn images, really understand them through a voice captioning the images of the
environment they specifically put in a subtle way.

Developing technology to help people with visual impairments to better understand
and interact with their environment is certainly a worthwhile and important goal. An image
captioning application that allows users to input an image and receive a verbal description of
the scene depicted in the image could be a useful tool in this regard.

To develop such an application, it would be necessary to use machine learning
techniques to train a model to recognize the objects and scenes depicted in an image and
generate a corresponding textual description. This would involve gathering a large and
diverse dataset of images and their corresponding captions, and using this data to train the
model to understand the visual features of an image and the words used to describe them.
Once the model has been trained, it could be incorporated into an application that
allows users to input an image and receive a verbal description of the scene depicted in the
image. This could be achieved using a variety of different technologies and approaches, such
as through the use of a smartphone or other device with a built-in camera, or through the use
of an external camera or scanner.

Developing technology that is easy and intuitive to use for people with visual
impairments is important in order to ensure that they can fully benefit from the capabilities
and features of the application. This may involve designing the application to be compatible
with assistive technologies and devices that are commonly used by people with visual
impairments, such as screen readers or Braille displays.

In addition to ensuring that the application is easy to use, it is also important to
design it in a way that is as independent as possible for the user. This may involve
incorporating features that allow the user to easily navigate the application and input and
receive information without the need for assistance from others.

The goal of an image captioning application for the visually impaired should be to
enable users to lead more independent and fulfilling lives by giving them the tools and 
information they need to better understand and interact with their environment. By
designing the application to be easy and intuitive to use, and by making it as independent as
possible, it is possible to help people with visual impairments gain greater control over their
lives and more fully participate in society.

It is true that no single technology or solution can completely solve all of the
challenges faced by people with disabilities. However, by developing tools and technologies
that can help to make their lives a little easier and more manageable, it is possible to greatly
improve their quality of life and give them greater independence and control over their
surroundings.

An image captioning application for the visually impaired could be a valuable tool in
this regard, as it could allow users to better understand and interact with their environment
by providing them with a verbal description of the scene depicted in an image. While this
application alone may not be able to solve all of the challenges faced by people with visual
impairments, it could be an important part of a broader approach that helps to make their
lives a little easier and more fulfilling.

An image captioning application for the visually impaired could be a valuable tool
for helping users to lead more independent and fulfilling lives. By providing them with a
verbal description of the scene depicted in an image, the application could allow users to
better understand and interact with their environment, which could in turn help them to feel
more connected and engaged with the world around them.

In addition, by designing the application to be easy and intuitive to use, and by
making it as independent as possible, it is possible to help users gain greater control over
their lives and more fully participate in society. This could be an important step towards
helping users lead lives that are more familiar and comfortable to them, and could make a
significant difference in their overall quality of life.

### 1.4 Introduction:
It sounds like you are proposing to develop an application that uses real-time object
detection to identify objects in images taken by the user and provide a verbal description of
the scene depicted in the image. This could be a useful tool for people with visual
impairments, as it would allow them to better understand and interact with their
environment.

Incorporating a text-to-speech API into the application could be a valuable way of
making it more accessible and user-friendly for people with visual impairments, as it would
allow them to receive information in verbal format rather than having to rely on visual
displays. This could make it easier for them to use the application and interact with the
objects and scenes depicted in the images.

Overall, it seems that your proposed application has the potential to be a valuable aid
for people with visual impairments, and could help to make their lives a little easier and
more fulfilling by giving them a better understanding of their environment.


## Chapter 2 : Literature Survey
There are indeed a number of existing applications and technologies that are
designed to help people with visual impairments to better understand and interact with their
environment. Some examples of these include:

### 2.1.1 Seeing AI:
It is certainly important for assistive technologies to be accurate and user-friendly in
order to be effective and useful for people with disabilities. It sounds like Seeing AI, while it
may have a range of different scanning modes, may not always provide accurate
descriptions of images, and may also have a user interface that is not particularly accessible
or easy to use for people with visual impairments.

In order to be effective, an image captioning application for the visually impaired
should aim to provide accurate and reliable descriptions of images, and should also be
designed with the needs and abilities of its users in mind. This may involve incorporating
features that make it easy for users to navigate and use the application, and designing the
user interface to be as accessible and intuitive as possible. By doing so, it is possible to
create an application that is truly useful and helpful for people with visual impairments, and
that can make a meaningful difference in their lives.

It is true that Seeing AI has received a lot of attention due in part to the fact that it
combines a number of different assistive technologies into a single app. This can be a
convenient and cost-effective solution for users who need access to multiple types of
assistive technology, as it allows them to access a range of different features and capabilities
without having to purchase and use multiple separate applications.

However, it is important to note that while a single app that provides multiple
assistive technologies may be convenient, it is also important for the app to be accurate and
user-friendly in order to be truly effective and helpful for people with disabilities. If an app
is not reliable or is difficult to use, it may not be as useful or valuable to its users, regardless
of the number of different features it provides.

It is always a good idea for consumers to carefully compare different products and
technologies in order to determine which one is the best fit for their needs and preferences.
In the case of assistive technologies for people with visual impairments, it is important to
consider not only the cost of the technology, but also its accuracy, reliability, and ease of
use.

While an app like Seeing AI that provides multiple assistive technologies in a single
package may be convenient, it is possible that stand-alone competitors may offer certain
features or capabilities that make them a better value or more suitable for certain users. By
carefully considering the specific needs and preferences of the user, it may be possible to
identify the assistive technology that is the best fit for their needs.

### 2.1.2 TapTapSee:
TapTapSee from Net Ideas, LLC is a free iOS app boasting an extremely simple
interface. Indeed, there are only three buttons, all clearly labeled. At the extreme upper left of
the screen is the "Repeat" button, and at the extreme upper right is the "About" button. Tap
anywhere else on the screen to locate the button most essential to this app: "Take Picture."

Aim your iPhone's camera at the item you'd like to recognize and double tap the
screen. The shutter click sounds, and VoiceOver announces, "Picture One taken." Your photo
is automatically uploaded to the company's servers, and when a match is found, VoiceOver
speaks the item name.

The item name is not displayed on the screen, but you can press the "Repeat" button to
have it re-voiced. The "Repeat" and "About" buttons at the upper left and right of the screen
are very small and can be difficult to find. A better way to access it is with a one-finger swipe
to the right to reach the "Repeat" button and a second swipe to reach the "About" button.

It is true that many existing technologies and applications that are designed to assist
people with visual impairments can be expensive, due to the cost of hardware components or
the need for specialized equipment. In some cases, these technologies may also be difficult to
use or may not provide accurate results due to factors such as variations in lighting or distance
from the object being observed.

By developing an application that can perform similar tasks in a software format, it
may be possible to create a more cost-effective and user-friendly solution for people with
visual impairments. This could potentially make it easier for them to access the assistive
technology they need in order to better understand and interact with their environment.

It is important to note that while there are many technologies and applications
available that are designed to assist people with visual impairments, some of these
technologies may not be as effective as others due to a variety of factors. These factors can
include the accuracy of the technology, the ease of use of the technology, and the ability of the
technology to adapt to different conditions or environments.

In order to effectively assist people with visual impairments, it is important to
consider not only the capabilities of the technology, but also how it can be tailored to meet the
specific needs and preferences of the user. This may involve taking into account factors such
as the user's level of visual impairment, their specific needs and goals, and the environments
in which they will be using the technology. By considering these factors and developing
technologies that are tailored to meet the specific needs of users with visual impairments, it
may be possible to create more effective and user-friendly assistive technologies.

### 2.2 Proposed Research paper on Object Detection and Captioning
![image](https://user-images.githubusercontent.com/124458271/218272909-9d23d06b-1a1c-4bcf-bc3d-c083d4e7c3b7.png)

## Chapter 3 : Image Detection and Captioning
### 3.1 Theories and Algorithm:
### 3.1.1 Convolutional Neural Network
A convolutional neural network (CNN) is a type of artificial neural network that is
specifically designed to process data that has a grid-like topology, such as an image. CNNs
are composed of multiple layers of interconnected nodes, and they use a variation of
multilayer perceptrons designed to require minimal preprocessing. They are particularly
well-suited for image recognition tasks, as they are able to automatically learn hierarchical
representations of visual data through the use of convolutional layers. These layers apply a
convolution operation to the input data, which allows the network to learn patterns and
features at different scales, making it possible to detect objects and features in images
despite variations in position, scale, and other factors.

### 3.2 Technologies:
### 3.2.1 Image Captioning:
The task of providing a verbal description of an image's content is known as image
captioning. Natural language processing and computer vision are intertwined in this
endeavor. An encoder-decoder framework is used by the majority of image captioning
systems, in which an input image is encoded into an intermediate representation of the
information contained in the image and then decoded into a sequence of descriptive text.
Nocaps and COCO are the most widely used benchmarks, and models are typically
evaluated using a BLEU or CIDER metric.

There are several approaches to image captioning, including template-based
methods, in which a fixed set of phrases is used to describe different types of images, and
machine learning-based methods, in which a model is trained to generate descriptions based
on a large dataset of images and corresponding captions.

Image captioning has a wide range of applications, including helping visually
impaired individuals to better understand and interact with their environment, and enabling
computers to generate more accurate and descriptive descriptions of images for use in
search engines and other applications. It is a rapidly growing field, with ongoing research
and development in areas such as natural language processing and computer vision.

### 3.2.2 Text-to-Speech:
One popular text-to-speech API is the Google Text-to-Speech API. This API allows
developers to use Google's text-to-speech technology in their own applications. It supports a
variety of languages and voices, and allows users to specify the speed and pitch of the
generated speech. Other text-to-speech APIs that you might consider include the Amazon
Polly API and the IBM Watson Text-to-Speech API. It's important to choose an API that
supports the languages and voices that you need, and that has good documentation and
support.

### 3.2.3 Application Development:
Once the image captioning and text-to-speech components have been implemented,
it's time to create the user interface for the application. When designing the user interface,
it's important to keep in mind the needs and capabilities of visually impaired users. This
may include using large, easy-to-read text and buttons, providing audio feedback for
actions, and allowing users to navigate the interface using keyboard commands or other
assistive technologies.

![image](https://user-images.githubusercontent.com/124458271/218273000-1a7d029d-d7cb-461e-9265-41ce1ffe717b.png)  ![image](https://user-images.githubusercontent.com/124458271/218273007-c11e1740-07de-4f8c-97ef-b2333e999fbd.png)

It's also a good idea to test the user interface with a group of visually impaired
individuals to ensure that it is intuitive and easy to use. This can help identify any issues or
areas for improvement before the application is released to the public.

Once the user interface is complete, the final step is to package everything into a
standalone application that can be installed on users' devices. This may involve using a tool
like PyInstaller to package the Python code into an executable file, or using a
platform-specific tool like Xcode for iOS or Android Studio for Android.

### 3.2.4 Libraries in Python:
● Keras:   
Keras is a popular open-source library for developing and training machine
learning models, and it is often used for image captioning and other computer vision
tasks. The Flickr8k dataset is a widely used dataset for image captioning and other
image analysis tasks, and it consists of around 8,000 images with corresponding
captions.

● TensorFlow:   
Tensorflow is another popular open-source library for machine
learning, and it is often used in conjunction with Keras for tasks like object detection
and classification. TensorFlow can be used to train machine learning models on large
datasets, and it also provides tools for storing and managing the trained models.
TensorFlow can also be used to perform object detection and classification on new
images or videos, using the trained models to identify and classify the objects of
interest.

Keras and TensorFlow are powerful tools for developing and training machine
learning models for image captioning and other computer vision tasks. They can be
used together to build and train machine learning models that can accurately identify
and classify objects in images and videos, making them useful for a wide range of
applications.

● Google TTS API:   
Google Cloud Text-to-Speech is a cloud-based service that
allows developers to synthesize natural-sounding speech from text using a wide
range of voices and languages. It is part of the Google Cloud Platform, and it uses
machine learning technology to generate high-quality speech in a variety of styles
and accents.

One of the key features of Google Cloud Text-to-Speech is the ability to choose from
over 100 voices in a variety of languages and accents. This allows developers to
select the voice that best fits the needs of their application, whether it is a male or
female voice, a specific accent, or a particular language.

Google Cloud Text-to-Speech is available for free as a part of the Google Cloud
Platform, but there are limits on the amount of text that can be synthesized for free.
If usage exceeds these limits, developers can purchase additional usage at a
reasonable cost.

Google Cloud Text-to-Speech is a powerful and flexible tool for synthesizing
natural-sounding speech from text, and it is widely used in a variety of applications,
including mobile apps, smart home devices, and virtual assistants.

### 3.3 Experimental Setup:
Object detection is a key component of many computer vision and image processing
systems, and it has a wide range of applications. In general, object detection algorithms aim
to identify and locate objects of interest (called "target objects") in digital images or videos.
This can be done using a variety of techniques, including machine learning algorithms,
template matching, and feature extraction.

There are several different types of object detection algorithms, including:
Class-specific object detectors: These algorithms are designed to detect objects from a
specific class, such as cars or pedestrians. They typically use machine learning algorithms to
learn what the target objects look like, and then use this knowledge to identify and locate
them in new images or videos.

Generic object detectors: These algorithms are designed to detect a wide range of objects,
regardless of their class. They often use a combination of machine learning and
feature-based techniques to identify objects in images or videos.

Multi-object detectors: These algorithms are designed to detect multiple objects of different
classes in a single image or video. They can be used to identify and locate all the objects of
interest in an image or video, or to identify a specific set of target objects.

Object detection is an active research area in computer vision and image processing,
and there are many ongoing efforts to develop new and more accurate object detection
algorithms. These algorithms have the potential to revolutionize a wide range of fields, from
robotics and autonomous vehicles to surveillance and security.

The YOLO (You Only Look Once) object detection algorithm is a popular, real-time
object detection system that is able to identify and locate objects in images and videos with high accuracy. One of the main advantages of YOLO is its ability to process images and
videos in real-time, making it suitable for use in a wide range of applications that require
fast object detection.

Another advantage of YOLO is its ability to detect multiple objects in a single image
or video. This is possible because YOLO divides the image into a grid of cells, and each cell
is responsible for predicting the presence and location of objects within its region. This
allows YOLO to detect multiple objects of different classes within the same image or video,
and to provide detailed information about their locations.

Overall, the YOLO object detection algorithm is a powerful and effective tool for
detecting objects in images and videos. It has been widely adopted in a variety of
applications, including robotics, surveillance, and autonomous vehicles, and it continues to
be an active area of research and development.

### 3.4 Image detection and captioning process:
Various steps involved in order to produce a caption from the image ,they are:
● Object Detection using You Look Only Once(YOLO)  
● Feature Extraction using a CNN model  
● Preprocessing the training captions iv. Build Tokenizers and generate Vocabulary for
both captions and the detected objects   
● Generate Model for training  
● Evaluate the model to achieve the results as BLEU score and generating captions for
test images   

The first five steps are part of the training process, while the final step is for testing and
evaluation. Figure 3.1 below illustrates the entire procedure for educating and testing the
Image Caption Generating system. The training process is shown in the left part, while
the testing procedure is shown in the right portion. file produced by the training process
can be used to create captions for test photos and to evaluate the model that was created.

![image](https://user-images.githubusercontent.com/124458271/218273210-c7bde185-5bb1-4d25-99cd-2317632b2887.png)

### 3.4.1 Object Detection using You Look Only Once(YOLO):  
Detecting the image by the popular algorithm “You Only Look Once” (YOLO).
YOLO algorithm employs Convolutional Neural Networks(CNN) to provide real time
object detection. Hence the prediction in the entire application will be done by this
algorithm. It detects objects in a single forward propagation with the computer vision.
Computer Vision basically deals with anything that humans can see and perceive.
Computer Vision enables computers and systems to derive complete meaningful
information from digital photos and digital videos and other visual inputs.
### 3.4.2 Feature Extraction From CNN:  
The process of extracting image features from an image is called feature extraction.
The vectorized picture content is an example of an image feature. Convolution, Max
Pooling, and Dense Layers are among the CNN layers. When utilizing a pre-trained
model, the final dense layer is deleted from the model.
### 3.4.2.1 Feature Extraction using VGG-16  
A convolutional neural network with 16 layers called a VGG-16 is used for image
classification. On the ImageNet dataset, VGG-16 has already been trained. The figure
below provides a description of the layers and architecture of the VGG-16. It generates
feature vectors of size 4096 from an input image with dimensions of 224 by 224. The
max pooling layers reduce the image's size by half while the convolution layers process
the image data. Pooling layers alter the image's size, whereas convolution layers only
alter the image's data.

![image](https://user-images.githubusercontent.com/124458271/218273271-09f35b99-335b-45c5-b933-855d14400437.png)

The maximum values for subsequent procedures are . With stride 1, the convolution
layer uses a 3x3 filter, whereas with stride 2, the max pooling layer uses a 2x2 filter .
Figure 3.3 depicts the layered architecture of VGG16, which employs convolution layers
for image data processing and pooling layers for image dimension reduction to produce
4096 size vectors.

### 3.4.3 Preprocessing the training captions:
Some symbols in training dataset captions include apostrophes, white space, and
punctuation marks. Preprocessing is required to generate a proper word vocabulary for
these captions by creating tokens for each word. Each dataset word is transformed into a
token and the apostrophe attached to it is removed in this step. Additionally, it removes
any other punctuation and makes each letter lower case.

### 3.4.5 Generate Model for Training   
### 3.4.5.1 Recurrent Neural Network(RNN)  
RNN's primary function is to predict the next word in a sentence to produce a
meaningful description. In RNN, the network receives a neuron first, and then the current
state is determined by combining the current input with the previous state. The caption is
created by combining all of the words that were predicted earlier. The output is compared
to the initial output because the method makes use of the backpropagation mechanism.
The error is backpropagated to the network in the event that it occurs in order to update
the weights that will assist in retrieving weights that provide the least amount of error.
Figure 3.4 shows how the previous output hi and the current input Xi are used to predict
the next word as an output in the RNN architecture.   

![image](https://user-images.githubusercontent.com/124458271/218273320-fb5ebad8-1df4-4066-9bbe-bcc81c52f1fd.png)

### 3.4.5.2 Long Short Term Memory(LSTM)
Since RNN cannot retain previously predicted words for an extended period of time,
a storage method is required. The memory layer of LSTM networks is made up of a
collection of recurrently connected blocks. It has a cell as well as the input gate, output
gate, and forget gate, three units. The movement of information into and out of the cell is
controlled by LSTM gates. The Forgot gates determine whether the information should
be kept or thrown away.The cell state's information should be updated by the Input gate,
which is the storage mechanism that keeps the task-relevant information in the cell state.
The next hidden state that can be used to evaluate the subsequent word is selected by the
Output gate. The LSTM cell with each of the described types of gates and cell state is
depicted in Figure 3.5. The forget gate, input gate, and output gate decide which
information should be removed from the cell stat using either the sigmoid activation
function or the tangent hypotential function.   

![image](https://user-images.githubusercontent.com/124458271/218273340-b0979ef5-122c-4d80-8027-6953e3b5a920.png)

### 3.4.5.3 Model Training  
Detected Objects, Image Features, captions that have been preprocessed, and the
vocabulary are all necessary for the generation of a model.

The experiment carried out serves as the foundation for the model that is depicted in
this report. For training, the model uses image features, detected objects, and word-by-word training captions as its three inputs. The VGG16 feature vector consists of
4096 size vectors, while the maximum lengths of the detected objects and captions are 49
and 34, respectively. During the training process, caption management is handled with the
help of Embedding and LSTM. By converting each of the three inputs to the same size 256,
they are combined. Figure 3.5 depicts the generated model.

![image](https://user-images.githubusercontent.com/124458271/218273385-f6b15f90-f3e6-495f-a97d-131a660590db.png)

### 3.4 Proposed Methodology:

![image](https://user-images.githubusercontent.com/124458271/218273433-02bed02a-2b74-400c-9026-0b4606d338cb.png)

## Chapter 4 : Dataset Details and Evaluation Matrix
### 4.1 Dataset Details  
For Image Captioning, there are three datasets with a variety of image types
available. There are 8K images in the Flickr8K Dataset, including 6K training images, 1K
validation images, and 1K testing images. There are 30K images in the Flickr30K Dataset,
including 1K testing images, 1K validation images, and 28K training images. Microsoft
introduced the object detection and image captioning dataset MSCOCO. MSCOCO has 328
thousand images, including 40775 testing images, 40504 validation images, and 82783
training images.

![image](https://user-images.githubusercontent.com/124458271/218273471-fb94bb2a-c624-4cb4-aefd-1a44954719f7.png)

### 4.2 Evaluation Matrix
The Image Captioning technique's produced captions can be evaluated using a
variety of metrics. The authors use BLEU (Bilingual Evaluation Understudy) as their
primary metric for evaluating machine-generated text. Scores for each text segment are
calculated after comparing the produced text segments to the reference text set. Averaging
that particular text can be used to calculate the entire evaluation. Despite its widespread use
in machine translation, the BLEU score is only appropriate for short captions. METEOR
(Metric for Evaluation of Translation with explicit Ordering), which is used to evaluate text
generated by machines and is similar to BLEU, is an alternative. It contrasts the reference
text with the word segment. It also matches words that are synonyms, which improves
sentence correlation. A metric for evaluating image descriptions is CIDEr
(Consensus-Based Image Description Evaluation). Additionally, it provides a consensus
between human-suggested and generated descriptions.

## Chapter 5 : Implementation Results And Observation of Result
This Section includes the results of each step in ‘Image caption using YOLO technology’
that was conducted on Flickr8k dataset.
### 5.1. Object Detection Using YOLO

![image](https://user-images.githubusercontent.com/124458271/218273553-2f3ca637-7d6e-4cee-88aa-d8a37d696ddb.png)

The figure 5.1 shows the object detected by the YOLO with the bounding boxes and their
confidence performed on the Flickr8K dataset.

### 5.2. Captions Generated
![image](https://user-images.githubusercontent.com/124458271/218273738-e7001697-4bfd-48c3-96b1-79eb8785bb27.png)

![image](https://user-images.githubusercontent.com/124458271/218273833-0afb28fa-db4f-4d1e-b2b0-8f3fbc5eb2ef.png)

## Chapter 6 : Project Scope
Navigation:  
This type of feature refers to technology that uses a combination of GPS (Global
Positioning System), mapping data, and the device's sensors to detect the user's location and
the presence of nearby obstacles. GPS is a satellite-based navigation system that allows
devices to determine their precise location and time by receiving signals from satellites in
orbit. Mapping data includes information about the layout and features of an area, such as
roads, buildings, and landmarks. The device's sensors, such as the accelerometer and
gyroscope, can detect movements and changes in orientation.

By combining GPS, mapping data, and sensor data, these technologies can determine
the user's location and the presence of nearby obstacles. This information can then be used to
provide audio or haptic feedback to the user, alerting them to the presence of an obstacle and
its distance. This can be especially useful for blind and low vision individuals, as it can help
them avoid collisions and navigate safely in unfamiliar environments.

Some of these technologies also have features like "wayfinding," which can help the
user navigate to a specific location by providing turn-by-turn directions and alerting them
when they are approaching a turn or intersection. Wayfinding technologies can be particularly
useful for blind and low vision individuals, as they can help them navigate unfamiliar
environments and reach their destination safely and efficiently.

There are other technologies that can be used to assist blind and low vision individuals
with obstacle detection and wayfinding. One example of such technology is wearable devices
that use ultrasonic sensors to detect obstacles. Ultrasonic sensors emit high frequency sound
waves that bounce off objects in the environment and return to the sensor. By analyzing the
time it takes for the sound waves to return, the device can determine the distance and location
of an obstacle. These devices can then provide haptic feedback to the user, alerting them to
the presence of an obstacle and its distance.

Another example of technology that can assist blind and low vision individuals with
obstacle detection and wayfinding is smart canes. These are specialized canes that use
sensors, such as ultrasonic sensors, to detect obstacles and provide haptic feedback to the
user. Some smart canes also have features like wayfinding, which can help the user navigate
to a specific location by providing turn-by-turn directions and alerting them when they are
approaching a turn or intersection.

Overall, the ability to detect and avoid obstacles is an important safety feature for
blind and low vision individuals, and there are already a number of technologies that can help
with this. However, there is always room for improvement and innovation in this area, and it
would be great to see more applications and devices that help users navigate safely and
independently.

Denomination Identification for a wide range of currencies: Denomination identification
is a feature that can help blind or low vision individuals identify the value of different
denominations of currency. There are already a few mobile apps that have this feature, which
use the device's camera to take a picture of the bill and then use image recognition technology
to determine the denomination. Some of these apps also have text-to-speech functionality, so
they can read the denomination out loud to the user. This can be very helpful for blind or low
vision individuals who may have difficulty identifying different denominations of currency.
There are also other ways that this feature could be implemented. For example, it
could be integrated into a smart wallet or purse, using sensors to detect the presence of
different denominations of bills. It could also be implemented as a standalone device, similar
to a currency identifier, that the user can hold up to a bill to have its denomination announced
out loud.

Overall, denomination identification is a useful feature that can make cash
transactions easier and more convenient for blind and low vision individuals. It can help them
manage their money more effectively and avoid misunderstandings or errors when paying for
goods and services.

## References
1. D.-J. Kim, D. Yoo, B. Sim, and I. S. Kweon, “Sentence learning on deep
convolutional networks for image Caption Generation,” in 2016 13th International
Conference on Ubiquitous Robots and Ambient Intelligence (URAI), Aug. 2016, pp.
246–247, doi: 10.1109/URAI.2016.7625747.
2. M. Tanti, A. Gatt, and K. P. Camilleri, “What is the Role of Recurrent Neural
Networks (RNNs) in an Image Caption Generator?,” ArXiv170802043 Cs, Aug.
2017, [Online]. Available: http://arxiv.org/abs/1708.02043.
3. M. Nguyen, “Illustrated Guide to LSTM’s and GRU’s: A step by step explanation,”
Medium, Jul. 10, 2019.
(https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-bystep-e
xplanation-44e9eb85bf21)
4. P. Shah, V. Bakrola, and S. Pati, “Image captioning using deep neural architectures,”
in 2017 International Conference on Innovations in Information, Embedded and
Communication Systems (ICIIECS), Mar. 2017, pp. 1–4, doi:
10.1109/ICIIECS.2017.8276124.
5. D. Hutchison et al., “Every Picture Tells a Story: Generating Sentences from
Images,” in Computer Vision – ECCV 2010, vol. 6314, K. Daniilidis, P. Maragos,
and N. Paragios, Eds. Berlin, Heidelberg: Springer Berlin Heidelberg, 2010, pp.
15–29.
6. ]C. Amritkar and V. Jabade, “Image Caption Generation Using Deep Learning
Technique,” in 2018 Fourth International Conference on Computing Communication
Control and Automation (ICCUBEA), Aug. 2018, pp. 1–4, doi:
10.1109/ICCUBEA.2018.8697360.
7. G. Nishad, “Automatic Image Captioning : Building an image-caption generator
from scratch !,” Medium, Mar. 12, 2019.
https://blog.goodaudience.com/automatic-imagecaptioning-building-an-image-captio
n-generator-from-scratch-4bdd8744bc38 .
8. “Understanding object detection in deep learning - The SAS Data Science Blog.”
https://blogs.sas.com/content/subconsciousmusings/2018/11/19/understanding-object
detection-in-deep-learning/.
9. F. Fang, H. Wang, and P. Tang, “Image Captioning with Word Level Attention,” in
2018 25th IEEE International Conference on Image Processing (ICIP), Oct. 2018,
pp. 1278–1282, doi: 29 10.1109/ICIP.2018.8451558.
10. “VGG16 - Convolutional Network for Classification and Detection.”
https://neurohive.io/en/popular-networks/vgg16/ (accessed Jul. 20, 2020).
11. “tf.keras.layers.Bidirectional | TensorFlow Core v2.2.0,” TensorFlow.
https://www.tensorflow.org/api_docs/python/tf/keras/layers/Bidirectional .
12. N. K. Kumar, D. Vigneswari, A. Mohan, K. Laxman, and J. Yuvaraj, “Detection and
Recognition of Objects in Image Caption Generator System: A Deep Learning
Approach,” in 2019 5th International Conference on Advanced Computing
Communication Systems (ICACCS), Mar. 2019, pp. 107–109, doi:
10.1109/ICACCS.2019.8728516.
13. “Image captioning.” https://kaggle.com/hsankesara/image-captioning .
14. “[1707.07102] OBJ2TEXT: Generating Visually Descriptive Language from Object
Layouts.” https://arxiv.org/abs/1707.07102
15. “deep learning - What’s the commercial usage of ‘image captioning’?,” Artificial
Intelligence Stack Exchange.
https://ai.stackexchange.com/questions/10114/whats-the-commercialusage-of-image
-captioning.
16. A. Poghosyan and H. Sarukhanyan, “Short-term memory with read-only unit in
neural image caption generator,” in 2017 Computer Science and Information
Technologies (CSIT), Sep. 2017, pp. 162–167, doi:
10.1109/CSITechnol.2017.8312163.
17. J. Hui, “Real-time Object Detection with YOLO, YOLOv2 and now YOLOv3,”
Medium, Aug. 27, 2019.
https://medium.com/@jonathan_hui/real-time-object-detection-with-yoloyolov2-28b
1b93e2088.
18. M. Z. Hossain, F. Sohel, M. F. Shiratuddin, and H. Laga, “A Comprehensive Survey
of Deep Learning for Image Captioning,” ArXiv181004020 Cs Stat, Oct. 2018.
[Online]. Available: http://arxiv.org/abs/1810.04020.
19. P. Kuznetsova, V. Ordonez, A. Berg, T. Berg, and Y. Choi, “Collective Generation of
Natural Image Descriptions,” in Proceedings of the 50th Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long Papers), Jeju Island,
Korea, Jul. 2012, pp. 359– 368. [Online].
Available:https://www.aclweb.org/anthology/P12- 1038.
20. S. Li, G. Kulkarni, T. L. Berg, A. C. Berg, and Y. Choi, “Composing Simple Image
Descriptions using Web-scale N-grams,” p. 9.
21. K. Papineni, S. Roukos, T. Ward, and W. Zhu, “BLEU: a Method for Automatic
Evaluation of Machine Translation,” 2002, pp. 311–318.
22. R. Thakur, “Step by step VGG16 implementation in Keras for beginners,” Medium,
Aug. 20, 2019.
https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-forbegi
nners-a833c686ae6c
23. J. Lu, C. Xiong, D. Parikh, and R. Socher, “Knowing When to Look: Adaptive
Attention via A Visual Sentinel for Image Captioning,” ArXiv161201887 Cs, Jun.
2017, Accessed: Jul. 20, 2020. [Online]. Available: http://arxiv.org/abs/1612.01887
24. K. Team, “Keras documentation: Text data preprocessing.”
https://keras.io/api/preprocessing/text/ .
25. M. Hodosh, P. Young, and J. Hockenmaier, “Framing Image Description as a
Ranking Task Data, Models and Evaluation Metrics Extended Abstract,” p. 5
26. “YOLO object detection using Opencv with Python,” Pysource, Jun. 27, 2019.
https://pysource.com/2019/06/27/yolo-object-detection-using-opencv-with-python/ .
27. Prabhu, “Understanding of Convolutional Neural Network (CNN) — Deep
Learning,” Medium, Nov. 21, 2019.
https://medium.com/@RaghavPrabhu/understanding-ofconvolutional-neural-networ
k-cnn-deep-learning-99760835f148
28. J. Brownlee, “How to Develop a Deep Learning Photo Caption Generator from
Scratch,” Machine Learning Mastery, Jun. 26, 2019.
https://machinelearningmastery.com/develop-adeep-learning-caption-generation-mo
del-in-python/ .
29. R. Vedantam, C. L. Zitnick, and D. Parikh, “CIDEr: Consensus-based Image
Description Evaluation,” ArXiv14115726 Cs, Jun. 2015, Accessed: Jul. 20, 2020.
[Online]. Available: http://arxiv.org/abs/1411.5726
30. “COCO - Common Objects in Context.” http://cocodataset.org/#home .
31. “Flickr8K.” https://kaggle.com/shadabhussain/flickr8k (accessed Nov. 25, 2019).












