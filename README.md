# DogClassification

In this project, I have built a pipeline that can be used within a web or mobile app to process real-world, user-supplied images.  Given an image of a dog, the algorithm will identify an estimate of the canine’s breed.  If supplied an image of a human, the code will identify the resembling dog breed and mention that a human was detected.

### Sample output #1

![sample_dog_output](https://user-images.githubusercontent.com/73464664/150930210-671e0cbf-d495-4cca-8c3b-a1d56503ff8d.png)

### Sample output #2

<img width="200" alt="sample_human_output" src="https://user-images.githubusercontent.com/73464664/150930216-8f9efefd-c8fd-4b76-acc2-ce2030a321bd.png">

In this project, I have used the pytorch library for help in the implementation of the neural network
## Project Instructions

### Instructions on implementation

1. Clone the repository and navigate to the downloaded folder.
2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/dogImages`.  The `dogImages/` folder should contain 133 folders, each corresponding to a different dog breed.
3. Download the [human dataset](http://vis-www.cs.umass.edu/lfw/lfw.tgz).  Unzip the folder and place it in the repo, at location `path/to/dog-project/lfw`.  If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder. 
4. Make sure you have already installed the necessary Python packages according to the README in the program repository.
5. Open a terminal window and navigate to the project folder.
	
	```
		jupyter notebook dog_app.ipynb
	```
6. Run each cell. The cells which train the network take a lot of time to process based on the GPU/ CPU installed on the machine.

### Steps followed

* [Step 0](#step0): Import Datasets
* [Step 1](#step1): Detect Humans
* [Step 2](#step2): Detect Dogs
* [Step 3](#step3): Create a CNN to Classify Dog Breeds (from Scratch)
* [Step 4](#step4): Create a CNN to Classify Dog Breeds (using Transfer Learning)
* [Step 5](#step5): Write the Algorithm
* [Step 6](#step6): Test the Algorithm

### Model Architecture
The architechture of this network was achieved by a lot of trial and error. This network contains 4 convulational layers, 4 Max-pooling layers, and 4 fully connected layers. I have implemented the RELU function as the activation function make the output of the first 3 fully connected layers linear. The architecture of the convulational layers is as follows: It receives a tensor with a depth of 3(due to RGB) which adds 61 filters, thus increasing the depth to 64. In a similar manner, the 64 layers are converted to 192, then 256 and finally 512. I have added a padding of 2 units at every convulational step to facilitate that every pixel of the original image is observed and taken into consideration. The kernel/filter size for the first layer is 11X11, 5X5 in the second layer and 3X3 in the 3rd and 4th layer. The stride for the first layer is 4 and for the other 3, the default 1 is used. For the MaxPool layer, the kernel/filter size is 3X3 and the stride is 2, thus, reducing the size of the sides of the image by a factor of 2 at every step.
In the fully connected layers, I first flattened the image to a single dimensional tensor of 512 X 4 X 4. I then implemented the 4 linear layers. The first layer gave the output of 5000 nodes, the second one gave an output of 2500 nodes, the third one gave an output of 500 nodes and the fourth and final layer gave an output of 133 nodes which is also the number of dog breeds identified. At every step, I used the dropout function with a probability of removing each datapoint of 25%. I used the RELU function in the first 3 layers and did not use any activation function for the last one.

<img width="500" alt="sample_cnn" src="https://user-images.githubusercontent.com/73464664/150933477-1de324d1-a071-462b-8ae0-5e999f50ad96.png">

### Python libraries

* Numpy
* Pandas
* cv2
* matplotlib
* tqdm
* glob
* torch
* torchvision.models.models
* PIL.Image
* torchvision.transforms.transforms
* torchvision.datasets
* os
* torch.nn
* torch.nn.Functional
* torch.optim

## Conclusion

<p> I believe that the algorithm has a lot of room of improvement. Some of my opinions are: </p>

1. The architecture of the network can be tweaked and improved. I chose the Resnet-50 networks as it is one of the best networks which uses an architecture which is very different from the normal neural networks as it is residual in nature. I just replaced the last fully connected layer in order to make the model have 133 classes which wsa required as per the question. However, as there are 50 layers in the network, I can also replace some of the previous fully connected layers which fit better with this network. 
2. Another potential problem which I found was computing power. The ResNet-50 is a very big network and thus, even training for 1 epoch took a very large amount of time which I why I was forced to train only for 3 epochs. Maybe, if I had better computing power, I could run the network for a few more epochs and check if the loss decreased any further. Similarly, for the network designed from scratch by me could be improved if I had access to better computing power as the I could have tested many more combinationas for the learning rate, epochs, etc.
3. Another reason which I believe was obstructing better results was the fact that I was unable to prep the data correctly or as much as required for better training. I think that the images are very messy and there are many factors such as color, light, and other factors which caused discrepancies in the training process. Thus, I believe that extensive methods can be applied to prep the data in such a way that the training goes as smoothly as possible and thus increase the efficiency of the system.
