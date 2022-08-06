# Accelerated Optimization via Geometric Numerical Integration

**Authors: [Valentin Duruisseaux](https://sites.google.com/view/valduruisseaux) and [Melvin Leok](https://mathweb.ucsd.edu/~mleok/)**


<br />

This repository provides the code for our paper

[**Practical Perspectives on Symplectic Accelerated Optimization.**](https://arxiv.org/abs/2207.11460)
Valentin Duruisseaux and Melvin Leok.
*2022.*

It contains simple implementations of the optimization algorithms in MATLAB and Python, and more sophisticated Python code implementations which allow the optimizers to be called conveniently within the TensorFlow and PyTorch frameworks.


<br />

<hr><hr>

# Table of Contents

*  [**Simple MATLAB Codes**](#simple-matlab-codes)

*  [**Simple Julia Codes**](#simple-julia-codes)

*  [**Simple Python Codes**](#simple-python-codes)

*  [**PyTorch Codes**](#pytorch-codes)

	* [BrAVO Algorithms](#bravo-algorithms)

	* [Applications](#applications)

		* [Fashion-MNIST Image Classification](#fashion-mnist-image-classification)

		* [CIFAR-10 Image Classification](#cifar-10-image-classification)

		* [Dynamics Learning and Control](#dynamics-learning-and-control)


*  [**TensorFlow Codes**](#tensorflow-codes)

	* [BrAVO Algorithms](#bravo-algorithms)

	* [Applications](#applications)

		* [Binary Classification](#binary-classification)

		* [Fermat-Weber Location Problem](#fermat-weber-location-problem)

		* [Data Fitting](#data-fitting)

		* [Natural Language Processing: arXiv Classification](#natural-language-processing-arxiv-classification)

		* [Timeseries Forecasting for Weather Prediction](#timeseries-forecasting-for-weather-prediction)


*  [**Additional Information**](#additional-information)







<br /><br />

<hr><hr>

## Simple MATLAB Codes

See the directory [`./Simple_MATLAB/`](Simple_MATLAB)

A simple implementation of the ExpoSLC-RTL and PolySLC-RTL algorithms from 
[**Practical Perspectives on Symplectic Accelerated Optimization**](https://arxiv.org/abs/2207.11460) as MATLAB functions is given in
```
	ExpoSLC_RTL.m     and     PolySLC_RTL.m
```

We also provide two simple MATLAB scripts to show how these optimizers can be used:
```
	Expo_Script.m     and     Poly_Script.m
```



<br /><br />

<hr><hr>

## Simple Julia Codes

See the directory [`./Simple_Julia/`](Simple_Julia)

A simple implementation of the ExpoSLC-RTL and PolySLC-RTL algorithms from 
[**Practical Perspectives on Symplectic Accelerated Optimization**](https://arxiv.org/abs/2207.11460) as Julia functions is given in
```
	ExpoSLC_RTL.jl     and     PolySLC_RTL.jl
```

We also provide two simple Julia scripts to show how these optimizers can be used:
```
	Expo_Script.jl     and     Poly_Script.jl
```



<br /><br />


<hr><hr>

## Simple Python Codes

See the directory [`./Simple_Python/`](Simple_Python)

A simple implementation of the ExpoSLC-RTL and PolySLC-RTL algorithms from 
[**Practical Perspectives on Symplectic Accelerated Optimization**](https://arxiv.org/abs/2207.11460) as Python functions is stored in 
```
	SLC_Optimizers.py
```

We also provide two simple python scripts to show how these optimizers can be used, once **SLC_Optimizers.py** is imported:
```
	ExpoScript.py     and     PolyScript.py
```


Usage:
```
	python ./Simple_Python/ExpoScript.py
```


<br /><br />



<hr><hr>


## PyTorch Codes

See the directory [`./PyTorch_Codes/`](PyTorch_Codes)

### BrAVO Algorithms

We have implemented the BrAVO algorithms in **BrAVO_torch.py** within the PyTorch framework.

They can be called in a similar way as the ADAM algorithm. For instance,
```
	optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
```
can be replaced by
```
	optimizer = BrAVO_torch.eBravo(model.parameters(), lr = 1)
```


<br />

### Applications

We have tested the BrAVO algorithms on a collection of optimization problems from machine learning, with a variety of model architectures, loss functions to minimize, and applications.



#### Fashion-MNIST Image Classification

We consider the popular multi-label image classification problem based on the Fashion-MNIST dataset. 

> *"Fashion-MNIST is a dataset of Zalando's article images consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes (t-shirt/top, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot)."*

We learn the 55,050 parameters of a Neural Network classification model.

Usage:
```
	python ./PyTorch_Codes/FashionMNIST.py
```

<br />

#### CIFAR-10 Image Classification

We consider the popular multi-label image classification problem based on the CIFAR-10 dataset.

> *"The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 mutually exclusive classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck), with 6000 images per class."*

We learn the 62,006 parameters of a Convolutional Neural Network classification model which is very similar to the LeNet-5 architecture.

Usage:
```
	python ./PyTorch_Codes/CIFAR10.py
```

<br />

#### Dynamics Learning and Control

We have test our algorithms for dynamics learning and control. We consider a Hamiltonian-based neural ODE network (with 231310 parameters) for dynamics learning and control on the SO(3) manifold, applied to a fully-actuated pendulum.

More details about this application and the code used can be found at

[**Hamiltonian-based Neural ODE Networks on the SE(3) Manifold For Dynamics Learning and Control.**](https://thaipduong.github.io/SE3HamDL/)
Thai Duong and Nikolay Atanasov.
*Proceedings of Robotics: Science and Systems, July 2021.*



<br /><br />
<hr>

## TensorFlow Codes

See the directory [`./TensorFlow_Codes/`](TensorFlow_Codes)

### BrAVO Algorithms

We have implemented the BrAVO algorithms in **BrAVO_tf.py** within the TensorFlow framework.

They can be called in a similar way as the ADAM algorithm. For instance,
```
	optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
```
can be replaced by
```
	optimizer = BrAVO_tf.eBravo(learning_rate = 1)
```



<br />

### Applications

We have tested the BrAVO algorithms on a collection of optimization problems from machine learning, with a variety of model architectures, loss functions to minimize, and applications.


#### Binary Classification

Given a set of feature vectors   <img src="https://latex.codecogs.com/svg.image?\inline&space;\large&space;x_1,&space;\ldots&space;,&space;x_m&space;\in&space;\mathbb{R}^n" title="https://latex.codecogs.com/svg.image?\inline \large x_1, \ldots , x_m \in \mathbb{R}^n" />  and associated labels  <img src="https://latex.codecogs.com/svg.image?\inline&space;y_1,&space;\ldots&space;,&space;y_m&space;\in&space;\{-1,1\}" title="https://latex.codecogs.com/svg.image?\inline y_1, \ldots , y_m \in \{-1,1\}" /> , we want to find a vector  <img src="https://latex.codecogs.com/svg.image?\inline&space;\large&space;w&space;\in&space;\mathbb{R}^n" title="https://latex.codecogs.com/svg.image?\inline \large w \in \mathbb{R}^n" />   such that   <img src="https://latex.codecogs.com/svg.image?\inline&space;\text{sign}&space;(&space;w^\top&space;x&space;)" title="https://latex.codecogs.com/svg.image?\inline \text{sign} ( w^\top x )" />   is a good model for  <img src="https://latex.codecogs.com/svg.image?\inline&space;\large&space;y(x)" title="https://latex.codecogs.com/svg.image?\inline \large y(x)" /> . 
	
Usage:
```
	python ./TensorFlow_Codes/BinaryClassification.py
```


<br />


#### Fermat-Weber Location Problem

Given a set of points <img src="https://latex.codecogs.com/svg.image?\inline&space;y_1,&space;\ldots&space;,y_m&space;&space;\in&space;\mathbb{R}^n" title="https://latex.codecogs.com/svg.image?\inline y_1, \ldots ,y_m \in \mathbb{R}^n" /> and associated positive weights <img src="https://latex.codecogs.com/svg.image?\inline&space;\large&space;w_1,&space;\ldots&space;,&space;w_m&space;\in&space;\mathbb{R}" title="https://latex.codecogs.com/svg.image?\inline \large w_1, \ldots , w_m \in \mathbb{R}" /> , we want to find the location <img src="https://latex.codecogs.com/svg.image?\inline&space;\large&space;x\in&space;\mathbb{R}^n" title="https://latex.codecogs.com/svg.image?\inline \large x\in \mathbb{R}^n" /> whose sum of weighted distances from the points <img src="https://latex.codecogs.com/svg.image?\inline&space;y_1,&space;\ldots&space;,y_m" title="https://latex.codecogs.com/svg.image?\inline y_1, \ldots ,y_m" /> is minimized.
	
	
Usage:
```
	python ./TensorFlow_Codes/LocationProblem.py
```

<br />



#### Data Fitting

Given data points from a noisy version of a chosen function, we learn the 4,355 parameters of a Neural Network to obtain a model which fits the data points as well as possible

Usage:
```
	python ./TensorFlow_Codes/DataFitting.py
```


<br />



#### Natural Language Processing: arXiv Classification

We consider the Natural Language Processing problem of constructing a multi-label text classifier which can provide suggestions for the most appropriate subject areas for arXiv papers based on their abstracts. 

Usage:
```
	python ./TensorFlow_Codes/NLP_arXivClassification.py
```


<br />



#### Timeseries Forecasting for Weather Prediction

We consider timeseries forecasting for weather prediction, using a Long Short-Term Memory (LSTM) model (with 5153 parameters).

Usage:
```
	python ./TensorFlow_Codes/WeatherForecasting.py
```








<br /><br /><br />
<hr>

# Additional Information

If you use this code in your research, please consider citing:

```bibTeX
@article{Duruisseaux2022Practical,
  title = {Practical Perspectives on Symplectic Accelerated Optimization},
  author = {Duruisseaux, Valentin and Leok, Melvin},
  year={2022},
  url={https://arxiv.org/abs/2207.11460}
}
```

The software is available under the [MIT License](https://github.com/vduruiss/AccOpt_via_GNI/blob/main/LICENSE).


If you have any questions, feel free to contact [Valentin Duruisseaux](https://sites.google.com/view/valduruisseaux) or [Melvin Leok](https://mathweb.ucsd.edu/~mleok/).


