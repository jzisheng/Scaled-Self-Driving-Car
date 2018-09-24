![Alt Text](https://raw.githubusercontent.com/jzisheng/Scaled-Self-Driving-Car/master/car_driving.gif)

# Scaled Self Driving Car


For my senior research project I explored implementations of NVIDIA's deep convolutional neural network model in autonomous cars. I evaluated the following configurations of NVIDIA's model:
* Categorical Output and Recurrent Neural Network(LSTM)
* Categorical Output and Single State Convolutional Network
* Single Output and Recurrent Neural Network(LSTM)
* Single Output and Single State Convolutional Network

 The performance of these models were evaluated using a scaled self driving car platform.

## Scaled Self Driving Car Platform

The scaled self driving car platform is built on the Donkeycar open source platform. This platform combines a RC Car, Raspberry Pi, Python, and various Python packages(Tornado, Keras, Tensorflow, OpenCV) to create a scaled autonomous vehicle. This section details the components used to build the platform.

The RC Car chosen for this project is a Traxxas Slash 4x4. The Traxxas Slash is a consumer grade remote control car modeled at 1/10th scale. At 1/10th scale there is substantial space for a Raspberry Pi, servo board, and battery to be mounted.

The mounting platform is built using a piece of flat plywood and a 3D printed camera mount. The 3D Printed Camera mount is uploaded on this project's github repository page

In this research project all neural networks are implemented using Keras. Keras is an open source neural network library written in Python. A tensorflow backend is used for building the networks. 


## Getting Started

## Evaluation
To evaluate the models I used a 

## Installation instructions

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

This project is a fork of the original donkeycar, with modifications. As of right now the project is still under development

### Installing
Pull the repository, and then install the python environment for donkeycar(these are the same as the original donkeycar repository)

```
git clone https://github.com/jzisheng/Scaled-Self-Driving-Car
pip install -e donkeycar
```

Create a car folder.
```
donkey createcar --path ~/d2
```

Start your car.

```python ~/d2/manage.py drive```

Now you can control your car by going to <ip_address_of_your_pi>:8887/drive


## Deployment

Note that the configuration file included in this repository is set up for a a Traxxas Slash 4x4. Make sure to calibrate your motor else your car will go flying
## Built With

* [Donkeycar](https://github.com/wroscoe/donkey/) - Donkey car open source platform



## Authors

* **Zisheng Jason Chang** - *Implementation* - [jzisheng](https://github.com/jzisheng)

## Credits
* **Tawn Kramer** - *LSTM and Sequence Generators* - [tawnkramer](https://github.com/tawnkramer/donkey/tree/master/donkeycar)


