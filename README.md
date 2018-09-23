![Alt Text](https://raw.githubusercontent.com/jzisheng/Scaled-Self-Driving-Car/master/car_driving.gif)

# Scaled Self Driving Car

This senior project explores implementations of deep convolutional neural networks for autonomous vehicles. In this project I describe the application of NVIDIA’s end to end learning model, and the adding recurrent LSTM layers to NVIDIA’s model. All implementations are modified versions of NVIDIA's published convolutional neural network. This project explores the following variations:
* Categorical Output vs. Single Output
* Recurrent Neural Network vs. Single State Convolutional Network.

The models’ performances are then evaluated on a scaled self driving car and compared to a human driver. NVIDIA's model combined with a RNN is able to keep the car within 6.1 cm of a human driver's path. The performance of NVIDIA’s original model to the new models will be evaluated by using a scaled self driving car platform. 

This research project examines the application and performance of artificial neural networks in autonomous vehicles. It focuses on building upon the foundation of fully autonomous vehicles: how the vehicle detects and navigates roads.  The performance of the models will be assessed based on the car’s ability to generalize and estimate its own confidence.


## Getting Started

The scaled self driving car platform is built on the Donkeycar open source platform. This platform combines a RC Car, Raspberry Pi, Python, and various Python packages(Tornado, Keras, Tensorflow, OpenCV) to create a scaled autonomous vehicle. This section details the components used to build the platform.

Below are all the components of hte SSDC

Traxxas Slash 4x4 The RC Car chosen for this project is a Traxxas Slash 4x4. The Traxxas Slash is a consumer grade remote control car modeled at 1/10th scale. At 1/10th scale there is substantial space for a Raspberry Pi, servo board, and battery to be mounted. 

Platform and Camera Mount: The mounting platform is built using a piece of flat plywood and a 3D printed camera mount. The 3D Printed Camera mount is uploaded on this project's github repository page \cite{ssdc}

Keras:In this research projrol car through a pulse width modulation servo board. Two channels on the servo board are connected to the RC Car: channel one controls the speed of the motor, and channel two controls the steering.

Traxxas Slash 4x4 The RC Car chosen for this project is a Traxxas Slash 4x4. The Traxxas Slash is a consumer grade remote control car modeled at 1/10th scale. At 1/10th scale there is substantial space for a Raspberry Pi, servo board, and battery to be mounted. 

Platform and Camera Mount The mounting platform is built using a piece of flat plywood and a 3D printed camera mount. The 3D Printed Camera mount is uploaded on this project's github repository page \cite{ssdc}

Keras In this research project all neural networks are implemented using Keras. Keras is an open source neural network library written in Python. A tensorflow backend is used for building the networks. The core data structure of Keras are models made up of a linear stack of layers. For example, to construct a simple Sequential model with 100 inputs and 10 categorical outputs:


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


