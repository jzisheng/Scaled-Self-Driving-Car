![Alt Text](https://raw.githubusercontent.com/jzisheng/Scaled-Self-Driving-Car/master/car_driving.gif)

# Scaled Self Driving Car

For my senior research project I explored implementations of NVIDIA's deep convolutional neural network model in autonomous cars. I evaluated the following configurations of NVIDIA's model:

* Single Output and Recurrent Neural Network(LSTM)
* Single Output and Single State Convolutional Network
* Categorical Output and Recurrent Neural Network(LSTM)
* Categorical Output and Single State Convolutional Network

The control for this project is the Single Output and Single State Convolutional Network, the model that most closely resembles NVIDIA's.


## Scaled Self Driving Car Platform

The scaled self driving car platform is built on the Donkeycar open source platform. This platform combines a RC Car, Raspberry Pi, Python, and various Python packages(Tornado, Keras, Tensorflow, OpenCV) to create a scaled autonomous vehicle. This section details the components used to build the platform.

The RC Car chosen for this project is a Traxxas Slash 4x4. The Traxxas Slash is a consumer grade remote control car modeled at 1/10th scale. At 1/10th scale there is substantial space for a Raspberry Pi, servo board, and battery to be mounted.

The mounting platform is built using a piece of flat plywood and a 3D printed camera mount.

In this research project all neural networks are implemented using Keras. Keras is an open source neural network library written in Python. A tensorflow backend is used for building the networks. 

## Getting Started

These instructions will get you a copy of the project up and running your local machine. This project is a fork of the original donkeycar, with modifications.

#### Installing

On the Raspberry Pi clone the repository, and install the python environment for Donkeycar

```
git clone https://github.com/jzisheng/Scaled-Self-Driving-Car
pip install -e donkeycar
```
Start your car.

```python ~/d2/manage.py drive```

Now you can control your car by going to `<ip_address_of_your_pi>:8887/drive`. By default, the vehicle will start recording as soon as there is throttle. To load a model for the autopilot, pass the model path and model type parameters:

```
python manage.py drive [--model=<model_path>] [--model_type=<model_type>]
```

The table below shows the callable `model_type`s.

| Model Type                                                | Parameter |
|-----------------------------------------------------------|-----------|
| Single Output and Single State Convolutional Network      | linear    |
| Categorical Output and Single State Convolutional Network | hres_cat  |
| Single Output and Recurrent Neural Network(LSTM)          | rnn       |
| Categorical Output and Recurrent Neural Network(LSTM)     | rnn_cat   |

#### Training

For training models, I recommend cloning this repository to your local machine, and exporting datasets from the Raspberry Pi to your machine.

```python ~/d2/manage.py train```

```manage.py (train) [--tub=<tub1,tub2,..tubn>]  [--model=<model>] [--model_type=<model_type>]  [--no_cache]```


## Evaluation
To evaluate the models I tracked the position of the car using a RGB tracker.



## Deployment

Note that the configuration file included in this repository is set up for a a Traxxas Slash 4x4. Make sure to calibrate your motor else your car will go flying
## Built With

* [Donkeycar](https://github.com/wroscoe/donkey/) - Donkey car open source platform



## Authors

* **Zisheng Jason Chang** - *Implementation* - [jzisheng](https://github.com/jzisheng)

## Credits
* **Tawn Kramer** - *LSTM and Sequence Generators* - [tawnkramer](https://github.com/tawnkramer/donkey/tree/master/donkeycar)


