![Alt Text](https://raw.githubusercontent.com/jzisheng/Scaled-Self-Driving-Car/master/car_driving.gif)

Videos of it in action:

https://vimeo.com/292911659 

https://vimeo.com/292911732 


# Scaled Self Driving Car

This research project examines the application and performance of artificial neural networks in autonomous vehicles. In this project I described the application of NVIDIA’s end to end learning model, and the expansion of recurrent LSTM layers on top of NVIDIA’s model. I evaluated the following configurations of NVIDIA's model:

* Single Output and Recurrent Neural Network(LSTM)
* Single Output and Single State Convolutional Network
* Categorical Output and Recurrent Neural Network(LSTM)
* Categorical Output and Single State Convolutional Network

The performance of NVIDIA’s original model to the new model was evaluated by using a scaled self driving car platform. The full paper can be found [here](https://digitalcommons.bard.edu/senproj_s2018/402/). 


## Scaled Self Driving Car Platform

The scaled self driving car platform is built on the Donkeycar open source platform. This platform combines a RC Car, Raspberry Pi, Python, and various Python packages(Tornado, Keras, Tensorflow, OpenCV) to create a scaled autonomous vehicle. This section details the components used to build the platform.

The RC Car chosen for this project is a Traxxas Slash 4x4. The Traxxas Slash is a consumer grade remote control car modeled at 1/10th scale. At 1/10th scale there is substantial space for a Raspberry Pi, servo board, and battery to be mounted.

The mounting platform is built using a piece of flat plywood and a 3D printed camera mount.

In this research project all neural networks are implemented using Keras. Keras is an open source neural network library written in Python. A tensorflow backend is used for building the networks.


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. This project is a fork of the original donkeycar, with modifications. 

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

For training models, I recommend cloning this repository to your local machine, and exporting datasets from the Raspberry Pi to your machine. Datasets from the car will be stored in a `tub` format. More on this in the `Datasets` section.

```python manage.py (train) [--tub=<tub1,tub2,..tubn>]  [--model=<model_path>] [--model_type=<model_type>]  [--no_cache]```

Below is a example call for training a `rnn`(Single Output and Recurrent Neural Network) model assuming the car's data has been exported into `./data/Tub_*` directory, and the final model saved in the directory `./models/rnn_8track1`

```python manage.py train --tub=./data/* --model=./models/rnn_8track1 --model_type=rnn```


## Built With

* [Donkeycar](https://github.com/wroscoe/donkey/) - Donkey car open source platform



## Authors

* **Zisheng Jason Chang** - *Implementation* - [jzisheng](https://github.com/jzisheng)

## Credits
* **Tawn Kramer** - *LSTM and Sequence Generators* - [tawnkramer](https://github.com/tawnkramer/donkey/tree/master/donkeycar)

