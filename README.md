![Alt Text](https://raw.githubusercontent.com/jzisheng/Scaled-Self-Driving-Car/master/car_driving.gif)

# Scaled Self Driving Car

This research project examines the application and performance of artificial neural networks in autonomous vehicles. It focuses on building upon the foundation of fully autonomous vehicles: how the vehicle detects and navigates roads. In this project I describe the application of NVIDIA’s end to end learning model, and the expansion of recurrent LSTM layers on top of NVIDIA’s model. The performance of NVIDIA’s original model to the new model will be evaluated by using a scaled self driving car platform. The full paper can be found [here](https://digitalcommons.bard.edu/senproj_s2018/402/).

## Getting Started

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


## Built With

* [Donkeycar](https://github.com/wroscoe/donkey/) - Donkey car open source platform



## Authors

* **Zisheng Jason Chang** - *Implementation* - [jzisheng](https://github.com/jzisheng)

## Credits
* **Tawn Kramer** - *LSTM and Sequence Generators* - [tawnkramer](https://github.com/tawnkramer/donkey/tree/master/donkeycar)


