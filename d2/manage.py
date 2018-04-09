#!/usr/bin/env python3
"""
Scripts to drive a donkey 2 car and train a model for it. 

Usage:
    manage.py (drive) [--model=<model>] [--model_type=<model_type>]
    manage.py (train) [--tub=<tub1,tub2,..tubn>]  [--model=<model>] [--model_type=<model_type>]  [--no_cache]

Options:
    -h --help        Show this screen.
    --tub TUBPATHS   List of paths to tubs. Comma separated. Use quotes to use wildcards. ie "~/tubs/*"
    --js             Use physical joystick.
"""

# python manage.py drive --model ~/d2/models/linear_8track --model_type linear

# python manage.py drive --model ~/d2/models/rnn_8track --model_type rnn

# python manage.py train --tub /home/jason/sproj/datasets/8track/data/tub_2_18-04-03,/home/jason/sproj/datasets/8track/data/tub_3_18-04-03 --model=./models/linear_ltrack --model_type=rnn_bin

# python manage.py train --tub /home/jason/sproj/datasets/8track/data/tub_2_18-04-03,/home/jason/sproj/datasets/8track/data/tub_3_18-04-03 --model=./models/linear_ltrack --model_type=rnn_bin

# python manage.py train --tub  --model_type categorical
# tub_1_18-04-09  tub_2_18-04-09  tub_3_18-04-09

#python manage.py train --tub /home/jason/sproj/datasets/8track/data/tub_2_18-04-03,/home/jason/sproj/datasets/8track/data/tub_3_18-04-03,/home/jason/sproj/datasets/ltrack/data-4-9/tub_2_18-04-09,/home/jason/sproj/datasets/ltrack/data-4-9/tub_3_18-04-09 --model=./models/rnn_ltrack/ --model_type=rnn


import os, sys
from docopt import docopt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os.path import dirname
sys.path.append("/home/jason/sproj/donkeycar")

import donkeycar as dk
#import parts
from donkeycar.parts.camera import PiCamera
from donkeycar.parts.transform import Lambda
from donkeycar.parts.keras import KerasCategorical, KerasRNN_LSTM, KerasHresCategorical, KerasLinear
from donkeycar.parts.actuator import PCA9685, PWMSteering, PWMThrottle
from donkeycar.parts.datastore import TubHandler, TubGroup
from donkeycar.parts.controller import LocalWebController, JoystickController


def drive(cfg, model_path=None,model_type='categorical',  use_joystick=False):
    '''
    Construct a working robotic vehicle from many parts.
    Each part runs as a job in the Vehicle loop, calling either
    it's run or run_threaded method depending on the constructor flag `threaded`.
    All parts are updated one after another at the framerate given in
    cfg.DRIVE_LOOP_HZ assuming each part finishes processing in a timely manner.
    Parts may have named outputs and inputs. The framework handles passing named outputs
    to parts requesting the same named input.
    '''

    #Initialize car
    V = dk.vehicle.Vehicle()
    cam = PiCamera(resolution=cfg.CAMERA_RESOLUTION)
    V.add(cam, outputs=['cam/image_array'], threaded=True)
    
    if use_joystick or cfg.USE_JOYSTICK_AS_DEFAULT:
        #modify max_throttle closer to 1.0 to have more power
        #modify steering_scale lower than 1.0 to have less responsive steering
        ctr = JoystickController(max_throttle=cfg.JOYSTICK_MAX_THROTTLE,
                                 steering_scale=cfg.JOYSTICK_STEERING_SCALE,
                                 auto_record_on_throttle=cfg.AUTO_RECORD_ON_THROTTLE)
    else:        
        #This web controller will create a web server that is capable
        #of managing steering, throttle, and modes, and more.
        ctr = LocalWebController()

    
    V.add(ctr, 
          inputs=['cam/image_array'],
          outputs=['user/angle', 'user/throttle', 'user/mode', 'recording'],
          threaded=True)
    
    #See if we should even run the pilot module. 
    #This is only needed because the part run_condition only accepts boolean
    def pilot_condition(mode):
        if mode == 'user':
            return False
        else:
            return True
        
    pilot_condition_part = Lambda(pilot_condition)
    V.add(pilot_condition_part, inputs=['user/mode'], outputs=['run_pilot'])
    
    #Run the pilot if the mode is not user.
    kl = KerasCategorical()
    if model_type == 'linear':
        kl = KerasLinear()
    if model_type == 'hres_cat':
        kl = KerasHresCategorical()
    # Change model type accordingly
    if(model_type == 'rnn'):
        kl = KerasRNN_LSTM()
    if model_path:
        #kl.load(model_path)
        #kl = dk.utils.get_model_by_type(model_type, cfg)
        kl.load(model_path)
        
    
    V.add(kl, inputs=['cam/image_array'], 
          outputs=['pilot/angle', 'pilot/throttle'],
          run_condition='run_pilot')
    
    
    #Choose what inputs should change the car.
    def drive_mode(mode, 
                   user_angle, user_throttle,
                   pilot_angle, pilot_throttle):
        if mode == 'user': 
            return user_angle, user_throttle
        
        elif mode == 'local_angle':
            return pilot_angle, user_throttle
        
        else: 
            return pilot_angle, pilot_throttle
        
    drive_mode_part = Lambda(drive_mode)
    V.add(drive_mode_part, 
          inputs=['user/mode', 'user/angle', 'user/throttle',
                  'pilot/angle', 'pilot/throttle'], 
          outputs=['angle', 'throttle'])
    
    
    steering_controller = PCA9685(cfg.STEERING_CHANNEL)
    steering = PWMSteering(controller=steering_controller,
                                    left_pulse=cfg.STEERING_LEFT_PWM, 
                                    right_pulse=cfg.STEERING_RIGHT_PWM)
    
    throttle_controller = PCA9685(cfg.THROTTLE_CHANNEL)
    throttle = PWMThrottle(controller=throttle_controller,
                                    max_pulse=cfg.THROTTLE_FORWARD_PWM,
                                    zero_pulse=cfg.THROTTLE_STOPPED_PWM, 
                                    min_pulse=cfg.THROTTLE_REVERSE_PWM)
    
    V.add(steering, inputs=['angle'])
    V.add(throttle, inputs=['throttle'])
    
    #add tub to save data
    inputs=['cam/image_array', 'user/angle', 'user/throttle', 'user/mode',
                'pilot/angle', 'pilot/throttle']
    types=['image_array', 'float', 'float',  'str','float', 'float']
    
    th = TubHandler(path=cfg.DATA_PATH)
    tub = th.new_tub_writer(inputs=inputs, types=types)
    V.add(tub, inputs=inputs, run_condition='recording')
    
    #run the vehicle
    V.start(rate_hz=cfg.DRIVE_LOOP_HZ, 
            max_loop_count=cfg.MAX_LOOPS)
    
    print("You can now go to <your pi ip address>:8887 to drive your car.")


def train(cfg, tub_names, model_name,model_type):
    '''
    use the specified data in tub_names to train an artifical neural network
    saves the output trained model as model_name
    '''

    X_keys = ['cam/image_array']
    y_keys = ['user/angle', 'user/throttle']
    
    binning = dk.utils.linear_bin
    if model_type == "hres_cat":
        binning = dk.utils.linear_bin_hres

    def rt(record):
        record['user/angle'] = binning(record['user/angle'])
        return record

    kl = KerasCategorical()
    if model_type == 'linear':
        kl = KerasLinear();
    if model_type=='categorical':
        kl = KerasCategorical()
    if model_type=='hres_cat':
        kl = KerasHresCategorical()
    print('tub_names', tub_names)
    if not tub_names:
        tub_names = os.path.join(cfg.DATA_PATH, '*')
    tubgroup = TubGroup(tub_names)

    train_gen, val_gen = tubgroup.get_train_val_gen(X_keys, y_keys, record_transform=rt,
                                                    batch_size=cfg.BATCH_SIZE,
                                                    train_frac=cfg.TRAIN_TEST_SPLIT)

    if model_type == 'linear':
        train_gen, val_gen = tubgroup.get_train_val_gen(X_keys, y_keys,
                                                    batch_size=cfg.BATCH_SIZE,
                                                    train_frac=cfg.TRAIN_TEST_SPLIT)


    model_path = os.path.expanduser(model_name)

    total_records = len(tubgroup.df)
    total_train = int(total_records * cfg.TRAIN_TEST_SPLIT)
    total_val = total_records - total_train
    print('train: %d, validation: %d' % (total_train, total_val))
    steps_per_epoch = total_train // cfg.BATCH_SIZE
    print('steps_per_epoch', steps_per_epoch)
    
    print(val_gen)
    history,save_best = kl.train(train_gen=train_gen,
                                 val_gen=val_gen,
                                 saved_model_path=model_path,
                                 steps=steps_per_epoch,
                                 train_split=cfg.TRAIN_TEST_SPLIT)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss : %f' % save_best.best)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(model_path + '_'+model_type+'_loss_%f.png' % save_best.best)


if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = dk.load_config()
    
    if args['drive']:
        model_path = args['--model']
        # use_joystic=args['--js']
        # use_joystick=args['--js']
        model_type = args['--model_type']
        print(model_type)
        drive(cfg, model_path=model_path,model_type=model_type, use_joystick=False)
        # drive(cfg, model_path=model_path,model_type=model_type, use_joystick=False):
        # drive(cfg, model_path, use_joystick, model_type)

    if args['train']:
        from train import rnn_train
        tub = args['--tub']
        model = args['--model']
        model_type = args['--model_type']
        cache = not args['--no_cache']
        if model_type == 'linear':
            train(cfg,tub,model,model_type)
        if model_type == 'rnn' or model_type == 'rnn_bin':
            rnn_train(cfg, tub, model,model_type)
        elif model_type == 'hres_cat' or 'categorical':
            train(cfg,tub,model,model_type)
        else:
            print("Invalid model name: rnn_bin | rnn | hres_cat | categorical")
