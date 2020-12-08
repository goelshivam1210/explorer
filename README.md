# Explorer: Reinforcement learning approach to handle non-stationary environments

## Comp 150 Reinforcement Learning - Tufts University Class Project

### Dependencies
Python 3.7.4 is used for development<br>
Use a Conda environment to install the requirements
```
pip install -r requirements.txt
```

### To run training
```
python train.py -C <resume_training{True/False}> -M <model_episode_number> -E <episode_number> -R <render{True/False}> -P <print result every "X" episode>
```

### For live plotting
```
cd plot
python plot.py -W <window_size> -P <pause_time(s)>
```

### For Evaluation
```
python evaluate.py -E <number_of _episodes_to_test> -M <model_episode_number> -R <Render(True/False)>
```

### For playing
```
sudo python keyboard_interface.py
```
