# RL_racing_car
Racing car RL project

The environment is based on openAi gymnasium Racing Car. 

In <b>src</b> directory:
- <b>car_racing.py</b> is the environment as it is provided by gymansium
- To play (manual actions), run *$ python car_racing.py*
- <b>car_racing_mod.py</b> is a modified version with obstacles.
- To play (manual action), run *$ python car_racing_mod.py* 


To run the code, in terminal: 
- To train: *$python main.py train*
- To test saved model: *$python main.py test*

In both case, you will be asked to input the used parameters.

The action space is Discrete and is not the same as the original one. 
You can change it in the function *custom_step* in the file *agents/help_func.py*.
**Don't forget to change the value of NUMBER_ACTIONS in the same file.**



**Note**:
$ *pip install gymansium* 
and not gym
You will also need swig, Box2D
