
PyTorch implementation of CDT3

If you use our code or data please cite the "A CNN-based Policy for Optimizing Continuous Action Control by Learning State Sequences".

Method is tested on [MuJoCo](http://www.mujoco.org/) continuous control tasks in [OpenAI gym](https://github.com/openai/gym). 
Networks are trained using [PyTorch 1.2](https://github.com/pytorch/pytorch) and Python 3.7. 


The experiment on a single environment can be run by calling:
```
python main_cnn.py --env HalfCheetah-v2
```
Numerical results can be found in the paper.

