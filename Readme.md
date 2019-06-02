# Task scheduling using reinforcement learning  (Keras and Python)



###              DSTI project – A18 



​            		    Vipin Kumar :  vipin.kumar@edu.dsti.institute

​				Benoit Roberge : benoit.roberge@edu.dsti.institute

​                                Mirel Isaj  : mirel.isaj@edu.dsti.institute

**Reinforcement Learning Basics**

**![img](https://lh3.googleusercontent.com/9NXYcTd6-36_JRTsWIV-316fBV2D-uYWvaS3GzxenCfRhWhqTOm-8P4iFiAn9vJ_7o-OY_FBykWezpFDGxB8NuTqAGoV2L2-hMhcQCuuYJwTRdWsVihy-rleWZ_-sWu8ihdSv569)**

At each time step t, the agent observers the current state from the environment and takes an action. From this interaction, the environment gives the agent a, a reward signal to tell it how well it is doing. The goal of the agent is to maximise the reward over time. 

**Examples of computer resource management:**

-  Cluster Scheduling: Managing a large number of machines in a cluster and figuring out how to schedule jobs on them

- Video Streaming– need to adapt bit rate according to network resources

- VM placement in the cloud, congestion control for networks. 

**Cluster Scheduling:**

Cluster scheduling in a multi-resource task allocation problem. Each machine has multiple types of resources such as CPU and memory. Users submit jobs and scheduler picks which jobs to run each time. Either to minimise completion time or maximise cluster utilisation.

**Problem Statement:**

For this problem we want to schedule a task based on CPU and Memory allocation. The selection will be from a batch of M task(5 tasks), The scheduling considers only the CPU and Memory and assumes there is no resource fragmentation and one knows the resources consumed by the task in advance.

This is a very simple attempt to apply and learn the concept of reinforcement deep learning.

**Solution attempted:**

We are going to attempt to solve the above problem with by combining Q - learning with reinforcement learning. This solution will use Keras “an open-source neural-network library written in Python” to model the agent.

**Agent and Environments:**

**State**: The state is defined as all the tasks in queue. In our case we have a total of 5 task in each batch. This result into total of 5 states to choose from. 

**Actions**:For each action, an agent can pick one task or pick nothing. If the agent picks nothing, the selection phase is over and all the selected task will be processed. In our example, there are initially 5 possible tasks to select (6 possible actions with the pass). A task selected cannot be selected again, hence, the number of possible action decrease with each selection until the end of the selection phase.

**Q values:** 

Below table shows an example of q values for each action. This is summarised as:

- The input: A vector of three values. Selection state, CPU and Memory
- The five possible action states
- The Q table from the agent based on the input and action state selected

To illustrate a quick example: 

Action ` [1 , 0, 0, 0, 0] (task [37, 39])` will result into a q value vector as an output of deep network: 

Q value =` [.98 ,0.20 ,  0.12 , 0.11 , 0.05]` for **action**  ` [1 , 0, 0, 0, 0] `

![img](https://lh3.googleusercontent.com/aq5mBCYtNrUYe1jdA0sFKurIN5MbB6IK9ZeMIYkUb4Djqp6hW2ECiwblQXZkGo72_56GVd1iLXPm9AH5AmfL1WtdsLDmhvyam-2sb-mTpSOyO-YxAxbOevqVvZbZdVRwd_o3wIKh)

**Environment:**

The environment acts based on the actions performed by the agent. The environment return the reward and the new state.  The new state is all the task in queue in the previous selection minus the task selected.

**Transition to new state:**

Let's assume a is the action performed by agent.

a = `[1,0,0,0,0]`

```
 a * state =  
              [[0, 37, 39] —> a = 0
              [0,  8 40] 
	          [0,  51  5]
              [0, 13 11] 
       		  [0, 82 15]]
						
```

**Deciding the next state:** 

Reward for current action:  `37 + 39  = 76`

Total amount of CPU : `90`

Total amount of RAM : `90`

Remaining CPU = `90 - 37`

Remaining RAM = `90 - 39`

The agent then will either pick a random action or will look into the deep q table in order to select the action which the best reward is predicted. If the resource of the selected task exceed the available resource or if the agent pass, the selection phase is over and the environment reset.

**Reward**: For a wrong selection, the environment penalize the agent by setting the reward to a negative value. For a right selection reward is the total number of resource count. In our example, ` task 0`

Action will fetch a reward of `37 + 39  = 76` and if the same task is selected again a penalty of `-7.6` would be awarded. 

**Keras model and environment code:**

There are two part of the program, Class env encodes the environment behaviour and using Keras sequential model we train our agent based on reward provided by the environment.

```python
import numpy as np
class env:
    rand_seed = 10
   	'''
   	Init function to initialize environment state, it takes number of resources and no of 
  	Tasks in a batch and creates a state representation of the resources.
  	'''
    def __init__(self, resources, num_tasks):
        np.random.seed(self.rand_seed)
        ### initially, no task are chosen
        self.state_ = np.zeros((num_tasks, 1), dtype=int)
        self.limit_ = 0
        for lim in resources:
            self.state_ = np.append(self.state_, np.random.randint(lim, size=(num_tasks, 1)), axis=1)
            
        self.reward_ = 0
        for lim in resources:
            self.limit_+=lim   #Set the total resource limit as sum of resources
       
        self.num_tasks_ = num_tasks
        self.resources_ = resources
    '''
    Get_initial_state: Reset the environment state for the new batch of incoming jobs
    '''   
    def get_initial_state(self):
        state = np.zeros((self.num_tasks_, 1), dtype=int)
        for lim in self.resources_: 
            state = np.append(state, np.random.randint(lim, size=(self.num_tasks_,1)), axis=1)
        self.state_ = state
        self.reward_ = 0
        self.num_tasks_ = num_tasks
        return self.state_
'''
     Return the sum of a one dimensional vector
'''
    def getSum(self, x ):
        if x[0] == 1:
            return sum(x)-1
        else:
            return 0
   
'''
This function defines the logic behind changing the states and deciding rewards based on the input provided by agent in terms of action.
Input: index of the selected task in state table
Output: 
New state:
Reward: 
Done:  Job processing done or not flag.
'''

    def get_next_step(self, actionIdx):
        ### update reward
        self.reward_ = 0
        reward = 0
        done = False
        #update the state based on action
        #case 1, if same task selected then penalize the agent
        if self.state_[actionIdx][0] == 1:
            sum_res = np.sum(self.state_[actionIdx], axis=0)
            reward = -sum_res/10;
        else:
            self.state_[actionIdx][0] = 1
       
        #collect all the resources for this batch until now.
        totReward = sum(np.apply_along_axis( self.getSum, axis=1, arr=self.state_ ))
        # get the sum for selected task
        sum_res = np.sum(self.state_[actionIdx], axis=0) 
        if totReward <= self.limit_:
            reward = sum_res
        else:
            reward = -sum_res/10;
            done = True
            
        return self.state_,reward,done 

resources = [80, 60] #resource limits
num_tasks = 5
en = env(resources,num_tasks)
action = 1
en.get_next_step(action)

from keras.models import Sequential
from keras.layers import InputLayer
from keras.models import Sequential
from keras.layers import Activation, Dense

resources = [80, 80] #resource limits
num_tasks = 5
input_layer_n = num_tasks
output_layer_n = 5
dense_layer_n = 10

envs = env(resources, num_tasks)
'''
Agent using Deep network, Keras Model
Input layers neurons: no of resources + task selection state
Dense layer: Arbitrary no picked for this, currently it is 10.
Output layer: Single dimensional vector of task selection score, called Q values for each state. No of output neurons = 5
'''
model = Sequential()
#model.add(InputLayer(batch_input_shape=(1, input_layer_n)))
model.add(InputLayer(batch_input_shape=(1, 3)))
model.add(Dense(dense_layer_n, activation='sigmoid'))
model.add(Dense(output_layer_n, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

num_job_samples = 1
# now execute the q learning
y = 0.95
eps = 0.5
decay_factor = 0.999
r_avg_list = []

'''
Training part of the model built in previous section.
No of job samples = 1024
'''

for i in range(num_job_samples):   #batch of 5 jobs and total of 1000 samples
    s = envs.get_initial_state()# stating state would be no selected jobs
    eps *= decay_factor
    if i % 100 == 0:
        print("Episode {} of {}".format(i + 1, num_job_samples))
    scheduled = False
    r_sum = 0

    while not scheduled:
        if np.random.random() < eps:
            a = np.random.randint(0, 5) #This will pick a random action
        else:
            #predicted value is a vector of q values
            a = np.argmax(model.predict(s)) 
        #print(a)
        
        new_s, r, scheduled = envs.get_next_step(a) 
        print(np.array(new_s))
        #model.predict(np.array(new_s))
        target = r + y * np.max(model.predict(np.array(new_s)))
        #target = r + y * np.max(model.predict(np.array(new_s))
        target_vec = model.predict(s)
        #Update only the current state value
        target_vec[a] = target
        model.fit(s, target_vec.reshape(-1, 25), epochs=1, verbose=0)
        s = new_s
        r_sum += r
    r_avg_list.append(r_sum / 1000)

```



**Testing :**

We ran it using https://colab.research.google.com/

```sh
Using TensorFlow backend.
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
Episode 1 of 1
[[ 0 11 78]
 [ 0 54 49]
 [ 0 62 51]
 [ 1 33 54]
 [ 0 72 77]]
```



**Error encountered and resolution:**

```sh
ValueError                                Traceback (most recent call last)
<ipython-input-2-e3c1ed82778d> in <module>()
     50         #Update only the current state value
     51         target_vec[a] = target
---> 52         model.fit(s, target_vec.reshape(-1, 25), epochs=1, verbose=0)
     53         s = new_s
     54         r_sum += r

2 frames
/usr/local/lib/python3.6/dist-packages/keras/engine/training_utils.py in standardize_input_data(data, names, shapes, check_batch_axis, exception_prefix)
    136                             ': expected ' + names[i] + ' to have shape ' +
    137                             str(shape) + ' but got array with shape ' +
--> 138                             str(data_shape))
    139     return data
    140 

ValueError: Error when checking target: expected dense_2 to have shape (5,) but got array with shape (25,)
```

