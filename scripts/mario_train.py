import tensorflow as tf
from tensorflow.keras.layers import Dense,Dropout,Conv2D, Flatten,MaxPooling2D ,Activation,Input,LeakyReLU
from tensorflow.keras.models import Sequential,load_model,Model
import gym
import numpy as np
import random
from collections import deque
from tensorflow.keras.utils import normalize as normal_values
from tensorflow.keras.optimizers import Adam
import cv2
import matplotlib
import matplotlib.pyplot as plt
from skimage import transform 
from skimage.color import rgb2gray 
import math
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros 
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)        
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)        
        if self.env.unwrapped._flag_get:
            reward += 100
            done = True            
        if self.env.unwrapped._is_dying:
            reward -= 50
            done = True                 
        self.was_real_done = done
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)    
        return obs

class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.
    This is incredibly important and effects performance
    drastically.
    """
    def reward(self, reward):

        return reward * 0.05 

class PreprocessFrame(gym.ObservationWrapper):
    """
    Here we do the preprocessing part:
    - Set frame to gray
    - Resize the frame to 96x96x1
    """
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.width = 96
        self.height = 96
        self.observation_space = gym.spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        # Set frame to gray
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Resize the frame to 96x96x1
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        frame = frame[:, :, None]

        return frame

class StochasticFrameSkip(gym.Wrapper):
    def __init__(self, env, n, stickprob):
        gym.Wrapper.__init__(self, env)
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()
        self.supports_want_render = hasattr(env, "supports_want_render")

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac):
        done = False
        totrew = 0
        for i in range(self.n):
            # First step after reset, use action
            if self.curac is None:
                self.curac = ac
            # First substep, delay with probability=stickprob
            elif i==0:
                if self.rng.rand() > self.stickprob:
                    self.curac = ac
            # Second substep, new action definitely kicks in
            elif i==1:
                self.curac = ac
            if self.supports_want_render and i<self.n-1:
                ob, rew, done, info = self.env.step(self.curac, want_render=False)
            else:
                ob, rew, done, info = self.env.step(self.curac)
            totrew += rew
            if done: break
        return ob, totrew, done, info

    def seed(self, s):
        self.rng.seed(s) 

class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]

env=gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env=JoypadSpace(env,COMPLEX_MOVEMENT)

env = EpisodicLifeEnv(env)
env = RewardScaler(env)
env = PreprocessFrame(env)
env = StochasticFrameSkip(env,4,0.5)
env = ScaledFloatFrame(env)
env = FrameStack(env, 4)

class Mario_Agent:
 
  def __init__(self,Actor_path=None,Critic_path=None,old_Actor_path=None):
    #self.env=env #import env
    self.state_shape = env.observation_space.shape # the state space
    self.action_shape = env.action_space.n # the action space
    self.width = 96
    self.Average_rewards = []
    self.height = 96
    self.gamma=[0.99] # decay rate of past observations
    self.learning_rate=3e-5 # learning rate in deep learning
    self.lambda_=0.95
    self.epochs=5
    self.tau=0.005
    self.batch_size=64
    self.beta=0.001 #Entropy Loss ratio
    self.clipping_ratio = 0.2
    if not Actor_path:
      self.Actor_model=self._create_model('Actor')    #Target Model is model used to calculate target values
      self.old_Actor_model=self._create_model('Actor')
      self.Critic_model=self._create_model('Critic')  #Training Model is model to predict q-values to be used.
    else:
      self.Actor_model=load_model(Actor_path) #import model
      self.Critic_model=load_model(Critic_path) #import model
      self.old_Actor_model=load_model(old_Actor_path)
    
        # record observations
    self.states=deque()
    self.rewards=deque()
    self.actions=deque()
    self.last_state=np.zeros((1,*self.state_shape))
  
  def remember(self, state,reward,action,last_state,done):
    '''stores observations'''
    self.states.append(state)
    self.actions.append(action)
    self.rewards.append(reward)
    if done:
      self.last_state[0]= last_state

  def _create_model(self,model_type):

 
    ''' builds the model using keras'''
    
    state_input=Input(shape=(self.state_shape))

    layer_1 = Conv2D(filters = 32,kernel_size = (8, 8),strides = (4, 4),padding = 'same',kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),bias_initializer=tf.keras.initializers.Zeros())(state_input)
    layer_2 = LeakyReLU()(layer_1)
    layer_3 = Conv2D(filters = 64,kernel_size = (4, 4),strides = (2, 2),padding = 'same',kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),bias_initializer=tf.keras.initializers.Zeros())(layer_2)
    layer_4 = LeakyReLU()(layer_3)
    layer_5 = Conv2D(filters = 64,kernel_size = (3, 3),strides = (1, 1),padding = 'same',kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),bias_initializer=tf.keras.initializers.Zeros())(layer_4)
    layer_6 = LeakyReLU()(layer_5)
    layer_7 = Flatten()(layer_6)

    layer_8=Dense(512,activation='relu',kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),bias_initializer=tf.keras.initializers.Zeros())(layer_7)
    
 
    if model_type=='Actor':
      output =(Dense(self.action_shape, activation='softmax'))(layer_8)
      model = Model(inputs=[state_input],outputs=[output])
    else:
      output=Dense(1, activation=None,bias_initializer=tf.keras.initializers.Ones(),kernel_initializer=tf.keras.initializers.Orthogonal(.01))(layer_8)
      model = Model(inputs=[state_input],outputs=[output])
      model.compile(optimizer=Adam(learning_rate=self.learning_rate),loss="mse")
    return model

  def get_action(self, state,status="Training"):
    action_probs=self.Actor_model(state)
    action_t = random.choices(list(range(self.action_shape)), k=1, weights=action_probs[0])[0]
    if status=='Testing':
      return np.argmax(action_probs)
    return action_t

  def get_GAEs(self,v_preds,rewards):
    T = len(v_preds)-1
    gaes = np.zeros((T,1),dtype='float32')
    future_gae = 0.0
    for t in reversed(range(T)):
      delta = rewards[t] + np.asarray(self.gamma) * v_preds[t + 1] - v_preds[t]
      gaes[t] = future_gae = delta + np.asarray(self.gamma) * np.asarray(self.lambda_) *np.asarray(future_gae)
    return gaes

  def update_models(self):

    states_mb=np.zeros((self.batch_size,*self.state_shape))
    V_s_mb=np.zeros((self.batch_size,1))
    actions_mb=np.zeros((self.batch_size,1))
    rewards_mb=np.zeros((self.batch_size,1))

    batch_indices = np.random.choice(len(self.states),self.batch_size)

    for i,j in enumerate(batch_indices):
      
      state_=tf.reshape(tf.convert_to_tensor(self.states[j]),(1,96,96,4))
      states_mb[i]=state_
      actions_mb[i]=(self.actions[j])
      rewards_mb[i]=self.rewards[j]

      
      vs_=self.Critic_model.predict(state_)
      V_s_mb[i]=vs_

    
    
   
    V_last_state=self.Critic_model.predict(tf.reshape(tf.convert_to_tensor(self.last_state),(1,96,96,4)))
    v_all=np.concatenate((V_s_mb,V_last_state),axis=0)
    
    Advantages=self.get_GAEs(v_all,rewards_mb)
    critic_targets = Advantages +  V_s_mb
    self.Critic_model.fit(states_mb, critic_targets,epochs=self.epochs,verbose=2)
    
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def train_step(states_1_mb,Advantages,actions):
      with tf.GradientTape() as tape:
        Advantages=tf.stop_gradient(Advantages)
        pred_actions=self.Actor_model(states_1_mb,training=True)

        actions_onehot=tf.one_hot(actions, self.action_shape)
        
        old_pred_actions =(self.old_Actor_model)(states_1_mb,training=True)
        log_prob_ratio = tf.math.log(tf.reduce_sum(pred_actions * actions_onehot, axis=1)+1e-30) - tf.math.log(tf.reduce_sum(old_pred_actions * actions_onehot, axis=1)+1e-30)                                   
        prob_ratio = tf.math.exp(log_prob_ratio) 
        
        clip_ratio = tf.clip_by_value(prob_ratio, clip_value_min=1 - self.clipping_ratio, clip_value_max=1 + self.clipping_ratio)
        surrogate=(tf.math.minimum(prob_ratio*Advantages,clip_ratio*Advantages))
        entropy_loss=-(pred_actions * tf.math.log(pred_actions+1e-10))
        ppo_loss = -(tf.math.reduce_mean(surrogate+self.beta* entropy_loss))                      
      grads = tape.gradient(ppo_loss,self.Actor_model.trainable_variables)
      grads = [(tf.clip_by_value(grad, -1.0, 1.0)) for grad in grads]
      optimizer.apply_gradients(zip(grads, self.Actor_model.trainable_variables))
      return ppo_loss

    ppo_loss=[]
    for epoch in range(self.epochs):
      loss=train_step(states_mb,Advantages,actions_mb)
      ppo_loss.append(loss)
    print("PPO_loss:",np.mean(ppo_loss)) 
   
    actor_weights = np.array(self.Actor_model.get_weights())
    actor_target_weights = np.array(self.old_Actor_model.get_weights())
    new_weights = self.tau*actor_weights + (1-self.tau)*actor_target_weights
    self.old_Actor_model.set_weights(new_weights)

    self.states.clear();self.rewards.clear();self.actions.clear()

  def evaluate(self,ep):
    video_frames = []
    score = 0
    state=env.reset()
    state = np.expand_dims(state , axis = 0)     
    state = tf.convert_to_tensor(state, dtype=tf.float32)
    while 1:
      video_frames.append(cv2.cvtColor(env.render(mode = 'rgb_array'), cv2.COLOR_RGB2BGR))
      action=self.get_action(state,'Testing')
      state, reward, done, info=env.step(action)
      state = np.expand_dims(state,axis = 0)
      state = tf.convert_to_tensor(state, dtype=tf.float32)     
      score += reward
      if done:
        break
    video_name = 'test_' + str(ep)+'.mp4'
    _, height, width, _ = np.shape(video_frames)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter(video_name, fourcc, 5, (width,height))
    for image in video_frames:
      video.write(image) 
    cv2.destroyAllWindows()
    video.release()        
    print('Test #%s , Score: %0.1f' %(ep, score))    

  def train(self,episodes):
    for episode in range(episodes):
      state=env.reset()
      state = np.expand_dims(state , axis = 0)     
      state = tf.convert_to_tensor(state, dtype=tf.float32)
      episode_reward=0
      done=False  
      print("Episode Started")
      while not done:
        action=self.get_action(state)
        next_state, reward, done, info=env.step(action)
        self.remember(state,reward,action,next_state,done)
        state = next_state
        state = np.expand_dims(state , axis = 0)
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        episode_reward+=reward
        if done:
          break
      self.Average_rewards.append(episode_reward)
      avg_reward = np.mean(self.Average_rewards[-100:])
      print("Episode:{}  Reward:{} Average_reward:{}".format(episode,episode_reward,avg_reward))
      print("Updating the model")
      self.update_models()
      if episode%20==0 and episode!=0:
        self.evaluate(episode)
      if episode%100==0 and episode!=0:
        self.Actor_model.save('Actor_{}.h5'.format(episode))
        self.old_Actor_model.save('old_Actor_{}.h5'.format(episode))
        self.Critic_model.save('Critic_{}.h5'.format(episode))

Agent=PPO()
Agent.train(100000)