import datetime
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gymnasium as gym

from time import time, sleep
from tqdm import trange
from keras.models import Model, clone_model, save_model
from keras.losses import mean_squared_error
from keras.optimizers import Adam

from replay_buffer import ReplayBuffer
from running_average import RunningAverageMetric, RunningAverageReturn

# Create Single Agent Randomised Ensemble Double Q-learning

class AgentREDQ:
    def __init__(
            self,
            policy_network: tf.keras.models.Model,
            q_network: tf.keras.models.Model,
            env: gym.Env):
        
        self.policy_network = policy_network
        self.q_network = q_network

        self.env = env
        self._init_environment()

        # Initialise the running averages for episodic mean return and std of temporal difference targets
        self.td_targets_std_instance = RunningAverageMetric(sample_size=100, non_zero=True)
        self.mean_return_instance = RunningAverageReturn(sample_size=3)


    # Function to create Q function ensemble network

    def _init_ensemble(self, q_network: tf.keras.models.Model, ensemble_size: int):
        state_input = q_network.layers[0].input
        action_input = q_network.layers[1].input

        inputs = q_network.layers[2]([state_input, action_input])

        class SubmodelLayer(tf.keras.layers.Layer):
            def __init__(self, **kwargs):
                super(SubmodelLayer, self).__init__(**kwargs)

            def build(self, input_shape):
                self.backbone = clone_model(Model(inputs=q_network.layers[3].input, outputs=q_network.layers[-1].output))
                super(SubmodelLayer, self).build(input_shape)

            def call(self, inputs):
                return self.backbone(inputs)
            
        # Create a list to hold the outputs of each submodel
        backbone_outputs = [SubmodelLayer()(inputs) for _ in range(ensemble_size)]

        critic_network = Model(inputs=[state_input, action_input], outputs=backbone_outputs)

        return critic_network


    # Function to reset the environment and get the initial state

    def _reset_environment(self):
        init_state, _ = self.env.reset(seed=42)
        init_state = tf.convert_to_tensor(init_state, dtype=tf.float32)
        init_state = tf.expand_dims(init_state, axis=0)
        self.state = init_state
        

    # Function to initialise the environment with a tensorflow compatible step function

    def _init_environment(self):
        self._reset_environment()
        state_dim = self.env.observation_space.shape[0]

        # Function to perform environment update
        def env_step(action: np.ndarray):
            state, reward, done, trunc, _ = self.env.step(action[0, :])
            done = done or trunc
            return state.astype(np.float32), reward.astype(np.float32), np.asarray(done, dtype=np.float32)

        # Need to wrap the env step function inside a tensorflow compatible function
        # We need to be able to obtain the raw numpy data inside the action tensor
        def tf_env_step(action: tf.Tensor):
            state, reward, done = tf.numpy_function(env_step, [action], [tf.float32, tf.float32, tf.float32])
            state = tf.reshape(state, shape=(1, state_dim))
            
            return state, reward, done
        
        self.tf_env_step = tf_env_step
    

    # Function to get a pre training environment transition (uniformly sampled action) and add to the agents replay buffer

    def _pretrain_env_step(self):
        action = tf.convert_to_tensor(self.env.action_space.sample(), dtype=tf.float32)
        action = tf.expand_dims(action, axis=0)
        new_state, reward, done = self.tf_env_step(action)

        # Append transition to replay buffer
        transition = (self.state, action, reward, new_state, done)
        self.replay_buffer._append(transition)

        # If terminated (done=True) reset environment otherwise update state
        if done == 1:
            self._reset_environment()
        else:
            self.state = new_state  

    # Function to sample an action from the policy network

    def _sample_action(self, states: tf.Tensor):
        outputs = self.policy_network(states)
        
        # Get the mean and stddev vector from policy network
        means = outputs[0]
        stddevs = outputs[1] + tf.constant(1e-6)

        # Get the approximated multivariate normal policy distribution 
        policy_distribution = tfp.distributions.MultivariateNormalDiag(
            loc=means,
            scale_diag=stddevs
        )

        # We sample the actions values from the multivariate normal dist and get the log likelihoods
        sample_values = policy_distribution.sample(seed=42)
        log_probs = policy_distribution.log_prob(sample_values)

        # We squash these values into the range (-1, 1) via tanh
        action_values = tf.math.tanh(sample_values)

        # Compute the log-likelihoods of these squashed actions
        action_log_probs = (
            log_probs
            -
            tf.math.reduce_sum(tf.math.log(1 + 1e-6 - action_values**2), axis=-1)
        )

        # action_values, shape: (None, action_dim)
        # action_log_ls, shape: (None, )
        return action_values, action_log_probs


    # Function to get a during training environment transition (action sampled using policy network) and add to the agents replay buffer

    def _train_env_step(self, total_steps: int, start_steps: int):
        # Sample action using agents policy network, update environment and add transition to replay buffer
        # Note: for an initial number of start_steps environment steps we sample from the action space uniformly at random
        if total_steps < start_steps:
            action = tf.convert_to_tensor(self.env.action_space.sample(), dtype=tf.float32)
            action = tf.expand_dims(action, axis=0)
        else:
            action, _ = self._sample_action(self.state)

        new_state, reward, done = self.tf_env_step(action)

        transition = (self.state, action, reward, new_state, done)
        self.replay_buffer._append(transition)

        # If terminated (done=True) reset environment
        if done == 1:
            self._reset_environment()
        else:
            self.state = new_state


    # Function to calculate the mean return evaluation metric of current agent

    def mean_return(self, eval_episodes: int):
        # Get initial state and function to perform environment update
        self._init_environment()

        episode_returns = []
        for _ in range(eval_episodes):

            episode_return = 0
            done = 0
            # Loop over transitions
            while not done:
                # Get most likely action from the stochastic policy
                action = tf.math.tanh(self.policy_network(self.state)[0])

                # Perform action and get reward
                self.state, reward, done = self.tf_env_step(action)

                episode_return += reward

            self._reset_environment()
            episode_returns.append(episode_return)

        return tf.math.reduce_mean(episode_returns).numpy()
        

    # Function to calculate temporal difference targets and record the mean and std in running average instances

    @tf.function
    def _compute_targets(
            self,
            rewards: tf.Tensor, 
            new_states: tf.Tensor, 
            done_flags: tf.Tensor, 
            discount_factor: float,
            entropy_reg: float,
            num_q_evals: int):
        
        # Sample new action and get action prob
        # Shapes are: (None, action_dim), (None, )
        new_actions, new_action_log_probs = self._sample_action(new_states)

        # Get the tensor of Q values from the target network ensemble
        # Shape: (None, self.ensemble_size)
        q_values = tf.concat(self.target_network([new_states, new_actions]), axis=-1)

        # Concatenate list of Q values from each submodel head in ensemble along last dimension then
        # gather the elements using a subset of values calculated by each head for each element in batch
        # Shape: (None, self.ensemble_size) -> (None, num_q_evals)
        q_indices = tf.random.categorical(tf.ones(shape=q_values.shape, dtype=tf.float32), num_samples=num_q_evals, dtype=tf.int32)
        q_values_subset = tf.gather(q_values, q_indices, axis=-1, batch_dims=True)

        # Get the minimum Q values from the randomly chosen subset of Q functions in the ensemble
        min_q_values = tf.math.reduce_min(q_values_subset, axis=-1)
        
        # Calculate temporal difference targets
        td_targets = rewards + discount_factor * (1 - done_flags) * (min_q_values - entropy_reg * new_action_log_probs)

        # Compute the standard deviation of this batch of targets
        td_targets_std = tf.math.reduce_std(td_targets)

        # Update the td targets std running average instance
        self.td_targets_std_instance._tf_update(td_targets_std)

        # Repeat td targets along -1 axis because we want them to be the same for each network in ensemble
        return tf.repeat(tf.expand_dims(td_targets, axis=-1), repeats=self.ensemble_size, axis=-1)
    

    # Function to compute a single critic network loss

    def _compute_critic_loss(
            self,
            states: tf.Tensor, 
            actions: tf.Tensor, 
            td_targets: tf.Tensor):
        
        # Shape: (None, self.ensemble_size)
        q_values = tf.concat(self.critic_network([states, actions]), axis=-1)
        
        # Shape: (None, )
        critic_loss = mean_squared_error(y_true=td_targets, y_pred=q_values)

        return tf.math.reduce_mean(critic_loss)
    

    # Function to compute policy network loss

    def _compute_policy_loss(
            self,
            states: tf.Tensor,
            entropy_reg: float):
        
        # Sample actions and action_probs from policy 
        # Note: sampling is differentiable thanks to the reparameterisation trick
        actions, action_log_probs = self._sample_action(states)

        q_values = tf.concat(self.critic_network([states, actions]), axis=-1)

        # Get the mean Q values along the ensemble of Q functions to get a better approximator
        mean_q_values = tf.math.reduce_mean(q_values, axis=-1)

        # Calculate the policy loss value 
        # Note: negative sign in the front of calculation
        policy_loss = -tf.math.reduce_mean(mean_q_values - entropy_reg * action_log_probs)

        return policy_loss
    

    # Function to update target Q network using Polyak interpolation

    def _update_target_network(self, polyak: float):

        critic_network_weights = self.critic_network.get_weights()
        target_network_weights = self.target_network.get_weights()

        target_network_new_weights = []

        for i in range(len(critic_network_weights)):
            target_network_new_weights.append(polyak * target_network_weights[i] + (1 - polyak) * critic_network_weights[i])

        self.target_network.set_weights(target_network_new_weights)


    # Function to perform a single critic network update
    @tf.function
    def _critic_update(
        self,
        optimizer: tf.keras.optimizers.Optimizer,
        states: tf.Tensor,
        actions: tf.Tensor,
        td_targets: tf.Tensor):

        with tf.GradientTape() as tape:
            critic_loss = self._compute_critic_loss(states, actions, td_targets)

        critic_grads = tape.gradient(critic_loss, self.critic_network.trainable_variables)
        optimizer.apply_gradients(zip(critic_grads, self.critic_network.trainable_variables))

        return critic_loss
    

    # Function to perform a single policy update
    @tf.function
    def _policy_update(
            self,
            optimizer: tf.keras.optimizers.Optimizer,
            states: tf.Tensor,
            entropy_reg: float):

        with tf.GradientTape() as tape:
            policy_loss = self._compute_policy_loss(states, entropy_reg)

        policy_grads = tape.gradient(policy_loss, self.policy_network.trainable_variables)
        optimizer.apply_gradients(zip(policy_grads, self.policy_network.trainable_variables))

        return policy_loss

    # Function to perform a single training step of the agent

    def _training_step(            
            self,
            batch_size: int,
            critic_optimizer: tf.keras.optimizers.Optimizer,
            policy_optimizer: tf.keras.optimizers.Optimizer,
            discount_factor: float,
            entropy_reg: float,
            polyak: float,
            num_q_evals: int):
        
        t0 = time()
        
        # Gets transition tensors from replay buffer sample
        states, actions, rewards, new_states, done_flags = self.replay_buffer._sample(batch_size)

        t1 = time()

        # Get temporal difference targets
        td_targets = self._compute_targets(
            rewards,
            new_states,
            done_flags,
            discount_factor,
            entropy_reg,
            num_q_evals
        )

        t2 = time()

        # Perform critic update
        critic_loss = self._critic_update(
            critic_optimizer,
            states,
            actions,
            td_targets
        )

        t3 = time()

        # Update target networks
        self._update_target_network(polyak)

        t4 = time()

        # Perform policy update
        policy_loss = self._policy_update(
            policy_optimizer,
            states,
            entropy_reg
        )

        t5 = time()
        print(t1-t0)
        print(t2-t1)
        print(t3-t2)
        print(t4-t3)
        print(t5-t4)
        print()

        return critic_loss, policy_loss



    def train(
            self,
            steps_per_epoch: int = 1000,
            epochs: int = 50,
            update_after: int = 5000,
            start_steps: int = 0,
            train_steps_per_env_step: int = 20,
            ensemble_size: int = 5,
            num_q_evals: int = 2,
            eval_episodes: int = 1,
            batch_size: int = 256,
            replay_size: int = 10 ** 6,
            optimizer: tf.keras.optimizers.Optimizer = Adam,
            learning_rate: float = 1e-4,
            discount_factor: float = 0.99,
            entropy_reg: float = 0.20,
            polyak: float = 0.995,
            log_warnings: bool = False):
        


        
        # !!! INIT PHASE !!!

        # Set tensorflow logger to ignore warnings by default
        if not log_warnings:
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        # Create default tensorflow summary writer for logging to tensorboard
        current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H%M")
        log_dir = f'logging/{current_time}'
        summary_writer = tf.summary.create_file_writer(log_dir)

        # Set seeds for reproducibility
        np.random.seed(seed=42)
        tf.random.set_seed(seed=42)

        tf.keras.mixed_precision.set_global_policy('mixed_float16')

        # Initialise critic ensemble network
        self.critic_network = self._init_ensemble(self.q_network, ensemble_size)

        # Initialise target ensemble network
        self.target_network = clone_model(self.critic_network)

        self.ensemble_size = ensemble_size

        self.step = 0
        self.train_steps_per_env_step = train_steps_per_env_step

        self.policy_loss = tf.convert_to_tensor(0, dtype=tf.float32)
        self.critic_loss = tf.convert_to_tensor(0, dtype=tf.float32)        

        # Initialise replay buffer
        self.replay_buffer = ReplayBuffer(replay_size)

        # Intialise optimizers (same learning_rate)
        critic_optimizer = optimizer(learning_rate)
        policy_optimizer = optimizer(learning_rate)

        # Set initial max return seen so far to 0
        max_return = 0.0

        mean_return = self.mean_return(eval_episodes)
        with summary_writer.as_default():
            tf.summary.scalar("mean_return", mean_return, step=0)



        # !!! SETUP PHASE !!!
        # Loop over pre-training steps (partly fills the replay buffer)
        print("Setup Phase  -  Filling Replay Buffer")
        sleep(0.5)
        t = trange(update_after)
        for _ in t:
            # Pre-training environment transition steps 
            self._pretrain_env_step()

        


        # !!! TRAINING PHASE !!!

        tf.summary.trace_on(graph=True, profiler=True)
        # Call only one tf.function when tracing.
        self._training_step(
            batch_size,
            critic_optimizer,
            policy_optimizer,  
            discount_factor,
            entropy_reg,
            polyak,
            num_q_evals
        )        
        with summary_writer.as_default():
            tf.summary.trace_export(
                name="my_func_trace",
                step=0,
                profiler_outdir=log_dir
            )


        print()
        print(f"Training Phase  -  Training Model {current_time}")
        sleep(0.5)
        for epoch in range(epochs):
            # Loop over transition steps in epoch
            t = trange(steps_per_epoch)
            t.set_description_str(f"Epoch {epoch+1}/{epochs}")
            for i in t:
                total_steps = epoch * steps_per_epoch + i

                # Mid-training environment transition steps
                self._train_env_step(total_steps, start_steps)

                # Perform network updates
                for _ in range(train_steps_per_env_step):

                    critic_loss, policy_loss = self._training_step(
                        batch_size,
                        critic_optimizer,
                        policy_optimizer,
                        discount_factor,
                        entropy_reg,
                        polyak,
                        num_q_evals
                    )

                    self.critic_loss = critic_loss
                    self.policy_loss = policy_loss      
                        
                    t.set_postfix(critic_loss=self.critic_loss.numpy(), policy_loss=self.policy_loss.numpy())

                with summary_writer.as_default():
                    tf.summary.scalar("critic_loss", self.critic_loss, step=total_steps)
                    tf.summary.scalar("policy_loss", self.policy_loss, step=total_steps)
                    tf.summary.scalar("td_std", self.td_targets_std_instance._get_running_average(), step=total_steps)


            # Print end of epoch metrics
            new_return = self.mean_return(eval_episodes)
            with summary_writer.as_default():
                tf.summary.scalar("mean_return", new_return, step=epoch+1)

            if new_return > max_return:
                print(f"mean_return={new_return}  -  improved from {max_return}  -  agent saved to {log_dir}\n")
                max_return = new_return

                save_model(self.policy_network, filepath=f"{log_dir}/agent/policy_network", include_optimizer=False)
                save_model(self.critic_network, filepath=f"{log_dir}/agent/critic_network", include_optimizer=False)
            else:
                print(f"mean_return={new_return}  -  no improvement")

        # !!! END OF TRAINING !!!

        # Close environment
        self.env.close()