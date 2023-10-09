import datetime
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gymnasium as gym

from threading import Thread
from time import sleep
from tqdm import trange
from keras.layers import Input, Dense, Concatenate
from keras.models import Model, _clone_layer, clone_model, save_model
from keras.optimizers import Adam

from ensemble_utils_layers import EnsembleDenseLayer, EnsembleSplitLayer, EnsembleOutputLayer
from replay_buffer import ReplayBufferUniform, ReplayBufferPrioritised
from running_average import RunningAverage

# Create Single Agent Randomised Ensemble Double Q-learning

class AgentREDQ:
    def __init__(self):
        self.policy_network = None
        self.critic_network = None

        self.env = None
        self.state = None

        self.replay_buffer = None

        # Initialise the running averages
        self.td_targets_std_instance = RunningAverage(sample_size=100, non_zero=True)
        self.entropy_instance = RunningAverage(sample_size=20)

        self.critic_loss_instance = RunningAverage(sample_size=100)
        self.policy_loss_instance = RunningAverage(sample_size=100)

        self.mean_return_instance = RunningAverage(sample_size=5)


    # Function to reset the environment and get the initial state

    def _reset_environment(self):
        init_state, _ = self.env.reset(seed=42)
        init_state = tf.convert_to_tensor(init_state, dtype=tf.float32)
        init_state = tf.expand_dims(init_state, axis=0)
        self.state = init_state
        

    # Function to initialise the environment with a tensorflow compatible step function

    def _init_environment(self, env):
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
        
        self.env = env
        self.tf_env_step = tf_env_step
    

    # Function to get a pre training environment transition (uniformly sampled action) and add to the agents replay buffer

    def _pretrain_env_step(self):
        action = tf.convert_to_tensor(self.env.action_space.sample(), dtype=tf.float32)
        action = tf.expand_dims(action, axis=0)
        new_state, reward, done = self.tf_env_step(action)

        # Append transition to replay buffer
        transition = (self.state, action, self.reward_scale * reward, new_state, done)
        self.replay_buffer._append(transition)

        # If terminated (done=True) reset environment otherwise update state
        if done == 1:
            self._reset_environment()
        else:
            self.state = new_state  

    # Function to sample an action from the policy network

    def _sample_action(self, policy_network: tf.keras.models.Model, states: tf.Tensor):
        outputs = policy_network(states)
        
        # Get the mean and stddev vector from policy network
        means = outputs[0]
        stddevs = outputs[1] + tf.constant(1e-4)

        # Get the approximated multivariate normal policy distribution 
        policy_distribution = tfp.distributions.MultivariateNormalDiag(
            loc=means,
            scale_diag=stddevs
        )

        # We sample the actions values from the multivariate normal dist and get the log probs
        sample_values = policy_distribution.sample(seed=42)
        log_probs = policy_distribution.log_prob(sample_values)

        # We squash these values into the range (-1, 1) via tanh
        action_values = tf.math.tanh(sample_values)

        # Compute the log probs of these squashed actions
        action_log_probs = (
            log_probs
            -
            tf.math.reduce_sum(tf.math.log(1 + 1e-4 - action_values**2), axis=-1)
        )

        # action_values, shape: (None, action_dim)
        # action_log_probs, shape: (None, )
        return action_values, action_log_probs


    # Function to get a during training environment transition (action sampled using policy network) and add to the agents replay buffer

    def _train_env_step(self, policy_network: tf.keras.models.Model, env_steps: int, start_steps: int):
        # Sample action using agents policy network, update environment and add transition to replay buffer
        # Note: for an initial number of start_steps environment steps we sample from the action space uniformly at random
        if env_steps < start_steps:
            action = tf.convert_to_tensor(self.env.action_space.sample(), dtype=tf.float32)
            action = tf.expand_dims(action, axis=0)
            entropy = 0
        else:
            action, action_log_probs = self._sample_action(policy_network, self.state)
            entropy = -action_log_probs
            

        new_state, reward, done = self.tf_env_step(action)

        transition = (self.state, action, self.reward_scale * reward, new_state, done)
        self.replay_buffer._append(transition)

        # If terminated (done=True) reset environment
        if done == 1:
            self._reset_environment()
        else:
            self.state = new_state

        return entropy


    # Function to calculate the mean return evaluation metric of current agent

    def mean_return(self, policy_network: tf.keras.models.Model, eval_episodes: int):
        # Get initial state and function to perform environment update
        self._init_environment(self.env)

        episode_returns = []
        for _ in range(eval_episodes):

            episode_return = 0
            done = 0
            # Loop over transitions
            while not done:
                # Get most likely action from the stochastic policy
                action = tf.math.tanh(policy_network(self.state)[0])

                # Perform action and get reward
                self.state, reward, done = self.tf_env_step(action)

                episode_return += reward

            self._reset_environment()
            episode_returns.append(episode_return)

        return tf.math.reduce_mean(episode_returns).numpy()
        

    # Function to calculate temporal difference targets and record the mean and std in running average instances

    def _compute_targets(
            self,
            policy_network: tf.keras.models.Model,
            target_network: tf.keras.models.Model,
            rewards: tf.Tensor, 
            new_states: tf.Tensor, 
            done_flags: tf.Tensor, 
            discount_factor: float,
            entropy_reg: float,
            num_q_evals: int):
        
        # Sample new action and get action prob
        # Shapes are: (None, action_dim), (None, )
        new_actions, new_action_log_probs = self._sample_action(policy_network, new_states)

        # Get the tensor of Q values from the target network ensemble
        # Shape: (None, ensemble_size)
        q_indices = tf.random.shuffle(tf.range(self.ensemble_size))[:num_q_evals]
        q_mask = tf.zeros(shape=(self.ensemble_size,), dtype=tf.float32)
        # Use scatter_update to set the elements at the given indices to 1
        q_mask = tf.tensor_scatter_nd_update(q_mask, tf.expand_dims(q_indices, axis=1), tf.ones_like(q_indices, dtype=tf.float32))

        q_values = tf.concat(target_network([new_states, new_actions, q_mask]), axis=-1)

        # Get the minimum Q values from the randomly chosen subset of Q functions in the ensemble
        min_q_values = tf.math.reduce_min(q_values, axis=-1)

        # Calculate temporal difference targets
        td_targets = rewards + discount_factor * (1 - done_flags) * (min_q_values - entropy_reg * new_action_log_probs)

        return td_targets
    

    # Function to compute a single critic network loss

    def _compute_critic_loss(
            self,
            critic_index: tf.Tensor,
            q_values: tf.Tensor,
            td_targets: tf.Tensor):

        # Gather q_values from current critic
        q_values = tf.gather(q_values, critic_index, axis=-1)    

        tf.io.write_file("log_q_" + tf.strings.format("{}", critic_index), tf.strings.format("{}", q_values))
        tf.io.write_file("log_td_" + tf.strings.format("{}", critic_index), tf.strings.format("{}", td_targets))
        
        critic_loss = tf.math.reduce_mean(
            tf.keras.losses.huber(y_true=td_targets, y_pred=q_values)
        )

        return critic_loss
    

    # Function to compute policy network loss

    def _compute_policy_loss(
            self,
            policy_network: tf.keras.models.Model,
            critic_network: tf.keras.models.Model,
            states: tf.Tensor,
            entropy_reg: float):
        
        # Sample actions and action_probs from policy 
        # Note: sampling is differentiable thanks to the reparameterisation trick
        actions, action_log_probs = self._sample_action(policy_network, states)

        q_mask = tf.ones(shape=self.ensemble_size, dtype=tf.float32)
        q_values = tf.concat(critic_network([states, actions, q_mask]), axis=-1)

        # Get the mean Q values along the ensemble of Q functions to get a better approximator
        #min_q_values = tf.math.reduce_min(q_values, axis=-1)
        mean_q_values = tf.math.reduce_mean(q_values, axis=-1)

        # Calculate the policy loss value 
        # Note: negative sign in the front of calculation
        policy_loss = -tf.math.reduce_mean(mean_q_values - entropy_reg * action_log_probs)

        return policy_loss

    def _update_target_network(
            self,
            target_network: tf.keras.Model, 
            critic_network: tf.keras.Model, 
            polyak: float):

        # Updates target networks using polyak interpolation
        critic_network_weights = critic_network.get_weights()

        target_network_weights = target_network.get_weights()

        target_network_new_weights = []

        for i in range(len(critic_network_weights)):
            target_network_new_weights.append(polyak * target_network_weights[i] + (1 - polyak) * critic_network_weights[i])

        target_network.set_weights(target_network_new_weights)


    # Function to perform a single training step of the agent

    @tf.function
    def _training_step(            
            self,
            update_policy: bool,
            policy_network: tf.keras.models.Model,
            critic_network: tf.keras.models.Model,
            target_network: tf.keras.models.Model,
            policy_optimizer: tf.keras.optimizers.Optimizer,
            critic_optimizer: tf.keras.optimizers.Optimizer,
            sample: list[tf.Tensor],
            discount_factor: float,
            entropy_reg: float,
            num_q_evals: int):
        
        # Unpack batched transition tensors from sample
        states, actions, rewards, new_states, done_flags = sample

        # Get temporal difference targets
        td_targets = self._compute_targets(
            policy_network,
            target_network,
            rewards,
            new_states,
            done_flags,
            discount_factor,
            entropy_reg,
            num_q_evals
        )

        # Calculate std of td_targets
        td_targets_std = tf.math.reduce_std(td_targets)

        critic_index = tf.constant(0, dtype=tf.int32)
        with tf.GradientTape(persistent=True) as tape:

            # Calculate the q values from all critic networks
            q_mask = tf.ones(self.ensemble_size, dtype=tf.float32)
            q_values = tf.concat(critic_network([states, actions, q_mask]), axis=-1)

            # Update the individual critic networks 
            while critic_index < self.ensemble_size:

                critic_loss = self._compute_critic_loss(
                    critic_index,
                    q_values,
                    td_targets
                )

                critic_grads = tape.gradient(critic_loss, critic_network.trainable_variables)
                critic_optimizer.apply_gradients(zip(critic_grads, critic_network.trainable_variables))

                critic_index += 1

            pred = tf.equal(update_policy, True)
            def true_fn():
                # Calculate the policy loss with auto-diff
                policy_loss = self._compute_policy_loss(
                    policy_network,
                    critic_network,
                    states,
                    entropy_reg
                )
                policy_grads = tape.gradient(policy_loss, policy_network.trainable_variables)
                policy_optimizer.apply_gradients(zip(policy_grads, policy_network.trainable_variables))
                return policy_loss
                
            def false_fn():
                # Calculate the policy loss without auto-diff
                with tape.stop_recording():
                    policy_loss = self._compute_policy_loss(
                        policy_network,
                        critic_network,
                        states,
                        entropy_reg
                    )
                return policy_loss

            # Update policy network if ready
            policy_loss = tf.cond(pred, true_fn, false_fn)

        return critic_loss, policy_loss, td_targets_std


    def networks(
            self,
            state_dim: int,
            action_dim: int,
            policy_network_hidden_units: list = [256, 256],
            critic_network_hidden_units: list = [256, 256],
            ensemble_size: int = 5,
            display_summaries: bool = False):
                

        self.ensemble_size = ensemble_size

        # Initialise policy network
        state_input = Input(shape=(state_dim))
        x = state_input

        for hidden_units in policy_network_hidden_units:
            x = Dense(units=hidden_units, activation="relu")(x)

        means = Dense(units=action_dim)(x)
        stddevs = Dense(units=action_dim, activation="softplus")(x)

        self.policy_network = Model(inputs=[state_input], outputs=[means, stddevs])
        
        # Initialise critic ensemble network
        state_input = tf.keras.layers.Input(shape=(state_dim))
        action_input = tf.keras.layers.Input(shape=(action_dim))
        mask_input = tf.keras.layers.Input(shape=(ensemble_size))

        data_input = tf.keras.layers.Concatenate()([state_input, action_input])
        mask_input_split = EnsembleSplitLayer()(mask_input)

        x = [EnsembleDenseLayer(critic_network_hidden_units[0], activation="relu")([data_input, mask_input_split[i]]) for i in range(ensemble_size)]
        for hidden_units in critic_network_hidden_units[1:]:
            x = [EnsembleDenseLayer(hidden_units, activation="relu")(x[i]) for i in range(ensemble_size)]
        x = [EnsembleDenseLayer(1)(x[i]) for i in range(ensemble_size)]

        outputs = [EnsembleOutputLayer()(x[i]) for i in range(ensemble_size)]

        self.critic_network = tf.keras.models.Model(inputs=[state_input, action_input, mask_input], outputs=outputs)

        if display_summaries:
            self.policy_network.summary()
            self.critic_network.summary()

        return self


    def buffer(
            self,
            buffer_type: str = "uniform",
            replay_size: int = 10 ** 5,
            prioritised_replay_size: int = None):
    

        if buffer_type == "uniform":
            self.replay_buffer = ReplayBufferUniform(replay_size)

        elif buffer_type == "prioritised":
            if prioritised_replay_size != None and prioritised_replay_size < replay_size:
                self.replay_buffer = ReplayBufferPrioritised(replay_size, prioritised_replay_size)
            else:
                print("Missing required argument. If buffer_type='prioritised' then must have prioritised_replay_size << replay_size.")
        
        else:
            print("Invalid argument value. buffer_type must be the string 'uniform' or 'prioritised'.")

        return self


    def environment(self, env: gym.Env):
        self._init_environment(env)

        return self


    def train(
            self,
            steps_per_epoch: int = 4000,
            epochs: int = 20,
            update_after: int = 4000,
            start_steps: int = 0,
            train_steps_per_env_step: int = 10,
            num_q_evals: int = 2,
            eval_episodes: int = 1,
            batch_size: int = 256,
            optimizer: tf.keras.optimizers.Optimizer = Adam,
            learning_rate: float = 1e-4,
            reward_scale: float = 0.01,
            discount_factor: float = 0.99,
            entropy_reg: float = 0.2,
            polyak: float = 0.995,
            log_warnings: bool = False):
        


        
        # !!! INIT PHASE !!!

        # Check that all necessary components for training have been initialised
        init_error = False
        if self.policy_network is None:
            print("Policy and/or critic network not initialised. Please use method 'network' to initialise")
            init_error = True
        elif self.replay_buffer is None:
            print("Replay buffer not initialised. Please use method 'buffer' to initialise")
            init_error = True
        elif self.env is None:
            print("Environment not initialised. Please use method 'environment' to initialise")
            init_error = True        

        if init_error:
            return

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

        # Initialise optimizers
        policy_optimizer = optimizer(learning_rate)
        critic_optimizer = optimizer(learning_rate)

        # Initialise networks
        policy_network = self.policy_network
        critic_network = self.critic_network
        target_network = clone_model(critic_network)

        self.reward_scale = reward_scale

        # Set initial max return seen so far to 0
        max_return = 0.0
        new_return = self.mean_return(policy_network, eval_episodes)
        self.mean_return_instance._update(new_return)




        # !!! SETUP PHASE !!!
        # Loop over pre-training steps (partly fills the replay buffer)
        print("Setup Phase  -  Filling Replay Buffer")
        sleep(0.5)
        t = trange(update_after)
        for _ in t:
            # Pre-training environment transition steps 
            self._pretrain_env_step()

        


        # !!! TRAINING PHASE !!!

        update_policy = False
        sample = [None] * 5
        self.replay_buffer._sample(sample, batch_size)
        tf.summary.trace_on(graph=True, profiler=True)
        # Call only one tf.function when tracing.
        self._training_step(
            update_policy,
            policy_network,
            critic_network,
            target_network,
            policy_optimizer,
            critic_optimizer,
            sample,
            discount_factor,
            entropy_reg,
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
        env_steps = 0
        for epoch in range(epochs):
            # Loop over transition steps in epoch
            t = trange(steps_per_epoch)
            t.set_description_str(f"Epoch {epoch+1}/{epochs}")
            for _ in t:
                # Mid-training environment transition steps
                entropy = self._train_env_step(policy_network, env_steps, start_steps)
                self.entropy_instance._update(entropy.numpy())

                # Perform network updates
                for update_step in range(train_steps_per_env_step):

                    # Perform sampling from buffer (on child thread)
                    new_sample = [None] * 5
                    child_thread = Thread(target=self.replay_buffer._sample, args=(new_sample, batch_size))
                    child_thread.start()

                    # Determine whether to update the policy network
                    update_policy = bool(update_step == train_steps_per_env_step-1)

                    # Perform training step (on main thread)
                    critic_loss, policy_loss, td_targets_std = self._training_step(
                        update_policy,
                        policy_network,
                        critic_network,
                        target_network,
                        policy_optimizer,
                        critic_optimizer,
                        sample,
                        discount_factor,
                        entropy_reg,
                        num_q_evals
                    )     

                    # Update target network
                    self._update_target_network(target_network, critic_network, polyak)

                    child_thread.join()
                    sample = new_sample

                    self.critic_loss_instance._update(critic_loss.numpy())
                    self.policy_loss_instance._update(policy_loss.numpy())
                    self.td_targets_std_instance._update(td_targets_std.numpy())

                    t.set_postfix(critic_loss=f"{critic_loss.numpy():.3f}", policy_loss=f"{policy_loss.numpy():.3f}")
                
                if env_steps % int(steps_per_epoch // 10) == 0:
                    new_return = self.mean_return(policy_network, eval_episodes)
                    self.mean_return_instance._update(new_return)


                with summary_writer.as_default():
                    tf.summary.scalar("critic_loss", self.critic_loss_instance._get_running_average(), step=env_steps)
                    tf.summary.scalar("policy_loss", self.policy_loss_instance._get_running_average(), step=env_steps)
                    tf.summary.scalar("td_std", self.td_targets_std_instance._get_running_average(), step=env_steps)
                    tf.summary.scalar("entropy", self.entropy_instance._get_running_average(), step=env_steps)
                    tf.summary.scalar("episode_return", self.mean_return_instance._get_running_average(), step=env_steps)

                env_steps += 1

            if new_return > max_return:
                print(f"mean_return={new_return}  -  improved from {max_return}  -  agent saved to {log_dir}\n")
                max_return = new_return

                save_model(policy_network, filepath=f"{log_dir}/agent/policy_network", include_optimizer=False)
                save_model(critic_network, filepath=f"{log_dir}/agent/critic_network", include_optimizer=False)
            else:
                print(f"mean_return={new_return}  -  no improvement\n")

        # !!! END OF TRAINING !!!

        # Close environment
        self.env.close()