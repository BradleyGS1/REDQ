import datetime
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gymnasium as gym

from threading import Thread
from time import time, sleep
from tqdm import trange
from keras.layers import Input
from keras.losses import mean_squared_error
from keras.models import Model, _clone_layer, clone_model, save_model
from keras.optimizers import Adam

from replay_buffer import ReplayBuffer
from running_average import RunningAverage

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

        # Initialise the running averages
        self.td_targets_std_instance = RunningAverage(sample_size=100, non_zero=True)
        self.critic_loss_instance = RunningAverage(sample_size=100)
        self.policy_loss_instance = RunningAverage(sample_size=100)
        self.mean_return_instance = RunningAverage(sample_size=3)


    # Function to create Q function ensemble network

    def _init_ensemble(self, q_network: tf.keras.models.Model, ensemble_size: int):
        state_input = Input(q_network.layers[0].input_shape[0][1])
        action_input = Input(q_network.layers[1].input_shape[0][1])

        join = _clone_layer(q_network.layers[2])([state_input, action_input])

        backbone_layers = [clone_model(Model(inputs=q_network.layers[3].input, outputs=q_network.layers[-1].output)) for _ in range(ensemble_size)]

        backbone_layers = [backbone_layer(join) for backbone_layer in backbone_layers]

        ensemble_network = Model(inputs=[state_input, action_input], outputs=backbone_layers)

        return ensemble_network


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

    def _sample_action(self, policy_network: tf.keras.models.Model, states: tf.Tensor):
        outputs = policy_network(states)
        
        # Get the mean and stddev vector from policy network
        means = outputs[0]
        stddevs = outputs[1] + tf.constant(1e-8)

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

        # Compute the log-likelihoods of these squashed actions
        action_log_probs = (
            log_probs
            -
            tf.math.reduce_sum(tf.math.log(1 + 1e-8 - action_values**2), axis=-1)
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
        else:
            action, _ = self._sample_action(policy_network, self.state)

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
        q_values = tf.concat(target_network([new_states, new_actions]), axis=-1)

        # Concatenate list of Q values from each submodel head in ensemble along last dimension then
        # gather the elements using a subset of values calculated by each head for each element in batch
        # Shape: (None, ensemble_size) -> (None, num_q_evals)
        q_indices = tf.random.categorical(tf.zeros(shape=q_values.shape, dtype=tf.float32), num_samples=num_q_evals, dtype=tf.int32)
        q_values_subset = tf.gather(q_values, q_indices, axis=-1, batch_dims=True)

        # Get the minimum Q values from the randomly chosen subset of Q functions in the ensemble
        min_q_values = tf.math.reduce_min(q_values_subset, axis=-1)
        
        # Calculate temporal difference targets
        td_targets = rewards + discount_factor * (1 - done_flags) * (min_q_values - entropy_reg * new_action_log_probs)

        # Repeat td targets along -1 axis because we want them to be the same for each network in ensemble
        td_targets = tf.repeat(tf.expand_dims(td_targets, axis=-1), repeats=q_values.shape[-1], axis=-1)

        return td_targets
    

    # Function to compute a single critic network loss

    def _compute_critic_loss(
            self,
            critic_network: tf.keras.models.Model,
            states: tf.Tensor, 
            actions: tf.Tensor, 
            td_targets: tf.Tensor):
        
        # Shape: (None, ensemble_size)
        q_values = tf.concat(critic_network([states, actions]), axis=-1)

        critic_loss = tf.math.reduce_mean(tf.math.reduce_sum((q_values - td_targets) ** 2, axis=-1))

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

        q_values = tf.concat(critic_network([states, actions]), axis=-1)

        # Get the mean Q values along the ensemble of Q functions to get a better approximator
        mean_q_values = tf.math.reduce_mean(q_values, axis=-1)

        # Calculate the policy loss value 
        # Note: negative sign in the front of calculation
        policy_loss = -tf.math.reduce_mean(mean_q_values - entropy_reg * action_log_probs)

        return policy_loss
    

    # Function to update target Q network using Polyak interpolation

    def _update_target_network(self, target_network: tf.keras.models.Model, critic_network: tf.keras.models.Model, polyak: float):
    
        for target_var, source_var in zip(target_network.trainable_variables, critic_network.trainable_variables):
            target_var.assign(polyak * target_var + (1 - polyak) * source_var)


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
            states: tf.Tensor,
            actions: tf.Tensor,
            rewards: tf.Tensor,
            new_states: tf.Tensor,
            done_flags: tf.Tensor,
            discount_factor: float,
            entropy_reg: float,
            polyak: float,
            num_q_evals: int):
        

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

        # Calculate critic and policy loss with autodiff enabled
        with tf.GradientTape(persistent=True) as tape:

            critic_loss = self._compute_critic_loss(
                critic_network,
                states,
                actions,
                td_targets
            )

            policy_loss = self._compute_policy_loss(
                policy_network,
                critic_network,
                states,
                entropy_reg
            )

        # Update critic network
        critic_grads = tape.gradient(critic_loss, critic_network.trainable_variables)
        critic_optimizer.apply_gradients(zip(critic_grads, critic_network.trainable_variables))

        pred = tf.equal(update_policy, True)

        def true_fn():
            policy_grads = tape.gradient(policy_loss, policy_network.trainable_variables)
            policy_optimizer.apply_gradients(zip(policy_grads, policy_network.trainable_variables))
        def false_fn():
            pass

        # Update policy network if ready
        tf.cond(pred, true_fn, false_fn)

        # Update target network
        self._update_target_network(target_network, critic_network, polyak)

        return critic_loss, policy_loss, td_targets_std


    def train(
            self,
            steps_per_epoch: int = 1000,
            epochs: int = 50,
            update_after: int = 4000,
            start_steps: int = 1000,
            train_steps_per_env_step: int = 20,
            ensemble_size: int = 10,
            num_q_evals: int = 3,
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

        # Initialise policy network
        policy_network = self.policy_network

        # Initialise critic ensemble network
        #critic_network = self._init_ensemble(self.q_network, ensemble_size)

        state_input = Input(shape=(11))
        action_input = Input(shape=(3))

        from keras.layers import Dense, Concatenate

        input = Concatenate()([state_input, action_input])

        hidden_units = 256

        x_layers = [Dense(units=hidden_units, activation="relu")(input) for _ in range(ensemble_size)]
        y_layers = [Dense(units=hidden_units, activation="relu")(x) for x in x_layers]

        q_values = [Dense(units=1)(y) for y in y_layers]

        critic_network = Model(inputs=[state_input, action_input], outputs=q_values)

        critic_network.summary()

        # Initialise target ensemble network
        target_network = clone_model(critic_network)

        for layer in target_network.layers:
            layer.trainable = False

        # Initialise replay buffer
        self.replay_buffer = ReplayBuffer(replay_size)

        # Intialise optimizers (same learning_rate)
        policy_optimizer = optimizer(learning_rate)
        critic_optimizer = optimizer(learning_rate)

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

        update_policy = False
        sample = [None] * 5
        self.replay_buffer._sample(sample, batch_size)
        states, actions, rewards, new_states, done_flags = sample

        tf.summary.trace_on(graph=True, profiler=True)
        # Call only one tf.function when tracing.
        self._training_step(
            update_policy,
            policy_network,
            critic_network,
            target_network,
            policy_optimizer,
            critic_optimizer,
            states,
            actions,
            rewards,
            new_states,
            done_flags,
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
                env_steps = epoch * steps_per_epoch + i

                # Mid-training environment transition steps
                self._train_env_step(policy_network, env_steps, start_steps)

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
                        states,
                        actions,
                        rewards,
                        new_states,
                        done_flags,
                        discount_factor,
                        entropy_reg,
                        polyak,
                        num_q_evals
                    )     

                    child_thread.join()
                    sample = new_sample

                    self.critic_loss_instance._update(critic_loss.numpy())
                    self.policy_loss_instance._update(policy_loss.numpy())
                    self.td_targets_std_instance._update(td_targets_std.numpy())

                    t.set_postfix(critic_loss=critic_loss.numpy(), policy_loss=policy_loss.numpy())

                with summary_writer.as_default():
                    tf.summary.scalar("critic_loss", self.critic_loss_instance._get_running_average(), step=env_steps)
                    tf.summary.scalar("policy_loss", self.policy_loss_instance._get_running_average(), step=env_steps)
                    tf.summary.scalar("td_std", self.td_targets_std_instance._get_running_average(), step=env_steps)


            # Print end of epoch metrics
            new_return = self.mean_return(eval_episodes)
            with summary_writer.as_default():
                tf.summary.scalar("mean_return", new_return, step=epoch+1)

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