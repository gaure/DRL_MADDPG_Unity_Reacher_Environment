# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg_agent import AgentDDPG
from utils import ReplayBuffer
import numpy as np
import torch
import torch.nn.functional as F
from utils import OUNoise

device = 'cpu'


class MADDPG:
    def __init__(self, action_size, discount_factor=0.95, tau=0.02):
        super(MADDPG, self).__init__()

        # Create the multi agent as a list of ddpg agents
        self.maddpg_agents = [AgentDDPG(24, 2, 0),
                             AgentDDPG(24, 2, 0)]
        
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0
        self.total_reward = 0.0
        self.count = 0
        self.update_every = 1
        self.batch_size = 128
        self.agent_number = len(self.maddpg_agents)
        self.t_step = 0
        # Initialize the Replay Memory
        self.buffer_size = 1000000
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)
        self.action_size = action_size
        self.total_reward = np.zeros((1, 2))

        # Initialize the Gaussian Noise process
        self.exploration_mu = 0
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.2
        self.noise = OUNoise(self.action_size,
                             self.exploration_mu,
                             self.exploration_theta,
                             self.exploration_sigma)

    def get_actors_local(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agt.actor_local for ddpg_agt in self.maddpg_agents]
        return actors

    def get_actors_target(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agt.actor_target for ddpg_agt in self.maddpg_agents]
        return target_actors

    def acts_local(self, states, noise_factor=0.0):
        """get actions from all actor_local agents in the MADDPG object"""
        """agents acts based on their individual observations"""
        actions = [agent.act_local(state).numpy() + noise_factor*self.noise.sample()
                   for agent, state in zip(self.maddpg_agents, states)]
        return actions

    def acts_target(self, states, noise_factor=0.0):
        """get actions from all actor_target agents in the MADDPG object"""
        target_actions = [agent.act_target(state).numpy() + noise_factor*self.noise.sample()
                          for agent, state in zip(self.maddpg_agents, states)]
        return target_actions

    # Actor interact with the environment through the step
    def step(self, states, actions, rewards, next_states, dones):
        # Add to the total reward the reward of this time step
        self.total_reward += rewards
        # Increase your count based on the number of rewards
        # received in the episode
        self.count += 1
        # Stored experience tuple in the replay buffer
        self.memory.add(states,
                        actions,
                        rewards,
                        next_states,
                        dones)

        # Learn every update_times time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:

            # Check to see if you have enough to produce a batch
            # and learn from it

            if len(self.memory) > self.batch_size:
                for agent_idx in range(len(self.maddpg_agents)):
                    experiences = self.memory.sample()
                    # Train the networks using the experiences
                    self.learn(experiences, agent_idx)

    def learn(self, experiences, agent_idx):
        """update the critics and actors of all the agents """

        # Reshape the experience tuples in separate arrays of states, actions
        # rewards, next_state, done
        # Your are converting every member of the tuple in a column or vector
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Convert the numpy arrays to tensors
        states = torch.from_numpy(states).float().unsqueeze(0).to(device)
        actions = torch.from_numpy(actions).float().unsqueeze(0).to(device)
        next_states = torch.from_numpy(next_states).float().unsqueeze(0).to(device)

        # Get the agent to train
        agent = self.maddpg_agents[agent_idx]

        # Using the buffer get ALL the actor_target actions based on ALL
        # the agents next states. The acts_target function returns a list of
        # detached tensor returned from the ddpg agent act_target
        next_state_actions = self.acts_target(next_states)
        next_state_actions = torch.from_numpy(np.squeeze(np.array(next_state_actions))).float().unsqueeze(0).to(device)

        # Using ALL the actor_targets next_state_actions based on the next states
        # and ALL the next_states, calculate the "current agent" target critic
        # q_target_next_state_action_values. The critic sees all the agents actions in
        # each of their state to calculate the q_value of the current agent.
        agent.critic_target.eval()
        with torch.no_grad():
            q_target_next_state_action_value = agent.critic_target(next_states, next_state_actions).detach()
        agent.critic_target.train()

        # Calculate the current agent q_targets value
        q_targets = torch.from_numpy(rewards[agent_idx] + self.discount_factor * q_target_next_state_action_value.numpy()
                                     * (1 - dones[agent_idx]))

        # --- Optimize the current critic_local agent --- #

        # Using ALL the agents actions and ALL states where they took
        # you calculate the critic_local q_expected values
        q_expected = agent.critic_local(states, actions)

        # Set the grad to zero, for the critic agent optimizer selected before
        agent.critic_optimizer.zero_grad()

        # Calculate the loss function MSE
        critic_loss = F.smooth_l1_loss(q_expected, q_targets)
        critic_loss.backward(retain_graph=True)

        # Clip the gradient to improve training
        torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), 1)

        # Optimize the critic_local model using the optimizer defined
        # in its init function
        agent.critic_optimizer.step()

        # --- Optimize the current actor_local model --- #

        # Get the current actor_local actions using only the corresponding
        # state, and for other actor states you detach the tensor from the
        # calculation to save computation time
        actor_actions = [self.maddpg_agents[i].actor_local(state) if i == agent_idx
                         else self.maddpg_agents[i].actor_local(state).detach()
                         for i, state in enumerate(states)][0]

        loss_actor = -1 * torch.sum(agent.critic_local.forward(states, actor_actions))

        # Set the grad to zero, for the actor local current agent optimization
        agent.actor_optimizer.zero_grad()
        loss_actor.backward()

        # Optimize the actor_local current agent
        agent.actor_optimizer.step()

        # Soft update the target and local models
        self.soft_update(agent.critic_local, agent.critic_target)
        self.soft_update(agent.actor_local, agent.actor_target)

    def soft_update(self, local_model, target_model):
        # Soft update targets
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save_models_weights(self):
        for idx, agent in enumerate(self.maddpg_agents):
            file_name = "./checkpoints_agent_" + str(idx) + ".pkl"
            torch.save(agent.actor_local.state_dict(), file_name)
