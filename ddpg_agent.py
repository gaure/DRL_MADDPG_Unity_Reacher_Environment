from actor import Actor
from critic import Critic
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AgentDDPG:

    def __init__(self, state_size, action_size, seed):
        """

        :state_size: size of the state vector
        :action_size: size of the action vector
        """

        self.state_size = state_size
        self.action_size = action_size
        self.t_step = 0
        self.score = 0.0
        self.best = 0.0
        self.seed = seed
        self.learning_rate_actor = 0.0001
        self.learning_rate_critic = 0.001

        # Instances of the policy function or actor and the value function or critic
        # Actor critic with Advantage

        # Actor local and target network definitions
        self.actor_local = Actor(self.state_size,
                                 self.action_size,
                                 self.seed).to(device)

        self.actor_target = Actor(self.state_size,
                                  self.action_size,
                                  self.seed).to(device)

        # Critic local and target
        self.critic_local = Critic(self.state_size,
                                   self.action_size,
                                   self.seed).to(device)

        self.critic_target = Critic(self.state_size,
                                    self.action_size,
                                    self.seed).to(device)
        # Actor Optimizer
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.learning_rate_actor)

        # Critic Optimizer
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.learning_rate_critic)

        # Make sure local and target start with the same weights
        self.actor_target.load_state_dict(self.actor_local.state_dict())
        self.critic_target.load_state_dict(self.critic_local.state_dict())

        # Parameters for the Algorithm
        self.gamma = 0.99  # Discount factor
        self.tau = 0.001  # Soft update for target parameters Actor Critic with Advantage

    # Actor determines what to do based on the policy
    def act_local(self, state):
        # Given a state return the action recommended by the policy actor_local
        # Reshape the state to fit the torch tensor input
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        # Pass the state to the actor local model to get an action
        # recommend for the policy in a state
        # set the actor_local model to predict not to train
        self.actor_local.eval()
        # set the model so this operation is not counted in the
        # gradiant calculation.
        with torch.no_grad():
            actions = self.actor_local(state)
        # set the model back to training mode
        self.actor_local.train()

        # Return actions tensor
        return actions.detach()

    def act_target(self, states):
        # Pass the state to the actor target model to get an action
        # recommend for the policy in a state
        # set the actor_target model to predict not to train
        self.actor_target.eval()
        # set the model so this operation is not counted in the
        # gradiant calculation.
        with torch.no_grad():
            actions = self.actor_target(states)
        # set the model back to training mode
        self.actor_target.train()

        # Return actions tensor
        return actions.detach()

    def get_episode_score(self):
        """
        Calculate the episode scores
        :return: None
        """
        # Update score and best score
        self.score = self.total_reward / float(self.count) if self.count else 0.0
        if self.score > self.best:
            self.best = self.score

    def save_model_weights(self):
        torch.save(self.actor_local.state_dict(), './checkpoints.pkl')

