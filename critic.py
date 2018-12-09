import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):

    def __init__(self,
                 state_size,
                 action_size,
                 seed,
                 fc1_units=400,
                 fc2_units=300):
        """

        :state_size: (int) Dimension of each state
        :action_size: (int) Dimension of each action
        :fc1_units: (int) number of units of first hidden layer
        :fc2_units: (int) number of units of first hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.state_size = state_size
        self.action_size = action_size

        # critic input state layers
        self.state_fc1 = nn.Linear(self.state_size, fc1_units)
        self.sln1 = nn.LayerNorm(fc1_units)

        # critic temp networks to join actions and states
        self.temp1 = nn.Linear(fc1_units, fc2_units)
        self.temp2 = nn.Linear(action_size, fc2_units)
        self.temp_norm = nn.LayerNorm(fc2_units)

        # critic output
        self.fc_output = nn.Linear(fc2_units, 1)

    def forward(self, state, action):
        # state path
        s = F.relu(self.sln1(self.state_fc1(state)))

        # merging path
        t1 = self.temp1(s)
        t2 = self.temp2(action)

        # add the actions to the states by concatenation
        merged = F.relu(self.temp_norm(t1 + t2))

        # get the q_value
        q_value = self.fc_output(merged)

        return q_value
