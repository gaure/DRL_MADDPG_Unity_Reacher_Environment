import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):

    def __init__(self,
                 state_size,
                 action_size,
                 seed,
                 fc1_units=400,
                 fc2_units=300):
        """
        :seed: (float): torch manual seed value
        :state_size: (int): Dimension of each state (# features)
        :action_size: (int): Dimension of each action
        :layers dimentions: fc1_units...fcn_units
        """
        self.state_size = state_size
        self.action_size = action_size

        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        # layers creation
        self.fc1 = nn.Linear(self.state_size, fc1_units)
        self.ln1 = nn.LayerNorm(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.ln2 = nn.LayerNorm(fc2_units)
        self.output = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        # Continues action range between -1 and 1
        actions = torch.tanh(self.output(x))

        scaled_actions = torch.clamp(actions, -1, 1)

        # return a tensor with the actions recommended by the policy
        # on the inpute state of the state
        return scaled_actions
