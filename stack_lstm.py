import torch
import torch.nn as nn
import torch.nn.init as init


class StackLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, std=0.02):
        super().__init__()
        self.h0 = nn.Parameter(torch.zeros(hidden_size).unsqueeze(0))
        self.c0 = nn.Parameter(torch.zeros(hidden_size).unsqueeze(0))
        init.normal_(self.h0.data, std=std)
        init.normal_(self.c0.data, std=std)

        self.stack_state_hist = [(self.h0, self.c0)]
        self.stack = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
        self.position_id_stack = [None]  # Tracks position idx of stack elements

    def forward(self, *input):
        return self.push(*input)

    def push(self, inputs, position_idx):
        next_hidden_state, next_cell_state = self.stack(inputs, self.stack_state_hist[-1])
        self.stack_state_hist.append((next_hidden_state, next_cell_state))
        self.position_id_stack.append(position_idx)
        return next_hidden_state

    def pop(self):
        if len(self.stack_state_hist) > 1:
            return self.stack_state_hist.pop(), self.position_id_stack.pop()  # Returns (hidden_state, cell_state), pos
        else:
            raise IndexError("pop from empty StackLSTM.")

    def peek(self):
        if len(self.stack_state_hist) > 1:
            return self.stack_state_hist[-1], self.position_id_stack[-1]  # Returns (hidden_state, cell_state), pos
        else:
            raise IndexError("peek from empty StackLSTM.")

    def top(self):
        return self.stack_state_hist[-1], self.position_id_stack[-1]

    def reset_state(self):
        self.stack_state_hist = [(self.h0, self.c0)]
        self.position_id_stack = [None]

    def __bool__(self):
        return bool(self.__len__())

    def __len__(self):
        return len(self.stack_state_hist) - 1  # Subtract one we don't count (h0, c0)
