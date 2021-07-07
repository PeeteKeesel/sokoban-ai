import numpy as np


class ReplayMemory:
    """
    Replay memory stores experience tuples from previous episodes. It can be
    used by obtaining batches of experience tuples from the replay memory for
    training a policy network. Old tuples are overwritten once memory limit is
    reached.
    """

    def __init__(self, size, column_types, column_shapes=None, batch_size=32):
        """

        Arguments:
             size:          Number of experience tuples to be stored.
             column_types:  Dictionary mapping comlumn names to the data type
                            of the data to be stored in it.
             column_shapes: Shapes of the data stored in the cells of each
                            column as lists. Empty lists for scalar values.
                            If None, all shapes are considered scalar.
             batch_size:    Size of the batch which is samples from the Replay
                            Memory.
        """
        pass

    def add(self, row):
        """
        Add new row of data to the replay memory.

        Arguments:
            row: Dictionary containing the new row's data for each column.
        """
        pass

    def add_all(self, rows):
        """
        Add multiple rows of data to the replay memory.

        Arguments:
            rows: Dictionary of list's containing the new row's data for each
                  column.
        """
        pass

    def get_minibatch(self):
        """
        Returns a batch of experience tuples for training.

        Returns:
            Dictionary containing a numpy array for each column.
        """
        pass

    def __str__(self):
        """
        Returns the content of the Replay Memory as a string.
        """
        pass