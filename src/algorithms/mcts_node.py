from anytree import NodeMixin


class MctsNode(NodeMixin):
    """
    This class represents a node in the Monte-Carlo search tree.
    """

    def __init__(self, name, state, last_pos=None, move_box=True, done=False,
                 action=None, parent=None, children=None):
        super(MctsNode, self).__init__()
        self.name = name
        self.state = state
        self.done = done
        self.Q = 0
        self.N = 0
        self.last_pos = last_pos
        self.move_box = move_box
        self.parent = parent
        self.action = action
        if children:
            self.children = children