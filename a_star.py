import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


class Node:
    """
        A node class for A* Pathfinding
        parent is parent of the current Node
        position is current position of the Node in the maze
        g is cost from start to current Node
        h is heuristic based estimated cost for current Node to end Node
        f is total cost of present node i.e. :  f = g + h
    """

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

# This function return the path of the search


def return_path(current_node, maze):
    path = []
    no_rows, no_columns = np.shape(maze)
    # here we create the initialized result maze with -1 in every position
    result = [[-1 for i in range(no_columns)] for j in range(no_rows)]
    current = current_node
    while current is not None:
        path.append(current.position)
        current = current.parent
    # Return reversed path as we need to show from start to end path
    path = path[::-1]
    start_value = 0
    # we update the path of start to end found by A-star serch with every step incremented by 1
    for i in range(len(path)):
        result[path[i][0]][path[i][1]] = start_value
        start_value += 1

    return path


# Heuristic functions h(n): an estimate of the remaining cost-to-go from node n to the goal.

EUCLIDEAN = "euclidean"
MANHATTAN = "manhattan"


def heuristic(position, goal, method=EUCLIDEAN):
    dx = abs(position[0] - goal[0])
    dy = abs(position[1] - goal[1])

    if method == MANHATTAN:
        # h_M(n) = dx + dy
        return dx + dy

    # Default: h_E(n) = sqrt(dx^2 + dy^2), the straight-line distance.
    # It never over-estimates the real cost, so the A* path stays optimal.
    return sqrt(dx ** 2 + dy ** 2)


def search(maze, start, end, heuristic_method=EUCLIDEAN):

    print("searching ....")

    maze = maze.T

    """
        Returns a list of tuples as a path from the given start to the given end in the given maze
        :param maze: 2D cost map (values > 0.8 are treated as walls)
        :param start: start cell index (row, col)
        :param end: goal cell index (row, col)
        :param heuristic_method: EUCLIDEAN or MANHATTAN
        :return: list of cell positions from start to end
    """

    # Create the start and end nodes.
    start = tuple(start)
    end = tuple(end)


    start_node = Node(None, start)
    start_node.g = 0                                        # cost from start Node
    start_node.h = heuristic(start, end, heuristic_method)  # heuristic estimated cost to end Node
    start_node.f = start_node.g + start_node.h

    
    end_node = Node(None, end)
    end_node.g = float('inf')  # set a large value if not defined
    end_node.h = 0             # heuristic estimated cost to end Node
    end_node.f = end_node.g + end_node.h

    # Initialize both yet_to_visit and visited dictionary
    # in this dict we will put all node that are yet_to_visit for exploration.
    # From here we will find the lowest cost node to expand next
    yet_to_visit_dict = {}  # key is the position (tuple), value is the node
    # in this list we will put all node those already explored so that we don't explore it again
    # key is the position (tuple), value is True (boolean)
    visited_dict = {}

    # Add the start node
    yet_to_visit_dict[start_node.position] = start_node

    # Adding a stop condition. This is to avoid any infinite loop and stop
    # execution after some reasonable number of steps
    outer_iterations = 0
    max_iterations = (len(maze) // 2) ** 10

   
    # Neighbour offsets [d_row, d_col] -> 8-point connectivity.
    # The first 4 entries alone give 4-point connectivity; the last 4 add the
    # diagonal moves.
    
    move = [[-1, 0],   # go up
            [0, -1],   # go left
            [1, 0],    # go down
            [0, 1],    # go right
            [-1, -1],  # go up left
            [1, -1],   # go down left
            [-1, 1],   # go up right
            [1, 1]]    # go down right

    """
        1) We first get the current node by comparing all f cost and selecting the lowest cost node for further expansion
        2) Check max iteration reached or not . Set a message and stop execution
        3) Remove the selected node from yet_to_visit dict and add this node to visited dict
        4) Perofmr Goal test and return the path else perform below steps
        5) For selected node find out all children (use move to find children)
            a) get the current postion for the selected node (this becomes parent node for the children)
            b) check if a valid position exist (boundary will make few nodes invalid)
            c) if any node is a wall then ignore that
            d) add to valid children node list for the selected parent
            
            For all the children node
                a) if child in visited dict then ignore it and try next node
                b) calculate child node g, h and f values
                c) if child in yet_to_visit dict then ignore it
                d) else move the child to yet_to_visit dict
    """
    # Size of the (transposed) map, used for the boundary check below
    no_rows, no_columns = np.shape(maze)

    # Loop until you find the end

    while len(yet_to_visit_dict) > 0:

        # Every time any node is referred from yet_to_visit list, counter of limit operation incremented
        outer_iterations += 1

        # Take the "best" node in the open set O (yet_to_visit_dict): the one
        # with the lowest f = g + h
        current_node = None
        current_fscore = None
        for position, node in yet_to_visit_dict.items():
            if current_fscore is None or node.f < current_fscore:
                current_fscore = node.f
                current_node = node

        # if we hit this point return the path such as it may be no solution or
        # computation cost is too high
        if outer_iterations > max_iterations:
            print("giving up on pathfinding too many iterations")
            return return_path(current_node, maze)

        # Move the best node from the open set O to the closed set C
        # (pop current node out of yet_to_visit dict, add to visited dict)
        yet_to_visit_dict.pop(current_node.position)
        visited_dict[current_node.position] = True

        # test if goal is reached or not, if yes then return the path
        if current_node == end_node:

            return return_path(current_node, maze)

        # Generate children from all adjacent squares
        children = []

        for new_position in move:

            # Neighbour position = current position + move offset
            node_position = (current_node.position[0] + new_position[0],
                             current_node.position[1] + new_position[1])

            # Skip neighbours that fall outside the map boundary
            if (node_position[0] < 0 or node_position[0] >= no_rows or
                    node_position[1] < 0 or node_position[1] >= no_columns):
                continue

            # Make sure walkable terrain
            if maze[node_position[0], node_position[1]] > 0.8:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children

        for child in children:

            # Already in the closed set C -> its best cost is final, skip it
            if visited_dict.get(child.position, False):
                continue

            # g(n): actual cost so far = parent's cost + length of this edge.
            # The edge length is the Euclidean step size: 1 for up/down/left/
            # right and sqrt(2) for a diagonal step (the "edge weight").
            step_cost = sqrt((child.position[0] - current_node.position[0]) ** 2 +
                             (child.position[1] - current_node.position[1]) ** 2)
            child.g = current_node.g + step_cost

            # h(n): heuristic estimate of the cost-to-go to the goal
            child.h = heuristic(child.position, end_node.position, heuristic_method)

            # f(n) = g(n) + h(n): estimated total cost through this child,
            # used to pick the next best node from the open set
            child.f = child.g + child.h

            # If the child is already in O with an equal or cheaper g, keep the
            # old entry which is the shortest path to it.
            # Otherwise the new entry replaces it, updating cost + back pointer.
            child_node_in_yet_to_visit = yet_to_visit_dict.get(
                child.position, False)
            if (child_node_in_yet_to_visit is not False) and (child.g >= child_node_in_yet_to_visit.g):
                continue

            # Add the child to the yet_to_visit list
            yet_to_visit_dict[child.position] = child
