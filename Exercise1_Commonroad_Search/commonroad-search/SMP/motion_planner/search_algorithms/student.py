import numpy as np

from SMP.motion_planner.node import PriorityNode

from SMP.motion_planner.plot_config import DefaultPlotConfig
from SMP.motion_planner.search_algorithms.best_first_search import AStarSearch


class StudentMotionPlanner(AStarSearch):
    """
    Motion planner implementation by students.
    Note that you may inherit from any given motion planner as you wish, or come up with your own planner.
    Here as an example, the planner is inherited from the GreedyBestFirstSearch planner.
    """

    def __init__(self, scenario, planningProblem, automata, plot_config=DefaultPlotConfig):
        super().__init__(scenario=scenario, planningProblem=planningProblem, automaton=automata,
                         plot_config=plot_config)

    def heuristic_function(self, node_current: PriorityNode) -> float:
        ########################################################################
        # todo: Implement your own heuristic cost calculation here.            #
        # Hint:                                                                #
        #   Use the State of the current node and the information from the     #
        #   planning problem, as well as from the scenario.                    #
        #   Some helper functions for your convenience can be found in         #
        #   ./search_algorithms/base_class.py                             #
        ########################################################################

        path_last = node_current.list_paths[-1]

        distStartState = self.calc_heuristic_distance(path_last[0])
        distLastState = self.calc_heuristic_distance(path_last[-1])

        if self.reached_goal(path_last):
            return 0.0

        if self.position_desired is None:
            return self.time_desired.start - node_current.list_paths[-1][-1].time_step

        else:
            factor = 1

            if distLastState < 0.5:
                factor = factor * 0.00001

            velocity = node_current.list_paths[-1][-1].velocity
            cost_time = self.calc_time_cost(path_last)

            if np.isclose(velocity, 0):
                return np.inf

            else:
                cost = 0.7 * (self.calc_euclidean_distance(
                    current_node=node_current) / velocity) + 0.1 * cost_time + 0.5 * distLastState
                return cost * factor
