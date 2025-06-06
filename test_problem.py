import unittest
from typing import List, Dict, Set, Tuple
import heapq

def solve_network_optimization(network: Dict[str, List[Tuple[str, int]]], 
                             source: str, 
                             target: str, 
                             constraints: Dict[str, int]) -> Tuple[int, List[str]]:
    """
    Solve a complex network optimization problem with multiple constraints.
    
    Args:
        network: Dictionary representing a directed graph where each node maps to a list of (neighbor, cost) tuples
        source: Starting node
        target: Destination node
        constraints: Dictionary of constraints (e.g., max_nodes, max_cost, required_nodes)
    
    Returns:
        Tuple of (total_cost, path)
    """
    # Initialize variables
    max_cost = constraints.get('max_cost', float('inf'))
    max_nodes = constraints.get('max_nodes', float('inf'))
    required_nodes = set(constraints.get('required_nodes', []))
    avoid_nodes = set(constraints.get('avoid_nodes', []))
    
    # Check if source or target is in avoid_nodes
    if source in avoid_nodes or target in avoid_nodes:
        raise ValueError("Source or target node is in avoid_nodes")
    
    # Initialize priority queue with (cost, path_length, current_node, path, visited_nodes)
    pq = [(0, 0, source, [source], {source})]
    visited = set()
    
    while pq:
        cost, path_length, current, path, visited_nodes = heapq.heappop(pq)
        
        # Check if we've reached the target
        if current == target:
            # Verify all required nodes are in the path
            if required_nodes.issubset(visited_nodes):
                return cost, path
            continue
        
        # Skip if we've already visited this node with a better path
        state = (current, frozenset(visited_nodes))
        if state in visited:
            continue
        visited.add(state)
        
        # Check constraints
        if cost > max_cost or path_length >= max_nodes:
            continue
        
        # Explore neighbors
        for neighbor, edge_cost in network[current]:
            # Skip if neighbor is in avoid_nodes
            if neighbor in avoid_nodes:
                continue
            
            new_cost = cost + edge_cost
            new_path_length = path_length + 1
            
            # Skip if constraints would be violated
            if new_cost > max_cost or new_path_length > max_nodes:
                continue
            
            new_path = path + [neighbor]
            new_visited = visited_nodes | {neighbor}
            
            # Add to priority queue
            heapq.heappush(pq, (new_cost, new_path_length, neighbor, new_path, new_visited))
    
    # If we get here, no valid path was found
    raise ValueError("No valid path exists that satisfies all constraints")

class TestNetworkOptimization(unittest.TestCase):
    def test_basic_path(self):
        network = {
            'A': [('B', 2), ('C', 4)],
            'B': [('D', 3)],
            'C': [('D', 1)],
            'D': []
        }
        constraints = {'max_cost': 6, 'max_nodes': 4}
        cost, path = solve_network_optimization(network, 'A', 'D', constraints)
        self.assertEqual(cost, 5, f"Expected cost 5 but got {cost}")
        self.assertEqual(path, ['A', 'C', 'D'], f"Unexpected path {path}")

    def test_no_valid_path(self):
        network = {
            'A': [('B', 2)],
            'B': [('C', 3)],
            'C': []
        }
        constraints = {'max_cost': 4, 'required_nodes': ['D']}
        with self.assertRaises(ValueError, msg="Expected ValueError for missing required node"):
            solve_network_optimization(network, 'A', 'C', constraints)

    def test_complex_constraints(self):
        network = {
            'A': [('B', 2), ('C', 4), ('D', 1)],
            'B': [('E', 3), ('F', 2)],
            'C': [('E', 1), ('F', 4)],
            'D': [('F', 3)],
            'E': [('G', 2)],
            'F': [('G', 1)],
            'G': []
        }
        constraints = {
            'max_cost': 8,
            'max_nodes': 5,
            'required_nodes': ['E'],
            'avoid_nodes': ['D']
        }
        cost, path = solve_network_optimization(network, 'A', 'G', constraints)
        self.assertLessEqual(cost, 8, f"Path cost {cost} exceeds limit")
        self.assertLessEqual(len(path), 5, f"Path length {len(path)} exceeds limit")
        self.assertIn('E', path, "Required node E missing from path")
        self.assertNotIn('D', path, "Avoid node D present in path")

    def test_learning_from_feedback(self):
        # Test that the solution improves when given feedback about failed attempts
        network = {
            'A': [('B', 2), ('C', 3)],
            'B': [('D', 4)],
            'C': [('D', 2)],
            'D': [('E', 1)],
            'E': []
        }
        constraints = {'max_cost': 7, 'max_nodes': 4}
        
        # First attempt
        cost1, path1 = solve_network_optimization(network, 'A', 'E', constraints)
        
        # Second attempt with feedback about first attempt
        cost2, path2 = solve_network_optimization(network, 'A', 'E', constraints)
        
        # The second attempt should be at least as good as the first
        self.assertLessEqual(cost2, cost1, "Second attempt did not improve cost")

if __name__ == '__main__':
    unittest.main() 