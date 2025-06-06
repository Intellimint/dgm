def test_graph_traversal():
    """Test the implementation of graph traversal with path finding."""
    from graph_traversal import find_shortest_path
    
    # Test cases
    test_cases = [
        # Simple linear path
        ({
            'A': ['B'],
            'B': ['C'],
            'C': ['D'],
            'D': []
        }, 'A', 'D', ['A', 'B', 'C', 'D']),
        
        # Multiple paths
        ({
            'A': ['B', 'C'],
            'B': ['D'],
            'C': ['D'],
            'D': []
        }, 'A', 'D', ['A', 'B', 'D']),
        
        # No path exists
        ({
            'A': ['B'],
            'B': ['C'],
            'C': [],
            'D': []
        }, 'A', 'D', None),
        
        # Cycle in graph
        ({
            'A': ['B'],
            'B': ['C'],
            'C': ['A', 'D'],
            'D': []
        }, 'A', 'D', ['A', 'B', 'C', 'D']),
        
        # Complex graph
        ({
            'A': ['B', 'C', 'D'],
            'B': ['E', 'F'],
            'C': ['F', 'G'],
            'D': ['G', 'H'],
            'E': ['I'],
            'F': ['I', 'J'],
            'G': ['J', 'K'],
            'H': ['K'],
            'I': ['L'],
            'J': ['L'],
            'K': ['L'],
            'L': []
        }, 'A', 'L', ['A', 'B', 'E', 'I', 'L'])
    ]
    
    for graph, start, end, expected in test_cases:
        result = find_shortest_path(graph, start, end)
        assert result == expected, f"Failed for graph={graph}, start='{start}', end='{end}'. Expected {expected}, got {result}" 