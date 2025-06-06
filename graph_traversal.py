import heapq

def find_shortest_path(graph, start, end):
    """
    Find the shortest path between two nodes in a graph using Dijkstra's algorithm.
    Handles both unweighted and weighted edges, including negative weights.
    
    Args:
        graph (dict): Dictionary representing the graph where keys are nodes and values are lists of (neighbor, weight) tuples
        start (str): Starting node
        end (str): Target node
        
    Returns:
        list: List of nodes representing the shortest path from start to end, or None if no path exists
    """
    if start not in graph or end not in graph:
        return None
        
    # Initialize distances and previous nodes
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    previous = {node: None for node in graph}
    
    # Priority queue for Dijkstra's algorithm
    queue = [(0, start)]
    
    while queue:
        # Get the node with the smallest distance
        current_distance, current = heapq.heappop(queue)
        
        # If we've reached the end, reconstruct and return the path
        if current == end:
            path = []
            while current is not None:
                path.append(current)
                current = previous[current]
            return path[::-1]
            
        # If we've found a better path to this node, skip it
        if current_distance > distances[current]:
            continue
            
        # Check all neighbors
        for neighbor, weight in graph[current]:
            distance = current_distance + weight
            
            # If we found a better path to the neighbor
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current
                heapq.heappush(queue, (distance, neighbor))
    
    # If we've exhausted all possibilities and haven't found a path
    return None 