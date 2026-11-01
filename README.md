# Path-planning-visualizer-
A Python Pygame-based pathfinding visualizer that compares A* Search and Dijkstra's Algorithm side by side, featuring interactive obstacles, shortest-path visualization, and a performance dashboard.
[README.md](https://github.com/user-attachments/files/34237704/README.md)
# Path Planning Visualizer: A* vs Dijkstra

An interactive Pygame project that runs A* and Dijkstra on the same grid and compares their performance.

## Requirements
- Python 3.10+ //only use py 3.13v
- Pygame 2.6+

## Install
```bash
py -m pip install -r requirements.txt
```

## Run
```bash
py main.py
```

## Controls
- Left click: place start, then goal, then obstacles
- Left click and drag: draw obstacles
- Right click: erase a wall, start, or goal
- `SPACE`: run A* and Dijkstra side by side
- `R`: clear search results but keep the map
- `C`: clear the entire grid
- `ESC`: quit

## Comparison metrics
- Number of explored nodes
- Shortest path length
- Algorithm-only search time (animation time excluded)
- Node-exploration bar chart

Both algorithms use the same four-directional movement rules, grid, obstacles, start, and goal. A* uses Manhattan distance, an admissible heuristic for this movement model.
