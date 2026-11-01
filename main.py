import heapq
import itertools
import math
import time
import pygame

# -------------------- SETTINGS --------------------
ROWS = COLS = 20
CELL = 22
GRID_W = COLS * CELL
GRID_H = ROWS * CELL
GAP = 24
MARGIN = 24
PANEL_H = 210
WIDTH = MARGIN * 2 + GRID_W * 2 + GAP
HEIGHT = MARGIN * 2 + GRID_H + PANEL_H

FPS = 60
ANIMATION_EVENTS_PER_FRAME = 5

# Colors
BG = (18, 22, 30)
PANEL = (29, 35, 46)
GRID_BG = (245, 247, 250)
GRID_LINE = (210, 216, 225)
WALL = (48, 55, 68)
START_COLOR = (40, 190, 105)
GOAL_COLOR = (225, 75, 90)
VISITED = (105, 175, 235)
FRONTIER = (248, 195, 80)
PATH = (174, 115, 235)
WHITE = (245, 247, 250)
MUTED = (175, 185, 200)
ASTAR_COLOR = (90, 190, 150)
DIJKSTRA_COLOR = (100, 165, 240)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Path Planning Visualizer | A* vs Dijkstra")
font = pygame.font.SysFont("consolas", 18, bold=True)
small = pygame.font.SysFont("consolas", 14)
tiny = pygame.font.SysFont("consolas", 12)
clock = pygame.time.Clock()

LEFT_X = MARGIN
RIGHT_X = MARGIN + GRID_W + GAP
GRID_Y = MARGIN + 34
PANEL_Y = GRID_Y + GRID_H + 18

walls = set()
start = None
goal = None
results = {}
animations = {}
animation_indices = {"Dijkstra": 0, "A*": 0}
finished = {"Dijkstra": False, "A*": False}
message = "Left click: set start, goal, walls | Right click: erase"
running_search = False


# -------------------- PATHFINDING --------------------
def neighbors(node):
    r, c = node
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nxt = (r + dr, c + dc)
        if 0 <= nxt[0] < ROWS and 0 <= nxt[1] < COLS and nxt not in walls:
            yield nxt


def heuristic(node, target):
    # Manhattan distance is admissible for four-directional movement.
    return abs(node[0] - target[0]) + abs(node[1] - target[1])


def search(use_astar):
    """Run one search and return animation events plus measured search statistics."""
    name = "A*" if use_astar else "Dijkstra"
    counter = itertools.count()
    initial_priority = heuristic(start, goal) if use_astar else 0
    heap = [(initial_priority, next(counter), 0, start)]
    distances = {start: 0}
    parents = {}
    settled = set()
    events = [("frontier", start)]
    found = False

    began = time.perf_counter()
    while heap:
        _, _, cost, current = heapq.heappop(heap)
        if current in settled or cost != distances[current]:
            continue
        settled.add(current)
        events.append(("visit", current))

        if current == goal:
            found = True
            break

        for nxt in neighbors(current):
            new_cost = cost + 1
            if new_cost < distances.get(nxt, math.inf):
                distances[nxt] = new_cost
                parents[nxt] = current
                priority = new_cost + (heuristic(nxt, goal) if use_astar else 0)
                heapq.heappush(heap, (priority, next(counter), new_cost, nxt))
                events.append(("frontier", nxt))

    elapsed_ms = (time.perf_counter() - began) * 1000

    route = []
    if found:
        node = goal
        while node != start:
            route.append(node)
            node = parents[node]
        route.append(start)
        route.reverse()
        for node in route:
            events.append(("path", node))

    return events, {
        "name": name,
        "explored": len(settled),
        "path_length": len(route) - 1 if found else None,
        "time_ms": elapsed_ms,
        "found": found,
    }


def clear_results():
    global results, animations, animation_indices, finished, running_search
    results = {}
    animations = {}
    animation_indices = {"Dijkstra": 0, "A*": 0}
    finished = {"Dijkstra": False, "A*": False}
    running_search = False


def clear_everything():
    global start, goal, message
    walls.clear()
    start = goal = None
    clear_results()
    message = "Grid cleared. Place a start and goal."


def run_comparison():
    global results, animations, running_search, message
    if start is None or goal is None:
        message = "Place both start and goal before comparing."
        return
    clear_results()
    # Both algorithms receive the exact same start, goal, walls and movement rules.
    animations["Dijkstra"], results["Dijkstra"] = search(False)
    animations["A*"], results["A*"] = search(True)
    running_search = True
    message = "Both searches are running on the same grid."


# -------------------- DRAWING --------------------
def draw_text(text, x, y, color=WHITE, selected_font=small):
    surface = selected_font.render(str(text), True, color)
    screen.blit(surface, (x, y))


def grid_origin(name):
    return (LEFT_X if name == "Dijkstra" else RIGHT_X, GRID_Y)


def draw_grid(name):
    ox, oy = grid_origin(name)
    state = {"visited": set(), "frontier": set(), "path": set()}
    if name in animations:
        # Reconstruct the visible state from the events already animated.
        upto = animation_indices[name]
        for kind, cell in animations[name][:upto]:
            if kind == "frontier":
                state["frontier"].add(cell)
            elif kind == "visit":
                state["frontier"].discard(cell)
                state["visited"].add(cell)
            elif kind == "path":
                state["path"].add(cell)

    for r in range(ROWS):
        for c in range(COLS):
            cell = (r, c)
            color = GRID_BG
            if cell in walls:
                color = WALL
            elif cell in state["path"]:
                color = PATH
            elif cell in state["visited"]:
                color = VISITED
            elif cell in state["frontier"]:
                color = FRONTIER
            if cell == start:
                color = START_COLOR
            elif cell == goal:
                color = GOAL_COLOR
            rect = pygame.Rect(ox + c * CELL, oy + r * CELL, CELL, CELL)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, GRID_LINE, rect, 1)


def draw_metric_bar(label, value, maximum, x, y, width, color):
    draw_text(label, x, y, MUTED, tiny)
    bar_y = y + 17
    pygame.draw.rect(screen, (55, 63, 77), (x, bar_y, width, 10), border_radius=5)
    ratio = 0 if maximum <= 0 else min(1, value / maximum)
    if ratio > 0:
        pygame.draw.rect(screen, color, (x, bar_y, max(4, int(width * ratio)), 10), border_radius=5)


def draw_dashboard():
    pygame.draw.rect(screen, PANEL, (MARGIN, PANEL_Y, WIDTH - 2 * MARGIN, PANEL_H), border_radius=12)
    draw_text("PERFORMANCE DASHBOARD", MARGIN + 16, PANEL_Y + 12, WHITE, font)
    draw_text("Same map • same start/goal • four-directional movement", MARGIN + 16, PANEL_Y + 39, MUTED, tiny)

    if not results:
        draw_text(message, MARGIN + 16, PANEL_Y + 78, WHITE, small)
        draw_text("Controls: SPACE = compare | C = clear all | R = clear search | ESC = quit", MARGIN + 16, PANEL_Y + 112, MUTED, tiny)
        draw_text("Mouse: left click places start → goal → walls; right click erases", MARGIN + 16, PANEL_Y + 136, MUTED, tiny)
        return

    d = results["Dijkstra"]
    a = results["A*"]
    col1 = MARGIN + 16
    col2 = MARGIN + (WIDTH - 2 * MARGIN) // 2 + 8
    row_y = PANEL_Y + 70

    for x, stat, color in ((col1, d, DIJKSTRA_COLOR), (col2, a, ASTAR_COLOR)):
        draw_text(stat["name"], x, row_y, color, font)
        path_value = stat["path_length"] if stat["found"] else "No path"
        draw_text(f"Explored nodes: {stat['explored']}", x, row_y + 27)
        draw_text(f"Path length: {path_value}", x, row_y + 49)
        draw_text(f"Search time: {stat['time_ms']:.4f} ms", x, row_y + 71)

    max_explored = max(d["explored"], a["explored"], 1)
    bar_y = PANEL_Y + 164
    bar_w = (WIDTH - 2 * MARGIN - 48) // 2 - 12
    draw_metric_bar("DIJKSTRA — NODES EXPLORED", d["explored"], max_explored, col1, bar_y, bar_w, DIJKSTRA_COLOR)
    draw_metric_bar("A* — NODES EXPLORED", a["explored"], max_explored, col2, bar_y, bar_w, ASTAR_COLOR)

    if finished["Dijkstra"] and finished["A*"]:
        if d["path_length"] == a["path_length"]:
            comparison = "Both found paths of equal length."
        else:
            comparison = "Check movement rules: path lengths differ."
        if a["explored"] < d["explored"]:
            comparison += f" A* explored {d['explored'] - a['explored']} fewer nodes."
        elif d["explored"] < a["explored"]:
            comparison += f" Dijkstra explored {a['explored'] - d['explored']} fewer nodes."
        else:
            comparison += " Both explored the same number of nodes."
        draw_text(comparison, col1, PANEL_Y + 190, WHITE, tiny)


def draw():
    screen.fill(BG)
    draw_text("DIJKSTRA", LEFT_X, MARGIN + 5, DIJKSTRA_COLOR, font)
    draw_text("A*", RIGHT_X, MARGIN + 5, ASTAR_COLOR, font)
    draw_grid("Dijkstra")
    draw_grid("A*")
    draw_dashboard()
    pygame.display.flip()


# -------------------- INPUT / MAIN LOOP --------------------
def handle_mouse(pos, button):
    global start, goal, message
    if running_search:
        return
    x, y = pos
    if y < GRID_Y or y >= GRID_Y + GRID_H:
        return

    # Convert either displayed grid's coordinates into the same logical cell.
    if LEFT_X <= x < LEFT_X + GRID_W:
        local_x = x - LEFT_X
    elif RIGHT_X <= x < RIGHT_X + GRID_W:
        local_x = x - RIGHT_X
    else:
        return

    cell = ((y - GRID_Y) // CELL, local_x // CELL)
    clear_results()
    if button == 3:
        if cell == start:
            start = None
        elif cell == goal:
            goal = None
        else:
            walls.discard(cell)
    elif button == 1:
        if start is None and cell != goal:
            start = cell
        elif goal is None and cell != start:
            goal = cell
        elif cell != start and cell != goal:
            walls.add(cell)
    message = "Press SPACE to compare A* and Dijkstra."


def animate():
    global running_search, message
    for name in ("Dijkstra", "A*"):
        if finished[name]:
            continue
        events = animations[name]
        end = min(animation_indices[name] + ANIMATION_EVENTS_PER_FRAME, len(events))
        animation_indices[name] = end
        if end >= len(events):
            finished[name] = True
    if all(finished.values()):
        running_search = False
        message = "Comparison complete. Review the dashboard below."


running = True
while running:
    clock.tick(FPS)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_c:
                clear_everything()
            elif event.key == pygame.K_r:
                clear_results()
                message = "Search results cleared. Map kept."
            elif event.key == pygame.K_SPACE:
                run_comparison()
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button in (1, 3):
            handle_mouse(event.pos, event.button)

    # Click-and-drag wall drawing works on either grid and edits the shared map.
    if pygame.mouse.get_pressed()[0] and not running_search and start is not None and goal is not None:
        mx, my = pygame.mouse.get_pos()
        if GRID_Y <= my < GRID_Y + GRID_H:
            if LEFT_X <= mx < LEFT_X + GRID_W:
                local_x = mx - LEFT_X
            elif RIGHT_X <= mx < RIGHT_X + GRID_W:
                local_x = mx - RIGHT_X
            else:
                local_x = None
            if local_x is not None:
                cell = ((my - GRID_Y) // CELL, local_x // CELL)
                if cell != start and cell != goal and cell not in walls:
                    clear_results()
                    walls.add(cell)
                    message = "Obstacle added. Press SPACE to compare."

    if running_search:
        animate()
    draw()

pygame.quit()
