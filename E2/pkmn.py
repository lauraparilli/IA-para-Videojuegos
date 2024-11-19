import heapq
from math import sqrt
import pygame
import time


## Player frames and animation

frame_width = 16
frame_height = 32
frames_per_direction = 3
cell_size = 30

def get_frames(row, spritesheet):
    frames = []
    for i in range(frames_per_direction):
        frame = spritesheet.subsurface((i * frame_width, row * frame_height, frame_width, frame_height))
        frames.append(frame)
    return frames


def get_animations(spritesheet):
    animations = {
    "down": get_frames(0, spritesheet),
    "left": get_frames(2, spritesheet),
    "right": [pygame.transform.flip(frame, True, False) for frame in get_frames(2, spritesheet)],
    "up": get_frames(1, spritesheet)
    }
    return animations

#
# NPC CLASS

class NPC(pygame.sprite.Sprite):
    def __init__(self, image_t, position,size=(1,1)):
        super().__init__()
        self.image = pygame.image.load('images/ditto.png').convert_alpha()
        self.image_close= pygame.image.load(image_t).convert_alpha()
        self.rect = self.image.get_rect(topleft=position)
        self.node_path = []
        self.move_delay = 1
        self.is_waiting = False
        self.position_updated = False
        self.position = position
        self.size = size
        self.speed = 1
        self.normal_speed = 1
        self.fast_speed = 2
        self.current_node = None
        self.target_set = False

    def set_path(self, path):
        self.node_path = path
    
    def update(self, player, graph, npcs_group):
        
        if time.time() - player.last_movement_time >= self.move_delay:
            self.is_waiting = False
        else:
            self.is_waiting = True

        if self.is_waiting:
            self.set_target(graph,(player.rect.x, player.rect.y))
            self.target_set = True
            
        decision_tree = DecisionTree(self)
        decision_tree.make_decision(player.rect.topleft, graph)
        
        # Speed update if the npc is in the blue floor
        node_x = self.rect.centerx // cell_size
        node_y = self.rect.centery // cell_size
        
        current_node = graph.grid[node_x][node_y]
        if current_node.f_type == 2:
            self.speed = self.fast_speed
        else:
            self.speed = self.normal_speed
        
        if self.node_path and not self.is_waiting:
            next_node = self.node_path[0]
            
            if next_node.f_type == 0:
                self.node_path = []
                return

            dx = next_node.x * cell_size - self.rect.x
            dy = next_node.y * cell_size - self.rect.y

            if abs(dx) > self.speed or abs(dy) > self.speed:
                self.rect.x += self.speed if dx > 0 else -self.speed
                self.rect.y += self.speed if dy > 0 else -self.speed
            else:
                self.rect.topleft = (next_node.x * cell_size, next_node.y * cell_size)
                self.current_node = next_node
                self.node_path.pop(0)
        self.enforce_window_bounds();
        
        if not self.node_path:
            for npc in npc_group:
                if npc != self and self.rect.colliderect(npc.rect):
                    self.avoid_collision(npc)
                
    def enforce_window_bounds(self):
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > width:
            self.rect.right = width
        if self.rect.top < 0:
            self.rect.top = 0
        if self.rect.bottom > height:
            self.rect.bottom = height
            
    def avoid_collision(self, other_npc):
        overlap_x = self.rect.centerx - other_npc.rect.centerx
        overlap_y = self.rect.centery - other_npc.rect.centery

        if abs(overlap_x) > abs(overlap_y):
            #horizontal collision
            if overlap_x > 0:
                self.rect.x += self.speed
            else:
                self.rect.x -= self.speed
        else:
        # vertical collision
            if overlap_y > 0:
                self.rect.y += self.speed
            else:
                self.rect.y -= self.speed
                
    def set_target(self, graph, target_position):
        start_node = graph.grid[self.rect.x // cell_size][self.rect.y // cell_size]
        goal_node = graph.grid[target_position[0] // cell_size][target_position[1] // cell_size]
        self.node_path = astar_pathfinding(graph, start_node, goal_node)
        self.current_node = start_node
    
    def is_player_near(self, player_position, threshold=20):
        distance = sqrt((self.position[0] - player_position[0]) ** 2 + (self.position[1] - player_position[1]) ** 2)
        return distance < threshold
    
    def change_image(self):
        self.image = self.image_close

#
# PLAYER CLASS

height_player, width_player = 30,60

class Player(pygame.sprite.Sprite):
    def __init__(self, spritesheet,x,y):
        super().__init__()
        self.direction = "down"
        self.animations = get_animations(spritesheet)
        self.original_image = self.animations[self.direction][0]
        self.image = pygame.transform.scale(self.original_image, (height_player, width_player))
        self.rect = self.image.get_rect(center=(x,y))

        self.pos_x = float(self.rect.x)
        self.pos_y = float(self.rect.y)
        self.frame_index = 0
        self.animation_speed = 0.02
        self.frame_time = 1
        self.is_moving = False
        self.speed = 1
        self.fast_speed = 2
        self.normal_speed = 1
        self.last_movement_time = time.time()
                
                
    def update(self):
        keys = pygame.key.get_pressed()
        self.is_moving = False
        
        # Speed update if the npc is in the blue floor
        cell_x = self.rect.centerx // cell_size
        cell_y = self.rect.centery // cell_size
        
        current_node = graph.grid[cell_x][cell_y]
        if current_node.f_type == 2:
            self.speed = self.fast_speed
        else:
            self.speed = self.normal_speed

            # Movimiento con verificación de paredes
        if keys[pygame.K_LEFT] and self.rect.left > 0:
            target_x = (self.rect.left - self.speed) // cell_size
            target_y = cell_y
            if graph.grid[target_y][target_x].f_type != 0:  # Si no es pared
                self.rect.x -= self.speed
                self.direction = "left"
                self.is_moving = True

        elif keys[pygame.K_RIGHT] and self.rect.right < width:
            target_x = (self.rect.right + self.speed - 1) // cell_size
            target_y = cell_y
            if graph.grid[target_y][target_x].f_type != 0:  # Si no es pared
                self.rect.x += self.speed
                self.direction = "right"
                self.is_moving = True

        if keys[pygame.K_UP] and self.rect.top > 0:
            target_x = cell_x
            target_y = (self.rect.top - self.speed) // cell_size
            if graph.grid[target_y][target_x].f_type != 0:  # Si no es pared
                self.rect.y -= self.speed
                self.direction = "up"
                self.is_moving = True

        elif keys[pygame.K_DOWN] and self.rect.bottom < height:
            target_x = cell_x
            target_y = (self.rect.bottom + self.speed - 1) // cell_size
            if graph.grid[target_y][target_x].f_type != 0:  # Si no es pared
                self.rect.y += self.speed
                self.direction = "down"
                self.is_moving = True

        # movement animation
        if self.is_moving:
            self.last_movement_time = time.time()
            self.frame_time += self.animation_speed
            if self.frame_time >= 1:
                self.frame_time = 0
                self.frame_index = (self.frame_index + 1) % frames_per_direction
            self.image = pygame.transform.scale(
                self.animations[self.direction][self.frame_index],
                (height_player, width_player),
            )  # Escalar la imagen
        else:
            self.frame_index = 0
            self.image = pygame.transform.scale(
                self.animations[self.direction][self.frame_index],
                (height_player, width_player),
            )

class Node:
    def __init__(self, x, y, f_type = 1):
        self.x = x
        self.y = y
        self.f_type = f_type
        self.connections = []

    def __lt__(self, other):
        return False

    def __str__(self):
        return f"Node(x={self.x}, y={self.y}, f_type={self.f_type})"

    def __repr__(self):
        return self.__str__()
    
class Connection:
    def __init__(self, from_node, to_node, cost):
        self.from_node = from_node
        self.to_node = to_node
        self.cost = cost

class Graph:
    
    def __init__(self, grid_width, grid_height):
        self.grid = [[Node(x, y) for y in range(grid_height)] for x in range(grid_width)]
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.create_connections()

    def create_connections(self):
        directions = [
            (-1, 0), (1, 0),  # izquierda, derecha
            (0, -1), (0, 1)  # abajo, arriba
        ]

        for x in range(self.grid_width):
            for y in range(self.grid_height):
                self.add_connections_for_node(self.grid[x][y], directions)

    def add_connections_for_node(self, node, directions):
        if node.f_type == 0:
            return
        for dx, dy in directions:
            nx, ny = node.x + dx, node.y + dy
            if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                neighbor = self.grid[nx][ny]
                if neighbor.f_type == 1:
                    cost = 1
                elif neighbor.f_type ==2:
                    cost = 0.5
                node.connections.append(Connection(node, neighbor, cost))

    def mark_as_wall(self, walls_positions,floor_type):
        if(floor_type ==0):
            for x,y in walls_positions:
                if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                    node = self.grid[x][y]
                    node.f_type = 0
                    node.connections = []
                    directions = [
                        (-1, 0), (1, 0), (0, -1), (0, 1)
                    ]
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                            neighbor = self.grid[nx][ny]
                            neighbor.connections = [
                            conn for conn in neighbor.connections if conn.to_node != node
                        ]
        elif (floor_type==2):
            for(x,y) in walls_positions:
                if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                    node = self.grid[x][y]
                    node.f_type = 2

def heuristic(node1, node2):
    dx = abs(node1.x - node2.x)
    dy = abs(node1.y - node2.y)
    return sqrt(dx**2 + dy**2)

def astar_pathfinding(graph, start_node, goal_node):
    open_list = []
    heapq.heappush(open_list, (0, start_node))
    came_from = {}
    cost_so_far = {start_node: 0}
    
    if goal_node.f_type == 0:
        return None

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == goal_node:
            path = []
            while current != start_node:
                path.append(current)
                current = came_from[current]
            path.append(start_node)
            path.reverse()
            return path

        for connection in current.connections:
            neighbor = connection.to_node
            if neighbor.f_type == 0:
                continue
            new_cost = cost_so_far[current] + connection.cost

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal_node)
                heapq.heappush(open_list, (priority, neighbor))
                came_from[neighbor] = current
    return None

class DecisionTree:
    def __init__(self, npc):
        self.npc = npc

    def make_decision(self, player_position, graph):
        
        if self.npc.is_player_near(player_position):
            self.npc.change_image()

        elif self.npc.rect.topleft == (10 * 30, 10 * 30):  # Verifica si está en (10,10)
            self.npc.change_image()

        # Si el NPC está en otra posición, se teletransporta
        elif self.npc.rect.topleft == (20 * 30, 20 * 30):  # Verifica si está en (20,20)
            self.npc.rect.x= 30
            self.npc.rect.y= 30

pygame.init()


# SCREEN
cell_size = 30
width, height = 920, 660
rows = height//16
cols = width//30
graph = Graph(rows, cols)
screen = pygame.display.set_mode((width, height))
background_color = (26, 135, 58)

pygame.display.set_caption("A* - Ditto, Ditto, Ditto")


# PLAYER
sprite_player = pygame.image.load('images/Male_Spritesheet.png')

# NPCS
ditto1 = NPC('images/Charmander.png', (100, 150))
ditto2 = NPC('images/Bulbasaur.png', (1100,90))
ditto3 = NPC('images/Pikachu.png', (800,389))

player1 = Player(sprite_player,34,34)

all_sprites = pygame.sprite.Group(player1, ditto1,ditto2,ditto3)
npc_group = pygame.sprite.Group(ditto1,ditto2,ditto3)

# FLOOR TYPES
wall_image = pygame.image.load('images/wall1.png').convert_alpha()
wall_image = pygame.transform.scale(wall_image, (cell_size, cell_size))

fast_floor = pygame.image.load('images/floor_1.png').convert_alpha()
fast_floor = pygame.transform.scale(fast_floor,(cell_size, cell_size))



graph.mark_as_wall([(3, 6),(4, 6),(4, 7),(5, 6),(6, 6),(6,7),(7, 7),(8,7),
                    (9,9),(10,10),(10,9),(10,11),
                    (13,10),(14,10),(15,10),
                    (15,20),(15,21),(15,22),(15,23),(16,20),(16,21),(16,22),(16,23),
                    (10,23),(11,23),(12,23),(13,23),(14,23), (15,23), (9,23),
                    (5,29),(6,29),(7,29),(5,30),(6,30),(7,30),(5,31),(6,31),(7,31),
                    (11,34),(12,34),(13,34),(14,34),(15,35),
                    (16,40),(17,40),(18,40),(19,40),(20,40),
                    (16,41),(17,41),(18,41),(19,41),(20,41)], 0)


graph.mark_as_wall( [(1,1),(1,2),(5,5),(10,6),(3,3),(10,9),(9,10),
            (11,10),(11,12),(11,13),(11,14),(11,11)],2)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    player1.update()

    # Update NPCs
    ditto1.update(player1, graph,npc_group)
    ditto2.update(player1, graph, npc_group)
    ditto3.update(player1, graph, npc_group)
        
    screen.fill((26, 135, 58))


    COLOR_NORMAL = (50, 200, 50)
    COLOR_BORDER = (148, 148, 148)

    for y in range(graph.grid_height):
        for x in range(graph.grid_width):
            node = graph.grid[x][y]

            if node.f_type == 0:  # not walkable node (wall)
                screen.blit(wall_image, (x * cell_size, y * cell_size))
            # walkables nodes
            elif node.f_type ==2: # speedy node
                screen.blit(fast_floor,(x*cell_size, y*cell_size))
            else:  # normal node
                color = COLOR_NORMAL
                rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
                pygame.draw.rect(screen, color, rect)

            pygame.draw.rect(screen, COLOR_BORDER, rect, 1)
    all_sprites.draw(screen)
    
    for node in graph.grid[x][y].connections:
        pygame.draw.line(screen, (0, 0, 0),
                     (node.from_node.x * cell_size + cell_size // 2,
                      node.from_node.y * cell_size + cell_size // 2),
                     (node.to_node.x * cell_size + cell_size // 2,
                      node.to_node.y * cell_size + cell_size // 2), 1)
    
    pygame.display.flip()

pygame.quit()