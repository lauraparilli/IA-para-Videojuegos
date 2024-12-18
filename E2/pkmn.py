import heapq
from math import sqrt
import pygame
import time
import random

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
pokemon_image = {
    "ditto" : 'images/ditto.png',
    "bulbasaur" : 'images/Bulbasaur.png',
    "charmander" : 'images/Charmander.png'
}

evolution_image = {
    "charmaleon" : 'images/Charmaleon.png',
    "ivysaur": 'images/Ivysaur.png'
}

pokeball_positions = [
            (22 * cell_size, 13 * cell_size),
            (10 * cell_size, 8 * cell_size),
            (15 * cell_size, 5 * cell_size),
            (3 * cell_size, 2 * cell_size),
            (18 * cell_size, 8 * cell_size)
        ]

lab_width = 5
lab_height = 3

lab_start_x, lab_start_y = 20, 2

lab_cells = [
    (x, y)
    for x in range(lab_start_x, lab_start_x + lab_width)
    for y in range(lab_start_y, lab_start_y + lab_height)
]

class NPC(pygame.sprite.Sprite):
    def __init__(self, position,size=(1,1)):
        super().__init__()
        self.pokemon = "ditto"
        self.image = pygame.image.load('images/ditto.png').convert_alpha()
        self.rect = self.image.get_rect(topleft=position)
        self.node_path = []
        self.non_tactical_node_path_ = []
        self.move_delay = 1
        self.is_waiting = False
        self.position_updated = False
        self.position = position
        self.size = size
        self.is_evolutioned = False
        self.evolved_in_lab = False
        self.speed = 1
        self.normal_speed = 1
        self.fast_speed = 2
        self.current_node = None
        self.target_set = False


    def set_path(self, path, non_tactical_path):
        self.node_path = path
        self.non_tactical_node_path = non_tactical_path

    def update(self, player, graph, npcs_group):

        if time.time() - player.last_movement_time >= self.move_delay:
            self.is_waiting = False
        else:
            self.is_waiting = True

        if self.is_waiting:
            self.set_target(graph,(player.rect.x, player.rect.y))
            self.target_set = True

        
        # Speed update if the npc is in the blue floor
        node_x = self.rect.centerx // cell_size
        node_y = self.rect.centery // cell_size
        
        current_node = graph.grid[node_x][node_y]
        if current_node.f_type == 2:
            self.speed = self.fast_speed
        else:
            self.speed = self.normal_speed

        if self.is_npc_in_lab(graph):
            if not self.evolved_in_lab:
                self.pokemon_evolution()
                self.evolved_in_lab = True
        else:
            self.evolved_in_lab = False
            
        if self.is_on_pokeball():
            random_pokemon = random.choice([poke for poke in pokemon_image.keys() if poke != self.pokemon])
            self.pokemon = random_pokemon
            self.change_image()
            
        self.got_player(graph,player)
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
            for npc in npcs_group:
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
            

    def is_on_pokeball(self):

        npc_position = (self.rect.x, self.rect.y)
        return npc_position in pokeball_positions
        
    def avoid_collision(self, other_npc):
        overlap_x = self.rect.centerx - other_npc.rect.centerx
        overlap_y = self.rect.centery - other_npc.rect.centery

        if abs(overlap_x) > abs(overlap_y):
            # horizontal collision
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
        self.node_path = astar_pathfinding(start_node, goal_node, self.pokemon)
        self.non_tactical_node_path = astar_pathfinding(start_node,goal_node,"ditto")

        
        self.current_node = start_node

    def got_player(self,graph, player):
        if graph.grid[player.rect.x // cell_size][player.rect.y // cell_size] == self.current_node:
            if not hasattr(self, "player_hit"):
                self.player_hit = False
            
            if not self.player_hit:
                player.lives -= 1
                self.position = (80, 80)
                self.rect.topleft = self.position
                self.player_hit = True
        else:
            self.player_hit = False
            
    def is_npc_in_lab(self,graph):
        npc_cell = self.current_node
        for  pos in lab_cells:
            if npc_cell == graph.grid[pos[0]][pos[1]] == npc_cell:
                return True
        return False
        
            
    def pokemon_evolution(self):
        if not self.is_evolutioned:
            if self.pokemon == "bulbasaur":
                self.pokemon = "ivysaur"
                self.image = pygame.image.load(evolution_image.get(self.pokemon)).convert_alpha()
            elif self.pokemon == "charmander":
                self.pokemon = "charmaleon"
                self.image = pygame.image.load(evolution_image.get(self.pokemon)).convert_alpha()
            else:
                return
            self.is_evolutioned = True
            
        elif self.is_evolutioned:
            if self.pokemon == "ivysaur":
                self.pokemon = "bulbasaur"
                self.image = pygame.image.load(pokemon_image.get(self.pokemon)).convert_alpha()
            elif self.pokemon == "charmaleon":
                self.pokemon = "charmander"
                self.image = pygame.image.load(pokemon_image.get(self.pokemon)).convert_alpha()
            else:
                return
            self.is_evolutioned = False

    def change_image(self):
        self.image = pygame.image.load(pokemon_image.get(self.pokemon)).convert_alpha()

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
        self.lives = 5


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
            if graph.grid[target_y][target_x].f_type != 0:
                self.rect.x += self.speed
                self.direction = "right"
                self.is_moving = True

        if keys[pygame.K_UP] and self.rect.top > 0:
            target_x = cell_x
            target_y = (self.rect.top - self.speed) // cell_size
            if graph.grid[target_y][target_x].f_type != 0:
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
            )
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
    def __init__(self, from_node = Node, to_node= Node):
        self.from_node = from_node
        self.to_node = to_node


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
                connection = Connection(node, neighbor)
                node.connections.append(connection)

    def set_floor_type(self, walls_positions,floor_type):
        if(floor_type == 0):
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
        else :
            for(x,y) in walls_positions:
                if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                    node = self.grid[x][y]
                    node.f_type = floor_type
                    node.connections = []
                    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    self.add_connections_for_node(node, directions)

def heuristic(node1, node2):
    dx = abs(node1.x - node2.x)
    dy = abs(node1.y - node2.y)
    return dx + dy

def astar_pathfinding(start_node, goal_node, npc):
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

            if neighbor.f_type == 2:
                cost = 10
            elif neighbor.f_type == 0:
                continue
            elif neighbor.f_type == 3:
                if npc == "bulbasaur" or npc == "ivysaur":
                    cost = 10000
                elif npc == "charmander" or npc == "charmaleon":
                    cost = 0
                else:
                    cost = 50
            elif neighbor.f_type == 4:
                if npc == "bulbasaur" or npc == "ivysaur":
                    cost = 0
                elif npc == "charmander" or npc == "charmaleon":
                    cost = 10000
                else:
                    cost = 50
            else: cost = 50
            
            new_cost = cost_so_far[current] + cost

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal_node)
                heapq.heappush(open_list, (priority, neighbor))
                came_from[neighbor] = current
    return None


pygame.init()

RED = (255, 27, 84)
BLUE = (27,43,148)
font = pygame.font.Font('pokemon-gb-font/PokemonGb-RAeo.ttf', 24)
font_small = pygame.font.Font('pokemon-gb-font/PokemonGb-RAeo.ttf', 12)

# SCREEN
cell_size = 30
width, height = 920, 660
rows = height//16
cols = width//30
graph = Graph(rows, cols)
screen = pygame.display.set_mode((width, height))
background_color = (38, 107, 24)

pygame.display.set_caption("Ditto, Ditto, Ditto")


def draw_game_over():

    screen.fill((16,20,20))
    game_over_text = font.render("GAME OVER", True, RED)
    restart_text = font.render("Press R to Restart", True, BLUE)
    quit_text = font_small.render("Press Q to Quit", True, BLUE)
    
    # Posiciones del texto
    screen.blit(game_over_text, (width // 2 - game_over_text.get_width() // 2,height // 2 - 100))
    screen.blit(restart_text, (width // 2 - restart_text.get_width() // 2, height // 2))
    screen.blit(quit_text, (width // 2 - quit_text.get_width() // 2, height // 2 + 50))


# PLAYER
sprite_player = pygame.image.load('images/Male_Spritesheet.png')
heart = pygame.image.load('images/heart.png')
heart = pygame.transform.scale(heart, (30, 30))

# NPCS
ditto1 = NPC( (90, 90))
ditto2 = NPC( (600,300))
ditto3 = NPC((800,389))

player1 = Player(sprite_player,34,34)

all_sprites = pygame.sprite.Group(player1, ditto1,ditto2,ditto3)
npc_group = pygame.sprite.Group(ditto1,ditto2,ditto3)

# FLOOR TYPES
wall_image = pygame.image.load('images/wall1.png').convert_alpha()
wall_image = pygame.transform.scale(wall_image, (cell_size, cell_size))

fast_floor = pygame.image.load('images/floor_1.png').convert_alpha()
fast_floor = pygame.transform.scale(fast_floor,(cell_size, cell_size))

# Tactical points
fire = pygame.image.load('images/fire.png').convert_alpha()
fire = pygame.transform.scale(fire, (cell_size, cell_size))

tall_grass = pygame.image.load('images/grass.png').convert_alpha()
tall_grass = pygame.transform.scale(tall_grass, (cell_size, cell_size))

# pokeball
pokeball = pygame.image.load('images/pokeball1.png').convert_alpha()
pokeball = pygame.transform.scale(pokeball,(cell_size-5,cell_size-5))

# buildings
oak_lab = pygame.image.load('images/oaks_lab.png').convert_alpha()
oak_lab = pygame.transform.scale(oak_lab, (5 * cell_size, 3 * cell_size))

# WALL POSITIONS
walls_positions = []

# Add borders
for x in range(cols):
    walls_positions.append((x, 0))  # Top border
    walls_positions.append((x, rows -1))  # Bottom border
for y in range(rows):
    walls_positions.append((0, y))  # Left border
    walls_positions.append((cols - 1, y))  # Right border

# Add concentrated wall blocks
for x in range(10, 15):
    for y in range(10, 15):
        walls_positions.append((x, y))

for y in range(20, 30):
    walls_positions.append((25, y))

for x in range(30, 40):
    walls_positions.append((x, 35))

graph.set_floor_type(walls_positions, 0)


fire_positions = [
    (5, 5), (6, 5), (7, 5), (8, 5), (9, 5),
    (5, 6), (6, 6), (7, 6), (8, 6), (9, 6),
    (5, 7), (6, 7), (7, 7), (8, 7), (9, 7),
    (5, 8), (6, 8), (7, 8), (8, 8), (9, 8),
    (5, 9), (6, 9), (7, 9), (8, 9), (9, 9),
]

grass_positions = [
    (22, 10), (23, 10), (24, 10), (25, 10),
    (22, 11), (23, 11), (24, 11), (25, 11),
    (22, 12), (23, 12), (24, 12), (25, 12),
    (22, 13), (23, 13), (24, 13), (25, 13),
    (22, 14), (23, 14), (24, 14), (25, 14),
    (22, 15), (23, 15), (24, 15), (25, 15),
]

graph.set_floor_type(fire_positions, 3)
graph.set_floor_type(grass_positions, 4)


graph.set_floor_type( [(1,1),(1,2),(5,5),(10,6),(3,3),(10,9),(9,10),
            (16,10),(16,12),(16,13),(16,14),(16,11),
            (17,10), (17,11), (17,12), (17,13), (17,14),
            (10,18),(11,18), (12,18), (13,18),(14,18) ],2)


game_state = "playing"
is_paused = False
running = True


while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:  # press p to pause the game
                is_paused = not is_paused
        if game_state == "game_over":
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # press r to restart the game
                    player1.lives = 5
                    game_state = "playing"
                elif event.key == pygame.K_q:  # press q to quit the game
                    running = False
                
    if is_paused:
        pause_text = font.render("PAUSE", True, RED)
        text_x = screen.get_width() // 2 - pause_text.get_width() // 2
        text_y = screen.get_height() // 2 - pause_text.get_height() // 2
        padding = 8
        background_rect = pygame.Rect(
            text_x - padding,
            text_y - padding,
            pause_text.get_width() + 2 * padding,
            pause_text.get_height() + 2 * padding
        )
        pygame.draw.rect(screen, (255, 255, 255), background_rect)
        screen.blit(pause_text, (text_x, text_y))

    if not is_paused:
        if game_state == "playing":
            if player1.lives <= 0:
                game_state = "game_over"
                
            player1.update()

            # Update NPCs
            ditto1.update(player1, graph,npc_group)
            ditto2.update(player1, graph, npc_group)
            ditto3.update(player1, graph, npc_group)

            screen.fill((38, 107, 24))


            COLOR_NORMAL = (38, 107, 24)
            COLOR_BORDER = (62, 72, 82)
            
            floor_images = {
                0: wall_image,
                2: fast_floor,
                3: fire,
                4: tall_grass
            }

            for y in range(graph.grid_height):
                for x in range(graph.grid_width):
                    node = graph.grid[x][y]
                    
                    rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
                    image = floor_images.get(node.f_type)
                    if image:
                        screen.blit(image, (x * cell_size, y * cell_size))
                    else:
                        rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
                        pygame.draw.rect(screen, COLOR_NORMAL, rect)
                    pygame.draw.rect(screen, COLOR_BORDER, rect, 1)
                    
            for poke in pokeball_positions:
                screen.blit(pokeball,(poke[0], poke[1], cell_size,cell_size))
                
                
            screen.blit(oak_lab, (20*cell_size, 2*cell_size))
            all_sprites.draw(screen)

            for node in graph.grid[x][y].connections:
                pygame.draw.line(screen, (0, 0, 0),
                            (node.from_node.x * cell_size + cell_size // 2,
                            node.from_node.y * cell_size + cell_size // 2),
                            (node.to_node.x * cell_size + cell_size // 2,
                            node.to_node.y * cell_size + cell_size // 2), 1)
            
            
            npc_colors_tactical = {
            "ditto": (217, 39, 209),
            "bulbasaur": (48, 217, 87),
            "ivysaur": (48, 217, 87),
            "charmander": (217, 90, 46),
            "charmaleon" : (217, 90, 46)
            }
            
            npc_colors_non_tactical = {
            "ditto": (255, 198, 255),
            "bulbasaur":(112, 217, 134),
            "ivysaur": (112, 217, 134),
            "charmander": (217, 152, 132),
            "charmaleon" : (217, 152, 132)
            }
            
            # show path
            for npc in npc_group:
                
                if npc.node_path:
                    if len(npc.node_path) > 1:  # at least two nodes in the path
                        for i in range(len(npc.node_path) - 1):
                            start_node = npc.node_path[i]
                            end_node = npc.node_path[i + 1]

                            # nodes coordenates
                            start_pos = (start_node.x * cell_size + cell_size // 2, start_node.y * cell_size + cell_size // 2)
                            end_pos = (end_node.x * cell_size + cell_size // 2, end_node.y * cell_size + cell_size // 2)
                            npc_color_tac = npc_colors_tactical.get(npc.pokemon)


                            pygame.draw.line(screen, npc_color_tac, start_pos, end_pos, 4)
                
                if npc.non_tactical_node_path :
                    if len(npc.non_tactical_node_path) > 1:  # at least two nodes in the path
                        for i in range(len(npc.non_tactical_node_path) - 1):
                            start_node = npc.non_tactical_node_path[i]
                            end_node = npc.non_tactical_node_path[i + 1]

                            # nodes coordenates
                            start_pos = (start_node.x * cell_size + cell_size // 2, start_node.y * cell_size + cell_size // 2)
                            end_pos = (end_node.x * cell_size + cell_size // 2, end_node.y * cell_size + cell_size // 2)
                            npc_color_non_tac = npc_colors_non_tactical.get(npc.pokemon)

                            pygame.draw.line(screen,npc_color_non_tac, start_pos, end_pos, 2)
                if npc.non_tactical_node_path:
                    last_node = npc.non_tactical_node_path[-1]
                    if (npc.rect.centerx // cell_size == last_node.x and npc.rect.centery // cell_size == last_node.y):
                        npc.non_tactical_node_path = []
                    


            for i in range(player1.lives):
                x = 10 + i * 40
                y = 10
                screen.blit(heart, (x, y))
        elif game_state == "game_over":
            draw_game_over()

    pygame.display.flip()

pygame.quit()