from controller import Supervisor
import random
import math
import sys

supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())
root_children = supervisor.getRoot().getField("children")

try:
    args = supervisor.getControllerArguments()
except AttributeError:
    args = sys.argv[1:]

if len(args) >= 5:
    n_objects = int(args[0])
    x_min = float(args[1])
    x_max = float(args[2])
    y_min = float(args[3])
    y_max = float(args[4])
else:
    n_objects = 15
    x_min, x_max = -3, 1.75
    y_min, y_max = -1, 1

# ===================================================================
# 🔵 ADIÇÃO: Desenhar retângulo delimitando a área de spawn
# ===================================================================

area_width = abs(x_max - x_min)
area_height = abs(y_max - y_min)

center_x = (x_min + x_max) / 2
center_y = (y_min + y_max) / 2

area_marker = f"""
Solid {{
  translation {center_x} {center_y} 0.002
  rotation 0 0 1 0
  name "spawn_area_marker"
  children [
    Shape {{
      appearance PBRAppearance {{
        baseColor 0 0 1
        transparency 0.6
      }}
      geometry Box {{
        size {area_width} {area_height} 0.001
      }}
    }}
  ]
}}
"""

# removendo marcador anterior (se existir)
for i in reversed(range(root_children.getCount())):
    node = root_children.getMFNode(i)
    name_field = node.getField("name")
    if name_field and name_field.getSFString() == "spawn_area_marker":
        node.remove()

# inserindo o novo marcador
root_children.importMFNodeFromString(-1, area_marker)

def draw_obstacle_safety_zones(obstacles):
    """Draw red safety zones on the ground around each obstacle."""
    
    # remove zonas antigas
    for i in reversed(range(root_children.getCount())):
        node = root_children.getMFNode(i)
        name_field = node.getField("name")
        if name_field and name_field.getSFString().startswith("safety_zone_"):
            node.remove()

    # cria novas zonas
    for idx, obs in enumerate(obstacles):
        radius = obs['radius'] + size

        zone_string = f"""
        Solid {{
          translation {obs['x']} {obs['y']} 0.001
          name "safety_zone_{idx}"
          children [
            Shape {{
              appearance PBRAppearance {{
                baseColor 1 0 0
                transparency 0.6
              }}
              geometry Cylinder {{
                radius {radius}
                height 0.001
              }}
            }}
          ]
        }}
        """

        root_children.importMFNodeFromString(-1, zone_string)
 
def draw_cube_safety_zones(cube_positions):
    """Draw red safety zones on the ground around each small cube."""

    # remove zonas antigas
    for i in reversed(range(root_children.getCount())):
        node = root_children.getMFNode(i)
        name_field = node.getField("name")
        if name_field and name_field.getSFString().startswith("cube_safety_"):
            node.remove()

    radius = min_dist / 2

    for idx, pos in enumerate(cube_positions):
        zone_string = f"""
        Solid {{
          translation {pos[0]} {pos[1]} 0.001
          name "cube_safety_{idx}"
          children [
            Shape {{
              appearance PBRAppearance {{
                baseColor 1 0 0
                transparency 0.4
              }}
              geometry Cylinder {{
                radius {radius}
                height 0.001
              }}
            }}
          ]
        }}
        """

        root_children.importMFNodeFromString(-1, zone_string)
    


# ===================================================================
# (Restante do seu código original — nada removido)
# ===================================================================

def get_existing_obstacles():
    """Extract positions of WoodenBoxes and PlasticFruitBoxes from the world."""
    obstacles = []
    n = root_children.getCount()
    for i in range(n):
        node = root_children.getMFNode(i)
        type_name = node.getTypeName()
        
        if type_name in ["WoodenBox", "PlasticFruitBox"]:
            trans_field = node.getField("translation")
            size_field = node.getField("size")
            
            if trans_field:
                pos = trans_field.getSFVec3f()
                if size_field:
                    size = size_field.getSFVec3f()
                    radius = math.sqrt(size[0]**2 + size[1]**2) / 2
                else:
                    radius = 0.3
                
                obstacles.append({
                    'x': pos[0],
                    'y': pos[1],
                    'radius': radius + 0.1
                })
    
    return obstacles

def delete_existing_objects():
    """Remove previously spawned objects."""
    n = root_children.getCount()
    for i in reversed(range(n)):
        node = root_children.getMFNode(i)
        name_field = node.getField("name")
        if name_field:
            node_name = name_field.getSFString()
            if node_name.startswith(("cube_", "cylinder_", "sphere_")):
                node.remove()

delete_existing_objects()

size = 0.03
mass = 0.03
min_dist = size * 2.5

existing_obstacles = get_existing_obstacles()
draw_obstacle_safety_zones(existing_obstacles)


positions = []
colors = [
    (0, 1, 0),  # green
    (1, 0, 0),  # red
    (0, 0, 1),  # blue
]
shapes = ["cube"]

def random_pos():
    """Generate random position on the floor."""
    return (
        random.uniform(x_min, x_max),
        random.uniform(y_min, y_max),
        size / 2 + 0.001
    )

def is_far_enough(pos):
    """Check if position is far from both spawned objects and existing obstacles."""
    for q in positions:
        dx, dy = pos[0] - q[0], pos[1] - q[1]
        if math.hypot(dx, dy) < min_dist:
            return False
    
    for obs in existing_obstacles:
        dx, dy = pos[0] - obs['x'], pos[1] - obs['y']
        if math.hypot(dx, dy) < (obs['radius'] + size):
            return False
    
    return True

spawned_count = 0
failed_spawns = 0
max_attempts = 100

for i in range(n_objects):
    tries = 0
    success = False
    
    while tries < max_attempts:
        tries += 1
        pos = random_pos()
        if is_far_enough(pos):
            success = True
            break
    
    if not success:
        print(f"Warning: Could not find valid position for object {i} after {max_attempts} attempts")
        failed_spawns += 1
        continue
    
    positions.append(pos)
    shape_type = random.choice(shapes)
    color = random.choice(colors)
    
    if shape_type == "cube":
        geometry = f"Box {{ size {size} {size} {size} }}"
        bounding = f"Box {{ size {size} {size} {size} }}"
    
    node_string = f"""
    Solid {{
      translation {pos[0]} {pos[1]} {pos[2]}
      name "{shape_type}_{i}"
      children [
        Shape {{
          appearance PBRAppearance {{
            baseColor {color[0]} {color[1]} {color[2]}
            roughness 0.8
            metalness 0
          }}
          geometry {geometry}
        }}
      ]
      boundingObject {bounding}
      physics Physics {{
        density -1
        mass {mass}
      }}
      recognitionColors [ {color[0]} {color[1]} {color[2]} ]
    }}
    """

    root_children.importMFNodeFromString(-1, node_string)
    spawned_count += 1
    
draw_cube_safety_zones(positions)

print(f"Spawn complete. The supervisor has spawned {spawned_count}/{n_objects} objects ({failed_spawns} failed).")

for _ in range(20):
    supervisor.step(timestep)
