# dataset_generator.py
from controller import Supervisor
import os, math

TIME_STEP = 16

# ✅ PARÂMETROS PARA SEMI-ESFERA COMPLETA (do chão até o topo) - APENAS CUBOS
RADIUS_CUBES = 0.045        
H_ANGLES_CUBES = 36        # 36 ângulos horizontais 
V_ANGLES_CUBES = 12        # 12 inclinações: de 5° até 90° (quase horizontal até top-down)

HEIGHT_OFFSET = 0.0
BASE_OUT_DIR = "dataset_cubes_only"
WAIT_STEPS_BEFORE = 8
WAIT_STEPS_BETWEEN = 1
SAVE_QUALITY = 100

# ✅ LIMITES DA ARENA
ARENA_CENTER_X = -0.79
ARENA_CENTER_Y = 0.0
ARENA_WIDTH = 7.0   
ARENA_HEIGHT = 4.0  


def vec_len(v):
    return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

def normalize(v):
    L = vec_len(v)
    if L == 0:
        return (0.0, 0.0, 0.0)
    return (v[0]/L, v[1]/L, v[2]/L)

def cross(a, b):
    return (a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0])

def rotation_matrix_from_basis(x_axis, y_axis, z_axis):
    return [
        [x_axis[0], y_axis[0], z_axis[0]],
        [x_axis[1], y_axis[1], z_axis[1]],
        [x_axis[2], y_axis[2], z_axis[2]]
    ]

def mat_trace(R):
    return R[0][0] + R[1][1] + R[2][2]

def axis_angle_from_rotation_matrix(R):
    trace = mat_trace(R)
    cos_ang = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    angle = math.acos(cos_ang)
    if abs(angle) < 1e-6:
        return (1.0, 0.0, 0.0, 0.0)
    denom = 2.0 * math.sin(angle)
    rx = (R[2][1] - R[1][2]) / denom
    ry = (R[0][2] - R[2][0]) / denom
    rz = (R[1][0] - R[0][1]) / denom
    return (rx, ry, rz, angle)

def look_at_axis_angle(cam_pos, target_pos):
    dx = target_pos[0] - cam_pos[0]
    dy = target_pos[1] - cam_pos[1]
    dz = target_pos[2] - cam_pos[2]
    f = normalize((dx, dy, dz))
    if vec_len(f) == 0:
        return (1, 0, 0, 0)
    up = (0.0, 0.0, 1.0)
    if abs(f[0]*up[0] + f[1]*up[1] + f[2]*up[2]) > 0.999:
        up = (0.0, 1.0, 0.0)
    y_axis = normalize(cross(up, f))
    z_axis = cross(f, y_axis)
    R = rotation_matrix_from_basis(f, y_axis, z_axis)
    return axis_angle_from_rotation_matrix(R)

def is_valid_camera_position(cx, cy, cz, object_z):
    """Verifica se a posição da câmera está dentro da arena E acima do chão"""
    x_min = ARENA_CENTER_X - ARENA_WIDTH/2 + 0.15  
    x_max = ARENA_CENTER_X + ARENA_WIDTH/2 - 0.15  
    y_min = ARENA_CENTER_Y - ARENA_HEIGHT/2 + 0.15 
    y_max = ARENA_CENTER_Y + ARENA_HEIGHT/2 - 0.15 
    
    # ✅ CÂMERA PODE FICAR BEM BAIXA (quase no nível do objeto)
    z_min = max(0.01, object_z - 0.02)  # um pouquinho acima do objeto
    
    if cx < x_min or cx > x_max or cy < y_min or cy > y_max or cz < z_min:
        return False
    
    return True

# ✅ DETECÇÃO DE CUBOS - APENAS 3 CORES
def detect_cubes(robot):
    cubes = []
    root = robot.getRoot()
    children = root.getField("children")
    for i in range(children.getCount()):
        node = children.getMFNode(i)
        if node is None or node.getTypeName() != "Solid":
            continue
        ch = node.getField("children")
        if ch is None or ch.getCount() == 0:
            continue
        shape = ch.getMFNode(0)
        geom = shape.getField("geometry").getSFNode()
        if geom is None or geom.getTypeName() != "Box":
            continue
        size = geom.getField("size").getSFVec3f()
        if not all(math.isclose(s, 0.03, rel_tol=1e-3, abs_tol=1e-6) for s in size):
            continue
            
        color_name = "unknown"
        try:
            rec = node.getField("recognitionColors")
            if rec is not None and rec.getCount() > 0:
                r, g, b = rec.getMFColor(0)
                key = (int(round(r)), int(round(g)), int(round(b)))
                
                # ✅ APENAS 3 CORES
                cube_color_map = {
                    (1, 0, 0): "red",      # Vermelho
                    (0, 1, 0): "green",    # Verde
                    (0, 0, 1): "blue"      # Azul
                }
                
                if key in cube_color_map:
                    color_name = cube_color_map[key]
                    print(f"[CUBE] Encontrado {color_name} em {node.getPosition()}")
        except Exception as e:
            print(f"[ERROR] Erro ao ler cor do cubo: {e}")
            
        if color_name != "unknown":
            cubes.append((node, color_name))
    return cubes

def wait_steps(robot, n=1):
    for _ in range(n):
        robot.step(TIME_STEP)

def get_next_image_counter(base_dir):
    if not os.path.exists(base_dir):
        return 0
    
    existing_files = []
    for filename in os.listdir(base_dir):
        if filename.startswith("img_") and filename.endswith(".png"):
            try:
                num_str = filename[4:10]
                existing_files.append(int(num_str))
            except:
                pass
    
    return max(existing_files) + 1 if existing_files else 0

# ----- main -----
def main():
    robot = Supervisor()

    cam_device = robot.getDevice("CAM")
    if cam_device is None:
        print("[ERROR] Dispositivo Camera 'CAM' não encontrado.")
        return
    cam_device.enable(TIME_STEP)

    cam_node = robot.getSelf().getField("children").getMFNode(0)
    if cam_node is None:
        print("[ERROR] Nó da câmera não encontrado.")
        return
    cam_trans = cam_node.getField("translation")
    cam_rot   = cam_node.getField("rotation")

    os.makedirs(BASE_OUT_DIR, exist_ok=True)
    labels_path = os.path.join(BASE_OUT_DIR, "labels.txt")
    img_counter = get_next_image_counter(BASE_OUT_DIR)
    labels_f = open(labels_path, "a", encoding="utf-8")

    wait_steps(robot, WAIT_STEPS_BEFORE)

    # ✅ DETECTAR APENAS CUBOS
    cubes = detect_cubes(robot)
    
    print(f"[INFO] ✅ DATASET APENAS CUBOS COM 3 CORES: red, green, blue")
    print(f"[INFO] ✅ SEMI-ESFERA COMPLETA (do chão até o topo)!")
    print(f"[INFO] Cubos encontrados: {len(cubes)}")
    print(f"[INFO] Total de imagens: {len(cubes)} × {H_ANGLES_CUBES} × {V_ANGLES_CUBES} = {len(cubes) * H_ANGLES_CUBES * V_ANGLES_CUBES}")
    print(f"[INFO] Começando do contador: {img_counter}")

    # ✅ PROCESSAR APENAS CUBOS (SEMI-ESFERA COMPLETA: do chão até o topo)
    for idx, (cube, color_name) in enumerate(cubes):
        print(f"[INFO] Processando cubo {idx+1}/{len(cubes)} ({color_name})")
        rot_field = cube.getField("rotation")
        try:
            orig_rot = rot_field.getSFRotation()
        except:
            orig_rot = [0,0,1,0]

        ox, oy, oz = cube.getPosition()
        target = (ox, oy, oz + HEIGHT_OFFSET)
        cube_images = 0

        # ✅ SEMI-ESFERA COMPLETA: de 5° (quase horizontal) até 90° (top-down)
        for vi in range(V_ANGLES_CUBES):
            if V_ANGLES_CUBES == 1:
                theta = math.pi/4.0
            else:
                # ✅ RANGE COMPLETO: de 5° até 90° (evita 0° para não dar problema)
                theta_min = math.pi/36.0  # 5° (quase horizontal/nível do chão)
                theta_max = math.pi/2.0   # 90° (top-down)
                theta = theta_min + (theta_max - theta_min) * vi / float(V_ANGLES_CUBES - 1)
            
            r_xy = RADIUS_CUBES * math.sin(theta)
            z_rel = RADIUS_CUBES * math.cos(theta)

            for hi in range(H_ANGLES_CUBES):
                phi = 2.0 * math.pi * hi / float(H_ANGLES_CUBES)

                cx = ox + r_xy * math.cos(phi)
                cy = oy + r_xy * math.sin(phi)
                cz = oz + z_rel + HEIGHT_OFFSET

                if not is_valid_camera_position(cx, cy, cz, oz):
                    continue

                cam_trans.setSFVec3f([cx, cy, cz])
                
                # ✅ APLICAR ROTAÇÃO COM VALIDAÇÃO
                try:
                    axis_x, axis_y, axis_z, angle = look_at_axis_angle((cx,cy,cz), target)
                    cam_rot.setSFRotation([axis_x, axis_y, axis_z, angle])
                    rot_field.setSFRotation([0,0,1,phi])
                    
                except Exception as e:
                    print(f"[WARNING] Erro na rotação, usando padrão: {e}")
                    cam_rot.setSFRotation([0, 0, 1, 0])
                    rot_field.setSFRotation([0,0,1,0])

                wait_steps(robot, WAIT_STEPS_BETWEEN)

                img_name = f"img_{img_counter:06d}.png"
                img_path = os.path.join(BASE_OUT_DIR, img_name)
                cam_device.saveImage(img_path, SAVE_QUALITY)
                labels_f.write(f"{img_name} {color_name}\n")
                labels_f.flush()
                
                cube_images += 1
                img_counter += 1
                
                if cube_images % 100 == 0:
                    print(f"  ✓ {cube_images} fotos do cubo {color_name} (θ={math.degrees(theta):.0f}°)")

        print(f"[DONE] Cubo {color_name}: {cube_images} imagens geradas")
        
        # ✅ RESTAURAR ROTAÇÃO ORIGINAL
        try:
            rot_field.setSFRotation(orig_rot)
        except:
            rot_field.setSFRotation([0,0,1,0])
        wait_steps(robot, 2)

    labels_f.close()
   
    while robot.step(TIME_STEP) != -1:
        pass

if __name__ == "__main__":
    main()