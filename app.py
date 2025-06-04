from flask import Flask, request, jsonify, send_file, render_template
import os
import threading
import math
import numpy as np
import threading
from array import array
from collections import deque

import segformer_b0 as seg
import path_finding as pf
import ppo
import firing as fire
from utils import shared_data
from collections import deque
import torch


app = Flask(__name__)

data_stack = deque()

# Segmentation 모델 선언
seg_model, image_processor = seg.init_model()
# threadingLock
threading_lock = threading.Lock()

# 전차 크기 정의 (x: 5미터, z: 11미터)
VEHICLE_WIDTH = int(5.0)
VEHICLE_LENGTH = int(11.0)

# 월드 크기 정의
WORLD_SIZE = 300  # 300x300 미터

# 적 감지 여부
enemy_detected = False
enemy_suspected = False
ready_to_attack = False
attack_confirmed = False
detected_buffer = 0
enemy_list = []
preemptive_strike = False

# 강화학습 모델 사용 시
use_reinforecement_model = True
device = None
model = None
env = None
command_to_number = {'W': 0, 'S' : 1, 'A': 2, 'D': 3}
number_to_command = {0: 'W', 1 : 'S', 2: 'A', 3: 'D'}
weight_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]

# 초기화
grid = pf.Grid(width=WORLD_SIZE, height=WORLD_SIZE)
pathfinding = pf.Pathfinding()
nav_config = pf.NavigationConfig()
nav_controller = pf.NavigationController(nav_config, pathfinding, grid)
obstacles_list = []

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)
latest_result = os.path.join(result_dir, "latest_result.png")

# 평시 정찰 코드
turret_rotate = 'Q'
distance_buffer = deque(maxlen=20)
for i in range(20):
    distance_buffer.append(999)

# destination 공유 코드
destination = None
destination_buffer = 0
trail = []
step_counter = 0
rng = np.random.default_rng(3)

# 각도 변환용
def change_degree(my_d):
    sin = np.sin(np.deg2rad(my_d))
    cos = np.cos(np.deg2rad(my_d))

    return sin, cos

def change_turret_degree(my_d):
    if my_d > 180:
        direction = -(360-my_d)
    else:
        direction = my_d
    return direction

# 상대좌표
def get_target_coord(now_x, now_y, turret_x, distance):
    rad = math.radians(turret_x)
    enemy_x = math.sin(rad) * distance + now_x
    enemy_y = math.cos(rad) * distance + now_y
    return enemy_x, enemy_y

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    global enemy_detected
    global detected_buffer
    global enemy_list
    global enemy_suspected

    image = request.files.get('image')
    if not image:
        print('no image received')
        return jsonify({"error": "No image received"}), 400

    image_path = './static/source/temp_image.jpg'
    image.save(image_path)

    pixels = seg.detect_vehicle(seg_model, image_processor, image_path)
    if pixels > 500:
        enemy_suspected = True
        print(f'enemy suspected')
    else:
        enemy_suspected = False
    if enemy_suspected:
        enemy_list = seg.get_vehicle_distance(seg_model, image_processor)
        if enemy_list == None:
            enemy_list = []
            enemy_detected = False
        else:
            enemy_detected = True
            print(f'enemy detected / {enemy_list}')

    filtered_results = []

    return (filtered_results), 200

@app.route('/latest_result')
def get_latest_result():
    if os.path.exists(latest_result):
        return send_file(latest_result, mimetype='image/png')
    else:
        return jsonify({"error": "No result available"}), 404


# Flask 라우팅
@app.route('/info', methods=['POST'])
def info():
    global trail
    data = request.get_json()
    try:
        shared_data.set_data(data)
        player_pos = data["playerPos"]
        x, z = float(player_pos["x"]), float(player_pos["z"])
        if step_counter % 5 == 0:
            trail.append([x, z])
        result = nav_controller.update_position(f"{x},0,{z}")
        return jsonify(result)
    except (KeyError, ValueError, TypeError) as e:
        print(f"Error in /info: {e}")
        return jsonify({"status": "ERROR", "message": "Invalid data"}), 400

@app.route('/update_position', methods=['POST'])
def update_position():
    data = request.get_json()
    if not data or "position" not in data:
        return jsonify({"status": "ERROR", "message": "위치 데이터 누락"}), 400
    result = nav_controller.update_position(data["position"])
    if result["status"] == "ERROR":
        return jsonify(result), 400
    return jsonify(result)

@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    global obstacles_list
    data = request.get_json()
    try:
        obstacles = data["obstacles"]
        for obstacle in obstacles:
            x_min = float(obstacle["x_min"])
            x_max = float(obstacle["x_max"])
            z_min = float(obstacle["z_min"])
            z_max = float(obstacle["z_max"])
            grid.set_obstacle(x_min, x_max, z_min, z_max)
            obstacles_list.append({
                "x_min": x_min,
                "x_max": x_max,
                "z_min": z_min,
                "z_max": z_max
            })
        # print(f"Obstacles Updated: {obstacles_list}")
        return jsonify({"status": "OK"})
    except (KeyError, ValueError, TypeError) as e:
        print(f"Error in /update_obstacle: {e}")
        return jsonify({"status": "ERROR", "message": "Invalid obstacle data"}), 400

@app.route('/set_destination', methods=['POST'])
def set_destination():
    data = request.get_json()
    if not data or "destination" not in data:
        return jsonify({"status": "ERROR", "message": "목적지 데이터 누락"}), 400
    result = nav_controller.set_destination(data["destination"])
    print('Destination:' , data["destination"], type(data["destination"]))
    if result["status"] == "ERROR":
        return jsonify(result), 400
    return jsonify(result)

@app.route('/get_move', methods=['GET'])
def get_move():
    global enemy_detected
    global enemy_suspected
    global enemy_list
    global destination_buffer
    global destination
    global step_counter
    global distance_buffer
    sim_data = shared_data.get_data()
    step_counter += 1
    # 강화학습 모델 사용 시 
    if use_reinforecement_model:
        x, y, z = sim_data['playerPos']['x'], sim_data['playerPos']['y'], sim_data['playerPos']['z']
        speed, t_x, t_y  = sim_data['playerSpeed'], sim_data['playerTurretX'], sim_data['playerTurretY']
        b_x, b_y, b_z = sim_data['playerBodyX'] ,sim_data['playerBodyY'], sim_data['playerBodyZ']
        d_x, d_z = destination[0], destination[1]
        distance = np.sqrt((x - d_x)**2 + (z - d_z)**2)
        # 각도 sin, cos 화 및 위치와 속도 정규화
        x, y, z, speed = x / 300, y / 300, z / 300, speed / 100
        b_x_sin, b_x_cos = change_degree(b_x)
        b_y_sin, b_y_cos = change_degree(b_y)
        b_z_sin, b_z_cos = change_degree(b_z)
        d_x, d_z = d_x / 300, d_z / 300
        if distance < 24:
            result = True
            print('Tank is arrived')
            sensor_data_for_reset = np.array([x,y,z,speed,b_x_sin,b_x_cos,b_y_sin,b_y_cos,b_z_sin,b_z_cos])
            destination_for_reset = np.array([d_x, d_z])
            env.reset(options={'sensor_data': sensor_data_for_reset, 'goal_position': destination_for_reset})
            return jsonify({"move": "STOP", 'weight': 1.0})
        else:
            result = False
        # PPO Agent 값 입력
        data = {
        'sensor_data': torch.tensor([x,y,z,speed,b_x_sin, b_x_cos, b_y_sin, b_y_cos, b_z_sin, b_z_cos], dtype=torch.float32).unsqueeze(0).to(device),
        'goal_position': torch.tensor([d_x, d_z], dtype=torch.float32).unsqueeze(0).to(device)
        }
        # env.step() 수행 시 꺼내서 조회할 데이터
        data_np = {'sensor_data': data['sensor_data'].cpu(), 'goal_position': data['goal_position'].cpu()}
        ppo.stack_data(data_np, result, distance)
        # PPO Agent 행동 도출 
        with torch.no_grad():
            action, value, log_prob = model.policy(data, deterministic=True)
        action_1 = number_to_command[action.detach().cpu().numpy()[0][0]]
        action_2 = weight_bins[action.detach().cpu().numpy()[0][1]]
        print('PPO Agent is making decisions')

        # Command 반환
        command = {"move": action_1, "weight": action_2} # 규칙 기반 출력 값
        
        # 적 발견에 따른 행동 분기 
        if enemy_detected and preemptive_strike:
            weight = 0.1
            enemies = len(enemy_list)
            if enemies == 0:
                print('enemy detected but enemy list is empty')
                return jsonify({"move": "STOP", 'weight': weight})
            if enemies > 0:
                # 사정거리 안에 있으면 그 자리에서 멈춰서 쏘자
                e_distance = enemy_list[0]['distance']
                distance_buffer.append(e_distance)
                distance = np.mean(distance_buffer)
                if distance < 90:
                    print('enemy in range. tank stop')
                    return jsonify({"move": "STOP", 'weight': weight})
                else:
                    # x = sim_data['playerPos']['x']
                    # y = sim_data['playerPos']['y']
                    # z = sim_data['playerPos']['z']
                    # turret_x = sim_data['playerTurretX']
                    # enemy_x, enemy_z = get_target_coord(x, z, turret_x, distance)
                    # if destination_buffer == 0:
                    #     destination = [enemy_x, enemy_z]
                    #     print(f'Destination has been changed: {enemy_x},{y},{enemy_z}')
                    #     destination_buffer += 1
                    # else:
                    #     destination_buffer += 1
                    #     if destination_buffer > 64:
                    #         destination_buffer = 0
                    # # command['weight'] = weight
                    # # command['move'] = 'W'
                    # print('enemy detected but out of range.')
                    return jsonify(command)
        else:
            if enemy_suspected and preemptive_strike:
                command['weight'] = 0.05
                print(f'Enemy suspected. Moving Command: {command}')
            else:
                print(f'No Enemy. Moving Command: {command}')
            return jsonify(command)

    # 강화학습 모델 미 사용 시
    else:
        if enemy_detected and preemptive_strike:
            weight = 0.1
            enemies = len(enemy_list)
            if enemies == 0:
                print('enemy detected but enemy list is empty')
                return jsonify({"move": "STOP", 'weight': weight})
            if enemies > 0:
                # 사정거리 안에 있으면 그 자리에서 멈춰서 쏘자
                distance = enemy_list[0]['distance']
                if distance < 105:
                    print('enemy in range. tank stop')
                    return jsonify({"move": "STOP", 'weight': weight})
                else:
                    x = sim_data['playerPos']['x']
                    y = sim_data['playerPos']['y']
                    z = sim_data['playerPos']['z']
                    turret_x = sim_data['playerTurretX']
                    enemy_x, enemy_z = get_target_coord(x, z, turret_x, distance)
                    if destination_buffer == 0:
                        nav_controller.set_destination(f'{enemy_x},{y},{enemy_z}')
                        print(f'Destination has been changed: {enemy_x},{y},{enemy_z}')
                        destination_buffer += 1
                    else:
                        destination_buffer += 1
                        if destination_buffer > 64:
                            destination_buffer = 0
                    command = nav_controller.get_move()
                    command['weight'] = weight
                    print('enemy detected but out of range.')
                    return jsonify(command)
        else:
            command = nav_controller.get_move()
            if enemy_suspected and preemptive_strike:
                command['weight'] = 0.05
                print(f'Enemy suspected. Moving Command: {command}')
            else:
                print(f'No Enemy. Moving Command: {command}')
            return jsonify(command)

@app.route('/get_action', methods=['GET'])
def get_action():
    global enemy_detected, enemy_suspected ,ready_to_attack
    global turret_rotate
    global enemy_list
    data = shared_data.get_data()
    turret_x = change_turret_degree(data['playerTurretX'])
    body_x =  change_turret_degree(data['playerBodyX'])
    heading = turret_x - body_x
    if heading > 30:
        turret_rotate = 'Q'
    elif heading < -30:
        turret_rotate = 'E'
    if enemy_detected and preemptive_strike:
        enemies = len(enemy_list)
        if enemies == 0:
            print('enemy detected but enemy list is empty')
            return jsonify({"turret": turret_rotate, "weight": 0.1})
        if enemies > 0:
            data['distance'] = enemy_list[0].get('distance')
            context = fire.Initialize(data)
            turret = fire.TurretControl(context)
            result = turret.normal_control()
            if result == None:
                return jsonify({"turret": "", "weight": 0.0})
            command = {"turret": result[0], "weight": result[1]}
            if command['turret'] == 'FIRE':
                ready_to_attack = True
            else:
                ready_to_attack = False
            if command['turret'] == 'FIRE' and not(attack_confirmed):
                command['turret'] = 'Q'
                command['weight'] = 0.0
                print('Attack not comfirmed')
            if command:
                print(f"🔫 Action Command: {command}")
                return jsonify(command)
            else:
                return jsonify({"turret": "", "weight": 0.0})      
    else:
        if enemy_suspected:
            weight = 0.1
        else:
            weight = 0.2
        return jsonify({"turret": turret_rotate, "weight": weight})

@app.route('/init', methods=['GET'])
def init():
    global rng
    global destination
    global is_env_start
    global trail

    with threading_lock:
        while True:
            random_coord = rng.integers(low=20, high=130, size=4)
            random_coord2 = rng.integers(low=170, high=280, size=4)
            x = int(random_coord[0])
            z = int(random_coord[1])
            des_x = int(random_coord2[0])
            des_z = int(random_coord2[1])

            distance = np.sqrt((x - des_x) ** 2 + (z - des_z) ** 2)
            if (distance > 40) and (des_x > 5 and des_x < 295 and des_z > 5 and des_z < 295):
                break
        # 초기 목적지를 탱크의 생성 위치로 설정 
        destination = [x, z]

        config = {
            "startMode": "start",  # Options: "start" or "pause"
            "blStartX": x,  #Blue Start Position
            "blStartY": 10,
            "blStartZ": z,
            "rdStartX": des_x, #Red Start Position
            "rdStartY": 10,
            "rdStartZ": des_z,
            "trackingMode": True,
            "detactMode": True,
            "logMode": True,
            "enemyTracking": False,
            "saveSnapshot": False,
            "saveStereoCamera": True,
            "saveLog": False,
            "saveLidarData": False
        }
        print("🛠️ Initialization config sent via /init:", config["blStartX"], config["blStartZ"], config["rdStartX"], config["rdStartZ"])
        is_env_start = False
        trail = []
    return jsonify(config)


@app.route('/api/data', methods=['GET'])
def get_data():
    data = shared_data.get_data()
    if not data:
        return jsonify({"status": "ERROR", "message": "No data available"}), 503

    sensor_data = {
        'x': data['playerPos']['x'],
        'y': data['playerPos']['y'],
        'z': data['playerPos']['z'],
        'speed': data['playerSpeed'],
        'e_x': data['enemyPos']['x'],
        'e_y': data['enemyPos']['y'],
        'e_z': data['enemyPos']['z']
    }
    destination_data = {'d_x': destination[0], 'd_z': destination[1]} if destination else {'d_x': None, 'd_z': None}
    enemy_data = {'detected': enemy_detected}

    return jsonify({
        'sensor_data': sensor_data,
        'destination_data': destination_data,
        'enemy_data': enemy_data,
        'trail': trail,
        'ready' : ready_to_attack,
        'preemptive' : preemptive_strike
    })

@app.route('/api/confirm', methods=['POST'])
def confirm_action():
    global attack_confirmed
    attack_confirmed = True
    print('attack confirmed :', attack_confirmed)
    return jsonify({"status": "SUCCESS", "confirmed": attack_confirmed})

@app.route('/api/deny', methods=['POST'])
def deny_action():
    global attack_confirmed
    attack_confirmed = False
    print('attack confirmed :', attack_confirmed)
    return jsonify({"status": "SUCCESS", "confirmed": attack_confirmed})


@app.route('/visualization', methods=['GET'])
def get_visualization():
    return render_template("visualization.html")  # 데이터는 클라이언트에서 API로 가져옴

@app.route('/update_goal', methods=['POST'])
def set_goal():
    global destination
    global preemptive_strike
    global use_reinforecement_model
    data = request.get_json()
    sim_data = shared_data.get_data()
    x = data['x']
    z = 300 - data['z']
    preemptive_strike = data['preemptive']
    if preemptive_strike:
        use_reinforecement_model = False
    else:
        use_reinforecement_model = True
    destination = [x, z]

    nav_controller.set_destination(f'{x},10,{z}')
    x, y, z = sim_data['playerPos']['x'], sim_data['playerPos']['y'], sim_data['playerPos']['z']
    speed, t_x, t_y  = sim_data['playerSpeed'], sim_data['playerTurretX'], sim_data['playerTurretY']
    b_x, b_y, b_z = sim_data['playerBodyX'] ,sim_data['playerBodyY'], sim_data['playerBodyZ']
    d_x, d_z = destination[0], destination[1]
    # 각도 sin, cos 화 및 위치와 속도 정규화
    x, y, z, speed = x / 300, y / 300, z / 300, speed / 100
    b_x_sin, b_x_cos = change_degree(b_x)
    b_y_sin, b_y_cos = change_degree(b_y)
    b_z_sin, b_z_cos = change_degree(b_z)
    d_x, d_z = d_x / 300, d_z / 300
    sensor_data_for_reset = np.array([x,y,z,speed,b_x_sin,b_x_cos,b_y_sin,b_y_cos,b_z_sin,b_z_cos])
    destination_for_reset = np.array([d_x, d_z])
    env.reset(options={'sensor_data': sensor_data_for_reset, 'goal_position': destination_for_reset})

    return jsonify({'result': 'success'}), 200


@app.route('/start', methods=['GET'])
def start():

    return jsonify({"control": ""})



if __name__ == '__main__':
    device = ppo.init_device()
    model, env = ppo.initialize_ppo()
    app.run(host='0.0.0.0', port=5055, debug=False)