from flask import Flask, request, jsonify, send_file, render_template
import os
import cv2
import shutil
import matplotlib.pyplot as plt
from PIL import Image
import segformer_b0 as seg
import path_finding as pf
import firing as fire
from utils import shared_data
import threading
import math
import numpy as np
import threading

app = Flask(__name__)

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
detected_buffer = 0
enemy_list = []

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

# destination 공유 코등
destination = None
destination_buffer = 0
trail = []
step_counter = 0
rng = np.random.default_rng(3)

# 각도 변환용
def change_degree(my_d):
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
        return jsonify({"error": "No image received"}), 400

    image_path = './static/source/temp_image.jpg'
    image.save(image_path)

    pixels = seg.detect_vehicle(seg_model, image_processor, image_path)
    if pixels > 500:
        enemy_suspected = True
        detected_buffer += 1
        print(f'enemy suspected')
    else:
        detected_buffer  = max(detected_buffer - 1, 0)
        enemy_suspected = False
    if detected_buffer > 3:
        enemy_detected = True
        enemy_list = seg.get_vehicle_distance(seg_model, image_processor)
        if enemy_list == None:
            enemy_list = []
        print(f'enemy detected / {enemy_list}')
    else:
        enemy_detected = False
        enemy_list = []

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
    global step_counter
    step_counter += 1
    if enemy_detected:
        weight = 0.1
        data = shared_data.get_data()
        enemies = len(enemy_list)
        if enemies == 0:
            print('enemy detected but enemy list is empty')
            return jsonify({"move": "W", 'weight': 0.1})
        if enemies > 0:
            # 사정거리 안에 있으면 그 자리에서 멈춰서 쏘자
            distance = enemy_list[0]['distance']
            if distance < 105:
                print('enemy in range. tank stop')
                return jsonify({"move": "STOP", 'weight': weight})
            else:
                x = data['playerPos']['x']
                y = data['playerPos']['y']
                z = data['playerPos']['z']
                turret_x = data['playerTurretX']
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
        # else:
        #     target_id = 0
        #     target_distance = 1000
        #     for i, enemy in enumerate(enemy_list):
        #         if enemy.get['distance'] < target_distance:
        #             target_id = i
        #                 # 사정거리 안에 있으면 그 자리에서 멈춰서 쏘자
        #     distance = enemy_list[target_id]['distance']
        #     if distance < 100:
        #         print('Stop the tank')
        #         return jsonify({"move": "STOP", 'weight': weight})
        #     else:
        #         x = data['playerPos']['x']
        #         y = data['playerPos']['y']
        #         z = data['playerPos']['z']
        #         turret_x = data['playerTurretX']
        #         enemy_x, enemy_z = get_target_coord(x, z, turret_x, distance)
        #         if destination_buffer == 0:
        #             nav_controller.set_destination(f'{enemy_x},{y},{enemy_z}')
        #             print(f'Destination has been changed: {enemy_x},{y},{enemy_z}')
        #             destination_buffer += 1
        #         else:
        #             destination_buffer += 1
        #             if destination_buffer > 32:
        #                 destination_buffer = 0
        #         command = nav_controller.get_move()
        #         command['weight'] = weight
        #         print(f'Moving Command: {command}')
        #         return jsonify(command)
    else:
        command = nav_controller.get_move()
        if enemy_suspected:
            command['weight'] = 0.2
            print(f'Enemy suspected. Moving Command: {command}')
        else:
            print(f'No Enemy. Moving Command: {command}')
        return jsonify(command)

@app.route('/api/data', methods=['GET'])
def get_data():
    global destination, enemy_detected, trail
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
        'trail': trail
    })

@app.route('/visualization', methods=['GET'])
def get_visualization():
    return render_template("visualization.html")  # 데이터는 클라이언트에서 API로 가져옴

@app.route('/update_goal', methods=['POST'])
def set_goal():
    global destination
    data = request.get_json()
    x = data['x']
    z = 300 - data['z']
    destination = [x, z]
    result = nav_controller.set_destination(f'{x},10,{z}')
    print(result)
    return jsonify({'result': 'success'}), 200


@app.route('/get_action', methods=['GET'])
def get_action():
    global enemy_detected
    global enemy_suspected
    global turret_rotate
    global enemy_list
    data = shared_data.get_data()
    turret_x = change_degree(data['playerTurretX'])
    body_x =  change_degree(data['playerBodyX'])
    heading = turret_x - body_x
    if heading > 30:
        turret_rotate = 'Q'
    elif heading < -30:
        turret_rotate = 'E'
    if enemy_detected:
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
    global final_destination
    global is_env_start

    with threading_lock:
        while True:
            random_coord = rng.integers(low=30, high=270, size=4)
            x = int(random_coord[0])
            z = int(random_coord[1])
            des_x = int(random_coord[2])
            des_z = int(random_coord[3])

            distance = np.sqrt((x - des_x) ** 2 + (z - des_z) ** 2)
            if (distance > 40) and (des_x > 5 and des_x < 295 and des_z > 5 and des_z < 295):
                break

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
    return jsonify(config)

@app.route('/start', methods=['GET'])
def start():

    return jsonify({"control": ""})



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5055, debug=True)