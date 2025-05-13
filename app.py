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

app = Flask(__name__)

# Segmentation 모델 선언
seg_model, image_processor = seg.init_model()

# 전차 크기 정의 (x: 5미터, z: 11미터)
VEHICLE_WIDTH = int(5.0)
VEHICLE_LENGTH = int(11.0)

# 월드 크기 정의
WORLD_SIZE = 300  # 300x300 미터

# 적 감지 여부
enemy_detected = False
detected_buffer = 0
destination_buffer = 0
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
    print(f'🔭 Detected Enemy : {enemy_detected}')
    image = request.files.get('image')
    if not image:
        return jsonify({"error": "No image received"}), 400

    image_path = 'temp_image.jpg'
    image.save(image_path)

    prediction = seg.predict_segmentation(image_path, seg_model, image_processor)
    seg.visualize_segmentation(image_path, prediction, latest_result)

    result = seg.get_vehicle_distance(seg_model, image_processor)
    enemy_list = result

    if result:
        enemy_detected = True
        detected_buffer = 0
        for i in result:
            id = i['id']
            distance = i['distance']
            piexles = i['pixels']
            print(f'🫡 ID {id} Distance: {distance} / Count: {piexles}')
    else:
        detected_buffer += 1
        if detected_buffer > 1:
            enemy_detected = False
            detected_buffer = 0
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
    data = request.get_json()
    try:
        shared_data.set_data(data)
        player_pos = data["playerPos"]
        x, z = float(player_pos["x"]), float(player_pos["z"])
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
    global enemy_list
    global destination_buffer
    if enemy_detected:
        data = shared_data.get_data()
        if enemy_list == None:
            print('Stop the tank')
            return jsonify({"move": "STOP"})
        enemies = len(enemy_list)
        if enemies == 1:
            # 사정거리 안에 있으면 그 자리에서 멈춰서 쏘자
            distance = enemy_list[0]['distance']
            if distance < 105:
                print('Stop the tank')
                return jsonify({"move": "STOP"})
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
                    if destination_buffer > 16:
                        destination_buffer = 0
                command = nav_controller.get_move()
                print(f'Moving Command: {command}')
                return jsonify(command)
        else:
            target_id = 0
            target_distance = 1000
            for i, enemy in enumerate(enemy_list):
                if enemy.get['distance'] < target_distance:
                    target_id = i
                        # 사정거리 안에 있으면 그 자리에서 멈춰서 쏘자
            distance = enemy_list[target_id]['distance']
            if distance < 100:
                print('Stop the tank')
                return jsonify({"move": "STOP"})
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
                    if destination_buffer > 16:
                        destination_buffer = 0
                command = nav_controller.get_move()
                print(f'Moving Command: {command}')
                return jsonify(command)
    else:
        command = nav_controller.get_move()
        print(f'Moving Command: {command}')
        return jsonify(command)

@app.route('/visualization', methods=['GET'])
def get_visualization():
    try:
        return send_file("path_visualization.html")
    except FileNotFoundError:
        return jsonify({"status": "ERROR", "message": "Visualization file not found. Please set a destination first."}), 404

@app.route('/get_action', methods=['GET'])
def get_action():
    global enemy_detected
    global turret_rotate
    global enemy_list
    data = shared_data.get_data()
    if enemy_detected:
        if enemy_list == None:
            return jsonify({"turret": "", "weight": 0.0})
        enemies = len(enemy_list)
        if enemies == 1:
            data['distance'] = enemy_list[0].get('distance')
            context = fire.Initialize(data)
            turret = fire.TurretControl(context)
            result = turret.normal_control()
            if result == None:
                return jsonify({"turret": "", "weight": 0.0})
            if len(result) == 2:
                command = {"turret": result[0], "weight": result[1]}
            else:
                command = {"turret": result}
            if command:
                print(f"🔫 Action Command: {command}")
                return jsonify(command)
            else:
                return jsonify({"turret": "", "weight": 0.0})
    else:
        turret_x = change_degree(data['playerTurretX'])
        body_x =  change_degree(data['playerBodyX'])
        heading = turret_x - body_x
        if heading > 45:
            turret_rotate = 'Q'
        elif heading < -45:
            turret_rotate = 'E'
        return jsonify({"turret": turret_rotate, "weight": 0.3})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5052, debug=True)