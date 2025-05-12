from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import time
import math
import heapq
import random
import numpy as np
import plotly.graph_objects as go

# 전차 크기 정의 (x: 5미터, z: 11미터)
VEHICLE_WIDTH = int(5.0)
VEHICLE_LENGTH = int(11.0)

# 월드 크기 정의
WORLD_SIZE = 300  # 300x300 미터

class Node:
    def __init__(self, grid_x, grid_z):
        self.grid_x = grid_x
        self.grid_z = grid_z
        self.g_cost = 0
        self.h_cost = 0
        self.parent = None
        self.is_obstacle = False

    @property
    def f_cost(self):
        return self.g_cost + self.h_cost

    def __lt__(self, other):
        return self.f_cost < other.f_cost

class Grid:
    def __init__(self, width=WORLD_SIZE, height=WORLD_SIZE):
        self.width = width
        self.height = height
        self.grid = [[Node(x, z) for z in range(height)] for x in range(width)]
        self.original_obstacles = []  # 원래 좌표 저장용 리스트

    def node_from_world_point(self, world_x, world_z):
        grid_x = max(0, min(int(world_x), self.width - 1))
        grid_z = max(0, min(int(world_z), self.height - 1))
        return self.grid[grid_x][grid_z]

    def set_obstacle(self, x_min, x_max, z_min, z_max):
        # 원래 좌표 저장
        self.original_obstacles.append({
            "x_min": x_min,
            "x_max": x_max,
            "z_min": z_min,
            "z_max": z_max
        })

        # A* 경로 탐색을 위한 확장 적용
        EXTENSION = 10
        x_min_ext = max(0, min(int(x_min) - EXTENSION, self.width - 1))
        x_max_ext = max(0, min(int(x_max) + EXTENSION, self.width - 1))
        z_min_ext = max(0, min(int(z_min) - EXTENSION, self.height - 1))
        z_max_ext = max(0, min(int(z_max) + EXTENSION, self.height - 1))
        for x in range(x_min_ext, x_max_ext + 1):
            for z in range(z_min_ext, z_max_ext + 1):
                self.grid[x][z].is_obstacle = True

    def get_neighbors(self, node):
        neighbors = []
        # 8방향 탐색 (상하좌우 + 대각선)
        for dx, dz in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            new_x, new_z = node.grid_x + dx, node.grid_z + dz
            if 0 <= new_x < self.width and 0 <= new_z < self.height:
                neighbor = self.grid[new_x][new_z]
                if not neighbor.is_obstacle:
                    # 대각선 이동 비용은 √2로 증가
                    move_cost = 10 if dx == 0 or dz == 0 else 14
                    neighbors.append((neighbor, move_cost))
        return neighbors

class Pathfinding:
    def find_path(self, start_pos, target_pos, grid):
        start_node = grid.node_from_world_point(start_pos[0], start_pos[1])
        target_node = grid.node_from_world_point(target_pos[0], target_pos[1])
        
        if start_node.is_obstacle or target_node.is_obstacle:
            print("Warning: Start or target position is on an obstacle.")
            return []

        # 전차 크기(5m x 11m) 고려
        half_width = VEHICLE_WIDTH // 2
        half_length = VEHICLE_LENGTH // 2
        for dx in range(-half_width, half_width + 1):
            for dz in range(-half_length, half_length + 1):
                check_x = start_node.grid_x + dx
                check_z = start_node.grid_z + dz
                if (0 <= check_x < grid.width and 0 <= check_z < grid.height and
                    grid.grid[check_x][check_z].is_obstacle):
                    print("Warning: Start position is near an obstacle.")
                    return []

        open_set = []
        heapq.heappush(open_set, (start_node.f_cost, id(start_node), start_node))
        open_set_nodes = {start_node}
        closed_set = set()

        while open_set:
            _, _, current_node = heapq.heappop(open_set)
            open_set_nodes.remove(current_node)
            closed_set.add(current_node)

            if current_node.grid_x == target_node.grid_x and current_node.grid_z == target_node.grid_z:
                return self.retrace_path(start_node, current_node)

            for neighbor, move_cost in grid.get_neighbors(current_node):
                if neighbor in closed_set:
                    continue
                new_cost = current_node.g_cost + move_cost
                if new_cost < neighbor.g_cost or neighbor not in open_set_nodes:
                    neighbor.g_cost = new_cost
                    # 유클리드 거리 기반 휴리스틱
                    dx_h = neighbor.grid_x - target_node.grid_x
                    dz_h = neighbor.grid_z - target_node.grid_z
                    neighbor.h_cost = math.sqrt(dx_h * dx_h + dz_h * dz_h) * 10
                    neighbor.parent = current_node
                    if neighbor not in open_set_nodes:
                        open_set_nodes.add(neighbor)
                        heapq.heappush(open_set, (neighbor.f_cost, id(neighbor), neighbor))
        return []

    def retrace_path(self, start_node, end_node):
        path = []
        current_node = end_node
        while current_node != start_node:
            path.append(current_node)
            current_node = current_node.parent
        path.append(start_node)
        path.reverse()
        return [(node.grid_x, node.grid_z) for node in path]

# 제어 관련 클래스
@dataclass
class NavigationConfig:
    MOVE_STEP: float = 0.1
    TOLERANCE: float = 15.0
    LOOKAHEAD_MIN: float = 1.0
    LOOKAHEAD_MAX: float = 10.0
    HEADING_SMOOTHING: float = 0.8
    STEERING_SMOOTHING: float = 0.7
    GOAL_WEIGHT: float = 2.0
    SLOW_RADIUS: float = 50.0
    MAX_SPEED: float = 1.0
    MIN_SPEED: float = 0.1
    SPEED_FACTOR: float = 0.8
    WEIGHT_FACTORS: Dict[str, float] = None
    WAYPOINT_OFFSET: float = 35

    def __post_init__(self):
        if self.WEIGHT_FACTORS is None:
            self.WEIGHT_FACTORS = {"D": 0.6, "A": 0.6, "W": 0.5, "S": 0.5}

class NavigationController:
    def __init__(self, config: NavigationConfig, pathfinding: Pathfinding, grid: Grid):
        self.config = config
        self.pathfinding = pathfinding
        self.grid = grid
        self.current_position: Optional[Tuple[float, float]] = None
        self.current_heading: float = 0.0
        self.destination: Optional[Tuple[float, float]] = None
        self.last_command: Optional[str] = None
        self.last_steering: float = 0.0
        self.last_update_time: float = time.time()
        self.initial_distance: Optional[float] = None
        self.waypoints: List[Tuple[float, float]] = []
        self.current_waypoint_idx: int = 0
        self.completed: bool = False
        self.actual_path: List[Tuple[float, float]] = []  # 실제 이동 경로 저장

    def update_position(self, position: str) -> Dict:
        try:
            x, y, z = map(float, position.split(","))
            new_position = (x, z)
            now = time.time()
            dt = now - self.last_update_time
            self.last_update_time = now

            if self.current_position:
                prev_x, prev_z = self.current_position
                dx, dz = x - prev_x, z - prev_z
                distance_moved = math.sqrt(dx**2 + dz**2)
                if distance_moved > 0.01:
                    new_heading = math.atan2(dx, dz)
                    self.current_heading = (
                        self.config.HEADING_SMOOTHING * self.current_heading +
                        (1 - self.config.HEADING_SMOOTHING) * new_heading
                    )
                    self.current_heading = math.atan2(
                        math.sin(self.current_heading), math.cos(self.current_heading)
                    )

            self.current_position = new_position
            # 실제 이동 경로에 현재 위치 추가
            self.actual_path.append(new_position)
            # 시각화 갱신
            self.visualize_path()
            return {
                "status": "OK",
                "current_position": self.current_position,
                "heading": math.degrees(self.current_heading)
            }
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}

    def set_destination(self, destination: str) -> Dict:
        try:
            x, y, z = map(float, destination.split(","))
            x = max(0, min(x, WORLD_SIZE))
            z = max(0, min(z, WORLD_SIZE))
            self.destination = (x, z)
            # 목적지 설정 시 실제 경로 초기화
            self.actual_path = []
            if self.current_position:
                self.actual_path.append(self.current_position)  # 시작 위치 추가
                # A* 경로 탐색 호출
                self.waypoints = self.pathfinding.find_path(self.current_position, self.destination, self.grid)
                self.current_waypoint_idx = 0
                self.completed = False
                if self.waypoints:
                    self.destination = self.waypoints[0]
                else:
                    self.destination = None
                    self.completed = True
                curr_x, curr_z = self.current_position
                self.initial_distance = math.sqrt((x - curr_x) ** 2 + (z - curr_z) ** 2)
            print(f"Waypoints set: {self.waypoints}")
            # 시각화 호출
            self.visualize_path()
            return {
                "status": "OK",
                "destination": {"x": x, "y": y, "z": z},
                "initial_distance": self.initial_distance,
                "waypoints": self.waypoints,
                "visualization_url": "http://127.0.0.1:5000/visualization"
            }
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}

    def _calculate_lookahead(self, distance: float) -> float:
        return min(
            self.config.LOOKAHEAD_MAX,
            max(self.config.LOOKAHEAD_MIN, distance * 0.5 + 5.0)
        )

    def _calculate_steering(self, curr_x: float, curr_z: float, dest_x: float, dest_z: float, lookahead_distance: float) -> Tuple[float, float]:
        goal_vector = np.array([dest_x - curr_x, dest_z - curr_z])
        goal_distance = np.linalg.norm(goal_vector)
        if goal_distance > 0:
            goal_vector = goal_vector / goal_distance

        target_vector = goal_vector * self.config.GOAL_WEIGHT
        target_vector_norm = np.linalg.norm(target_vector)
        if target_vector_norm > 0:
            target_vector = target_vector / target_vector_norm
            target_heading = math.atan2(target_vector[0], target_vector[1])
        else:
            target_heading = math.atan2(goal_vector[0], goal_vector[1])

        lookahead_x = curr_x + target_vector[0] * lookahead_distance
        lookahead_z = curr_z + target_vector[1] * lookahead_distance
        dx, dz = lookahead_x - curr_x, lookahead_z - curr_z
        target_heading = math.atan2(dx, dz)

        heading_error = target_heading - self.current_heading
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
        curvature = 2.0 * math.sin(heading_error) / max(lookahead_distance, 0.01)
        steering = (
            self.config.STEERING_SMOOTHING * self.last_steering +
            (1 - self.config.STEERING_SMOOTHING) * curvature
        )
        return steering, heading_error

    def _calculate_speed(self, distance: float, abs_steering: float) -> float:
        base_speed = self.config.MAX_SPEED - abs_steering * self.config.SPEED_FACTOR
        if distance < self.config.SLOW_RADIUS * 0.5:
            speed = self.config.MAX_SPEED * distance / (self.config.SLOW_RADIUS * 0.5)
        else:
            speed = base_speed
        return max(self.config.MIN_SPEED, min(self.config.MAX_SPEED, speed))

    def _calculate_weights(self, steering: float, speed: float, heading_error: float, progress: float) -> Dict[str, float]:
        abs_steering = abs(steering)
        dynamic_weights = {
            "D": self.config.WEIGHT_FACTORS["D"] * (1 + abs_steering * 2) if steering > 0 else 0.0,
            "A": self.config.WEIGHT_FACTORS["A"] * (1 + abs_steering * 2) if steering < 0 else 0.0,
            "W": self.config.WEIGHT_FACTORS["W"] * speed,
            "S": self.config.WEIGHT_FACTORS["S"] if heading_error > math.pi * 0.6 else 0.0
        }
        for cmd in dynamic_weights:
            if dynamic_weights[cmd] > 0:
                dynamic_weights[cmd] *= (1 + progress * 0.5)
        return dynamic_weights

    def _update_position(self, speed: float, command: str, curr_x: float, curr_z: float) -> None:
        move_distance = self.config.MOVE_STEP * speed
        new_x, new_z = curr_x, curr_z
        if command == "D":
            new_x += move_distance * math.cos(self.current_heading + math.pi/2)
            new_z += move_distance * math.sin(self.current_heading + math.pi/2)
        elif command == "A":
            new_x += move_distance * math.cos(self.current_heading - math.pi/2)
            new_z += move_distance * math.sin(self.current_heading - math.pi/2)
        elif command == "W":
            new_x += move_distance * math.sin(self.current_heading)
            new_z += move_distance * math.cos(self.current_heading)
        elif command == "S":
            new_x -= move_distance * math.sin(self.current_heading)
            new_z -= move_distance * math.cos(self.current_heading)
        self.current_position = (new_x, new_z)
        # 실제 이동 경로에 업데이트된 위치 추가
        self.actual_path.append(self.current_position)
        # 시각화 갱신
        self.visualize_path()

    def get_move(self) -> Dict:
        if self.current_position is None or self.completed:
            return {"move": "STOP", "weight": 1.0, "current_waypoint": self.current_waypoint_idx, "completed": self.completed}

        if self.destination is None and self.waypoints:
            self.destination = self.waypoints[self.current_waypoint_idx]

        if not self.destination:
            return {"move": "STOP", "weight": 1.0, "current_waypoint": self.current_waypoint_idx, "completed": self.completed}

        curr_x, curr_z = self.current_position
        dest_x, dest_z = self.destination
        distance = math.sqrt((dest_x - curr_x) ** 2 + (dest_z - curr_z) ** 2)

        if distance < self.config.TOLERANCE:
            if self.current_waypoint_idx == len(self.waypoints) - 1:
                self.completed = True
                self.destination = None
                self.initial_distance = None
                return {
                    "move": "STOP",
                    "weight": 1.0,
                    "current_waypoint": self.current_waypoint_idx,
                    "completed": self.completed
                }
            else:
                self.current_waypoint_idx += 1
                self.destination = self.waypoints[self.current_waypoint_idx]
                self.initial_distance = None
                dest_x, dest_z = self.destination
                distance = math.sqrt((dest_x - curr_x) ** 2 + (dest_z - curr_z) ** 2)

        lookahead_distance = self._calculate_lookahead(distance)
        steering, heading_error = self._calculate_steering(curr_x, curr_z, dest_x, dest_z, lookahead_distance)
        self.last_steering = steering
        speed = self._calculate_speed(distance, abs(steering))

        progress = max(0, 1 - distance / self.initial_distance) if self.initial_distance and distance > 0 else 0.0
        dynamic_weights = self._calculate_weights(steering, speed, heading_error, progress)

        commands = [cmd for cmd, w in dynamic_weights.items() if w > 0]
        if not commands:
            return {"move": "STOP", "weight": 1.0, "current_waypoint": self.current_waypoint_idx, "completed": self.completed}

        weights = [dynamic_weights[cmd] for cmd in commands]
        chosen_cmd = random.choices(commands, weights=weights, k=1)[0]
        self.last_command = chosen_cmd

        if chosen_cmd:
            self._update_position(speed, chosen_cmd, curr_x, curr_z)

        return {
            "move": chosen_cmd,
            "weight": dynamic_weights[chosen_cmd],
            "current_waypoint": self.current_waypoint_idx,
            "completed": self.completed
        }

    def visualize_path(self):
        if not self.current_position:
            print("Visualization skipped: No current position available.")
            return

        # Plotly Figure 생성
        fig = go.Figure()

        # 장애물 시각화 (원래 좌표를 사용해 사각형으로 표시)
        for i, obstacle in enumerate(self.grid.original_obstacles):
            x_min = obstacle["x_min"]
            x_max = obstacle["x_max"]
            z_min = obstacle["z_min"]
            z_max = obstacle["z_max"]
            # 사각형의 4개 꼭짓점 정의 (시계 방향)
            x_coords = [x_min, x_max, x_max, x_min, x_min]
            z_coords = [z_min, z_min, z_max, z_max, z_min]
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=z_coords,
                mode='lines',
                fill='toself',
                fillcolor='black',
                line=dict(color='black'),
                opacity=0.6,
                name=f'Obstacle {i+1}'
            ))

        # A* 경로 시각화 (파란 선)
        if self.waypoints:
            path_x = [point[0] for point in self.waypoints]
            path_z = [point[1] for point in self.waypoints]
            fig.add_trace(go.Scatter(x=path_x, y=path_z, mode='lines', line=dict(color='blue'), name='A* Path'))

        # 실제 이동 경로 시각화 (초록 선)
        if self.actual_path:
            actual_x = [point[0] for point in self.actual_path]
            actual_z = [point[1] for point in self.actual_path]
            fig.add_trace(go.Scatter(x=actual_x, y=actual_z, mode='lines', line=dict(color='green'), name='Actual Path'))

        # 전차 위치 (빨간 화살표)
        curr_x, curr_z = self.current_position
        fig.add_trace(go.Scatter(
            x=[curr_x],
            y=[curr_z],
            mode='markers+text',
            marker=dict(color='red', size=10, symbol='arrow', angle=math.degrees(self.current_heading)),
            text=['Tank'],
            textposition="top right",
            name='Tank'
        ))

        # 최종 목적지 (빨간 별)
        if self.waypoints:
            final_goal = self.waypoints[-1]
            fig.add_trace(go.Scatter(x=[final_goal[0]], y=[final_goal[1]], mode='markers', marker=dict(color='red', size=10, symbol='star'), name='Final Goal'))

        # 레이아웃 설정
        fig.update_layout(
            title='Path Visualization',
            xaxis_title='X',
            yaxis_title='Z',
            xaxis=dict(range=[0, WORLD_SIZE]),
            yaxis=dict(range=[0, WORLD_SIZE]),
            showlegend=True,
            width=800,
            height=800
        )

        # HTML 파일로 저장 (Plotly JS를 포함)
        fig.write_html("path_visualization.html", include_plotlyjs='cdn', full_html=True)
        # print("Visualization saved as path_visualization.html")