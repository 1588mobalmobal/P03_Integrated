import math
import time
from utils import shared_data

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # 벡터 크기 정하기
    def magnitude(self): return math.sqrt(self.x**2 + self.y**2)

    # 벡터 정규화
    def normalize(self):
        mag = self.magnitude()
        if mag == 0:
            return Vector(0, 0)
        return Vector(self.x / mag, self.y / mag)

class Initialize:
    EFFECTIVE_MAX_RANGE = 115.8  # Unit: meters
    EFFECTIVE_MIN_RANGE = 21.002 # Unit: meters
    BULLET_VELOCITY = 42.6  # Unit: meters/second

    def __init__(self, data=None):
        if data is None:
            data = {
                "enemyPos": {"x": 0, "y": 0, "z": 0},
                "playerPos": {"x": 0, "y": 0, "z": 0},
                "distance": 115,
                "enemySpeed": 0,
                "playerSpeed": 0,
                "time": 0,
                "enemyBodyX": 0,
                "playerTurretX": 0,
                "playerTurretY":0
            }
        self.shared_data = data
        self.turret_tolerance = 0.0110  # Unit: degrees; will be dynamically assigned later
        self.barrel_tolerance = 0.0110  # Unit: degrees; will be dynamically assigned later, too
        self.input_key_value = {
            "getRight": "E", "getLeft": "Q",
            "getRise": "R", "getFall": "F", "getFire": "FIRE"
        }

# 평면에서의 탄속 고려려 (42.6 m/s)
class Ballistics:
    def __init__(self, context):
        self.context = context

    def _calculation_of_barrel_angle_by_distance(self):
        # 원 회귀식; y=0.373x2+5.914x+41.24; y: distance, x: barrel_degree
        # 적과의 거리가 사정거리 내인지 확인할 것것
        distance = self.context.shared_data["distance"]
        if self.context.EFFECTIVE_MIN_RANGE <= distance <= self.context.EFFECTIVE_MAX_RANGE:
            # 포신 각도를 회귀식을 통해 구하기기
            # if not (20.995 <= distance <= 137.68):
            #     raise ValueError("Distance is outside the inverse function's domain [20.995, 137.68].")

            # 원 회귀식의 역함수
            discriminant = 1.492 * distance - 24.264784
            # if discriminant < 0:
            #     raise ValueError("Discriminant is negative. No real solutions exist.")

            barrel_angle_deg = (-5.914 + math.sqrt(discriminant)) / 0.746  # In degrees
            # if not (-5.0 + 1e-6 <= barrel_angle_deg <= 10.0 + 1e-6):
            #     raise ValueError("Calculated barrel angle is outside the range [-5, 10].")

            # Convert barrel angle to radians (for error calculation)
            barrel_angle = barrel_angle_deg * math.pi / 180

            # Calculate barrel angle error
            current_turret_angle_rad = self.context.shared_data["playerTurretY"] * math.pi / 180
            barrel_angle_error = current_turret_angle_rad - barrel_angle
            barrel_angle_error = math.atan2(math.sin(barrel_angle_error), math.cos(barrel_angle_error))

            return barrel_angle, barrel_angle_error
        else:
            # raise ValueError("Distance exceeds effective range")
            print('Out of the range')
            return 0, 0

class AimingBehavior:
    def __init__(self, context):
        self.context = context
        self.ballistics = Ballistics(context)

    def _calculate_turret_angle(self):
        goal_vector = Vector(
            self.context.shared_data["enemyPos"]["x"] - self.context.shared_data["playerPos"]["x"],
            self.context.shared_data["enemyPos"]["z"] - self.context.shared_data["playerPos"]["z"]
        )
        # print(goal_vector.x, goal_vector.y)
        goal_vector = goal_vector.normalize()
        # print(f"🎯 Goal Vector: ({goal_vector.x}, {goal_vector.y})")  # 목표 벡터 출력

        deg =  (math.atan2(goal_vector.x, goal_vector.y))*180/math.pi
        # print("deg: ", deg)
        goal_heading = (math.atan2(goal_vector.x, goal_vector.y) - math.pi / 2 )+ 1.5707
        player_heading_to_radians = self.context.shared_data["playerTurretX"] * math.pi / 180
        heading_error = goal_heading - player_heading_to_radians
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
        # print(f"🧭 Goal Heading: {goal_heading}, Player Heading: {player_heading_to_radians}, Heading Error: {heading_error}")  # 헤딩 정보 출력

        return goal_vector, heading_error

    def control_information(self):
        goal_vector, heading_error = self._calculate_turret_angle()
        barrel_angle, barrel_angle_error = self.ballistics._calculation_of_barrel_angle_by_distance()
        return goal_vector, heading_error, barrel_angle, -barrel_angle_error

class TurretControl:
    def __init__(self, context):
        self.context = context
        self.previous_play_time = 0
        self.aiming_behavior = AimingBehavior(context)
        self.target_vector, self.heading_error, self.barrel_angle, self.barrel_angle_error = self.aiming_behavior.control_information()

    def normal_control(self):
            # print(f"⏰ Previous Time: {self.previous_play_time}, Current Time: {self.context.shared_data['time']}")
            if self.previous_play_time < self.context.shared_data["time"]:
                self.target_vector, self.heading_error, self.barrel_angle, self.barrel_angle_error = self.aiming_behavior.control_information()
                # print(f"🔄 Updated - Heading Error: {self.heading_error}, Barrel Angle Error: {self.barrel_angle_error}")
                turret_weight = min(max(abs(self.heading_error) / math.pi, 0.2), 1)
                barrel_weight = min(max(abs(self.barrel_angle_error) / math.pi, 0.5), 1)
                # print(f"⚖️ Turret Weight: {turret_weight}, Barrel Weight: {barrel_weight}")
                if abs(self.heading_error) > self.context.turret_tolerance:
                    direction = "getRight" if self.heading_error > 0 else "getLeft"
                    # print(f"🛠️ Command: {direction}, Weight: {turret_weight}")
                    # 시뮬레이션: 방향 업데이트 (예: 1도/초 회전)
                    rotation_speed = 1.0  # 도/초
                    if direction == "getLeft":
                        self.context.shared_data["playerTurretX"] -= rotation_speed
                    else:
                        self.context.shared_data["playerTurretX"] += rotation_speed
                    shared_data.set_data(self.context.shared_data)  # 이제 shared_data 사용 가능
                    return self.context.input_key_value[direction], turret_weight
                elif abs(self.heading_error) <= self.context.turret_tolerance and self.context.EFFECTIVE_MIN_RANGE <= \
                    self.context.shared_data["distance"] <= self.context.EFFECTIVE_MAX_RANGE:
                    if abs(self.barrel_angle_error) > self.context.barrel_tolerance:
                        direction = "getRise" if self.barrel_angle_error > 0 else "getFall"
                        # print(f"🛠️ Command: {direction}, Weight: {barrel_weight}")
                        return self.context.input_key_value[direction], barrel_weight
                    else:
                        direction = "getFire"
                        # print(f"🛠️ Command: {direction}")
                        return self.context.input_key_value[direction]
                self.previous_play_time = self.context.shared_data["time"]
            # print("⏭️ No update, returning None")
            return None

# if __name__ == "__main__":
    # print(time.time())
    # context = Initialize()
    # turret = TurretControl(context)
    # print(id(context.shared_data) == id(turret.context.shared_data))
    # result = turret.normal_control()
    # print(result)
    # print(time.time())