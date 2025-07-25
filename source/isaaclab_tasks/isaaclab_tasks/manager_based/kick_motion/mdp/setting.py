from __future__ import annotations
import torch
import math
# ([7, 8, 11, 12, 15, 16, 17, 18, 19, 20, 21, 22], 
#  ['Left_Hip_Pitch', 'Right_Hip_Pitch', 'Left_Hip_Roll', 'Right_Hip_Roll', 'Left_Hip_Yaw', 'Right_Hip_Yaw', 'Left_Knee_Pitch', 'Right_Knee_Pitch', 'Left_Ankle_Pitch', 'Right_Ankle_Pitch', 'Left_Ankle_Roll', 'Right_Ankle_Roll'])

LEG_IDS = [7, 8, 11, 12, 15, 16, 17, 18, 19, 20, 21, 22]
LEG_NAMES = [   'Left_Hip_Pitch', 
                'Right_Hip_Pitch', 
                'Left_Hip_Roll', 
                'Right_Hip_Roll', 
                'Left_Hip_Yaw', 
                'Right_Hip_Yaw', 
                'Left_Knee_Pitch', 
                'Right_Knee_Pitch',
                'Left_Ankle_Pitch', 
                'Right_Ankle_Pitch', 
                'Left_Ankle_Roll',
                'Right_Ankle_Roll']

LEG_JOINT_LIMIT = [
                    (-math.pi,math.pi), # Left_Hip_Pitch
                    (-math.pi,math.pi), # Right_Hip_Pitch
                    (-math.pi,math.pi), # Left_Hip_Roll
                    (-math.pi,math.pi), # Right_Hip_Roll
                    (-0.523599,math.pi), # Left_Hip_Yaw 内側は30度まで
                    (-math.pi,0.523599), # Right_Hip_Yaw 内側は30度まで
                    (-math.pi,math.pi), # Left_Knee_Pitch
                    (-0.087,math.pi),      # Right_Knee_Pitch 膝は逆に5度まで
                    (-0.087,math.pi),      # Left_Ankle_Pitch 膝は逆に5度まで
                    (-math.pi,math.pi), # Right_Ankle_Pitch
                    (-math.pi,math.pi), # Left_Ankle_Roll
                    (-math.pi,math.pi)  # Right_Ankle_Roll
                    ]

assert len(LEG_JOINT_LIMIT) == len(LEG_NAMES) == len(LEG_IDS), "Joint 設定が間違っている！"


ARM_IDS = [1, 2, 5, 6, 9, 10, 13, 14]

ARM_NAMES =  [  'Left_Shoulder_Pitch',
                'Right_Shoulder_Pitch',
                'Left_Shoulder_Roll',
                'Right_Shoulder_Roll',
                'Left_Elbow_Pitch',
                'Right_Elbow_Pitch',
                'Left_Elbow_Yaw', 
                'Right_Elbow_Yaw']

ARM_JOINT_LIMIT = [
                    (-math.pi,math.pi), # Left_Shoulder_Pitch
                    (-math.pi,math.pi), # Right_Shoulder_Pitch
                    (-1.2,0.0), # Left_Shoulder_Roll (-70 ~ 0)  # 0の状態が地面と平行
                    (0.0,1.2), # Right_Shoulder_Roll (0 ~ 70)
                    (-math.pi,math.pi), # Left_Elbow_Pitch
                    (-math.pi,math.pi), # Right_Elbow_Pitch
                    (-math.pi,math.pi), # Left_Elbow_Yaw
                    (-math.pi,math.pi), # Right_Elbow_Yaw
                    ]

assert len(ARM_IDS) == len(ARM_NAMES) == len(ARM_JOINT_LIMIT)

# タプルの最初がminimum,後ろがmaximum
def getLegJointLimits(device)-> tuple[torch.tensor,torch.tensor]:
    from operator import itemgetter
    
    min = torch.tensor(list(map(itemgetter(0),LEG_JOINT_LIMIT)) ,device=device)
    max = torch.tensor(list(map(itemgetter(1),LEG_JOINT_LIMIT)) ,device=device)
    return (min,max)

def getArmJointLimits(device)->tuple[torch.tensor,torch.tensor]:
    from operator import itemgetter
    min = torch.tensor(list(map(itemgetter(0),ARM_JOINT_LIMIT)) ,device=device)
    max = torch.tensor(list(map(itemgetter(1),ARM_JOINT_LIMIT)) ,device=device)
    return (min,max)