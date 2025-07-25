from __future__ import annotations

import torch
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class BallTouchManager:
    def __init__(self):
        self.sim_dt : int = 0
        self.contact_tracking : torch.tensor = None 
        self.current_step : int = 0
        self.initialized : bool = False 

    def initialize(self,env:ManagerBasedRLEnv):
        if not self.initialized:
            self.contact_tracking = torch.zeros(env.num_envs,device=env.device)
            self.sim_dt = env.step_dt
            self.initialized = True
        else:
            raise RuntimeError("Initialized over twice!!!")

    def __getResult(self,touch_offset_step:int):
        return self.contact_tracking <= (self.current_step + touch_offset_step)

    # (num_envs)のBool tensorを返す。これはその環境がそのエピソードでボールと接触したかどうかを表す。毎step呼ばれる事を想定している。
    def getTouchState(self,env:ManagerBasedRLEnv,touch_offset_step:int = 0 ) -> torch.tensor:
        if not self.initialized:
            raise RuntimeError("BallTouchManager:: Use before initialized!!!")
        # 既にそのステップで更新を行っている場合は記録した結果を返す
        if self.current_step == env.common_step_counter:
            return self.__getResult(touch_offset_step)
        else: # 更新を行う
            self.current_step = env.common_step_counter
            MAX_STEP = 90000000 # 9000万ステップが代入されている所はまだ衝突していない
            # エピソードがリセットされた環境はリセットする
            self.contact_tracking = torch.where(env.termination_manager.terminated,MAX_STEP,self.contact_tracking)
            contact_threshould = 0.5
            sensor_right = env.scene.sensors["contact_balls_right"]
            sensor_left = env.scene.sensors["contact_balls_left"]
            force_right = torch.norm(sensor_right.data.force_matrix_w[:,0,0],p=2,dim=1)
            force_left = torch.norm(sensor_left.data.force_matrix_w[:,0,0],p=2,dim=1)
            has_contact = (force_right > contact_threshould) | (force_left > contact_threshould)
            # 今回ステップで衝突があった環境に今のステップを記録
            self.contact_tracking = torch.where(has_contact,self.current_step,self.contact_tracking)
            return self.__getResult(touch_offset_step)

_balltouch_manager = BallTouchManager() 

def getTouchState(env:ManagerBasedRLEnv,touch_offset_step:int = 0) -> torch.tensor:
    global _balltouch_manager
    if not _balltouch_manager.initialized :
        _balltouch_manager.initialize(env)
    
    return _balltouch_manager.getTouchState(env,touch_offset_step)