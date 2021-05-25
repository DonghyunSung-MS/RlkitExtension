import mujoco_py
import numpy as np
from rlkit.envs.mujoco_env import MujocoEnv
import mujoco_py

class Mani2dEnv(MujocoEnv):
    def __init__(self):
        self.init_serialization(locals())
        xml_path = "2dmanipulator.xml"

        super().__init__(
            xml_path,
            frame_skip=5,
            automatically_set_obs_and_action_space=True,
        )
    def step(self, a):
        cur_obs = self._get_obs()
        dist = np.linalg.norm(cur_obs[7:9] - cur_obs[-2:])
        
        reward = - dist
        # reward = 1.0 if dist < 0.01 else 0.0

        self.do_simulation(a, self.frame_skip)


        return self._get_obs(), reward, False, {}

    def _get_obs(self):
        qpos = self.sim.data.qpos.copy()
        qvel = self.sim.data.qvel.copy()

        robot_obs = np.hstack([qpos[:3], qvel[:3]])
        obj_obs = np.hstack([qpos[4:], qvel[4:]])
        target_obs = self.sim.data.site_xpos[0, :2]
        return np.concatenate([
            robot_obs,
            obj_obs,
            # target_obs,
        ])

    def reset_model(self):
        target_xy = list(self.np_random.uniform(low=np.array([-0.2, -0.2]), high=np.array([0.2, 0.2])))
        site_pos = self.model.body_pos
        site_pos = np.array(site_pos)
        site_pos[-1, :] = np.array([target_xy + [0.01]])
        self.model.body_pos[:] = site_pos

        high = np.array([np.pi, np.pi/2.0, np.pi/2.0, np.pi/2.0, 0.2, 0.2])
        qpos = self.init_qpos + self.np_random.uniform(low=-high, high=high)
        qvel = self.init_qvel #+ self.np_random.randn(self.model.nv) * .1

        self.set_state(qpos, qvel)
        return self._get_obs()

if __name__=="__main__":
    env = Mani2dEnv()
    print(env.observation_space.shape)
    # env.reset()
    while True:
        env.render()
        print(env.step([1., 0.0] + [np.random.randn()])[1])