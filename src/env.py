import numpy as np

class RoadEnv:
    """Simple discrete-time road environment for RL.

    Observations (continuous):
      - position (m)
      - speed (m/s)
      - time (s)
    Actions (discrete): accelerate by -a_max, 0, +a_max

    Reward:
      -1 per time step (minimize time), +100 on reaching goal,
      -100 on illegal crossing (crossing on red) or constraint violation.
    """
    def __init__(self, distances, cycles, offsets, green_durations, v_max, a_max, dt=1.0):
        self.distances = list(distances)
        self.cycles = list(cycles)
        self.offsets = list(offsets)
        self.green_durations = list(green_durations)
        self.v_max = float(v_max)
        self.a_max = float(a_max)
        self.dt = float(dt)

        # cumulative distances including 0 at start
        self.cum_dist = np.concatenate(([0.0], np.cumsum(self.distances)))
        self.n_lights = len(self.distances)
        self.actions = np.array([-self.a_max, 0.0, self.a_max])

        self.reset()

    def reset(self):
        self.pos = 0.0
        self.vel = 0.0
        self.t = 0.0
        self.next_light = 0  # index of next traffic light to cross
        self.done = False
        return self._get_obs()

    def _get_obs(self):
        # distance to next light (or to goal if none)
        if self.next_light < self.n_lights:
            next_pos = self.cum_dist[self.next_light + 1]
        else:
            next_pos = self.cum_dist[-1]
        dist_to_next = max(0.0, next_pos - self.pos)
        phase = None
        if self.next_light < self.n_lights:
            C = self.cycles[self.next_light]
            phase = (self.t - self.offsets[self.next_light]) % C
        else:
            phase = 0.0
        return np.array([self.pos, self.vel, dist_to_next, phase], dtype=float)

    def _is_green(self, light_idx, abs_time):
        C = self.cycles[light_idx]
        start = self.offsets[light_idx]
        green_start = start % C
        green_end = (start + self.green_durations[light_idx]) % C
        phase = (abs_time) % C
        # handle wrap-around
        if green_start <= green_end:
            return green_start <= phase <= green_end
        else:
            return phase >= green_start or phase <= green_end

    def step(self, action_idx):
        if self.done:
            raise RuntimeError("step() called on terminated environment")
        acc = float(self.actions[int(action_idx)])
        # enforce acceleration limits
        acc = np.clip(acc, -self.a_max, self.a_max)

        # update speed and position (simple Euler integration)
        new_vel = self.vel + acc * self.dt
        # constraint violations
        if new_vel < -1e-6 or new_vel > self.v_max + 1e-6:
            # illegal speed
            self.done = True
            return self._get_obs(), -100.0, True, {"reason": "speed_violation"}
        new_vel = np.clip(new_vel, 0.0, self.v_max)
        new_pos = self.pos + new_vel * self.dt
        new_time = self.t + self.dt

        reward = -1.0 * self.dt  # penalize time
        info = {}

        # check for crossing any lights between old pos and new_pos
        crossed = False
        while self.next_light < self.n_lights and new_pos >= self.cum_dist[self.next_light + 1]:
            crossed = True
            crossing_time = new_time  # approximate with end of step
            if not self._is_green(self.next_light, crossing_time):
                # illegal crossing on red
                self.pos = min(new_pos, self.cum_dist[self.next_light + 1])
                self.vel = 0.0
                self.t = new_time
                self.done = True
                return self._get_obs(), -100.0, True, {"reason": "red_violation", "light": self.next_light}
            # successful crossing: advance next_light
            self.next_light += 1

        # update state
        self.pos = new_pos
        self.vel = new_vel
        self.t = new_time

        # goal check
        if self.pos >= self.cum_dist[-1]:
            self.done = True
            reward += 100.0  # success bonus
            return self._get_obs(), reward, True, {"reason": "goal", "time": self.t}

        return self._get_obs(), reward, False, info

    def render(self):
        print(f"t={self.t:.1f}s pos={self.pos:.1f}m vel={self.vel:.2f}m/s next_light={self.next_light}")


if __name__ == "__main__":
    # simple smoke test
    env = RoadEnv([300.0], [60.0], [10.0], [20.0], v_max=20.0, a_max=2.0)
    s = env.reset()
    done = False
    while not done:
        s, r, done, info = env.step(1)  # maintain
    print('done', done, 'info', info)
