import os

from learned_model import *
from behaviors import *
from type import *


def create_agent_from_dict(data):
    idx = data["id"]
    x, y, v, yaw, a, l, w = data["x"], data["y"], data["v"], data["yaw"], data["a"], data["l"], data["w"]
    idm_p = IdmParameter(data["idm_p"]["acc_max"], data["idm_p"]["acc_min"],
                         data["idm_p"]["min_dis"], data["idm_p"]["acc_com"],
                         data["idm_p"]["thw"], data["idm_p"]["vd"])
    mobil_p = MOBILParameter(data["mobil_param"]["politeness_factor"],
                             data["mobil_param"]["delta_a"],
                             data["mobil_param"]["bias_a"])
    # yielding_p = data["yielding_param"]
    yielding_p = np.array([1.70968573, 0.00582201, 0.40742229, 1.02390871, -0.32091885, 1.48127062, -0.69499426,
                           -0.34547462, -0.56578542, -1.911662, -0.03998742, 0.02613949, -0.02392805])
    behavior_model = data["behavior_model"]
    if "lane_change" in behavior_model:
        behavior_model = "idm_mobil_lane_change"
    change_intention_threshold = data["change_intention_threshold"]
    random_yielding_seed = data["random_yielding_seed"]
    agent = Agent(idx, x, y, v, yaw, a, l, w, idm_p, yielding_p, mobil_p,
                  behavior_model,
                  change_intention_threshold,
                  random_yielding_seed)
    if "vy" in data.keys():
        vy = data["vy"]
        agent.set_lateral_velocity(vy)
    return agent


def create_agents_from_dicts(agents_dicts):
    copied_agents = []
    for a in agents_dicts:
        copied_agents.append(create_agent_from_dict(a))
    return copied_agents


def create_dicts_from_agents(agents):
    dicts = []
    for a in agents:
        dicts.append(a.to_dict())
    return dicts


class Agent:
    def __init__(self, idx, x, y, v, yaw, a, l, w, idm_param, yielding_param, mobil_param,
                 behavior_model,
                 change_intention_threshold=0.625,
                 random_yielding_seed=None):
        # yaw in degree
        self.id = idx
        self.x = x
        self.y = y
        self.v = v
        self.vy = 0.
        self.yaw = np.deg2rad(yaw)
        self.a = a
        self.l = l
        self.w = w
        self.corridor_history = []
        self.acceleration_history = [a]
        self.state_history = [[x, y, v, a]]
        self.hull = vehicle_pose_to_rect(x, y, yaw, l, w)
        self.polygon = Polygon(self.hull)
        self.rss_param = RSSParameter()  # TODO: for lane change mobil vehicles use aggressive rss param
        self.idm_param = idm_param
        self.yielding_param = yielding_param
        self.mobil_param = mobil_param
        self.behavior_model = behavior_model  # "idm", "merging"
        self.behavior_model_history = [behavior_model]
        self.change_intention_threshold = change_intention_threshold
        if random_yielding_seed and random_yielding_seed >= 0:
            self.random_yielding_seed = random_yielding_seed
        else:
            self.random_yielding_seed = int(random.random() * 100)
        self.yielding_model = YieldingModel(yielding_param) if yielding_param is not None else YieldingModel()
        # track the yielding intention: {id: YieldingIntention, ...}
        self.yielding_intention_to_others = {}
        if "merging" in self.behavior_model or "exit" in self.behavior_model:
            self.learned_merging_model = LearnedMergingModel()
            self.merging_flag = MergingFlag()
            self.exit_flag = ExitFlag()
        if self.behavior_model == "lane_change_learned":
            self.learned_lane_change_model = LearnedLaneChangeModel()
            self.current_decision = "Initial"
        if "lane_change" in self.behavior_model:
            self.rss_param = RSSParameter(0.4, 0.7, -6, -6, 1.8, 1.2)
        # for mobil lane change model
        self.commit_lane_change = "not_decided"
        self.commit_keep_lane_step = 0
        self.target_lane_id = -1

        # for MIQP planner
        self.left_leading_id = -1
        self.left_following_id = -1
        self.select_new_target_vehicles = False

    def get_position(self):
        return [self.x, self.y]

    def set_lateral_velocity(self, vy):
        self.vy = vy

    def set_behavior_model(self, model):
        self.behavior_model = model
        if "merging" in self.behavior_model or "exit" in self.behavior_model:
            self.learned_merging_model = LearnedMergingModel()
            self.merging_flag = MergingFlag()
            self.exit_flag = ExitFlag()
        if self.behavior_model == "lane_change_learned":
            self.learned_lane_change_model = LearnedLaneChangeModel()
            self.current_decision = "Initial"
        if "lane_change" in self.behavior_model:
            self.rss_param = RSSParameter(0.4, 0.7, -6, -6, 1.8, 1.2)
        self.behavior_model_history.append(model)

    def reset_behavior_model(self, model):
        self.set_behavior_model(model)
        self.behavior_model_history = [model]

    def to_dict(self):
        out = {"id": int(self.id), "x": float(self.x), "y": float(self.y), "v": float(self.v), "vy": float(self.vy),
               "yaw": float(np.rad2deg(self.yaw)), "a": float(self.a), "l": float(self.l), "w": float(self.w),
               "idm_p": {"acc_max": float(self.idm_param.acc_max),
                         "acc_min": float(self.idm_param.acc_min),
                         "min_dis": float(self.idm_param.min_dis),
                         "acc_com": float(self.idm_param.acc_com),
                         "thw": float(self.idm_param.thw),
                         "vd": float(self.idm_param.vd)},
               "mobil_param": {"politeness_factor": float(self.mobil_param.politeness_factor),
                               "delta_a": float(self.mobil_param.delta_a),
                               "bias_a": float(self.mobil_param.bias_a)},
               "yielding_param": [float(p) for p in self.yielding_model.param],
               "behavior_model": self.behavior_model,
               "change_intention_threshold": float(self.change_intention_threshold),
               "random_yielding_seed": int(self.random_yielding_seed)}
        return out

    def had_harsh_brake(self):
        for i in range(len(self.acceleration_history) - 1):
            if self.acceleration_history[i] < -5 and self.acceleration_history[i + 1] < -5 and i >= 3:
                return True
        return False

    def was_merging_vehicle(self):
        for b in self.behavior_model_history:
            if "merging" in b:
                return True
        return False

    def num_lane_change(self):
        if len(self.corridor_history) <= 1:
            return 0
        num_lc = 0
        for i in range(len(self.corridor_history) - 1):
            if self.corridor_history[i + 1] != self.corridor_history[i]:
                num_lc += 1
        return num_lc

    def average_utility(self):
        v_history = [s[2] for s in self.state_history]
        if len(v_history) == 0:
            return 0.
        u_sum = 0
        for v in v_history:
            u_sum += v / self.idm_param.vd if v <= self.idm_param.vd else max(1. - (v / self.idm_param.vd - 1.), 0.9)
        return u_sum / len(v_history)

    def minimum_comfort(self):
        # use the worst 1/5 acc
        if len(self.acceleration_history) == 0:
            return 0.
        abs_acc = [abs(a) for a in self.acceleration_history]
        abs_acc.sort(reverse=True)
        c_sum = 0
        for i in range(max(int(0.2 * len(abs_acc)), 1)):
            c_sum += 1 + abs_acc[i] / (-8.)
        return c_sum / max(int(0.2 * len(abs_acc)), 1)

    def plan(self, observed_env, dt):
        if self.behavior_model == "idm":
            return idm_behavior(observed_env, self)
        if self.behavior_model == "idm_mobil_lane_change" or self.behavior_model == "idm_mobil_lane_change_safe":
            return lane_change_behavior_mobil(observed_env, self)
        if self.behavior_model == "MIQP_merging":
            return MIQP_merging_behavior(observed_env, self, dt)

    def step(self, action, dt):
        if isinstance(action, Trajectory):
            # TODO: implement trajectory interface
            pass
        if isinstance(action, DecoupledAction):
            # move in frenet frame along the ref path
            self.x, self.y, self.v, self.yaw = step_along_path(action.ref_path, self, action.ax, action.vy, dt)
            if self.v == 0:
                self.a = 0.
            else:
                self.a = action.ax
            self.vy = action.vy
            self.acceleration_history.append(self.a)
            self.hull = vehicle_pose_to_rect(self.x, self.y, self.yaw, self.l, self.w)
            self.polygon = Polygon(self.hull)
            self.state_history.append([self.x, self.y, self.v, self.a])


class Neighbor:
    def __init__(self, left_ff, left_f, left_b, left_bb, f, b,
                 right_ff, right_f, right_b, right_bb):
        self.left_ff = left_ff
        self.left_f = left_f
        self.left_b = left_b
        self.left_bb = left_bb
        self.f = f
        self.b = b
        self.right_ff = right_ff
        self.right_f = right_f
        self.right_b = right_b
        self.right_bb = right_bb
        self.lefts = [left_ff, left_f, left_b, left_bb]
        self.rights = [right_ff, right_f, right_b, right_bb]
        self.lefts_and_front = [left_ff, left_f, left_b, left_bb, f]


class MatchedAgent:
    def __init__(self, agent, corridor_id):
        self.agent = agent
        self.corridor_id = corridor_id


class IdmMatch:
    def __init__(self, dis, agent):
        self.id = agent.id
        self.agent = agent
        self.dis = dis  # < 0 when after ego vehicle
        self.x = agent.x
        self.y = agent.y
        self.v = agent.v
        self.vy = agent.vy
        self.a = agent.a
        self.l = agent.l
        self.w = agent.w
        self.vd = agent.idm_param.vd


class SimpleIdmMatch:
    def __init__(self, dis, v):
        self.dis = dis
        self.v = v


class MergingFlag:
    def __init__(self):
        self.finish_merging = False
        self.t_finish_merging = None
        self.once_fallback = False
        self.flag_freeze = False

    def __repr__(self):
        return "Finish merging {}, t finish: {}, once fallback {}".format(
            self.finish_merging, self.t_finish_merging, self.once_fallback)


class ExitFlag:
    def __init__(self):
        self.finish_exit = False
        self.t_finish_exit = None
        self.once_fallback = False
        self.flag_freeze = False

    def __repr__(self):
        return "Finish merging {}, t finish: {}, once fallback {}".format(
            self.finish_exit, self.t_finish_exit, self.once_fallback)


class MergingAndExitStatistics:
    def __init__(self):
        self.num_total_agents = 0
        self.num_finish = 0
        self.total_t = []
        self.num_fallback = 0
        self.epoch_and_ego_id_history = []  # [[epoch_id, ego_id], ...]
        self.utility_ego = []
        self.comfort_ego = []
        self.utility_other = []
        self.comfort_other = []

    def average_t_finish_exit(self):
        if len(self.total_t) == 0:
            return -1
        return np.mean([x for x in self.total_t if x != -10])

    def average_utility_ego(self):
        if len(self.utility_ego) == 0:
            return -1
        return np.mean([x for x in self.utility_ego if x != -10])

    def average_comfort_ego(self):
        if len(self.comfort_ego) == 0:
            return -1
        return np.mean([x for x in self.comfort_ego if x != -10])

    def average_utility_other(self):
        if len(self.utility_other) == 0:
            return -1
        return np.mean([x for x in self.utility_other if x != -10])

    def average_comfort_other(self):
        if len(self.comfort_other) == 0:
            return -1
        return np.mean([x for x in self.comfort_other if x != -10])

    def save(self, path, name):
        np.save(os.path.join(path, name + "_utility_ego.npy"), np.array(self.utility_ego))
        np.save(os.path.join(path, name + "_comfort_ego.npy"), np.array(self.comfort_ego))
        np.save(os.path.join(path, name + "_utility_others.npy"), np.array(self.utility_other))
        np.save(os.path.join(path, name + "_comfort_others.npy"), np.array(self.comfort_other))
        np.save(os.path.join(path, name + "_finish_exit_t.npy"), np.array(self.total_t))
        np.save(os.path.join(path, name + "_epoch_and_ego_id_history.npy"), np.array(self.epoch_and_ego_id_history))

    def remove_invalid_data(self, index_list):
        self.utility_ego = [self.utility_ego[i] for i in range(len(self.utility_ego)) if i not in index_list]
        self.comfort_ego = [self.comfort_ego[i] for i in range(len(self.utility_ego)) if i not in index_list]
        self.utility_other = [self.utility_other[i] for i in range(len(self.utility_ego)) if i not in index_list]
        self.comfort_other = [self.comfort_other[i] for i in range(len(self.utility_ego)) if i not in index_list]
        self.total_t = [self.total_t[i] for i in range(len(self.utility_ego)) if i not in index_list]
        self.epoch_and_ego_id_history = [self.epoch_and_ego_id_history[i] for i in range(len(self.utility_ego)) if i not in index_list]

    def load(self, path, name):
        self.utility_ego = list(np.load(os.path.join(path, name + "_utility_ego.npy")))
        self.comfort_ego = list(np.load(os.path.join(path, name + "_comfort_ego.npy")))
        self.utility_other = list(np.load(os.path.join(path, name + "_utility_others.npy")))
        self.comfort_other = list(np.load(os.path.join(path, name + "_comfort_others.npy")))
        self.total_t = list(np.load(os.path.join(path, name + "_finish_exit_t.npy")))
        self.epoch_and_ego_id_history = list(np.load(os.path.join(path, name + "_epoch_and_ego_id_history.npy")))
        self.utility_ego = [x for x in self.utility_ego if x != -10]
        self.comfort_ego = [x for x in self.comfort_ego if x != -10]
        self.utility_other = [x for x in self.utility_other if x != -10]
        self.comfort_other = [x for x in self.comfort_other if x != -10]
        self.total_t = [x for x in self.total_t if x != -10]
        self.epoch_and_ego_id_history = [x for x in self.epoch_and_ego_id_history if x[0] != -1]
        assert len(self.utility_ego) == len(self.epoch_and_ego_id_history)

    def on_lane_number(self, num):
        id_range = []
        if num == 1:
            id_range = range(10)
        if num == 2:
            id_range = range(10, 17)
        if num == 3:
            id_range = range(19, 100)
        ue, ce, uo, co = [], [], [], []
        for i in range(len(self.epoch_and_ego_id_history)):
            if self.epoch_and_ego_id_history[i][1] in id_range:
                ue.append(self.utility_ego[i])
                ce.append(self.comfort_ego[i])
                uo.append(self.utility_other[i])
                co.append(self.comfort_other[i])
        self.utility_ego = ue
        self.comfort_ego = ce
        self.utility_other = uo
        self.comfort_other = co

    def __repr__(self, average=False):
        if not average:
            return "{} agents, num finish {}, num fallback {}, average t {}, " \
                   "utility ego {}, comfort ego {} utility obj {}, comfort obj {}".format(
                    self.num_total_agents,
                    self.num_finish,
                    self.num_fallback,
                    round(self.total_t[-1], 3) if self.total_t else 0.,
                    round(self.utility_ego[-1], 3) if len(self.utility_ego) else 0.,
                    round(self.comfort_ego[-1], 3) if len(self.comfort_ego) else 0.,
                    round(self.utility_other[-1], 3) if len(self.utility_other) else 0.,
                    round(self.comfort_other[-1], 3) if len(self.comfort_other) else 0.)
        else:
            return "{} agents, num finish {}, num fallback {}, average t {}, average utility ego {}, "\
                   "average comfort ego {}, average utility obj {}, average comfort obj {}".format(
                    self.num_total_agents,
                    self.num_finish,
                    self.num_fallback,
                    round(self.average_t_finish_exit(), 3),
                    round(self.average_utility_ego(), 3),
                    round(self.average_comfort_ego(), 3),
                    round(self.average_utility_other(), 3),
                    round(self.average_comfort_other(), 3))


class LaneChangeStatistics:
    def __init__(self):
        self.num_fallback = 0
        self.num_lane_change = 0
        self.lane_id_history = []
        self.epoch_and_ego_id_history = []  # [[epoch_id, ego_id], ...]
        self.utility_ego = []
        self.comfort_ego = []
        self.utility_other = []
        self.comfort_other = []

    def average_utility_ego(self):
        if len(self.utility_ego) == 0:
            return -1
        return np.mean([x for x in self.utility_ego if x != -10])

    def average_comfort_ego(self):
        if len(self.comfort_ego) == 0:
            return -1
        return np.mean([x for x in self.comfort_ego if x != -10])

    def average_utility_other(self):
        if len(self.utility_other) == 0:
            return -1
        return np.mean([x for x in self.utility_other if x != -10])

    def average_comfort_other(self):
        if len(self.comfort_other) == 0:
            return -1
        return np.mean([x for x in self.comfort_other if x != -10])

    def save(self, path, name):
        np.save(os.path.join(path, name + "_utility_ego.npy"), np.array(self.utility_ego))
        np.save(os.path.join(path, name + "_comfort_ego.npy"), np.array(self.comfort_ego))
        np.save(os.path.join(path, name + "_utility_others.npy"), np.array(self.utility_other))
        np.save(os.path.join(path, name + "_comfort_others.npy"), np.array(self.comfort_other))
        np.save(os.path.join(path, name + "_lane_id_history.npy"), np.array(self.lane_id_history))
        np.save(os.path.join(path, name + "_epoch_and_ego_id_history.npy"), np.array(self.epoch_and_ego_id_history))

    def remove_invalid_data(self, index_list):
        self.utility_ego = [self.utility_ego[i] for i in range(len(self.utility_ego)) if i not in index_list]
        self.comfort_ego = [self.comfort_ego[i] for i in range(len(self.utility_ego)) if i not in index_list]
        self.utility_other = [self.utility_other[i] for i in range(len(self.utility_ego)) if i not in index_list]
        self.comfort_other = [self.comfort_other[i] for i in range(len(self.utility_ego)) if i not in index_list]
        self.lane_id_history = [self.lane_id_history[i] for i in range(len(self.utility_ego)) if i not in index_list]
        self.epoch_and_ego_id_history = [self.epoch_and_ego_id_history[i] for i in range(len(self.utility_ego)) if i not in index_list]

    def load(self, path, name):
        self.utility_ego = list(np.load(os.path.join(path, name + "_utility_ego.npy")))
        self.comfort_ego = list(np.load(os.path.join(path, name + "_comfort_ego.npy")))
        self.utility_other = list(np.load(os.path.join(path, name + "_utility_others.npy")))
        self.comfort_other = list(np.load(os.path.join(path, name + "_comfort_others.npy")))
        self.lane_id_history = list(np.load(os.path.join(path, name + "_lane_id_history.npy")))
        self.epoch_and_ego_id_history = list(np.load(os.path.join(path, name + "_epoch_and_ego_id_history.npy")))
        self.utility_ego = [x for x in self.utility_ego if x != -10]
        self.comfort_ego = [x for x in self.comfort_ego if x != -10]
        self.utility_other = [x for x in self.utility_other if x != -10]
        self.comfort_other = [x for x in self.comfort_other if x != -10]
        self.epoch_and_ego_id_history = [x for x in self.epoch_and_ego_id_history if x[0] != -1]
        assert len(self.utility_ego) == len(self.epoch_and_ego_id_history)

    def on_lane_number(self, num):
        id_range = []
        if num == 1:
            id_range = range(10)
        if num == 2:
            id_range = range(10, 17)
        if num == 3:
            id_range = range(19, 100)
        ue, ce, uo, co = [], [], [], []
        for i in range(len(self.epoch_and_ego_id_history)):
            if self.epoch_and_ego_id_history[i][1] in id_range:
                ue.append(self.utility_ego[i])
                ce.append(self.comfort_ego[i])
                uo.append(self.utility_other[i])
                co.append(self.comfort_other[i])
        self.utility_ego = ue
        self.comfort_ego = ce
        self.utility_other = uo
        self.comfort_other = co

    def __repr__(self, average=False):
        valid_num = str(len(self.utility_ego))
        if not average:
            return valid_num + " sims, num lane change {}, num fallback {}, utility ego {}, comfort ego {} " \
                   "utility obj {}, comfort obj {}".format(
                   self.num_lane_change,
                   self.num_fallback,
                   round(self.utility_ego[-1], 3) if len(self.utility_ego) else 0.,
                   round(self.comfort_ego[-1], 3) if len(self.comfort_ego) else 0.,
                   round(self.utility_other[-1], 3) if len(self.utility_other) else 0.,
                   round(self.comfort_other[-1], 3) if len(self.comfort_other) else 0.)
        else:
            return valid_num + " sims, num lane change {}, num fallback {}, average utility ego {}, " \
                               "average comfort ego {}, average utility obj {}, average comfort obj {}".format(
                   self.num_lane_change,
                   self.num_fallback,
                   round(self.average_utility_ego(), 3),
                   round(self.average_comfort_ego(), 3),
                   round(self.average_utility_other(), 3),
                   round(self.average_comfort_other(), 3))



