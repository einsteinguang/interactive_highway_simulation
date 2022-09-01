import copy
import os

from learned_model import *
from behaviors import *


def create_agent_from_dict(data):
    idx = data["id"]
    x, y, v, yaw, a, l, w = data["x"], data["y"], data["v"], data["yaw"], data["a"], data["l"], data["w"]
    idm_p = IdmParameter(data["idm_p"]["acc_max"], data["idm_p"]["acc_min"],
                         data["idm_p"]["min_dis"], data["idm_p"]["acc_com"],
                         data["idm_p"]["thw"], data["idm_p"]["vd"])
    mobil_p = MOBILParameter(data["mobil_param"]["politeness_factor"],
                             data["mobil_param"]["delta_a"],
                             data["mobil_param"]["bias_a"])
    yielding_p = data["yielding_param"]
    behavior_model = data["behavior_model"]
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
        self.yielding_param = yielding_param if yielding_param is not None else np.array([-0.5, 1.81, -4.8, -1.1])
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
               "yielding_param": [float(self.yielding_model.param[0]),
                                  float(self.yielding_model.param[1]),
                                  float(self.yielding_model.param[2]),
                                  float(self.yielding_model.param[3])],
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

    def plan(self, observed_env):
        if self.behavior_model == "idm":
            return idm_behavior(observed_env, self)
        if self.behavior_model == "idm_mobil_lane_change":
            return lane_change_behavior_mobil(observed_env, self, False)
        if self.behavior_model == "idm_mobil_lane_change_safe":
            return lane_change_behavior_mobil(observed_env, self, True)
        if self.behavior_model == "merging_learned":
            return cloned_merging_behavior(observed_env, self)
        if self.behavior_model == "merging_closest_gap_policy":
            return merging_behavior_closest_gap(observed_env, self)
        if self.behavior_model == "lane_change_learned":
            return lane_change_behavior_learned(observed_env, self)
        if self.behavior_model == "exit":
            return exit_behavior_closest_gap(observed_env, self)
        if self.behavior_model == "exit_learned":
            return cloned_exit_behavior(observed_env, self)

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


def plot_customized_idm():
    dt = 0.1
    ego_v = 10
    ego_x = 65
    f_x = 50
    b_x = 20
    f_v = 8
    b_v = 8
    t, ego_xs, f_xs, b_xs = [], [], [], []
    ego_as = []
    idm_p = IdmParameter(acc_max=2., acc_min=-2, vd=10.)
    for i in range(200):
        t.append(i * dt)
        ego_xs.append(ego_x)
        f_xs.append(f_x)
        b_xs.append(b_x)
        gap = [SimpleIdmMatch(f_x - ego_x, f_v), SimpleIdmMatch(b_x - ego_x, b_v)]
        acc = idm_acc_f_b([gap[0]], gap[1], ego_v, 10, idm_p)
        ego_as.append(acc)
        ego_v = max(0, ego_v + dt * acc)
        if ego_v > 0:
            ego_x = ego_x + ego_v * dt + 0.5 * acc * dt**2
        f_x += dt * f_v
        b_x += dt * b_v
    f = plt.figure(figsize=(3.5, 3))
    from matplotlib import gridspec
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ax1 = plt.subplot(gs[0])
    plt.plot(t, f_xs, "r", label="leader")
    plt.plot(t, ego_xs, "green", label="ego vehicle")
    plt.plot(t, b_xs, "b", label="follower")
    plt.tick_params('x', labelbottom=False)
    plt.legend(loc="lower right", fontsize=12)
    plt.ylabel(r'$x$(m)', fontsize=14)

    ax2 = plt.subplot(gs[1])
    plt.plot(t, ego_as, "green")
    plt.xlabel(r'$t(\mathrm{s})$', fontsize=14)
    plt.ylabel(r'$a_\mathrm{lon}(\frac{\mathrm{m}}{\mathrm{s}^2}$)', fontsize=14)
    plt.show()


def plot_customized_idm_for_crossing():
    path_h = [[100, 2], [-100, 2]]
    path_h2 = [[-100, -2], [100, -2]]
    path_v = [[0, -100], [0, 100]]
    dt = 0.2
    idm_p = IdmParameter(acc_max=2., acc_min=-2, vd=10.)
    fig = plt.figure(figsize=(4 * 2.2, 4 * 2.2))
    ax = fig.add_subplot(1, 1, 1)
    ego = Agent(0, 0, -10, v=5, yaw=np.pi/2, a=0, l=5, w=2,
                idm_param=None, yielding_param=None, mobil_param=None, behavior_model="cross")
    a_h_1 = Agent(1, 25, 2, v=3, yaw=np.pi, a=0, l=5, w=2,
                  idm_param=None, yielding_param=None, mobil_param=None, behavior_model="cross")
    a_h_2 = Agent(2, 45, 2, v=3, yaw=np.pi, a=0, l=5, w=2,
                  idm_param=None, yielding_param=None, mobil_param=None, behavior_model="cross")
    a_h2_1 = Agent(3, -40, -2, v=3, yaw=0, a=0, l=5, w=2,
                   idm_param=None, yielding_param=None, mobil_param=None, behavior_model="cross")
    v_ego = []
    for i in range(200):
        a_h_1.step(DecoupledAction(0, 0, LineSimplified(path_h)), dt)
        a_h_2.step(DecoupledAction(0, 0, LineSimplified(path_h)), dt)
        a_h2_1.step(DecoupledAction(0, 0, LineSimplified(path_h2)), dt)
        acc = 0
        if ego.y < 0:
            front = []
            back = [SimpleIdmMatch(-a_h_1.x - ego.y + 7, a_h_1.v),
                    SimpleIdmMatch(-a_h_2.x - ego.y + 7, a_h_2.v),
                    SimpleIdmMatch(a_h2_1.x - ego.y + 7, a_h2_1.v)]
            acc_needed_to_stop_before_cz = -10
            if ego.y < -7:
                acc_needed_to_stop_before_cz = -ego.v * ego.v / 2 / (-7 - ego.y)
            acc = idm_acc_fs_bs(front, back, ego.v, 10, idm_p)
            acc = max(acc_needed_to_stop_before_cz, acc)
        if ego.y >= 0:
            acc = idm_acc_fs_bs([], [], ego.v, 10, idm_p)
        ego.step(DecoupledAction(acc, 0, LineSimplified(path_v)), dt)
        v_ego.append(round(ego.v, 3))
        print("Acc: ", acc)
        ax.add_patch(copy.copy(plt.Polygon(ego.hull, closed=True, fill=True, color="blue")))
        ax.add_patch(copy.copy(plt.Polygon(a_h_1.hull, closed=True, fill=True, color="black")))
        ax.add_patch(copy.copy(plt.Polygon(a_h_2.hull, closed=True, fill=True, color="black")))
        ax.add_patch(copy.copy(plt.Polygon(a_h2_1.hull, closed=True, fill=True, color="black")))
        plt.xlim((-100, 100))
        plt.ylim((-100, 100))
        plt.draw()
        plt.pause(0.01)
        ax.patches = []
        ax.clear()
    print(v_ego)


def plot_cautious_approach():
    dt = 0.1
    ego_ass = []
    ego_xss = []
    f_xs = []
    t = []
    f_x = 50
    for j in range(150):
        t.append(j * dt)
        f_xs.append(f_x)
    for i in np.arange(1, 5):
        ego_v = 10
        ego_x = 0
        ego_xs = []
        ego_as = []
        f_v = 0
        idm_p = IdmParameter(acc_max=i, acc_min=-i, vd=10.)
        for j in range(150):
            ego_xs.append(ego_x)
            acc = idm_acc_fs_bs([SimpleIdmMatch(f_x - ego_x, f_v)], [], ego_v, 10, idm_p)
            ego_as.append(acc)
            ego_v = max(0, ego_v + dt * acc)
            if ego_v > 0:
                ego_x = ego_x + ego_v * dt + 0.5 * acc * dt**2
        ego_xss.append(ego_xs)
        ego_ass.append(ego_as)
        # f_x += dt * f_v
    f = plt.figure(figsize=(3.5, 3))
    from matplotlib import gridspec
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ax1 = plt.subplot(gs[0])
    plt.plot(t, f_xs, "r", label="leader")
    for i, ego_xs in enumerate(ego_xss):
        plt.plot(t, ego_xs, label="max dcc: " + str(-i - 2))
    plt.tick_params('x', labelbottom=False)
    plt.legend(loc="lower right", fontsize=12)
    plt.ylabel(r'$x$(m)', fontsize=14)

    ax2 = plt.subplot(gs[1])
    for ego_as in ego_ass:
        plt.plot(t, ego_as)
    plt.xlabel(r'$t(\mathrm{s})$', fontsize=14)
    plt.ylabel(r'$a_\mathrm{lon}(\frac{\mathrm{m}}{\mathrm{s}^2}$)', fontsize=14)
    plt.show()


if __name__ == '__main__':
    # plot_customized_idm()
    plot_customized_idm_for_crossing()
    # plot_cautious_approach()
    pass

