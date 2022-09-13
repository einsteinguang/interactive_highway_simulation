from agent import *


def random_number_minimum(mean, var, min):
    return max(min, np.random.normal(mean, var))


def random_idm_p(desired_v):
    acc_max = np.random.uniform(1.8, 2.2)  # recommended: 1.4
    acc_min = np.random.uniform(-2., -4.)
    min_dis = random_number_minimum(4, 1, 2)
    acc_com = np.random.normal(-2.0, 0.1)
    thw = random_number_minimum(1.5, 0.1, 1.0)  # recommended: 1.5
    vd = desired_v + np.random.normal(0., 0.5)
    return IdmParameter(acc_max, acc_min, min_dis, acc_com, thw, vd)


def random_idm_p_truck(desired_v):
    acc_max = np.random.uniform(0.8, 1.1)  # recommended: 1.4
    acc_min = np.random.uniform(-1.0, -2.0)
    min_dis = random_number_minimum(4, 1, 2)
    acc_com = np.random.normal(-2.0, 0.1)
    thw = random_number_minimum(1.2, 0.1, 1.0)  # recommended: 1.5
    vd = desired_v + np.random.normal(0., 1.)
    return IdmParameter(acc_max, acc_min, min_dis, acc_com, thw, vd)


def random_idm_p_merging(desired_v):
    acc_max = np.random.uniform(2.0, 2.5)  # merging vehicle should move more dynamic as surroundings
    acc_min = np.random.uniform(-3., -4.)
    min_dis = random_number_minimum(4, 1, 2)
    acc_com = np.random.normal(-2.0, 0.1)
    thw = random_number_minimum(1.5, 0.1, 1.0)
    vd = desired_v + np.random.normal(0., 0.5)
    return IdmParameter(acc_max, acc_min, min_dis, acc_com, thw, vd)


def random_yielding_p():
    original_parameter = [1.70968573, 0.00582201, 0.40742229, 1.02390871,
                          -0.32091885, 1.48127062, -0.69499426, -0.34547462,
                          -0.56578542, -1.911662, -0.03998742, 0.02613949, -0.02392805]
    randomized_parameter = []
    for p in original_parameter:
        randomized_parameter.append(np.random.normal(p, abs(p) * 0.2))
    return randomized_parameter


def random_mobil_p():
    politeness_f = np.random.uniform(0, 1)
    delta_a = np.random.normal(0.5, 0.2)
    bias_a = delta_a + np.random.normal(0.1, 0.05)
    return MOBILParameter(politeness_f, delta_a, bias_a)


class RandomDensityAndVelocityParameter:
    def __init__(self, thw_mean, thw_var, mean_v_diff_to_limit, v_var, s_range=None, num_vehicles=None):
        self.thw_mean = thw_mean  # mean time headway between generated agents in one lane
        self.thw_var = thw_var  # variance of time headway between generated agents in one lane
        # mean velocity difference of initial velocity to speed limit of generated agents
        self.mean_v_diff_to_limit = mean_v_diff_to_limit
        # variance of initial velocity
        self.v_var = v_var
        # optional: longitudinal range of spawned agents of one lane.
        # If a lane has 1000m length, [0, 500] range will only generate agents in the first 500m
        self.s_range = s_range
        # optional: maximum number of agents in one lane
        self.num_vehicles = num_vehicles


class RandomTrafficGenerator:

    def __init__(self):
        self.vehicle_id = 0
        self.shapes = [[4.5, 1.9], [4.8, 2], [4.9, 2.1], [5.2, 2.2], [5.9, 2.4]]  # typical shapes from data
        self.truck_shape = [[11.5, 2.5], [12, 2.5], [10, 2.5]]

    def random_agents_on_map(self, m, parameter, unsafe_lc_ratio=0., truck_ratio=0.):
        agents_merging, agents_main = [], []
        v_limit_candidates = []
        truck_lane_id = m.truck_lane_id()
        for _, c in m.corridors.items():
            if c.type == "main":
                v_limit_candidates.append(c.v_limit)
        for i, c in m.corridors.items():
            if c.type == "main":
                if i == truck_lane_id:
                    agents_main.extend(self.random_agents_on_lane(c, parameter[i], v_limit_candidates, "idm",
                                                                  unsafe_lc_ratio=unsafe_lc_ratio,
                                                                  truck_ratio=truck_ratio))
                else:
                    agents_main.extend(self.random_agents_on_lane(c, parameter[i], v_limit_candidates, "idm",
                                                                  unsafe_lc_ratio=unsafe_lc_ratio))
            if c.type == "merging":
                agents_merging.extend(self.random_agents_on_lane(c, parameter[i], v_limit_candidates,
                                                                 "merging_closest_gap_policy"))
        return agents_merging, agents_main

    def random_agents_on_lane(self, corridor, params, v_limit_candidates, behavior_model,
                              unsafe_lc_ratio=0., truck_ratio=0.):
        vehicles = []
        s_range = [0, corridor.centerSimplified.path_length]
        num_vehicles = 1e10
        if params.s_range:
            s_range = [max(s_range[0], params.s_range[0]), min(s_range[1], params.s_range[1])]
        if params.num_vehicles:
            num_vehicles = params.num_vehicles
        x_frenet = np.random.uniform(s_range[0], 0.3 * corridor.v_limit * params.thw_mean)
        while x_frenet < s_range[1] and len(vehicles) < num_vehicles:
            y_frenet = np.random.normal(0, 0.2)
            v = np.random.normal(corridor.v_limit + params.mean_v_diff_to_limit, params.v_var)
            x, y, yaw = corridor.centerSimplified.to_global_coordinate(x_frenet, y_frenet)
            random_num = np.random.uniform(0, 1)
            if random_num < truck_ratio:
                l, w = random.choice(self.truck_shape)
                idm_p = random_idm_p_truck(min(v_limit_candidates))
            else:
                l, w = random.choice(self.shapes)
                idm_p = random_idm_p(random.choice(v_limit_candidates))
            change_intention_threshold = np.random.uniform(0.4, 0.7)  # Willingness of changing yielding intention
            b_model = behavior_model
            # 10% idm
            if behavior_model == "idm":
                random_num = np.random.uniform(0, 1)
                if 1 - unsafe_lc_ratio >= random_num > 0.1:
                    b_model = "idm_mobil_lane_change_safe"
                elif random_num > 1 - unsafe_lc_ratio:
                    b_model = "idm_mobil_lane_change"
            vehicle = Agent(self.vehicle_id, x, y, v, yaw, 0, l, w, idm_p, random_yielding_p(), random_mobil_p(),
                            b_model, change_intention_threshold)
            if "merging" in behavior_model:
                vehicle.idm_param = random_idm_p_merging(random.choice(v_limit_candidates))
            vehicles.append(vehicle)
            x_frenet += np.random.uniform(corridor.v_limit * (params.thw_mean - params.thw_var),
                                          corridor.v_limit * (params.thw_mean + params.thw_var)) + l
            self.vehicle_id += 1
        return vehicles
