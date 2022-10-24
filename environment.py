from agent import *


def check_unique_id(objects):
    ids = []
    for a in objects:
        ids.append(a.id)
    return len(ids) == len(set(ids))


def create_corridor_from_dict(d):
    idx, t, left, right, center, v_limit = d["id"], d["type"], d["left"], d["right"], d["center"], d["v_limit"]
    left_id = d["left_id"] if d["left_id"] != "none" else None
    right_id = d["right_id"] if d["right_id"] != "none" else None
    c = Corridor(idx, t, left, right, center, v_limit)
    c.set_left_and_right_corridor_id(left_id, right_id)
    return c


class Map:
    def __init__(self, corridors):
        # corridors: list of Corridor
        self.corridors = {}
        self.corridors_by_type = {}
        if not check_unique_id(corridors):
            print("Corridors have duplicate ids")
        for c in corridors:
            self.corridors[c.id] = c
            if c.type in self.corridors_by_type.keys():
                self.corridors_by_type[c.type].append(c)
            else:
                self.corridors_by_type[c.type] = [c]

    def corridor_match(self, vehicle):
        for c_id, corridor in self.corridors.items():
            if Point([vehicle.x, vehicle.y]).within(corridor.polygon):
                return corridor
        return False

    def on_merging_lane(self, point):
        for corridor in self.corridors_by_type["merging"]:
            if Point(point).within(corridor.polygon):
                return True
        return False

    def truck_lane_id(self):
        if "merging" in self.corridors_by_type.keys():
            merging_lane = self.corridors_by_type["merging"][0]
            return merging_lane.left_id
        elif "exit" in self.corridors_by_type.keys():
            exit_lane = self.corridors_by_type["exit"][0]
            return exit_lane.left_id
        else:
            for i, c in self.corridors.items():
                if c.right_id is None:
                    return i

    def to_dict(self):
        out = {}
        for i, c in self.corridors.items():
            left_id = "none" if c.left_id is None else c.left_id
            right_id = "none" if c.right_id is None else c.right_id
            c_dict = {"id": c.id, "type": c.type,
                      "left": c.left, "right": c.right, "center": c.center, "v_limit": c.v_limit,
                      "left_id": left_id, "right_id": right_id}
            out[i] = c_dict
        return out


class Environment:
    def __init__(self, m, agents):
        self.map = m
        self.agents = {}
        self.removed_agents = {}
        self.matched_agents = {}  # {agent_id: MatchedAgent, ...}
        self.matched_agents_sorted_by_arc_length = {}  # {corridor_id: [[dis, agent], ...]}
        for a in agents:
            self.agents[a.id] = a
        if not check_unique_id(agents):
            print("Agents have duplicate ids")
        self.match_agents_on_corridor()
        self.sim_time = 0

    def match_agents_on_corridor(self):
        self.matched_agents = {}
        self.matched_agents_sorted_by_arc_length = {}
        for i, agent in self.agents.items():
            matched_corridor = self.map.corridor_match(agent)
            if matched_corridor:
                agent.corridor_history.append(matched_corridor.id)
                if matched_corridor.type == "main" and "merging" in agent.behavior_model:
                    agent.set_behavior_model("idm_mobil_lane_change_safe")
                if matched_corridor.type == "exit" and "exit" in agent.behavior_model:
                    agent.set_behavior_model("idm")
                self.matched_agents[i] = MatchedAgent(agent, matched_corridor.id)
        for i in list(self.agents.keys()):
            if i not in self.matched_agents.keys():
                self.removed_agents[i] = self.agents.pop(i)
        for i in self.map.corridors.keys():
            self.matched_agents_sorted_by_arc_length[i] = self.agents_in_corridor_by_arc_length(i)

    def corridor_for_agent(self, idx):
        if idx not in self.agents.keys():
            print("Agent with id {} is not in agent list".format(idx))
            return None
        if idx not in self.matched_agents.keys():
            print("Agent with id {} is not matched on any corridor".format(idx))
            return None
        corridor_id = self.matched_agents[idx].corridor_id
        return self.map.corridors[corridor_id]

    def dis_to_lane_border(self, idx, direction):
        corridor = self.corridor_for_agent(idx)
        return corridor.distance_to_border(self.matched_agents[idx].agent.hull, direction)

    def limit_of_vehicles(self):
        xmin, xmax, ymin, ymax = 1e10, -1e10, 1e10, -1e10
        for i, agent in self.agents.items():
            xmin = min(agent.x, xmin)
            xmax = max(agent.x, xmax)
            ymin = min(agent.y, ymin)
            ymax = max(agent.y, ymax)
        return xmin - 30, xmax + 30, ymin - 10, ymax + 10

    def limit_of_maps(self):
        xmin, xmax, ymin, ymax = 1e10, -1e10, 1e10, -1e10
        for i, c in self.map.corridors.items():
            xmin = min(c.x_min, xmin)
            xmax = max(c.x_max, xmax)
            ymin = min(c.y_min, ymin)
            ymax = max(c.y_max, ymax)
        return xmin, xmax, ymin, ymax

    def agents_in_corridor_by_arc_length(self, corridor_id):
        agents_by_arc_length = []  # [[dis, agent], ...] sorted by distance
        center = self.map.corridors[corridor_id].centerLine
        lane_agents = [a.agent for _, a in self.matched_agents.items() if a.corridor_id == corridor_id]
        for obj in lane_agents:
            obj_arc_length = center.project(Point([obj.x, obj.y]))
            agents_by_arc_length.append([obj_arc_length, obj])
        agents_by_arc_length.sort(key=lambda x: x[0])
        return agents_by_arc_length

    def sorted_agents_on_corridor_within_range_of_vehicle(self, corridor_id, ego_id, max_l_dis):
        agents_by_arc_length = []  # [[dis, agent], ...] sorted by distance
        center = self.map.corridors[corridor_id].centerLine
        ego = self.agents[ego_id]
        ego_arc_length = center.project(Point([ego.x, ego.y]))
        lane_agents = [a.agent for _, a in self.matched_agents.items() if a.corridor_id == corridor_id]
        for obj in lane_agents:
            obj_arc_length = center.project(Point([obj.x, obj.y]))
            if abs(obj_arc_length - ego_arc_length) <= max_l_dis:
                agents_by_arc_length.append([obj_arc_length, obj])
        agents_by_arc_length.sort(key=lambda x: x[0])
        return [x[1] for x in agents_by_arc_length]

    def agent_to_idm_match(self, ego_id, obj_id):
        ego = self.agents[ego_id]
        obj = self.agents[obj_id]
        obj_center_line = self.map.corridors[self.matched_agents[obj_id].corridor_id].centerLine
        ego_arc = obj_center_line.project(Point([ego.x, ego.y]))
        obj_arc = obj_center_line.project(Point([obj.x, obj.y]))
        return IdmMatch(obj_arc - ego_arc, obj)

    def front_agent(self, idx):
        if idx not in self.agents.keys():
            print("Agent with id {} is not in agent list".format(idx))
            return None
        if idx not in self.matched_agents.keys():
            print("Agent with id {} is not matched on any corridor".format(idx))
            return None
        ego = self.agents[idx]
        corridor_id = self.matched_agents[idx].corridor_id
        center = self.map.corridors[corridor_id].centerLine
        ego_arc_length = center.project(Point([ego.x, ego.y]))
        agents_on_ego_lane = self.matched_agents_sorted_by_arc_length[corridor_id]
        for a in agents_on_ego_lane:
            if a[0] > ego_arc_length:
                return IdmMatch(a[0] - ego_arc_length, a[1])
        return None

    def following_agent(self, idx):
        if idx not in self.agents.keys():
            print("Agent with id {} is not in agent list".format(idx))
            return None
        if idx not in self.matched_agents.keys():
            print("Agent with id {} is not matched on any corridor".format(idx))
            return None
        ego = self.agents[idx]
        corridor_id = self.matched_agents[idx].corridor_id
        center = self.map.corridors[corridor_id].centerLine
        ego_arc_length = center.project(Point([ego.x, ego.y]))
        agents_on_ego_lane = reversed(self.matched_agents_sorted_by_arc_length[corridor_id])
        for a in agents_on_ego_lane:
            if a[0] < ego_arc_length:
                return IdmMatch(a[0] - ego_arc_length, a[1])
        return None

    def neighbor_around_agent(self, idx):
        left_ff, left_f, left_b, left_bb = None, None, None, None
        right_ff, right_f, right_b, right_bb = None, None, None, None
        if idx not in self.agents.keys():
            print("Agent with id {} is not in agent list".format(idx))
            return Neighbor(left_ff, left_f, left_b, left_bb, None, None, right_ff, right_f, right_b, right_bb)
        if idx not in self.matched_agents.keys():
            print("Agent with id {} is not matched on any corridor".format(idx))
            return Neighbor(left_ff, left_f, left_b, left_bb, None, None, right_ff, right_f, right_b, right_bb)
        ego = self.agents[idx]
        corridor_id = self.matched_agents[idx].corridor_id
        left_corridor_id = self.map.corridors[corridor_id].left_id
        right_corridor_id = self.map.corridors[corridor_id].right_id

        if left_corridor_id is not None:
            left_agents_by_arc = self.matched_agents_sorted_by_arc_length[left_corridor_id]
            left_center = self.map.corridors[left_corridor_id].centerExtended
            ego_arc_on_left = left_center.project(Point([ego.x, ego.y]))
            for a in left_agents_by_arc:
                if not left_f and a[0] >= ego_arc_on_left:
                    left_f = IdmMatch(a[0] - ego_arc_on_left, a[1])
                    continue
                if left_f and a[0] > left_f.dis + ego_arc_on_left:
                    left_ff = IdmMatch(a[0] - ego_arc_on_left, a[1])
                    break
            for a in reversed(left_agents_by_arc):
                if not left_b and a[0] < ego_arc_on_left:
                    left_b = IdmMatch(a[0] - ego_arc_on_left, a[1])
                    continue
                if left_b and a[0] < left_b.dis + ego_arc_on_left:
                    left_bb = IdmMatch(a[0] - ego_arc_on_left, a[1])
                    break
        if right_corridor_id is not None:
            right_agents_by_arc = self.matched_agents_sorted_by_arc_length[right_corridor_id]
            right_center = self.map.corridors[right_corridor_id].centerExtended
            ego_arc_on_right = right_center.project(Point([ego.x, ego.y]))
            for a in right_agents_by_arc:
                if not right_f and a[0] >= ego_arc_on_right:
                    right_f = IdmMatch(a[0] - ego_arc_on_right, a[1])
                    continue
                if right_f and a[0] > right_f.dis + ego_arc_on_right:
                    right_ff = IdmMatch(a[0] - ego_arc_on_right, a[1])
                    break
            for a in reversed(right_agents_by_arc):
                if not right_b and a[0] < ego_arc_on_right:
                    right_b = IdmMatch(a[0] - ego_arc_on_right, a[1])
                    continue
                if right_b and a[0] < right_b.dis + ego_arc_on_right:
                    right_bb = IdmMatch(a[0] - ego_arc_on_right, a[1])
                    break
        return Neighbor(left_ff, left_f, left_b, left_bb, self.front_agent(idx), self.following_agent(idx),
                        right_ff, right_f, right_b, right_bb)

    def all_finish_merging(self):
        for _, agent in self.agents.items():
            if "merging_flag" in agent.__dict__:
                if not agent.merging_flag.finish_merging:
                    return False
        return True

    def all_finish_exit(self):
        for _, agent in self.agents.items():
            if "exit" in agent.behavior_model:
                if self.map.corridors[agent.corridor_history[-1]].type != "exit":
                    return False
        return True

    def agents_once_on_merging_lane(self):
        agents = []
        for _, agent in self.agents.items():
            if "merging_flag" in agent.__dict__:
                agents.append(agent)
        return agents

    def agents_never_on_merging_lane(self):
        agents = []
        for _, agent in self.agents.items():
            if "merging_flag" not in agent.__dict__:
                agents.append(agent)
        return agents

    def average_normalized_acc_agents(self, agents):
        sum_average_normalized_acc_over_objects = 0
        for a in agents:
            average_acc = sum(a.acceleration_history) / len(a.acceleration_history)
            average_normalized_acc = (average_acc - a.idm_param.acc_min) / (a.idm_param.acc_max - a.idm_param.acc_min)
            sum_average_normalized_acc_over_objects += average_normalized_acc
        return sum_average_normalized_acc_over_objects / max(len(agents), 1)

    def set_flag(self, agent):
        # finish merging, time for finishing merging, once fallback
        for b in agent.behavior_model_history:
            if "merging" in b:
                if not agent.merging_flag.flag_freeze:
                    corridor_id = self.matched_agents[agent.id].corridor_id
                    if self.map.corridors[corridor_id].type == "main":
                        agent.merging_flag.finish_merging = True
                        agent.merging_flag.t_finish_merging = self.sim_time
                        agent.merging_flag.flag_freeze = True
                    if self.map.corridors[corridor_id].type == "merging":
                        if agent.v <= 2 and agent.had_harsh_brake(): # and d_to_merging_end < 20
                            agent.merging_flag.once_fallback = True
                break
        for b in agent.behavior_model_history:
            if "exit" in b:
                if not agent.exit_flag.flag_freeze:
                    corridor_id = self.matched_agents[agent.id].corridor_id
                    if self.map.corridors[corridor_id].type == "exit":
                        agent.exit_flag.finish_exit = True
                        agent.exit_flag.t_finish_exit = self.sim_time
                        agent.exit_flag.flag_freeze = True
                    if self.map.corridors[corridor_id].type == "main":
                        if agent.had_harsh_brake():
                            agent.merging_flag.once_fallback = True
                break

    def step(self, dt):
        observed_env = self
        for _, a in self.agents.items():
            act = a.plan(observed_env, dt)
            a.step(act, dt)
            self.set_flag(a)
        self.match_agents_on_corridor()
        self.sim_time += dt
