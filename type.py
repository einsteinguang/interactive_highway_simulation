import matplotlib.pyplot as plt

from utility import *
from mc_sim_highway import Point as PointMS
from mc_sim_highway import Line as LineMS
from mc_sim_highway import Corridor as CorridorMS


class DecoupledAction:
    # ax: along vehicle x direction, vy: positive to left
    def __init__(self, ax, vy, ref_path):
        # ref_path: LineSimplified
        self.ax = ax
        self.vy = vy
        self.ref_path = ref_path


class Trajectory:
    def __init__(self, trajectory, time_step):
        self.trajectory = trajectory
        self.time_step = time_step


class IdmParameter:
    def __init__(self,
                 acc_max=1.4, acc_min=-4., min_dis=2, acc_comfort=-2, thw=1.5, vd=None):
        self.acc_max = acc_max
        self.acc_min = acc_min
        self.min_dis = min_dis
        self.acc_com = acc_comfort
        self.thw = thw
        self.vd = vd  # desired velocity


class RSSParameter:
    def __init__(self, r_time_ego=0.4, r_time_other=0.7, a_min=-8, a_max=-10, soft_acc=1.8, soft_dcc=1.2):
        self.r_t_ego = r_time_ego
        self.r_t_other = r_time_other
        self.a_min = a_min
        self.a_max = a_max
        self.soft_acc = soft_acc
        self.soft_dcc = soft_dcc


class MOBILParameter:
    def __init__(self, politeness_factor=0.9, delta_a=0.5, bias_a=0.6):
        self.politeness_factor = politeness_factor
        self.delta_a = delta_a
        self.bias_a = bias_a


class YieldingFeature:
    def __init__(self, thw, ttc, acc):
        self.thw = thw  # d / ego_v, represents no unit distance
        self.ttc = ttc  # (ego_v - obj_v) / ego_v, represents thw changing rate
        self.acc = acc  # acc of ego, represents wish of ego vehicle to cooperate


class YieldingIntention:
    def __init__(self, idx, initial_probability, change_intention_threshold=0.625, random_intention_seed=None):
        # change_intention_threshold: change intention when new probability is < change_intention_threshold * initial_p
        self.id = idx
        self.initial_probability = initial_probability
        self.change_intention_threshold = change_intention_threshold  # Willingness of changing yielding intention
        if random_intention_seed:
            random.seed(random_intention_seed)
        random_number = random.random()
        self.yielding = random_number <= initial_probability

    def update_intention(self, new_probability):
        if not self.yielding:
            if (1 - new_probability) <= self.change_intention_threshold * (1 - self.initial_probability):
                self.yielding = True
        else:
            if new_probability <= self.change_intention_threshold * self.initial_probability:
                self.yielding = False

    def __repr__(self):
        return "Yielding {}, initial_p: {}".format(self.yielding, self.initial_probability)


class YieldingModel:
    def __init__(self, param=np.array([-0.5, 1.81, -4.8, -1.1])):
        self.param = param

    def logistic_function(self, f):
        # feature: [thw, tc, acc]
        # prevent OverflowError
        r = max(min(100, (self.param[0] + self.param[1] * f.thw + self.param[2] * f.ttc + self.param[3] * f.acc)), -100)
        return 1 / (1 + math.exp(-r))

    def yielding_probability(self, idm_match, ego):
        # yielding probability of ego to idm match
        if idm_match.dis + 0.5 * (ego.l + idm_match.l) < 0:
            return 0.
        thw, ttc = thw_and_ttc(idm_match.dis - 0.5 * (ego.l + idm_match.l), ego.v, idm_match.v)
        return self.logistic_function(YieldingFeature(thw, ttc, 0))


class EstimatedYieldingModel(YieldingModel):
    def __init__(self, param):
        super(EstimatedYieldingModel, self).__init__(param=param)

    def yielding_probability(self, idm_match, ego):
        # yielding probability of idm match to ego
        thw, ttc = thw_and_ttc(-idm_match.dis + 0.5 * (ego.l + idm_match.l), idm_match.v, ego.v)
        return self.logistic_function(YieldingFeature(thw, ttc, idm_match.acc))


class Corridor:
    def __init__(self, idx, t, left, right, center, v_limit):
        # left, right, center: [[x, y], [x, y], ...]
        self.id = idx
        self.type = t  # "merging", "main"
        self.left = left
        self.right = right
        self.center = center
        self.leftLine = LineString(left)
        self.rightLine = LineString(right)
        self.centerLine = LineString(center)
        self.leftSimplified = LineSimplified(left)
        self.rightSimplified = LineSimplified(right)
        self.centerSimplified = LineSimplified(center)
        self.centerExtended = LineString(extend_line(center, 100))
        self.v_limit = v_limit
        self.length = self.centerSimplified.path_length
        self.width = self.leftLine.distance(self.rightLine)
        self.border = []
        for p in left:
            self.border.append(p)
        for p in reversed(right):
            self.border.append(p)
        self.polygon = Polygon(self.border)
        self.left_id = None
        self.right_id = None
        self.border_patch = None
        self.center_patch = None
        self.create_border_and_center_patch()
        self.x_min, self.x_max, self.y_min, self.y_max = self.get_x_y_limit()

    def set_left_and_right_corridor_id(self, left_id, right_id):
        self.left_id = left_id
        self.right_id = right_id

    def create_border_and_center_patch(self):
        border_array = np.array(self.border).reshape((len(self.border), 2))
        center_array = np.array(self.center).reshape((len(self.center), 2))
        self.border_patch = plt.Polygon(border_array, color="grey", fill=False, linewidth=1.0, closed=True)
        self.center_patch = plt.Polygon(center_array, color="grey", fill=False,
                                        linewidth=1.5, closed=False, linestyle="--")

    def distance_to_corridor_end(self, position):
        return self.centerSimplified.path_length - self.centerSimplified.project(Point(position))

    def get_x_y_limit(self):
        x_min, x_max, y_min, y_max = 1e10, -1e10, 1e10, -1e10
        for p in self.border:
            x_min = min(p[0], x_min)
            x_max = max(p[0], x_max)
            y_min = min(p[1], y_min)
            y_max = max(p[1], y_max)
        return x_min, x_max, y_min, y_max

    def to_frenet_corridor(self, ref_line):
        # ref_line: LineSimplified object
        l_p0 = ref_line.to_frenet_frame(self.left[0])
        l_p1 = ref_line.to_frenet_frame(self.left[-1])
        r_p0 = ref_line.to_frenet_frame(self.right[0])
        r_p1 = ref_line.to_frenet_frame(self.right[-1])
        c_p0 = ref_line.to_frenet_frame(self.center[0])
        c_p1 = ref_line.to_frenet_frame(self.center[-1])
        return CorridorMS(self.id, self.type,
                          LineMS(PointMS(l_p0[0], l_p0[1]), PointMS(l_p1[0], l_p0[1])),
                          LineMS(PointMS(r_p0[0], r_p0[1]), PointMS(r_p1[0], r_p0[1])),
                          LineMS(PointMS(c_p0[0], c_p0[1]), PointMS(c_p1[0], c_p0[1])), self.v_limit)

    def to_frenet_corridor_limit_length(self, ref_line, x, length):
        # ref_line: LineSimplified object
        l_p0 = ref_line.to_frenet_frame(self.left[0])
        l_p1 = ref_line.to_frenet_frame(self.left[-1])
        r_p0 = ref_line.to_frenet_frame(self.right[0])
        r_p1 = ref_line.to_frenet_frame(self.right[-1])
        c_p0 = ref_line.to_frenet_frame(self.center[0])
        c_p1 = ref_line.to_frenet_frame(self.center[-1])
        l_p1 = [min(l_p1[0], x + length), l_p1[1]]
        r_p1 = [min(r_p1[0], x + length), r_p1[1]]
        c_p1 = [min(c_p1[0], x + length), c_p1[1]]
        return CorridorMS(self.id, self.type,
                          LineMS(PointMS(l_p0[0], l_p0[1]), PointMS(l_p1[0], l_p0[1])),
                          LineMS(PointMS(r_p0[0], r_p0[1]), PointMS(r_p1[0], r_p0[1])),
                          LineMS(PointMS(c_p0[0], c_p0[1]), PointMS(c_p1[0], c_p0[1])), self.v_limit)


class LineSimplified(object):

    def __init__(self, path, simplify=True):
        # append points in the end
        self.path = path
        self.path_line = LineString(self.path)
        if simplify:
            self.path_line = self.path_line.simplify(0.05)
        self.path_length = self.path_line.length
        dis = 0
        self.dis_list = []
        self.yaw_list = []
        self.dis_list.append(dis)
        self.yaw_list.append(np.arctan2(self.path[1][1] - self.path[0][1], self.path[1][0] - self.path[0][0]))
        for i in range(len(self.path) - 1):
            dis += distance(self.path[i], self.path[i + 1])
            self.dis_list.append(dis)
            self.yaw_list.append(
                np.arctan2(self.path[i + 1][1] - self.path[i][1], self.path[i + 1][0] - self.path[i][0]))

    def project(self, point):
        return self.path_line.project(Point(point))

    def get_yaw_at_length(self, length):
        index = np.searchsorted(self.dis_list, length)
        if index >= len(self.yaw_list):
            return self.yaw_list[-1]
        return self.yaw_list[index]

    def get_point_at_length(self, length):
        if length == 0:
            return self.path[0]
        if length < self.path_length:
            # return self.path_line.interpolate(length).coords[0]
            index = np.searchsorted(self.dis_list, length)
            seg_dis = self.dis_list[index] - self.dis_list[index - 1]
            ratio = (self.dis_list[index] - length) / seg_dis
            return np.array(self.path[index]) - ratio * (np.array(self.path[index]) - np.array(self.path[index - 1]))
        else:
            dis_end = self.dis_list[-1] - self.dis_list[-2]
            ratio = (length - self.path_length) / dis_end
            return np.array(self.path[-1]) + ratio * (np.array(self.path[-1]) - np.array(self.path[-2]))

    def is_left_of(self, point):
        if self.path_line.distance(Point(point)) <= 0:
            return False
        else:
            project_l = self.path_line.project(Point(point))
            projected_p = self.get_point_at_length(project_l)
            yaw_at_p = self.get_yaw_at_length(project_l)
            yaw_projected_line = np.arctan2(point[1] - projected_p[1], point[0] - projected_p[0])
            return np.sin(yaw_projected_line - yaw_at_p) > 0

    def signed_distance(self, point):
        dis = self.path_line.distance(Point(point))
        if dis <= 0:
            return 0
        project_l = self.path_line.project(Point(point))
        projected_p = self.get_point_at_length(project_l)
        yaw_at_p = self.get_yaw_at_length(project_l)
        yaw_projected_line = np.arctan2(point[1] - projected_p[1], point[0] - projected_p[0])
        sign = 1 if np.sin(yaw_projected_line - yaw_at_p) > 0 else -1
        return sign * dis

    def to_global_coordinate(self, l, d):
        yaw = self.get_yaw_at_length(l)
        p = offset_point(self.get_point_at_length(l), yaw, d)
        return p[0], p[1], yaw

    def to_frenet_frame(self, point):
        return [self.project(point), self.signed_distance(point)]

