import math
import random
import numpy as np
from copy import deepcopy
from shapely.geometry import Point, LineString, Polygon


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def vehicle_pose_to_rect(pos_x, pos_y, yaw, length, width):
    # get the hull point of vehicle given position and yaw angle
    head = [pos_x + length / 2 * math.cos(yaw), pos_y + length / 2 * math.sin(yaw)]
    tair = [pos_x - length / 2 * math.cos(yaw), pos_y - length / 2 * math.sin(yaw)]
    head_left = [head[0] - width / 2 * math.sin(yaw), head[1] + width / 2 * math.cos(yaw)]
    head_right= [head[0] + width / 2 * math.sin(yaw), head[1] - width / 2 * math.cos(yaw)]
    tair_left = [tair[0] - width / 2 * math.sin(yaw), tair[1] + width / 2 * math.cos(yaw)]
    tair_right= [tair[0] + width / 2 * math.sin(yaw), tair[1] - width / 2 * math.cos(yaw)]
    return [head_left, head_right, tair_right, tair_left]


def idm_acc_f_b(fs, b, ego_v, v_limit, idm_param, obey_v_limit=True):
    # fs: [idm_match, ...], b: idm_match, dis is pure distance, front > 0, back < 0
    acc_max = idm_param.acc_max
    acc_min = idm_param.acc_min
    min_dis = idm_param.min_dis
    acc_com = idm_param.acc_com
    thw = idm_param.thw
    # r_time = idm_param.reaction_time
    vd = min(idm_param.vd, 1.1 * v_limit) if obey_v_limit else idm_param.vd
    free_road_term = acc_max * (1 - (ego_v / vd) ** 4)
    # print("free term", free_road_term)
    d_term = 0
    d_terms_f = []
    for f in fs:
        d_desire = (0.7 * (min_dis + ego_v * thw) + ego_v * (ego_v - f.v) / (2 * math.sqrt(-acc_max * acc_com)))
        d_t = -acc_max * (d_desire / max(1, f.dis)) ** 2
        # print("dcc term", d_t)
        d_terms_f.append(d_t)
    d_term += min(d_terms_f) if d_terms_f else 0
    if b:
        d_desire = (0.7 * (min_dis + b.v * thw) + b.v * (b.v - ego_v) / (2 * math.sqrt(-acc_max * acc_com)))
        d_t = acc_max * (d_desire / max(1, -b.dis)) ** 2
        # print("acc term", d_t)
        d_term += d_t
    acc = free_road_term + d_term
    # print(acc)
    return max(min(acc_max, acc), acc_min)


def idm_acc_fs_bs(fs, bs, ego_v, v_limit, idm_param, obey_v_limit=True):
    # fs: [idm_match, ...], b: idm_match, dis is pure distance, front > 0, back < 0
    acc_max = idm_param.acc_max
    acc_min = idm_param.acc_min
    min_dis = idm_param.min_dis
    acc_com = idm_param.acc_com
    thw = idm_param.thw
    # r_time = idm_param.reaction_time
    vd = min(idm_param.vd, 1.1 * v_limit) if obey_v_limit else idm_param.vd
    free_road_term = acc_max * (1 - (ego_v / vd) ** 4)
    # print("free term", free_road_term)
    d_term = 0
    d_terms_f = []
    for f in fs:
        d_desire = (0.7 * (min_dis + ego_v * thw) + ego_v * (ego_v - f.v) / (2 * math.sqrt(-acc_max * acc_com)))
        d_t = -acc_max * (d_desire / max(1, f.dis)) ** 2
        # print("dcc term", d_t)
        d_terms_f.append(d_t)
    d_term += min(d_terms_f) if d_terms_f else 0
    d_terms_b = []
    for b in bs:
        d_desire = (0.7 * (min_dis + b.v * thw) + b.v * (b.v - ego_v) / (2 * math.sqrt(-acc_max * acc_com)))
        d_t = acc_max * (d_desire / max(1, -b.dis)) ** 2
        # print("acc term", d_t)
        d_terms_b.append(d_t)
    d_term += max(d_terms_b) if d_terms_b else 0
    acc = free_road_term + d_term
    # print(acc)
    return max(min(acc_max, acc), acc_min)


def offset_point(p, yaw, offset):
    left = [p[0] - offset * math.sin(yaw), p[1] + offset * math.cos(yaw)]
    return left


def step_along_path(ref_path, ego_v, ax, vy, dt):
    # ref_path: LineSimplified
    ego_x = ref_path.path_line.project(Point(ego_v.get_position()))
    sign = 1 if ref_path.is_left_of(ego_v.get_position()) else -1
    ego_y = sign * ref_path.path_line.distance(Point(ego_v.get_position()))
    ego_x_new = ego_x + ego_v.v * dt
    ego_y_new = ego_y + vy * dt
    ego_v_new = max(0, ego_v.v + ax * dt)
    ego_yaw_new = ref_path.get_yaw_at_length(ego_x_new)
    ego_position_new = offset_point(ref_path.get_point_at_length(ego_x_new), ego_yaw_new, ego_y_new)
    return ego_position_new[0], ego_position_new[1], ego_v_new, ego_yaw_new


def thw_and_ttc(d, ego_v, obj_v):
    # d should be pure distance of the gap
    thw = d / max(ego_v, 1e-10)
    # if d == 0:
    if ego_v > 0:
        ttc = (ego_v - obj_v) / ego_v
    else:
        ttc = (ego_v - obj_v) / 1e-3
    # else:
    #     ttc = (ego_v - obj_v) / d
    return thw, ttc


def extend_line(line, d):
    new_line = deepcopy(line)
    new_p = np.array(line[-1]) + d / distance(line[-1], line[-2]) * (np.array(line[-1]) - np.array(line[-2]))
    new_line.append([new_p[0], new_p[1]])
    return new_line


def extend_line_both_sides(line, d):
    new_line = []
    f_p = np.array(line[0]) + d / distance(line[0], line[1]) * (np.array(line[0]) - np.array(line[1]))
    b_p = np.array(line[-1]) + d / distance(line[-1], line[-2]) * (np.array(line[-1]) - np.array(line[-2]))
    new_line.append(f_p)
    for p in line:
        new_line.append(p)
    new_line.append(b_p)
    return new_line


def rss_dis(vf, vb, rt, aMin, aMax):
    return max(vb * rt + vb * vb / (-2 * aMin) - vf * vf / (-2 * aMax), 0.)


def idm_p_for_merging_exit_agents(random_seed, idm_p_origin, is_truck):
    random.seed(random_seed)
    random_number = random.random()
    idm_p_yielding = deepcopy(idm_p_origin)
    if not is_truck:
        idm_p_yielding.acc_max -= (0.6 + 0.4 * random_number)
        idm_p_yielding.acc_min += (0.8 + 0.4 * random_number)
    return idm_p_yielding



