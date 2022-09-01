from type import *

from mc_sim_highway import Agent as AgentMS
from mc_sim_highway import MapMerging, IdmParam, RSSParam, Vehicle, \
    YieldingParam, SimParameter, MapMultiLane, Action


def idm_param(ego_l):
    idm_p = IdmParam(1.6, -3.0, 2., -2, -8., 1.2)
    idm_p_truck = IdmParam(0.8, -1.5, 2., -2, -8., 1.2)
    if ego_l > 7:
        return idm_p_truck
    return idm_p


def rss_param(is_merging):
    rss_p = RSSParam(0.7, 0.4, -6., -6., 1.8, 1.2)
    rss_p_merging = RSSParam(0.7, 0.4, -8., -10., 1.8, 1.2)
    if is_merging:
        return rss_p_merging
    return rss_p


def merging_mcs_sim_from_env(env, agent, ego_corridor):
    # For ego vehicle, use known parameters
    idm_p_ego = IdmParam(agent.idm_param.acc_max,
                         agent.idm_param.acc_min,
                         agent.idm_param.min_dis,
                         agent.idm_param.acc_com,
                         -8., agent.idm_param.thw)
    yp_ego = YieldingParam(agent.yielding_param[0], agent.yielding_param[1],
                           agent.yielding_param[2], agent.yielding_param[3],
                           agent.mobil_param.politeness_factor, agent.mobil_param.delta_a, agent.mobil_param.bias_a)
    # For other vehicles, estimated parameters should be used, not the real parameter from the agents
    yp = YieldingParam()

    # ego_input: [ego_x, ego_y, vx, l, w], frenet frame of merging lane
    possible_actions = [-1]
    left_corridor_id = env.map.corridors[ego_corridor.id].left_id
    left_corridor = env.map.corridors[left_corridor_id]
    ref_line = LineSimplified(extend_line_both_sides(left_corridor.center, 1000))
    ego = ref_line.to_frenet_frame([agent.x, agent.y])
    ego_behavior = "merging" if "merging" in agent.behavior_model else "idm"
    ego_agent = AgentMS(agent.id, ego[0], ego[1], agent.v, agent.vy,
                        agent.idm_param.vd,
                        agent.a, agent.l, agent.w, idm_p_ego, rss_param(True), yp_ego, ego_behavior)
    agents = [ego_agent]
    neighbor = env.neighbor_around_agent(agent.id)
    for obj in neighbor.lefts_and_front:
        if obj:
            x, y = ref_line.to_frenet_frame([obj.x, obj.y])
            behavior = "merging" if "merging" in agent.behavior_model else "idm_lc"
            vehicle = AgentMS(obj.id, x, y, obj.v, obj.vy, obj.vd + np.random.normal(0., 1),
                              obj.a, obj.l, obj.w, idm_param(obj.l), rss_param(behavior == "merging"), yp, behavior)
            agents.append(vehicle)

    for obj in reversed(neighbor.lefts):
        if obj and len(possible_actions) < 4:
            possible_actions.append(obj.id)

    merging_corridor = ego_corridor.to_frenet_corridor(ref_line)
    main_corridor = left_corridor.to_frenet_corridor(ref_line)
    corridors = [merging_corridor, main_corridor]
    ll_corridor_id = left_corridor.left_id
    if ll_corridor_id is not None:
        ll_corridor = env.map.corridors[ll_corridor_id]
        ll_agents = env.sorted_agents_on_corridor_within_range_of_vehicle(ll_corridor_id, agent.id, 80)
        for obj in ll_agents:
            x, y = ref_line.to_frenet_frame([obj.x, obj.y])
            vehicle = AgentMS(obj.id, x, y, obj.v, obj.vy, obj.idm_param.vd + np.random.normal(0., 1),
                              obj.a, obj.l, obj.w, idm_param(obj.l), rss_param(False), yp, "idm_lc")
            agents.append(vehicle)
        corridors.append(ll_corridor.to_frenet_corridor(ref_line))
    # print(lateral_dis, ego_corridor.width, left_corridor.width)
    return corridors, agents, possible_actions


def exit_mcs_sim_from_env(env, agent, ego_corridor):
    # For ego vehicle, use known parameters
    idm_p_ego = IdmParam(agent.idm_param.acc_max,
                         agent.idm_param.acc_min,
                         agent.idm_param.min_dis,
                         agent.idm_param.acc_com,
                         -8., agent.idm_param.thw)
    yp_ego = YieldingParam(agent.yielding_param[0], agent.yielding_param[1],
                           agent.yielding_param[2], agent.yielding_param[3],
                           agent.mobil_param.politeness_factor, agent.mobil_param.delta_a, agent.mobil_param.bias_a)
    # For other vehicles, estimated parameters should be used, not the real parameter from the agents
    yp = YieldingParam()

    # ego_input: [ego_x, ego_y, vx, l, w], frenet frame of merging lane
    possible_actions = [-1]
    # find exit lane
    right_corridor_id = env.map.corridors[ego_corridor.id].right_id
    exit_lane_id = -1
    while right_corridor_id is not None:
        right_corridor = env.map.corridors[right_corridor_id]
        if right_corridor.type == "exit":
            exit_lane_id = right_corridor_id
            break
        right_corridor_id = right_corridor.right_id
    if exit_lane_id == -1:
        print("Exit mcs wrapper: no exit lane found!")
    exit_corridor = env.map.corridors[exit_lane_id]

    ref_line = LineSimplified(extend_line_both_sides(ego_corridor.center, 1000))
    ego = ref_line.to_frenet_frame([agent.x, agent.y])
    ego_behavior = "exit"
    ego_agent = AgentMS(agent.id, ego[0], ego[1], agent.v, agent.vy, agent.idm_param.vd,
                        agent.a, agent.l, agent.w, idm_p_ego, rss_param(True), yp_ego, ego_behavior)
    # limit length of ego corridor, max. to end of exit lane, fake merging lane
    exit_mcs_corridor = exit_corridor.to_frenet_corridor(ref_line)
    # limit length based on how many lanes between ego and exit lane
    max_length = min(ego_corridor.v_limit ** 2 / 2.5 ** 2,  exit_mcs_corridor.m.p2.x -
                     (abs(exit_mcs_corridor.id - ego_corridor.id) - 1) * 50 - ego[0])
    center_corridor = ego_corridor.to_frenet_corridor_limit_length(ref_line, ego[0], max_length)

    agents = [ego_agent]
    corridors = [center_corridor]
    ego_front = env.front_agent(agent.id)
    ego_follow = env.following_agent(agent.id)
    if ego_front:
        x, y = ref_line.to_frenet_frame([ego_front.x, ego_front.y])
        ego_f_agent = AgentMS(ego_front.id, x, y, ego_front.v, ego_front.vy, ego_front.vd + np.random.normal(0., 1),
                              ego_front.a, ego_front.l, ego_front.w, idm_param(ego_front.l), rss_param(False), yp, "idm_lc")
        agents.append(ego_f_agent)
        ego_ff = env.front_agent(ego_front.id)
        if ego_ff:
            x, y = ref_line.to_frenet_frame([ego_ff.x, ego_ff.y])
            ego_ff_agent = AgentMS(ego_ff.id, x, y, ego_ff.v, ego_ff.vy, ego_ff.vd + np.random.normal(0., 1),
                                   ego_ff.a, ego_ff.l, ego_ff.w, idm_param(ego_ff.l), rss_param(False), yp, "idm_lc")
            agents.append(ego_ff_agent)

    if ego_follow:
        x, y = ref_line.to_frenet_frame([ego_follow.x, ego_follow.y])
        ego_f_agent = AgentMS(ego_follow.id, x, y, ego_follow.v, ego_follow.vy, ego_follow.vd + np.random.normal(0., 1),
                              ego_follow.a, ego_follow.l, ego_follow.w, idm_param(ego_follow.l), rss_param(False), yp, "idm_lc")
        agents.append(ego_f_agent)

    neighbors = env.neighbor_around_agent(agent.id)
    left_neighbor = neighbors.lefts
    left_corridor_id = env.map.corridors[ego_corridor.id].left_id
    if left_corridor_id is not None:
        left_corridor = env.map.corridors[left_corridor_id]
        for obj in left_neighbor:
            if obj:
                x, y = ref_line.to_frenet_frame([obj.x, obj.y])
                vehicle = AgentMS(obj.id, x, y, obj.v, obj.vy, obj.vd + np.random.normal(0., 1),
                                  obj.a, obj.l, obj.w, idm_param(obj.l), rss_param(False), yp, "idm_lc")
                agents.append(vehicle)
        corridors.append(left_corridor.to_frenet_corridor(ref_line))
        ll_corridor_id = left_corridor.left_id
        if ll_corridor_id is not None:
            ll_corridor = env.map.corridors[ll_corridor_id]
            ll_agents = env.sorted_agents_on_corridor_within_range_of_vehicle(ll_corridor_id, agent.id, 80)
            for obj in ll_agents:
                x, y = ref_line.to_frenet_frame([obj.x, obj.y])
                vehicle = AgentMS(obj.id, x, y, obj.v, obj.vy, obj.idm_param.vd + np.random.normal(0., 1),
                                  obj.a, obj.l, obj.w, idm_param(obj.l), rss_param(False), yp, "idm_lc")
                agents.append(vehicle)
            corridors.append(ll_corridor.to_frenet_corridor(ref_line))

    for obj in reversed(neighbors.rights):
        if obj and len(possible_actions) < 4:
            possible_actions.append(obj.id)

    right_neighbor = neighbors.rights
    right_corridor_id = env.map.corridors[ego_corridor.id].right_id
    if right_corridor_id is not None:
        right_corridor = env.map.corridors[right_corridor_id]
        for obj in right_neighbor:
            if obj:
                behavior = "merging" if right_corridor.type == "merging" else "idm_lc"
                x, y = ref_line.to_frenet_frame([obj.x, obj.y])
                vehicle = AgentMS(obj.id, x, y, obj.v, obj.vy, obj.vd + np.random.normal(0., 1),
                                  obj.a, obj.l, obj.w, idm_param(obj.l), rss_param(behavior == "merging"), yp, behavior)
                agents.append(vehicle)
        corridors.append(right_corridor.to_frenet_corridor(ref_line))
        rr_corridor_id = right_corridor.right_id
        if rr_corridor_id is not None:
            rr_corridor = env.map.corridors[rr_corridor_id]
            rr_agents = env.sorted_agents_on_corridor_within_range_of_vehicle(rr_corridor_id, agent.id, 80)
            for obj in rr_agents:
                behavior = "merging" if rr_corridor.type == "merging" else "idm_lc"
                x, y = ref_line.to_frenet_frame([obj.x, obj.y])
                vehicle = AgentMS(obj.id, x, y, obj.v, obj.vy, obj.idm_param.vd + np.random.normal(0., 1),
                                  obj.a, obj.l, obj.w, idm_param(obj.l), rss_param(behavior == "merging"), yp, behavior)
                agents.append(vehicle)
            corridors.append(rr_corridor.to_frenet_corridor(ref_line))
    return corridors, agents, possible_actions


def lane_change_mcs_sim_from_env(env, agent, ego_corridor):
    # For ego vehicle, use known parameters
    idm_p_ego = IdmParam(agent.idm_param.acc_max,
                         agent.idm_param.acc_min,
                         agent.idm_param.min_dis,
                         agent.idm_param.acc_com,
                         -8., agent.idm_param.thw)
    yp_ego = YieldingParam(agent.yielding_param[0], agent.yielding_param[1],
                           agent.yielding_param[2], agent.yielding_param[3],
                           agent.mobil_param.politeness_factor, agent.mobil_param.delta_a, agent.mobil_param.bias_a)
    rss_p_ego = RSSParam(agent.rss_param.r_t_other, agent.rss_param.r_t_ego,
                         agent.rss_param.a_min, agent.rss_param.a_max,
                         agent.rss_param.soft_acc, agent.rss_param.soft_dcc)
    # For other vehicles, estimated parameters should be used, not the real parameter from the agents
    yp = YieldingParam()

    # ego_input: [ego_x, ego_y, vx, l, w], frenet frame of merging lane
    ref_line = LineSimplified(extend_line_both_sides(ego_corridor.center, 1000))
    ego = ref_line.to_frenet_frame([agent.x, agent.y])
    ego_behavior = "merging" if "merging" in agent.behavior_model else "idm"
    ego_agent = AgentMS(agent.id, ego[0], ego[1], agent.v, agent.vy, agent.idm_param.vd,
                        agent.a, agent.l, agent.w, idm_p_ego, rss_p_ego, yp_ego, ego_behavior)
    center_corridor = ego_corridor.to_frenet_corridor(ref_line)
    agents = [ego_agent]
    corridors = [center_corridor]
    ego_front = env.front_agent(agent.id)
    ego_follow = env.following_agent(agent.id)
    if ego_front:
        x, y = ref_line.to_frenet_frame([ego_front.x, ego_front.y])
        ego_f_agent = AgentMS(ego_front.id, x, y, ego_front.v, ego_front.vy, ego_front.vd + np.random.normal(0., 1),
                              ego_front.a, ego_front.l, ego_front.w, idm_param(ego_front.l), rss_param(False), yp, "idm_lc")
        agents.append(ego_f_agent)
        ego_ff = env.front_agent(ego_front.id)
        if ego_ff:
            x, y = ref_line.to_frenet_frame([ego_ff.x, ego_ff.y])
            ego_ff_agent = AgentMS(ego_ff.id, x, y, ego_ff.v, ego_ff.vy, ego_ff.vd + np.random.normal(0., 1),
                                   ego_ff.a, ego_ff.l, ego_ff.w, idm_param(ego_ff.l), rss_param(False), yp, "idm_lc")
            agents.append(ego_ff_agent)

    if ego_follow:
        x, y = ref_line.to_frenet_frame([ego_follow.x, ego_follow.y])
        ego_f_agent = AgentMS(ego_follow.id, x, y, ego_follow.v, ego_follow.vy, ego_follow.vd + np.random.normal(0., 1),
                              ego_follow.a, ego_follow.l, ego_follow.w, idm_param(ego_follow.l), rss_param(False), yp, "idm_lc")
        agents.append(ego_f_agent)

    neighbors = env.neighbor_around_agent(agent.id)
    left_neighbor = neighbors.lefts
    left_corridor_id = env.map.corridors[ego_corridor.id].left_id
    if left_corridor_id is not None:
        left_corridor = env.map.corridors[left_corridor_id]
        for obj in left_neighbor:
            if obj:
                x, y = ref_line.to_frenet_frame([obj.x, obj.y])
                vehicle = AgentMS(obj.id, x, y, obj.v, obj.vy, obj.vd + np.random.normal(0., 1),
                                  obj.a, obj.l, obj.w, idm_param(obj.l), rss_param(False), yp, "idm_lc")
                agents.append(vehicle)
        corridors.append(left_corridor.to_frenet_corridor(ref_line))
        ll_corridor_id = left_corridor.left_id
        if ll_corridor_id is not None:
            ll_corridor = env.map.corridors[ll_corridor_id]
            ll_agents = env.sorted_agents_on_corridor_within_range_of_vehicle(ll_corridor_id, agent.id, 80)
            for obj in ll_agents:
                x, y = ref_line.to_frenet_frame([obj.x, obj.y])
                vehicle = AgentMS(obj.id, x, y, obj.v, obj.vy, obj.idm_param.vd + np.random.normal(0., 1),
                                  obj.a, obj.l, obj.w, idm_param(obj.l), rss_param(False), yp, "idm_lc")
                agents.append(vehicle)
            corridors.append(ll_corridor.to_frenet_corridor(ref_line))

    right_neighbor = neighbors.rights
    right_corridor_id = env.map.corridors[ego_corridor.id].right_id
    if right_corridor_id is not None:
        right_corridor = env.map.corridors[right_corridor_id]
        for obj in right_neighbor:
            if obj:
                behavior = "merging" if right_corridor.type == "merging" else "idm_lc"
                x, y = ref_line.to_frenet_frame([obj.x, obj.y])
                vehicle = AgentMS(obj.id, x, y, obj.v, obj.vy, obj.vd + np.random.normal(0., 1),
                                  obj.a, obj.l, obj.w, idm_param(obj.l), rss_param(behavior == "merging"), yp, behavior)
                agents.append(vehicle)
        corridors.append(right_corridor.to_frenet_corridor(ref_line))
        rr_corridor_id = right_corridor.right_id
        if rr_corridor_id is not None:
            rr_corridor = env.map.corridors[rr_corridor_id]
            rr_agents = env.sorted_agents_on_corridor_within_range_of_vehicle(rr_corridor_id, agent.id, 80)
            for obj in rr_agents:
                behavior = "merging" if rr_corridor.type == "merging" else "idm_lc"
                x, y = ref_line.to_frenet_frame([obj.x, obj.y])
                vehicle = AgentMS(obj.id, x, y, obj.v, obj.vy, obj.idm_param.vd + np.random.normal(0., 1),
                                  obj.a, obj.l, obj.w, idm_param(obj.l), rss_param(behavior == "merging"), yp, behavior)
                agents.append(vehicle)
            corridors.append(rr_corridor.to_frenet_corridor(ref_line))
    return corridors, agents