import time

from mcs_wrapper import *

from mc_sim_highway import Action, SimParameter, MapMultiLane, Simulation, simulationMultiThread, \
    fixedDirectionLaneChangeBehavior, fixedGapMergingBehavior, closestGapMergingBehavior, laneChangeMobilBehavior
from mc_sim_highway import Environment as EnvironmentMS


def idm_behavior(observed_model, agent):
    corridor = observed_model.corridor_for_agent(agent.id)
    # idm behavior to front vehicle
    front_idm_match = observed_model.front_agent(agent.id)
    if front_idm_match:
        ax = idm_acc_f_b([front_idm_match], None, agent.v, corridor.v_limit, agent.idm_param)
    else:
        ax = idm_acc_f_b([], None, agent.v, corridor.v_limit, agent.idm_param)

    # lateral behavior -> attach to centerLine
    signed_distance = corridor.centerSimplified.signed_distance([agent.x, agent.y])
    vy_abs = min(max(abs(signed_distance) * 1.0, 0.1), 1.3)
    vy = -signed_distance / (abs(signed_distance) + 1e-10) * vy_abs

    vehicles_idm_matches = []
    # yielding behavior to vehicles on merging lane
    if corridor.right_id is not None and observed_model.map.corridors[corridor.right_id].type == "merging":
        neighbor = observed_model.neighbor_around_agent(agent.id)
        vehicles = [neighbor.right_bb, neighbor.right_b, neighbor.right_f, neighbor.right_ff]
        vehicles = [v for v in vehicles if v is not None]
        vehicles_idm_matches.extend(vehicles)
    # add left exit agents
    if corridor.left_id is not None and observed_model.map.corridors[corridor.left_id].type == "main":
        for a in observed_model.matched_agents_sorted_by_arc_length[corridor.left_id]:
            if "exit" in a[1].behavior_model:
                vehicles_idm_matches.append(observed_model.agent_to_idm_match(agent.id, a[1].id))

    for i, v in enumerate(vehicles_idm_matches):
        p = agent.yielding_model.yielding_probability(v, agent)
        # initialize or update intention to others
        if v.id not in agent.yielding_intention_to_others.keys():
            agent.yielding_intention_to_others[vehicles_idm_matches[i].id] = \
                YieldingIntention(vehicles_idm_matches[i].id, p,
                                  change_intention_threshold=agent.change_intention_threshold,
                                  random_intention_seed=agent.random_yielding_seed)
        else:
            agent.yielding_intention_to_others[vehicles_idm_matches[i].id].update_intention(p)
    # remove intention to currently uninterested vehicles
    vehicle_ids = [a.id for a in vehicles_idm_matches]
    for idx in list(agent.yielding_intention_to_others.keys()):
        if idx not in vehicle_ids:
            agent.yielding_intention_to_others.pop(idx)

    yielding_vehicles = []
    for v in vehicles_idm_matches:
        if agent.yielding_intention_to_others[v.id].yielding:
            yielding_vehicles.append(v)
    if len(yielding_vehicles) > 0:
        ax = min(ax, idm_acc_f_b(yielding_vehicles, None, agent.v, corridor.v_limit,
                                 idm_p_for_merging_exit_agents(agent.random_yielding_seed, agent.idm_param, agent.l>7)))

    if front_idm_match and rss_dis(front_idm_match.v, agent.v, agent.rss_param.r_t_ego,
                                   agent.rss_param.a_min, agent.rss_param.a_max) > \
            front_idm_match.dis - 0.5 * (front_idm_match.l + agent.l):
        ax = agent.rss_param.a_min
    return DecoupledAction(ax, vy, corridor.centerSimplified)


def lane_change_behavior_mobil(observed_model, agent, require_rss_safety):
    # copy from mc simulation
    ego_corridor = observed_model.corridor_for_agent(agent.id)
    action_idm = idm_behavior(observed_model, agent)
    corridors, mcs_agents = lane_change_mcs_sim_from_env(observed_model, agent, ego_corridor)
    for a in mcs_agents:
        if a.id == agent.id:
            a.targetLaneId = agent.target_lane_id
            a.commitToDecision = agent.commit_lane_change
            a.commitKeepLaneStep = agent.commit_keep_lane_step
            break
    action = laneChangeMobilBehavior(EnvironmentMS(MapMultiLane(corridors), mcs_agents),
                                     agent.id, Action(action_idm.ax, action_idm.vy), require_rss_safety)
    agent.commit_lane_change = action.commitToDecision
    agent.commit_keep_lane_step = action.commitKeepLaneStep
    agent.target_lane_id = action.targetLaneId
    return DecoupledAction(action.ax, action.vy, ego_corridor.centerSimplified)


def lane_change_behavior_learned(observed_model, agent):
    # manually add cost for lane change: right = +0.02 < keep = 0 < left = -0.02
    # possible_actions = ["left", "keep_lane", "right", "dcc", "acc"]
    possible_actions = ["keep_lane", "dcc", "acc"]
    ego_corridor = observed_model.corridor_for_agent(agent.id)
    if ego_corridor.left_id and observed_model.map.corridors[ego_corridor.left_id].type == "main":
        possible_actions.append("left")
    if ego_corridor.right_id and observed_model.map.corridors[ego_corridor.right_id].type == "main":
        possible_actions.append("right")
    corridors, mcs_agents = lane_change_mcs_sim_from_env(observed_model, agent, ego_corridor)
    sim_param = SimParameter(0.3, 40, 30)
    mcs_map = MapMultiLane(corridors)

    features = []
    debug_features = {}
    for action in possible_actions:
        feature = simulationMultiThread(20, sim_param, mcs_map, mcs_agents, agent.id, action, -2)
        f = [feature.successRate, feature.fallBackRate, feature.utility, feature.comfort,
             feature.expectedStepRatio, feature.utilityObjects, feature.comfortObjects]
        features.append(f)
        debug_features[action] = [round(x, 2) for x in f]
    q_values, decision = agent.learned_lane_change_model.inference(features, possible_actions, agent.current_decision)
    agent.current_decision = decision
    # print(debug_features, "q_values: ", q_values, " decision: ", decision)

    action = fixedDirectionLaneChangeBehavior(EnvironmentMS(mcs_map, mcs_agents), agent.id, decision)
    return DecoupledAction(action.ax, action.vy, ego_corridor.centerSimplified)


def cloned_merging_behavior(observed_model, agent):
    ego_corridor = observed_model.corridor_for_agent(agent.id)
    corridors, mcs_agents, possible_actions = merging_mcs_sim_from_env(observed_model, agent, ego_corridor)
    mcs_map = MapMultiLane(corridors)
    sim_param = SimParameter(0.3, 40, 10)
    features = []
    debug_features = {}
    for gap_id in possible_actions:
        feature = simulationMultiThread(30, sim_param, mcs_map, mcs_agents, agent.id, "left", gap_id)
        f = [feature.successRate, feature.fallBackRate, feature.utility, feature.comfort,
             feature.expectedStepRatio, feature.utilityObjects, feature.comfortObjects]
        features.append(f)
        debug_features[gap_id] = [round(x, 4) for x in f]

    action_index, q_values = agent.learned_merging_model.inference(features)
    decision = possible_actions[action_index]
    agent.current_decision = decision
    action = fixedGapMergingBehavior(EnvironmentMS(mcs_map, mcs_agents), agent.id, "left", decision)
    # print(agent.id, debug_features, "q_values: ", q_values, " decision: ", decision)
    return DecoupledAction(action.ax, action.vy, ego_corridor.centerSimplified)


def merging_behavior_closest_gap(observed_model, agent):
    # copy from mc simulation
    ego_corridor = observed_model.corridor_for_agent(agent.id)
    corridors, mcs_agents, _ = merging_mcs_sim_from_env(observed_model, agent, ego_corridor)
    action = closestGapMergingBehavior(EnvironmentMS(MapMultiLane(corridors), mcs_agents), agent.id, "left")
    return DecoupledAction(action.ax, action.vy, ego_corridor.centerSimplified)


def exit_behavior_closest_gap(observed_model, agent):
    # copy from mc simulation
    ego_corridor = observed_model.corridor_for_agent(agent.id)
    corridors, mcs_agents, _ = exit_mcs_sim_from_env(observed_model, agent, ego_corridor)
    action = closestGapMergingBehavior(EnvironmentMS(MapMultiLane(corridors), mcs_agents), agent.id, "right")
    return DecoupledAction(action.ax, action.vy, ego_corridor.centerSimplified)


def cloned_exit_behavior(observed_model, agent):
    ego_corridor = observed_model.corridor_for_agent(agent.id)
    corridors, mcs_agents, possible_actions = exit_mcs_sim_from_env(observed_model, agent, ego_corridor)
    mcs_map = MapMultiLane(corridors)
    sim_param = SimParameter(0.3, 40, 10)

    features = []
    debug_features = {}
    for gap_id in possible_actions:
        feature = simulationMultiThread(20, sim_param, mcs_map, mcs_agents, agent.id, "right", gap_id)
        f = [feature.successRate, feature.fallBackRate, feature.utility, feature.comfort,
             feature.expectedStepRatio, feature.utilityObjects, feature.comfortObjects]
        features.append(f)
        debug_features[gap_id] = [round(x, 4) for x in f]
    action_index, q_values = agent.learned_merging_model.inference(features)
    decision = possible_actions[action_index]

    agent.current_decision = decision
    action = fixedGapMergingBehavior(EnvironmentMS(mcs_map, mcs_agents), agent.id, "right", decision)
    # print(agent.id, debug_features, "q_values: ", q_values, " decision: ", decision)
    return DecoupledAction(action.ax, action.vy, ego_corridor.centerSimplified)


def visualize_one_sim_multilane(history, corridors, ego_id, ids, write_gif=False):
    obj_histories = {}
    for idx in ids:
        obj_histories[idx] = [[p.x, p.y, p.v, p.a] for p in history[idx]]
    # print([p[2] for p in obj_histories[3]])
    if not plt.fignum_exists(1):
        fig = plt.figure(figsize=(6.4 * 2, 4.8 * 2))
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = plt.figure(num=1)
        ax = fig.axes[0]
    step = 0
    for i in range(len(obj_histories[ids[0]])):
        for c in corridors:
            plt.plot([c.m.p1.x, c.m.p2.x], [c.m.p1.y, c.m.p2.y], "--", color="b")
        for idx, obj in obj_histories.items():
            color = "black"
            if idx == ego_id:
                color = "blue"
            obj_rect = plt.Rectangle((obj[i][0], obj[i][1] - 1), 7, 3, color=color)
            # ax.annotate(str(idx), (obj[i][0] + 2.5, obj[i][1]), color='white', weight='bold', fontsize=6, ha='center', va='center')
            ax.annotate(str(round(obj[i][2], 1)), (obj[i][0] + 2.5, obj[i][1]),
                        color='white', weight='bold', fontsize=6, ha='center', va='center')
            ax.add_patch(obj_rect)
        plt.axis('equal')
        # ax.set_xlim([-10, 250])
        # ax.set_ylim([-5, 15])
        plt.draw()
        plt.pause(0.1)
        ax.patches = []
        ax.clear()
        step += 1