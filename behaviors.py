import time
import yaml

from mcs_wrapper import *

from mc_sim_highway import Action, SimParameter, MapMultiLane, Simulation, simulationMultiThread, \
    fixedDirectionLaneChangeBehavior, fixedGapMergingBehavior, closestGapMergingBehavior, laneChangeMobilBehavior
from mc_sim_highway import Environment as EnvironmentMS
from interaction_aware_miqp_planner.params import PlannerParams
from interaction_aware_miqp_planner.planner.planner import Planner
from interaction_aware_miqp_planner.planner.utils_planner import const_velocity_prediction
from interaction_aware_miqp_planner.obstacle import Obstacle as MIQPObstacle
from interaction_aware_miqp_planner.vehicle import Vehicle as MIQPVehicle
from interaction_aware_miqp_planner.environment_model import EnvModel


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
    # assuming exiting agents already indicate and can be observed by other vehicles -> exit intention known
    if corridor.left_id is not None and observed_model.map.corridors[corridor.left_id].type == "main":
        for a in observed_model.matched_agents_sorted_by_arc_length[corridor.left_id]:
            if "exit" in a[1].behavior_model:
                vehicles_idm_matches.append(observed_model.agent_to_idm_match(agent.id, a[1].id))

    for i, v in enumerate(vehicles_idm_matches):
        merging_direction = "right" if "exit" in v.agent.behavior_model else "left"
        p = agent.yielding_model.yielding_probability(v, agent, front_idm_match, observed_model, merging_direction)
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


def MIQP_merging_behavior(observed_model, agent, dt):
    # Important: merging corridor should start at the same longitudinal position as the main corridor
    vehicle_param = {'id': 0, 'x_min': [0, 0, -4, 1.0, -2, -2], 'x_max': ['GRB.INFINITY', 35, 3, 6.0, 2, 2],
                     'u_min': [-3, -2], 'u_max': [3, 2], 'Q': [0, 1, 2, 1, 2, 4], 'R': [2, 2], 'W': 1.0,
                     "dimensions": [agent.l, agent.w]}
    merge_v = None
    obstacle = None
    if (agent.left_leading_id < 0 and agent.left_following_id < 0) or agent.select_new_target_vehicles:
        neighbor = observed_model.neighbor_around_agent(agent.id)
        if neighbor.left_b:
            merge_v = neighbor.left_b.agent
            agent.left_following_id = merge_v.id
        if neighbor.left_f:
            obstacle = neighbor.left_f.agent
            agent.left_leading_id = obstacle.id
    else:
        # TODO: solve when tracked following vehicle changes lane
        merge_v = observed_model.matched_agents[agent.left_following_id] if agent.left_following_id else None
        obstacle = observed_model.matched_agents[agent.left_leading_id] if agent.left_leading_id else None

    ego_vehicle = MIQPVehicle(vehicle_param)
    if merge_v:
        vehicle_param["dimensions"] = [merge_v.l, merge_v.w]
    merge_vehicle = MIQPVehicle(vehicle_param)
    vehicles = [ego_vehicle, merge_vehicle]

    params = PlannerParams()
    ego_corridor = observed_model.corridor_for_agent(agent.id)
    ego_arc_l_on_corridor = ego_corridor.centerExtended.project(agent.get_position())
    ego_y_on_corridor = ego_corridor.centerExtended.signed_distance(agent.get_position())
    ego_dis_to_corridor_end = ego_corridor.distance_to_corridor_end(agent.get_position())
    params.x_lane_ending = ego_arc_l_on_corridor + ego_dis_to_corridor_end
    # TODO: set values
    params.tau = dt
    params.N = 40
    params.tau_sim = 0.2
    params.x_earliest_merge = 0.
    params.lane_with = ego_corridor.width
    cooperation_models = [[1., 1.], [1., 100.]]
    planner = Planner(params, cooperation_models, vehicles)

    if ego_corridor.left_id is None:
        print("MIQP: no left lane to merge!")

    main_corridor = observed_model.map.corridors[ego_corridor.left_id]

    ego_ref_v = ego_corridor.v_limit
    ego_ref_y = ego_corridor.centerSimplified.path_line.distance(main_corridor.centerSimplified.path_line)
    merge_ref_v = main_corridor.v_limit
    merge_ref_y = ego_ref_y
    ego_ref_state = np.array([0, ego_ref_v, 0, ego_ref_y, 0, 0]).reshape(-1, 1)
    merge_ref_state = np.array([0, merge_ref_v, 0, merge_ref_y, 0, 0]).reshape(-1, 1)
    ego_ref_input = np.array([0., 0.]).reshape(-1, 1)
    merge_ref_input = np.array([0., 0.]).reshape(-1, 1)

    x_init_ego = np.array([ego_arc_l_on_corridor, agent.v, 0, ego_y_on_corridor, agent.vy, 0]).reshape(-1, 1)
    x_init_merge = np.array([0., 0, 0, merge_ref_y, 0, 0]).reshape(-1, 1)
    if merge_v:
        merge_arc_l_on_corridor = ego_corridor.centerExtended.project(merge_v.get_position())
        merge_y_on_corridor = ego_corridor.centerExtended.signed_distance(merge_v.get_position())
        x_init_merge = np.array([merge_arc_l_on_corridor, merge_v.v, 0, merge_y_on_corridor, merge_v.vy, 0]).reshape(-1, 1)

    planner.initialize([ego_ref_state, merge_ref_state],
                       [ego_ref_input, merge_ref_input],
                       [x_init_ego, x_init_merge])

    env_model = EnvModel()
    env_model.obstacles = []
    obstacle_params_0 = {"x_0": np.array([1000, 20, ego_corridor.width, 0]).reshape(-1, 1), "dimensions": [5., 2.]}
    if obstacle:
        obstacle_arc_l_on_corridor = ego_corridor.centerExtended.project(obstacle.get_position())
        obstacle_y_on_corridor = ego_corridor.centerExtended.signed_distance(obstacle.get_position())
        obstacle_params_0 = {"x_0": np.array([obstacle_arc_l_on_corridor, obstacle.v,
                                              obstacle_y_on_corridor, obstacle.vy]).reshape(-1, 1),
                             "dimensions": [obstacle.l, obstacle.w]}
    obstacle_0 = MIQPObstacle(obstacle_params_0)
    obstacle_0.prediction = const_velocity_prediction(obstacle_0, params.N, params.tau)
    env_model.obstacles.append(obstacle_0)
    env_model.ego_state = x_init_ego
    # create observation of the merge vehicle (observation = real)
    #  [x, vx, y, vy]
    observation_merge = np.array([x_init_merge[0], x_init_merge[1], x_init_merge[3], x_init_merge[4]]).reshape(-1, 1)
    env_model.observation = observation_merge

    ax, vy, select_new_neighbors = planner.run(env_model)
    agent.select_new_target_vehicles = select_new_neighbors
    return DecoupledAction(ax, vy, ego_corridor.centerSimplified)


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