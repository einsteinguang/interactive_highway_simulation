from simulation_multilane import *


def generate_random_lc_traffic():
    sim_param = SimulationParameter(0.2, 100, render=False, write_gif=False, show_debug_info=False)
    sim = Simulation(None, sim_param)

    traffic_generator = RandomTrafficGenerator()
    random_traffic_param = {0: RandomDensityAndVelocityParameter(1.0, 0.1, -2, 2, None, 2),
                            1: RandomDensityAndVelocityParameter(1.8, 0.4, 0, 2, [0, 500], None),
                            2: RandomDensityAndVelocityParameter(2.0, 1.0, 0, 2, [0, 500], None),
                            3: RandomDensityAndVelocityParameter(2.0, 1.0, 0, 2, [0, 500], None)}
    ratio_of_unsafe_agents = 0.
    ratio_of_truck = 0.4
    num_epochs = 500

    for i in range(num_epochs):
        v_limit = float(np.random.choice([100 / 3.6], 1)[0])
        merging_lane_length = v_limit ** 2 / 4 + 20
        merging_corridor = Corridor(0, "merging",
                                    [[-10, 3.5], [merging_lane_length + 10, 3.5]],
                                    [[-10, 0], [merging_lane_length, 0]],
                                    [[-10, 1.75], [merging_lane_length + 5, 1.75]],
                                    v_limit)
        main_corridor = Corridor(1, "main",
                                 [[-200, 7], [500, 7]],
                                 [[-200, 3.5], [500, 3.5]],
                                 [[-200, 5.25], [500, 5.25]],
                                 v_limit)
        main_corridor2 = Corridor(2, "main",
                                  [[-200, 10.5], [500, 10.5]],
                                  [[-200, 7], [500, 7]],
                                  [[-200, 8.75], [500, 8.75]],
                                  v_limit + 20 / 3.6)
        main_corridor3 = Corridor(3, "main",
                                  [[-200, 10.5 + 3.5], [500, 10.5 + 3.5]],
                                  [[-200, 10.5], [500, 10.5]],
                                  [[-200, 8.75 + 3.5], [500, 8.75 + 3.5]],
                                  v_limit + 30 / 3.6)
        merging_corridor.set_left_and_right_corridor_id(1, None)
        main_corridor.set_left_and_right_corridor_id(2, 0)
        main_corridor2.set_left_and_right_corridor_id(3, 1)
        main_corridor3.set_left_and_right_corridor_id(None, 2)
        lc_map = Map([merging_corridor, main_corridor, main_corridor2, main_corridor3])
        # random unsafe lc ratio, random truck ratio
        agents_merging, agents_main = \
            traffic_generator.random_agents_on_map(lc_map, random_traffic_param, ratio_of_unsafe_agents, ratio_of_truck)
        agents_dicts_merging = create_dicts_from_agents(agents_merging)
        agents_dicts_main = create_dicts_from_agents(agents_main)
        env = Environment(lc_map,
                          create_agents_from_dicts(agents_dicts_merging) +
                          create_agents_from_dicts(agents_dicts_main))
        sim.reset(env, sim_param)
        sim.dump_initial_scene()
        sim.epoch += 1
        traffic_generator.vehicle_id = 0


def evaluate_recorded_lc_scene(path, sim, ego_policy, epoch_id, ego_id, statistics):
    # path: scene path
    sim.reset(None, sim.param)
    sim.load_initial_scene(path)
    for _, a in sim.environment.agents.items():
        if a.id == ego_id:
            a.reset_behavior_model(ego_policy)
    sim.run(os.path.join(path, "fig"))
    # average utility ego
    print_output = ego_policy + " with id {}: ".format(ego_id)
    if ego_id not in sim.environment.agents.keys() and ego_id not in sim.environment.removed_agents.keys():
        statistics.utility_ego.append(-10)
        statistics.comfort_ego.append(-10)
        statistics.utility_other.append(-10)
        statistics.comfort_other.append(-10)
        statistics.epoch_and_ego_id_history.append([-1, -1])
        print_output += statistics.__repr__() + "\n"
        return print_output
    agent = sim.environment.agents[ego_id] if ego_id in sim.environment.agents.keys() \
        else sim.environment.removed_agents[ego_id]
    mean_utility_ego = agent.average_utility()
    min_comfort_ego = agent.minimum_comfort()
    if agent.had_harsh_brake():
        statistics.num_fallback += 1
    statistics.num_lane_change += agent.num_lane_change()
    statistics.lane_id_history.append(agent.corridor_history[-1])
    statistics.epoch_and_ego_id_history.append([epoch_id, ego_id])
    statistics.utility_ego.append(mean_utility_ego)
    statistics.comfort_ego.append(min_comfort_ego)
    # minimum utility and comfort others
    mean_utility_others = 1e10
    mean_comfort_others = 1e10
    for _, a in sim.environment.agents.items():
        if a.id != ego_id:
            mean_comfort_others = min(a.minimum_comfort(), mean_comfort_others)
            mean_utility_others = min(a.average_utility(), mean_utility_others)
    for _, a in sim.environment.removed_agents.items():
        if a.id != ego_id:
            mean_comfort_others = min(a.minimum_comfort(), mean_comfort_others)
            mean_utility_others = min(a.average_utility(), mean_utility_others)
    statistics.utility_other.append(mean_utility_others)
    statistics.comfort_other.append(mean_comfort_others)
    print_output += statistics.__repr__() + "\n"
    return print_output


def evaluate_recorded_lc_scenes():
    root_dirs = ["500sim_4lanes_1.8thw_safe_lc_truck", "500sim_4lanes_1.8thw_unsafe_lc_truck"]
    param = SimulationParameter(0.2, 50, render=True, write_gif=False, show_debug_info=False, scenario="lane_change")
    sim = Simulation(None, param)
    for root_dir in root_dirs:
        print_output = ""
        statistics_idm_only = LaneChangeStatistics()
        statistics_baseline = LaneChangeStatistics()
        statistics_learning = LaneChangeStatistics()
        for n in np.arange(0, 500):
            epoch_dir = os.path.join(root_dir, "epoch" + str(n))
            new_print = "processing epoch" + str(n) + "\n"
            if os.path.isdir(epoch_dir):
                ego_ids = []
                sim.reset(None, sim.param)
                sim.load_initial_scene(epoch_dir)
                # check 2. to 4. vehicles ids of each lane
                for c_id, sorted_agents in sim.environment.matched_agents_sorted_by_arc_length.items():
                    if sim.environment.map.corridors[c_id].type == "main":
                        for _, a in sorted_agents[2:5]:
                            # no trucks will be ego vehicle
                            if a.l <= 7:
                                ego_ids.append(a.id)
                for i in ego_ids:
                    sim.ego_id = i
                    print("ego id: ", i)
                    new_print += evaluate_recorded_lc_scene(epoch_dir, sim, "idm", n, i,
                                                            statistics_idm_only)
                    new_print += evaluate_recorded_lc_scene(epoch_dir, sim, "idm_mobil_lane_change_safe", n, i,
                                                            statistics_baseline)
                    new_print += evaluate_recorded_lc_scene(epoch_dir, sim, "lane_change_learned", n, i,
                                                            statistics_learning)
                print(new_print)
            #     print_output += new_print
            # statistics_idm_only.save(root_dir, "idm_only_final")
            # statistics_baseline.save(root_dir, "baseline_final")
            # statistics_learning.save(root_dir, "learning_final")
            with open(os.path.join(root_dir, 'statistics_lane_change.txt'), 'w') as f:
                f.write(print_output)


def plot_lc_statistics():
    root_dir = "500sim_4lanes_1.8thw_safe_lc_truck"
    statistics_idm_only = LaneChangeStatistics()
    statistics_baseline = LaneChangeStatistics()
    statistics_learning = LaneChangeStatistics()
    statistics_idm_only.load(root_dir, "idm_only_final")
    statistics_baseline.load(root_dir, "baseline_final")
    statistics_learning.load(root_dir, "learning_final")
    # statistics_idm_only.on_lane_number(1)
    # statistics_baseline.on_lane_number(1)
    # statistics_learning.on_lane_number(1)
    print(statistics_idm_only.__repr__(True) + "\n")
    print(statistics_baseline.__repr__(True) + "\n")
    print(statistics_learning.__repr__(True) + "\n")
    plt.figure()
    ax1 = plt.subplot(411)
    plt.hist(statistics_baseline.comfort_ego, 200, density=True, facecolor='r', alpha=0.5)
    plt.hist(statistics_learning.comfort_ego, 200, density=True, facecolor='b', alpha=0.5)
    plt.hist(statistics_idm_only.comfort_ego, 200, density=True, facecolor='g', alpha=0.5)
    ax2 = plt.subplot(412)
    plt.hist(statistics_baseline.utility_ego, 200, density=True, facecolor='r', alpha=0.5)
    plt.hist(statistics_learning.utility_ego, 200, density=True, facecolor='b', alpha=0.5)
    plt.hist(statistics_idm_only.utility_ego, 200, density=True, facecolor='g', alpha=0.5)
    ax3 = plt.subplot(413)
    plt.hist(statistics_baseline.comfort_other, 200, density=True, facecolor='r', alpha=0.5)
    plt.hist(statistics_learning.comfort_other, 200, density=True, facecolor='b', alpha=0.5)
    plt.hist(statistics_idm_only.comfort_other, 200, density=True, facecolor='g', alpha=0.5)
    ax4 = plt.subplot(414)
    plt.hist(statistics_baseline.utility_other, 200, density=True, facecolor='r', alpha=0.5)
    plt.hist(statistics_learning.utility_other, 200, density=True, facecolor='b', alpha=0.5)
    plt.hist(statistics_idm_only.utility_other, 200, density=True, facecolor='g', alpha=0.5)
    plt.show()


if __name__ == '__main__':
    generate_random_lc_traffic()
    # evaluate_recorded_lc_scenes()
    # plot_lc_statistics()