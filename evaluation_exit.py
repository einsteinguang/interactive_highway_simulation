from simulation_multilane import *


def generate_random_exit_traffic():
    sim_param = SimulationParameter(0.2, 100, render=False, write_gif=False, show_debug_info=False)
    sim = Simulation(None, sim_param)

    traffic_generator = RandomTrafficGenerator()
    random_traffic_param = {0: RandomDensityAndVelocityParameter(1.0, 0.1, -2, 2, None, 0),
                            1: RandomDensityAndVelocityParameter(1.6, 0.4, 0, 2, [0, 500], None),
                            2: RandomDensityAndVelocityParameter(2.0, 1.0, 0, 2, [0, 500], None),
                            3: RandomDensityAndVelocityParameter(2.0, 1.0, 0, 2, [0, 500], None)}
    ratio_of_unsafe_agents = 0.
    ratio_of_truck = 0.4
    num_epochs = 500

    for i in range(num_epochs):
        v_limit = float(np.random.choice([100 / 3.6], 1)[0])
        merging_corridor = Corridor(0, "exit",
                                    [[245, 3.5], [500, 3.5]],
                                    [[255, 0], [500, 0]],
                                    [[250, 1.75], [500 + 5, 1.75]],
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
        _, agents_main = \
            traffic_generator.random_agents_on_map(lc_map, random_traffic_param, ratio_of_unsafe_agents, ratio_of_truck)
        agents_dicts_main = create_dicts_from_agents(agents_main)
        env = Environment(lc_map, create_agents_from_dicts(agents_dicts_main))
        sim.reset(env, sim_param)
        sim.dump_initial_scene()
        sim.epoch += 1
        traffic_generator.vehicle_id = 0


def evaluate_recorded_exit_scene(path, sim, ego_policy, epoch_id, ego_id, statistics, max_fall_back_ratio=1.0):
    # path: scene path
    sim.reset(None, sim.param)
    sim.load_initial_scene(path)
    for _, a in sim.environment.agents.items():
        if a.id == ego_id:
            a.reset_behavior_model(ego_policy)
            if "learn" in ego_policy:
                a.learned_merging_model.learned_model_parameter.max_fall_back_rate = max_fall_back_ratio
    sim.run(os.path.join(path, "fig"))
    # average utility ego
    print_output = ego_policy + " with id {}: ".format(ego_id)
    if ego_id not in sim.environment.agents.keys() and ego_id not in sim.environment.removed_agents.keys():
        statistics.utility_ego.append(-10)
        statistics.comfort_ego.append(-10)
        statistics.utility_other.append(-10)
        statistics.comfort_other.append(-10)
        statistics.total_t.append(-10)
        statistics.epoch_and_ego_id_history.append([-1, -1])
        print_output += statistics.__repr__() + "\n"
        return print_output
    agent = sim.environment.agents[ego_id] if ego_id in sim.environment.agents.keys() \
        else sim.environment.removed_agents[ego_id]
    mean_utility_ego = agent.average_utility()
    min_comfort_ego = agent.minimum_comfort()

    statistics.num_total_agents += 1
    if agent.exit_flag.t_finish_exit is not None:
        statistics.total_t.append(agent.exit_flag.t_finish_exit)
    else:
        statistics.total_t.append(sim.param.dt * sim.param.total_steps)
    statistics.num_finish += agent.exit_flag.finish_exit
    statistics.num_fallback += agent.exit_flag.once_fallback
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


def evaluate_recorded_exit_scenes():
    root_dirs = ["500sim_4lanes_1.6thw_exit_truck"]
    param = SimulationParameter(0.2, 100, render=True, write_gif=False, show_debug_info=False, scenario="exit")
    sim = Simulation(None, param)
    for root_dir in root_dirs:
        print_output = ""
        statistics_baseline = MergingAndExitStatistics()
        statistics_learning = MergingAndExitStatistics()
        statistics_learning_02_fb = MergingAndExitStatistics()
        for n in np.arange(0, 500):
            epoch_dir = os.path.join(root_dir, "epoch" + str(n))
            new_print = "processing epoch" + str(n) + "\n"
            if os.path.isdir(epoch_dir):
                ego_ids = []
                sim.reset(None, sim.param)
                sim.load_initial_scene(epoch_dir)
                # check 2. to 4. vehicles ids of each lane
                for c_id, sorted_agents in sim.environment.matched_agents_sorted_by_arc_length.items():
                    if c_id == 2 or c_id == 3:
                        for _, a in sorted_agents[2:4]:
                            # no trucks will be ego vehicle
                            if a.l <= 7:
                                ego_ids.append(a.id)
                for i in ego_ids:
                    sim.ego_id = i
                    new_print += evaluate_recorded_exit_scene(epoch_dir, sim, "exit", n, i,
                                                              statistics_baseline)
                    new_print += evaluate_recorded_exit_scene(epoch_dir, sim, "exit_learned", n, i,
                                                              statistics_learning)
                    new_print += evaluate_recorded_exit_scene(epoch_dir, sim, "exit_learned", n, i,
                                                              statistics_learning_02_fb, 0.2)
                print(new_print)
                print_output += new_print
            # statistics_baseline.save(root_dir, "baseline")
            # statistics_learning.save(root_dir, "learning")
            # statistics_learning_02_fb.save(root_dir, "learning_02_fb")
            with open(os.path.join(root_dir, 'statistics_exit.txt'), 'w') as f:
                f.write(print_output)


def plot_exit_statistics():
    root_dir = "500sim_4lanes_1.6thw_exit_truck"
    statistics_baseline = MergingAndExitStatistics()
    statistics_learning = MergingAndExitStatistics()
    statistics_learning_02_fb = MergingAndExitStatistics()
    statistics_baseline.load(root_dir, "baseline")
    statistics_learning.load(root_dir, "learning")
    statistics_learning_02_fb.load(root_dir, "learning_02_fb")
    print(statistics_baseline.__repr__(True) + "\n")
    print(statistics_learning.__repr__(True) + "\n")
    print(statistics_learning_02_fb.__repr__(True) + "\n")
    plt.figure()
    ax1 = plt.subplot(411)
    plt.hist(statistics_baseline.comfort_ego, 200, density=True, facecolor='r', alpha=0.5)
    plt.hist(statistics_learning.comfort_ego, 200, density=True, facecolor='b', alpha=0.5)
    plt.hist(statistics_learning_02_fb.comfort_ego, 200, density=True, facecolor='g', alpha=0.5)
    ax2 = plt.subplot(412)
    plt.hist(statistics_baseline.utility_ego, 200, density=True, facecolor='r', alpha=0.5)
    plt.hist(statistics_learning.utility_ego, 200, density=True, facecolor='b', alpha=0.5)
    plt.hist(statistics_learning_02_fb.utility_ego, 200, density=True, facecolor='g', alpha=0.5)
    ax3 = plt.subplot(413)
    plt.hist(statistics_baseline.comfort_other, 200, density=True, facecolor='r', alpha=0.5)
    plt.hist(statistics_learning.comfort_other, 200, density=True, facecolor='b', alpha=0.5)
    plt.hist(statistics_learning_02_fb.comfort_other, 200, density=True, facecolor='g', alpha=0.5)
    ax4 = plt.subplot(414)
    plt.hist(statistics_baseline.utility_other, 200, density=True, facecolor='r', alpha=0.5)
    plt.hist(statistics_learning.utility_other, 200, density=True, facecolor='b', alpha=0.5)
    plt.hist(statistics_learning_02_fb.utility_other, 200, density=True, facecolor='g', alpha=0.5)
    plt.show()


if __name__ == '__main__':
    # generate_random_exit_traffic()
    evaluate_recorded_exit_scenes()
    # plot_exit_statistics()