import os, datetime
import imageio
import yaml

from environment import *
from random_traffic_generator import *


class SimulationParameter:
    def __init__(self, dt, total_steps, render=True, write_gif=True, show_debug_info=False, scenario="lane_change"):
        self.dt = dt
        self.total_steps = total_steps
        self.render = render
        self.write_gif = write_gif
        self.show_debug_info = show_debug_info
        self.scenario = scenario


def load_yaml(path):
    with open(path) as infile:
        data = yaml.safe_load(infile)
    return data


class Simulation:
    def __init__(self, env, param):
        self.environment = env
        self.param = param
        self.step = 0
        self.fig = plt.figure(figsize=(18, 4.))
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.write_dir = os.path.join(os.getcwd(), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.initial_agents_dict = {}
        if env:
            for i, a in self.environment.agents.items():
                self.initial_agents_dict[i] = a.to_dict()
        if param.write_gif:
            self.gif_num = 0
        self.epoch = 0
        self.ego_id = -1

    def reset(self, env, param):
        self.environment = env
        self.initial_agents_dict = {}
        if self.environment is not None:
            for i, a in self.environment.agents.items():
                self.initial_agents_dict[i] = a.to_dict()
        self.param = param
        self.step = 0
        self.ax.patches = []
        self.ax.clear()

    def add_map_patch(self):
        map_patch = []
        for _, corridor in self.environment.map.corridors.items():
            map_patch.append(corridor.border_patch)
            # map_patch.append(corridor.center_patch)
        return map_patch

    def add_agents_patch(self):
        agents_patch = []
        for _, agent in self.environment.agents.items():
            color = "black"
            # if "merging" in agent.behavior_model:
            #     color = "blue"
            if agent.id == self.ego_id:
                color = "blue"
                if "learn" in agent.behavior_model and "lane_change" in agent.behavior_model:
                    if agent.current_decision == "keep_lane":
                        agents_patch.append(plt.Arrow(agent.x + 5, agent.y, 12, 12 * math.sin(agent.yaw), color='green', width=5))
                    if agent.current_decision == "acc":
                        agents_patch.append(plt.Arrow(agent.x + 5, agent.y, 20, 20 * math.sin(agent.yaw), color='green', width=5))
                    if agent.current_decision == "dcc":
                        agents_patch.append(plt.Arrow(agent.x + 5, agent.y, 6, 6 * math.sin(agent.yaw), color='green', width=5))
                    if agent.current_decision == "left":
                        agents_patch.append(plt.Arrow(agent.x + 5, agent.y, -4 * math.sin(agent.yaw), 4 * math.cos(agent.yaw), color='green', width=5))
                    if agent.current_decision == "right":
                        agents_patch.append(plt.Arrow(agent.x + 5, agent.y, 4 * math.sin(agent.yaw), -4 * math.cos(agent.yaw), color='green', width=5))
                # self.ax.annotate(str(round(agent.v, 2)) + "m/s, vd: " + str(round(agent.idm_param.vd, 2)),
                #                  (agent.x, agent.y + 5), color='black', weight='bold', fontsize=9)
                # self.ax.annotate(str(round(agent.a, 1)) + "m/s2", (agent.x, agent.y + 2),
                #                  color='black', weight='bold', fontsize=7, ha='center', va='center')
                # self.ax.annotate(agent.current_decision, (agent.x + 5, agent.y + 5),
                #                  color='red', weight='bold', fontsize=10, ha='center', va='center')
            patch = plt.Polygon(agent.hull, closed=True, fill=True, color=color)
            # self.ax.annotate(str(agent.id), (agent.x, agent.y),
            #                  color='white', weight='bold', fontsize=10, ha='center', va='center')
            self.ax.annotate(str(round(agent.v, 1)) + "m/s", (agent.x, agent.y + 3),
                             color='red', weight='bold', fontsize=12, ha='center', va='center')
            # self.ax.annotate("vy: " + str(round(agent.vy, 2)) + "m/s", (agent.x, agent.y + 5),
            #                  color='red', weight='bold', fontsize=7, ha='center', va='center')
            if self.param.show_debug_info:
                offset = 4
                for i, intention in agent.yielding_intention_to_others.items():
                    if intention.yielding:
                        self.ax.annotate("Y to " + str(i), (agent.x, agent.y + offset),
                                         color='green', weight='bold', fontsize=7, ha='center', va='center')
                    else:
                        self.ax.annotate("NY to " + str(i), (agent.x, agent.y + offset),
                                         color='red', weight='bold', fontsize=7, ha='center', va='center')
                    offset += 2
            agents_patch.append(patch)
        return agents_patch

    def if_terminate(self):
        # set termination condition
        if self.param.scenario == "exit":
            if self.environment.all_finish_exit():
                return True
        if self.param.scenario == "merging":
            if self.environment.all_finish_merging():
                return True
        return False

    def load_initial_scene(self, path):
        self.step = 0
        self.ax.patches = []
        self.ax.clear()
        agents_dict = load_yaml(os.path.join(path, "agents.yml"))
        agents = []
        for _, a in agents_dict.items():
            agents.append(create_agent_from_dict(a))
        map_dict = load_yaml(os.path.join(path, "map.yml"))
        corridors = []
        for _, c in map_dict.items():
            corridors.append(create_corridor_from_dict(c))
        m = Map(corridors)
        self.environment = Environment(m, agents)
        self.initial_agents_dict = {}
        for i, a in self.environment.agents.items():
            self.initial_agents_dict[i] = a.to_dict()

    def dump_initial_scene(self):
        self.to_yaml(self.initial_agents_dict, "agents")
        self.to_yaml(self.environment.map.to_dict(), "map")

    def dump_print_outputs(self, output):
        if not os.path.exists(self.write_dir):
            os.makedirs(self.write_dir)
        with open(os.path.join(self.write_dir, 'console_output.txt'), 'w') as f:
            f.write(output)

    def to_yaml(self, data, data_name):
        if not os.path.exists(self.write_dir):
            os.makedirs(self.write_dir)
        data_dir = os.path.join(self.write_dir, "epoch" + str(self.epoch))
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        path = os.path.join(data_dir, data_name + '.yml')
        with open(path, 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)
        return path

    def visualize_current_scene(self, path=None):
        for p in self.add_map_patch():
            self.ax.add_patch(copy.copy(p))
        for p in self.add_agents_patch():
            self.ax.add_patch(copy.copy(p))
        plt.autoscale()
        plt.axis("equal")
        xmin, xmax, ymin, ymax = self.environment.limit_of_maps()
        plt.xlim((xmin, xmax))
        plt.ylim((ymin, ymax))
        plt.tight_layout()
        plt.show()
        # plt.pause(0.1)
        if self.param.write_gif and os.path.exists(path):
            plt.savefig(os.path.join(path, 'scene.png'))

    def run(self, write_fig_dir=None):
        if not write_fig_dir:
            write_fig_dir = self.write_dir
        if self.param.write_gif and self.param.render:
            if not os.path.exists(write_fig_dir):
                os.makedirs(write_fig_dir)

        terminate_counter = 0
        for step in range(self.param.total_steps):
            if self.param.render or self.param.write_gif:
                for p in self.add_map_patch():
                    self.ax.add_patch(copy.copy(p))
                for p in self.add_agents_patch():
                    self.ax.add_patch(copy.copy(p))

            self.environment.step(self.param.dt)
            self.step = step
            # plt.autoscale()
            plt.axis("equal")
            xmin, xmax, ymin, ymax = self.environment.limit_of_maps()
            plt.xlim((xmin, xmax))
            plt.ylim((ymin, ymax))
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.xlabel(r'$x$(m)', fontsize=15)
            plt.ylabel(r'$y$(m)', fontsize=15)
            plt.tight_layout()
            if self.param.write_gif:
                plt.savefig(os.path.join(write_fig_dir, 'gif_' + str(self.gif_num) + "_" + str(step) + '.png'))
            if self.param.render:
                plt.draw()
                plt.pause(0.001)
            if self.param.render or self.param.write_gif:
                self.ax.patches = []
                self.ax.clear()

            # run 1s more simulation after termination
            if self.if_terminate():
                if terminate_counter * self.param.dt > 0.4:
                    break
                terminate_counter += 1

        if self.param.write_gif:
            filenames = []
            for i in np.arange(1, self.step + 1):
                filenames.append(os.path.join(write_fig_dir, 'gif_' + str(self.gif_num) + "_" + str(i) + '.png'))
            images = []
            for filename in filenames:
                image = imageio.imread(filename)
                images.append(image)
            imageio.mimsave(os.path.join(write_fig_dir, 'gif_' + str(self.gif_num) + '.gif'), images)
            self.gif_num += 1


def test_dump_txt():
    sim_param = SimulationParameter(0.2, 100, render=True, write_gif=False)
    sim = Simulation(None, sim_param)
    statistics = MergingAndExitStatistics()
    print_output = "Finish {} simulation, {} %\n".format(0, 0 / 1) + \
                   "Baseline: " + statistics.__repr__() + "\n"
    sim.dump_print_outputs(print_output)


def test_load_scene():
    sim_param = SimulationParameter(0.2, 50, render=True, write_gif=True, show_debug_info=False)
    sim = Simulation(None, sim_param)
    sim.load_initial_scene("test_scene/epoch0")
    sim.run()


if __name__ == '__main__':
    test_load_scene()

