import numpy as np


def filter_actions(valid_features, q_values, sr_threshold, fallback_threshold):
    # limit success ratio
    sorted_q_values = sorted(q_values.items(), key=lambda kv: kv[1], reverse=True)  # [[0, q1], [1, q2], ...]
    sorted_action_ids = [key[0] for key in sorted_q_values]  # [0, 1, ...]
    if all([item[0] < sr_threshold for _, item in valid_features.items()]):
        return -1  # go after the last vehicle
    for action in sorted_action_ids:
        if valid_features[action][0] >= sr_threshold and valid_features[action][1] <= fallback_threshold:
            return action
    return -1


class ClonedBehaviorParameter:
    def __init__(self, min_success_rate, max_fall_back_rate):
        self.min_success_rate = min_success_rate
        self.max_fall_back_rate = max_fall_back_rate


class LearnedMergingModel:
    def __init__(self):
        self.weights = np.array([0.5, -0.7, 0.05, 0.05, -1, 0.1, 0.15])
        self.learned_model_parameter = ClonedBehaviorParameter(0.15, 1.)
        self.feature_placeholder = [0., 1., 0., 0., 1., 0., 0.]

    def inference(self, feature):
        valid_feature = {}
        for i, f in enumerate(feature):
            valid_feature[i] = f
        q_values = {}
        for i, f in enumerate(feature):
            q_value = np.dot(self.weights, np.array(f))
            q_values[i] = q_value
        # print("feature and q: ", valid_feature, q_values)
        return filter_actions(valid_feature, q_values,
                              self.learned_model_parameter.min_success_rate,
                              self.learned_model_parameter.max_fall_back_rate), q_values


class LearnedLaneChangeModel:
    def __init__(self):
        # learned weights
        # self.weights = np.array([0.183, -0.367, 0.23, 0.1, -0.22, 1, 0.1])
        self.weights = np.array([0.183, -0.367, 0.3, 0.1, -0.15, 1.0, 0.25])
        self.feature_placeholder = [0., 1., 0., 0., 1., 0., 0.]
        self.left_cost = -0.01
        self.right_cost = 0.03
        self.delta_to_last_decision = 0.015

    def inference(self, feature, commands, last_decision="initial"):
        # feature: [action_num, feature_size], left, keep, right
        if len(feature) != len(commands):
            print("Dimension of features do not match possible commands!")
        q_values = []
        q_last_decision = -1e10
        for i, f in enumerate(feature):
            q_value = np.dot(self.weights, np.array(f))
            if commands[i] == "left":
                q_value += self.left_cost
            if commands[i] == "right":
                q_value += self.right_cost
            if commands[i] == last_decision:
                q_last_decision = q_value
            q_values.append([commands[i], q_value])
        q_values.sort(key=lambda x: x[1], reverse=True)
        # print(q_values, last_decision)
        if q_values[0][0] != last_decision:
            for c, q in q_values:
                if q >= q_last_decision + self.delta_to_last_decision:
                    return q_values, c
        return q_values, last_decision


if __name__ == '__main__':
    lc_model = LearnedLaneChangeModel()
    possible_actions = ["left", "keep_lane", "dcc", "acc"]
    features = [[1.0, 0.0, 0.8347861709085435, 0.6743563285927423, 0.24833333333333338, 0.8191169496505016, 0.9227193316538841],
                [1.0, 0.0, 0.8853556042443036, 0.8439449702745763, 0.09999999999999999, 0.6586241190819563, 0.6891999923860896],
                [1.0, 0.0, 0.6976766932260383, 0.63619700339087, 0.09999999999999999, 0.8208634255022598, 0.9273373686498663],
                [1.0, 0.0, 0.7808685861030975, 0.9227713968177605, 0.09999999999999999, 0.6802358091597496, 0.7512569752282063]]
    qs, command = lc_model.inference(features, possible_actions)
    print(qs, command)

    possible_actions = ["keep_lane", "acc", "dcc"]
    features = [[1.0, 0.0, 0.9289858483420468, 0.8855733633991669, 0.1, 1.0, 1.0],
                [1.0, 0.0, 0.8842244456976619, 0.9954408577571839, 0.1, 1.0, 1.0],
                [1.0, 0.0, 0.861946719555924, 0.7104609461924002, 0.1, 1.0, 1.0]]
    qs, command = lc_model.inference(features, possible_actions)
    print(qs, command)

    possible_actions = ["left", "keep_lane", "dcc", "acc"]
    features = [[1.0, 0.0, 0.8619296717349689, 0.9142026620520505, 0.125, 0.9982689235522592, 0.9692335473951416],
                [1.0, 0.0, 0.7196382886956361, 0.9370796297108704, 0.09999999999999999, 0.9979878665033657, 0.969206543425642],
                [1.0, 0.0, 0.6978860317014086, 0.8001376554275782, 0.09999999999999999, 0.9981125958871385, 0.9695845002591762],
                [1.0, 0.0, 0.5856435906807819, 0.9828949735363534, 0.09999999999999999, 0.9983211486397954, 0.9697824316858275]]
    qs, command = lc_model.inference(features, possible_actions)
    print(qs, command)