

class DPCliffWalkingEnv:
    """ 悬崖漫步环境"""
    def __init__(self, n_col=12, n_row=4):
        self.n_col = n_col  # 定义网格世界的列
        self.n_row = n_row  # 定义网格世界的行
        # 转移矩阵P[state][action] = [(p, next_state, reward, done)]包含下一个状态和奖励
        self.map = self.create_map()

    def create_map(self):
        # 初始化
        init_map = [[[] for _ in range(4)] for _ in range(self.n_row * self.n_col)]
        # 4种动作, change[0]:上,change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
        # 定义在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        for i in range(self.n_row):
            for j in range(self.n_col):
                for a in range(4):
                    # 位置在悬崖或者目标状态,因为无法继续交互,任何动作奖励都为0
                    if i == self.n_row - 1 and j > 0:
                        init_map[i * self.n_col + j][a] = [(1, i * self.n_col + j, 0, True)]
                        continue
                    # 其他位置
                    next_x = min(self.n_col - 1, max(0, j + change[a][0]))
                    next_y = min(self.n_row - 1, max(0, i + change[a][1]))
                    next_state = next_y * self.n_col + next_x
                    reward = -1
                    done = False
                    # 下一个位置在悬崖或者终点
                    if next_y == self.n_row - 1 and next_x > 0:
                        done = True
                        if next_x != self.n_col - 1:  # 下一个位置在悬崖
                            reward = -100
                    init_map[i * self.n_col + j][a] = [(1, next_state, reward, done)]
        return init_map


class TDCliffWalkingEnv:
    def __init__(self, n_col, n_row):
        self.n_row = n_row
        self.n_col = n_col
        self.x = 0  # 记录当前智能体位置的横坐标
        self.y = self.n_row - 1  # 记录当前智能体位置的纵坐标

    def step(self, action):  # 外部调用这个函数来改变当前位置
        # 4种动作, change[0]:上, change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
        # 定义在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.x = min(self.n_col - 1, max(0, self.x + change[action][0]))
        self.y = min(self.n_row - 1, max(0, self.y + change[action][1]))
        next_state = self.y * self.n_col + self.x
        reward = -1
        done = False
        if self.y == self.n_row - 1 and self.x > 0:  # 下一个位置在悬崖或者目标
            done = True
            if self.x != self.n_col - 1:
                reward = -100
        return next_state, reward, done

    def reset(self):  # 回归初始状态,坐标轴原点在左上角
        self.x = 0
        self.y = self.n_row - 1
        return self.y * self.n_col + self.x
