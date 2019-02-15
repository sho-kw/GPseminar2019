import numpy as np
import random
from time import sleep

def main():
    maze = Maze(
        '''
        ########
        #S#...G#
        #...#..#
        #.O...O#
        ########
        ''',
        {'.':-1,
         'O':-50,
         'G':100})
    Q = np.random.rand(*maze.maze_size, 4)
    gamma = 0.9
    eps = 0.05
    #
    def choice_action(x, y):
        if random.random() < eps:
            return random.choice(range(4))
        else:
            return np.argmax(Q[x, y])
    def update_Q(x, y, action, reward, is_end, nx, ny):
        if is_end:
            Q[x, y, action] = reward
        else:
            Q[x, y, action] = reward + gamma * np.max(Q[nx, ny])
    #
    for episode in range(20):
        x, y = maze.reset()
        maze.print_with_agent(x, y)
        is_end = False
        step_count = 0
        while not is_end:
            sleep(0.2)
            action = choice_action(x, y)
            nx, ny, reward, is_end = maze.action(x, y, action)
            update_Q(x, y, action, reward, is_end, nx, ny)
            x, y = nx, ny
            maze.print_with_agent(x, y)
            step_count += 1
        print('goal ({} steps)'.format(step_count))
        sleep(1)


class Maze:
    def __init__(self, maze_map, reward_dict):
        self.maze_map = [line.strip() for line in maze_map.splitlines() if line.strip()]
        self.maze_size = (len(self.maze_map), len(self.maze_map[0]))
        self._reward_dict = reward_dict
    def _get_reward(self, x, y):
        block = self.maze_map[x][y]
        if block in self._reward_dict:
            return self._reward_dict[block]
        else:
            return 0
    def reset(self):
        for x in range(len(self.maze_map)):
            for y in range(len(self.maze_map[x])):
                if self.maze_map[x][y] == 'S':
                    return x, y
        raise RuntimeError('S not found')
    def action(self, x, y, direction):
        # basic move
        if direction == 0:
            nx, ny = x, y - 1
        elif direction == 1:
            nx, ny = x + 1, y
        elif direction == 2:
            nx, ny = x, y + 1
        elif direction == 3:
            nx, ny = x - 1, y
        # exceptional move
        penalty = 0
        if self.maze_map[nx][ny] == '#':
            nx, ny = x, y
            penalty = -10
        # return pos, reward, end_flag
        reward = self._get_reward(nx, ny) + penalty
        is_end = (self.maze_map[nx][ny] in ('O', 'G'))
        return nx, ny, reward, is_end
    def print_with_agent(self, x, y):
        for i in range(len(self.maze_map)):
            if x == i:
                print(self.maze_map[i][0:y], 'P', self.maze_map[i][y + 1:], sep='')
            else:
                print(self.maze_map[i])
        print()


if __name__ == '__main__':
    main()