from icecream import ic
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

from classes.target import Target

class Main():
    def __init__(self, path = None, N = 4) -> None:
        print("Initializing animator...")
        #self.path = path # TODO: make path 6D
        self.path = path
        if self.path is None:
            self.path = np.array(list(self.gen_path(N))).T # (3, N)
        self.N = N

        self.target = Target("target", 0, 0, 0, 0, 0, 0)
        self.animate_path()

    def animate_path(self):
        print("Animating path...")
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        line, = ax.plot(self.path[0, 0:1], self.path[1, 0:1], self.path[2, 0:1])

        # Setting the axes properties
        ax.set_xlim3d([-1.0, 1.0])
        ax.set_xlabel('X')

        ax.set_ylim3d([-1.0, 1.0])
        ax.set_ylabel('Y')

        ax.set_zlim3d([0.0, 10.0])
        ax.set_zlabel('Z')
        ani = animation.FuncAnimation(fig, self.update, self.N, fargs=(self.path, line), interval=100, blit=False)
        plt.show()

    def gen_path(self, n):
        phi = 0
        while phi < 2*np.pi:
            yield np.array([np.cos(phi), np.sin(phi), phi])
            phi += 2*np.pi/n

    def update(self, num, path, line):
        self.target.set_position(path[:, num])
        line.set_data(path[:2, :num])
        line.set_3d_properties(path[2, :num])

        if num == self.N-1:
            plt.close()
            exit()

if __name__ == '__main__':
    path = np.array([[0, 1, 2, 3, 4, 5],
                     [0, 1, 2, 3, 4, 5],
                     [0, 1, 2, 3, 4, 5],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0]])
    Main(path)