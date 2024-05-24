import math

from .dynamic_system import DynamicSystem


class SimplePendulum(DynamicSystem):
    def __init__(self, mass=0.3, length=0.5, friction=0.1):
        """
        Initializes a nSimplePendulum object.

        Args:
            mass (float, optional): The mass of the pendulum [kg]. Defaults to 0.3.
            length (float, optional): The length of the pendulum [m]. Defaults to 0.5.
            friction (float, optional): The kinetic friction coefficient of the pendulum [N/(m/s)]. Defaults to 0.1.
        """

        super(SimplePendulum, self).__init__()

        # initialize attributes
        self.mass = mass
        self.length = length
        self.friction = friction

        self.tag = "SimplePendulum"
        self.state_boundaries = [
            [0, -2],
            [2 * math.pi, 2],
        ]

        self.input_boundaries = [
            [-10],
            [10],
        ]

        self.state_dim = 2
        self.input_dim = 1

    def ode(self, t, x, u):
        """
        Calculates the derivative of the state variables of the dynamic system.

        Args:
            t (float): The current time.
            x (list): The current state variables of the system.
            u (list): The current control input.

        Returns:
            list: The derivative of the state variables.
        """

        x_dot_1 = x[1]
        x_dot_2 = (
            -(self.friction * x[1] / self.mass + 9.81 / self.length * math.sin(x[0]))
            + u[0] / (self.mass * self.length)
        )

        x_dot = [x_dot_1, x_dot_2]

        return x_dot
