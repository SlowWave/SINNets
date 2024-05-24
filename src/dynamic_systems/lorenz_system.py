from .dynamic_system import DynamicSystem


class LorenzSystem(DynamicSystem):
    def __init__(self, sigma=10, rho=28, beta=8/3):
        """
        Initializes a LorenzSystem object.

        Args:
            sigma (float, optional): The value of sigma in the Lorenz system. Defaults to 10.
            rho (float, optional): The value of rho in the Lorenz system. Defaults to 28.
            beta (float, optional): The value of beta in the Lorenz system. Defaults to 8/3.
        """

        super(LorenzSystem, self).__init__()

        # initialize attributes
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

        self.tag = "LorenzSystem"
        self.state_boundaries = [
            [-20, -20, -20],
            [20, 20, 20],
        ]

        self.input_boundaries = None

        self.state_dim = 3
        self.input_dim = 0

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

        x_dot_1 = self.sigma * (x[1] - x[0])
        x_dot_2  = x[0] * (self.rho - x[2]) - x[1]
        x_dot_3 = x[0] * x[1] - self.beta * x[2]

        x_dot = [x_dot_1, x_dot_2, x_dot_3]

        return x_dot
