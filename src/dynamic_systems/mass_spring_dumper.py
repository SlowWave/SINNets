from .dynamic_system import DynamicSystem


class MassSpringDumper(DynamicSystem):
    def __init__(self, mass=0.3, spring=0.5, dumper=0.1):
        """
        Initializes a MassSpringDumper object.

        Args:
            mass (float, optional): The mass of the mass-spring-dumper system [kg]. Defaults to 0.3.
            spring (float, optional): The spring constant of the mass-spring-dumper system [N/m]. Defaults to 0.5.
            dumper (float, optional): The damping coefficient of the mass-spring-dumper system [N/(m/s)]. Defaults to 0.1.
        """

        super(MassSpringDumper, self).__init__()

        # initialize attributes
        self.mass = mass
        self.spring = spring
        self.dumper = dumper

        self.tag = "MassSpringDumper"
        self.state_boundaries = [
            [-5, -2],
            [5, 2],
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
            -(self.spring * x[0] + self.dumper * x[1]) / self.mass + u[0] / self.mass
        )

        x_dot = [x_dot_1, x_dot_2]

        return x_dot
