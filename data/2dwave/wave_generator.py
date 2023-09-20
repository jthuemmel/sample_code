"""
This script provides a class solving the two-dimensional wave equation using
the central difference method. It can be used to generate circularly expanding
wave data that reflect at the field boundaries (no explicit boundary condition
has been implemented).
"""

import numpy as np

__author__ = "Matthias Karlbauer"


class WaveGenerator:

    def __init__(self, dt, dx, dy, timesteps, width, height, amplitude,
                 velocity, damp, wave_width_x, wave_width_y, p_absorption=0.0,
                 p_generation=0.0, std_absorption=0.0, std_generation=0.0,
                 obstacles_N = None, obstacles_given = None, obstacles_dim = None,
                 skip=1, waves=1, velocity_range=None, amplitude_range=None,
                 skip_rate_range=None, 
                 boundary_type = 'reflecting'):
        """
        Constructor method initializing the parameters for the wave equation.
        :param dt: Temporal simulation resolution (step size in time)
        :param dx: Spatial simulation resolution in x-direction (step size x)
        :param dy: Spatial simulation resolution in y-direction (step size y)
        :param timesteps: The timesteps of the simulation
        :param width: Width of the data field
        :param height: Height of the data field
        :param amplitude: Wave amplitude
        :param velocity: Propagation speed of the wave
        :param damp: Factor denoting how quickly the waves decay over time
        :param wave_width_x: Width of the wave front in x-direction
        :param wave_width_y: Width of the wave front in y-direction
        :param p_absorption: Probability that a cell absorbs activity
        :param p_generation: Probability that a cell generates activity
        :param std_absorption: Standard dev. of absorbed activity ~N(0, std)
        :param std_generation: Standard dev. of generated activity ~N(0, std)
        :param boundary: Switch between different boundary conditions
        :param obstacles_N: Number of obstacles to include (randomly sampled, unless given)
        :param obstacles_given: Obstacles to always include ([UpperLeft,object_width,object_height])
        :param obstacles_dim: Range of obstacle dimensions ((min,max width), (min,max height)) to sample from
        :param skip: (Optional) take only every #skip'th simulation step
        :param waves: (Optional) Number of generated waves for each sample
        :param velocity_range: (Optional) Range for variable wave velocities
        :param amplitude_range: (Optional) Range for variable wave amplitudes
        :param skip_rate_range: (Optional) Range vor variable skip rates
        :return: No return value
        """

        # Set the class parameters
        self.t = timesteps
        self.width = width
        self.height = height

        self.a = amplitude
        self.c = np.ones((width, height)) * velocity
        self.d = damp
        self.wave_width_x = wave_width_x
        self.wave_width_y = wave_width_y

        self.dt = dt
        self.dx = dx
        self.dy = dy

        self.skip = skip
        self.waves = waves

        self.p_absorption = p_absorption
        self.p_generation = p_generation
        self.std_absorption = std_absorption
        self.std_generation = std_generation
        
        self.boundary_type = boundary_type
        
        self.obstacles_N = obstacles_N
        self.obstacles_dim = obstacles_dim
        self.obstacles_given = obstacles_given
        
               
        self.obstacles = []
        #Initialize a field variable for the poisson events
        self.poisson_field = np.ones((width,height))

        # Initiate a field that accounts for quantity absorption/generation
        self.abs_gen_field = np.zeros((width, height))

        self.velocity_range = velocity_range
        self.amplitude_range = amplitude_range
        self.skip_rate_range = skip_rate_range

        # Calculate the number of simulation steps required to realize the
        # chosen skip_rate
        self.simulation_steps = (self.t * self.skip) + 1

        # Initialize the wave field as two-dimensional zero-array
        self.field = np.zeros([self.simulation_steps, width, height])
        
        

    def generate_sample(self):
        """
        Single wave sample generation using the parameters of this wave
        generator.
        :return: The generated wave sample as numpy array(t, x, y)
        """

        # Reset the abs_gen_field to neutral state (no absorption/generation)
        self.abs_gen_field = np.zeros_like(self.abs_gen_field)
        
        # If provided, choose a variable velocity for each sequence
        if self.velocity_range is not None:
            sampled_velocity = np.random.uniform(low=self.velocity_range[0],
                                        high=self.velocity_range[1])
            self.c = sampled_velocity * np.ones((self.width, self.height))

        # If provided, choose a variable amplitude for each sequence
        if self.amplitude_range is not None:
            self.a = np.random.uniform(low=self.amplitude_range[0],
                                       high=self.amplitude_range[1])

        # If provided, choose a variable skip rate for each sequence
        if self.skip_rate_range is not None:
            self.skip = np.random.randint(low=self.skip_rate_range[0],
                                          high=self.skip_rate_range[1] + 1)

        # If desired, create a map where wave activity can be absorbed
        # (reduced) by some factor
        if self.p_absorption > 0.0:
            tmp_rand = np.random.rand(self.width, self.height)
            abs_positions = np.zeros((self.width, self.height), dtype=int)
            abs_positions[tmp_rand < self.p_absorption] = 1
            abs_values = np.clip(
                a=np.random.randn(self.width,
                                  self.height) * self.std_absorption,
                a_min=-np.infty,
                a_max=0
            )

            self.abs_gen_field[abs_positions == 1] += \
                                                abs_values[abs_positions == 1]

        # If desired, create a map where wave activity can be generated
        # (increased)
        if self.p_generation > 0.0:
            tmp_rand = np.random.rand(self.width, self.height)
            gen_positions = np.zeros((self.width, self.height), dtype=int)
            gen_positions[tmp_rand < self.p_generation] = 1
            gen_values = np.clip(
                a=np.random.randn(self.width,
                                  self.height) * self.std_generation,
                a_min=0,
                a_max=np.infty
            )
            self.abs_gen_field[gen_positions == 1] += \
                                                gen_values[gen_positions == 1]
            
       #If desired, create obstacles
        if self.obstacles_N:
            #Reset the list of obstacles
            self.obstacles = []
            for i in range(self.obstacles_N):
                #If obstacles are given, create them
                if len(self.obstacles_given) >= i+1:
                    self.obstacles.append(self.obstacle(*self.obstacles_given[i]))
                #Else create random ones
                else:
                    dim_a = np.random.randint(self.obstacles_dim[0][0],self.obstacles_dim[0][1])
                    dim_b = np.random.randint(self.obstacles_dim[1][0],self.obstacles_dim[1][1])
                    loc = np.random.randint((0,0+dim_b),(self.width-dim_a,self.height))
                    self.obstacles.append(self.obstacle(loc,dim_a,dim_b))
                    
        # (Re)set the field to zero
        self.field = np.zeros_like(self.field)

        # Create the desired number of wave initializations
        for wave in range(self.waves):
            # Generate a random point in the field where the impulse will be
            # initialized
            start_pt = [np.random.randint(0, self.width),
                        np.random.randint(0, self.height)]

            # Compute the initial field activity by applying a 2D gaussian around
            # the start point
            for x in range(self.width):
                for y in range(self.height):
                    self.field[0, x, y] += self.f(x=x, y=y, start_pt=start_pt)

        # Iterate over all time steps to compute the activity at each position in
        # the grid over all time steps
        for t in range(self.simulation_steps - 1):

            # Iterate over all values in the field and update them
            for x in range(self.width):
                for y in range(self.height):
                    self.field[t + 1, x, y] = self.u(t=t, x=x, y=y)
                       
            # Potentially absorb/generate (reduce/amplify) wave activity in
            # certain cells
            self.field[t] += self.abs_gen_field
            
        # Only take every skip'th data point in time
        self.field = self.field[::self.skip]
        


        return self.field

    def f(self, x, y, start_pt):
        """
        Function to set the initial activity of the field. We use the Gaussian bell
        curve to initialize the field smoothly.
        :param x: The x-coordinate of the field (j, running over width)
        :param y: The y-coordinate of the field (i, running over height)
        :param start_pt: The point in the field where the wave origins
        :return: The initial activity at (x, y)
        """

        varx = self.wave_width_x
        vary = self.wave_width_y
        
        x_part = ((x - start_pt[0])**2) / (2 * varx)
        y_part = ((y - start_pt[1])**2) / (2 * vary)
        
        return self.a*np.exp(-(x_part + y_part))

    def g(self, x, y):
        """
        Function to determine the changes (derivative) over time in the field
        :param x: The x-coordinate of the field (j, running over width)
        :param y: The y-coordinate of the field (i, running over height)
        :return: The changes over time in the field at (x, y)
        """
        # Note: this function has not been implemented yet
        # x_part = x * f(x, y, varx, vary, a)
        # y_part = y * f(x, y, varx, vary, a)
        # return (x_part + y_part) / 2.
        return 0.0

    def u(self, t, x, y):
        """
        Function to calculate the field activity in time step t at (x, y)
        :param t: The current time step
        :param x: The x-coordinate of the field (j, running over width)
        :param y: The y-coordinate of the field (i, running over height))
        :return: The field activity at position x, y in time step t.
        """

        # Compute changes in x- and y-direction
        dxxu = self.dxx_u(t, x, y)
        dyyu = self.dyy_u(t, x, y)
        
        # Get the activity at x and y in time step t
        ut = self.field[t, x, y]

        # Catch initial condition, where there is no value of the field at time step
        # (t-1) yet
        if t == 0:
            ut_1 = self.dt_u(x, y)
        else:
            ut_1 = self.field[t - 1, x, y]

        # Propagation velocity in this cell
        c = self.c[x, y]

        # Incorporate the changes in x- and y-direction and return the activity
        return self.d*(((c**2)*(self.dt**2))*(dxxu + dyyu) + 2*ut - ut_1)

    def dxx_u(self, t, x, y):
        """
        The second derivative of u to x. Computes the lateral activity change in
        x-direction.
        :param t: The current time step
        :param x: The x-coordinate of the field (j, running over width)
        :param y: The y-coordinate of the field (i, running over height))
        :return: Field activity in t, considering changes in x-direction
        """
        if self.boundary_type == 'reflecting':
            # Boundary condition at left end of the field (reflecting waves)
            if x == 0 :
                dx_left = 0.0
            else:
                dx_left = self.field[t, x - self.dx, y]

            # Boundary condition at right end of the field (reflecting waves)
            if x == self.width - 1 :
                dx_right = 0.0
            else:
                dx_right = self.field[t, x + self.dx, y]
                
             
        elif self.boundary_type == 'periodic':
            # Periodic boundary condition   
            if x + self.dx > self.width -1: 
                x = -1
            dx_right = self.field[t, x + self.dx, y]
            dx_left = self.field[t, x - self.dx, y]
            
        #additionally, check object boundaries, objects always reflect.
        if self.obstacles_N > 0:
            for o in self.obstacles:
                if o.is_inside(x,y):
                    dx_left = 0.
                    dx_right = 0.
                elif o.is_left(x,y):
                    dx_left = 0.
                elif o.is_right(x,y):
                    dx_right = 0.
            
        # Calculate change in x-direction and return it
        ut_dx = dx_right - 2*self.field[t, x, y] + dx_left

        return ut_dx / np.square(self.dx)

    def dyy_u(self, t, x, y):
        """
        The second derivative of u to y. Computes the lateral activity change in
        y-direction.
        :param t: The current time step
        :param x: The x-coordinate of the field (j, running over width)
        :param y: The y-coordinate of the field (i, running over height))
        :return: Field activity in t, considering changes in y-direction
        """
        if self.boundary_type == 'reflecting':
            # Boundary condition at top end of the field (reflecting waves)
            if y == 0 :
                dy_above = 0.0
            else:
                dy_above = self.field[t, x, y - self.dy]

            # Boundary condition at bottom end of the field (reflecting waves)
            if y == self.height - 1:
                dy_below = 0.0
            else:
                dy_below = self.field[t, x, y + self.dy]
            
            
        elif self.boundary_type == 'periodic':
            # Periodic boundary condition
            if y + self.dy > self.height - 1: 
                y = -1
            dy_above = self.field[t, x, y - self.dy]
            dy_below = self.field[t, x, y + self.dy]

            
        #additionally, check object boundaries, objects always reflect.
        if self.obstacles_N > 0:
            for o in self.obstacles:
                if o.is_inside(x,y):
                    dy_above = 0.
                    dy_below = 0.
                elif o.is_above(x,y):
                    dy_above = 0.
                elif o.is_below(x,y):
                    dy_below = 0.
                
        
        # Calculate change in y-direction and return it
        ut_dy = dy_below - 2*self.field[t, x, y] + dy_above

        return ut_dy / np.square(self.dy)

    def dt_u(self, x, y):
        """
        First derivative of u to t, only required in the very first time step
        to compute u(-dt, x, y).
        :param x: The x-coordinate of the field (j, running over width)
        :param y: The y-coordinate of the field (i, running over height))
        :return: The value of the field at (t-1), x, y
        """
        return self.field[1, x, y] - 2*self.dt*self.g(x, y)
      
    class obstacle():
        
        def __init__(self,anchor,dims):
            """
            Obstacle object. The obstacles know their own location and 
            can return boundary conditions (wave outside).
            :param anchor: position of the upper left corner
            :param a: width of the object in pixels
            :param_b: height of the object in pixels
            """

            self.anchor = np.array(anchor)
            self.a = dims[0]
            self.b = dims[1]
            
            self.area = self.a*self.b
            
            #calculate the corners of the object
            #to account for pythons indexing we add or substract 1
            self.upper_left = np.array(anchor)
            self.lower_left = anchor + np.array([0,-self.b+1])
            self.upper_right = anchor + np.array([self.a-1,0])
            self.lower_right = anchor + np.array([+self.a-1,-self.b+1])
            

        def is_above(self,x,y):
            if self.lower_left[0] <= x <= self.lower_right[0] and y == self.lower_left[1]: return True
            else: return False
        def is_below(self,x,y):
            if self.upper_left[0] <= x <= self.upper_right[0] and y == self.upper_left[1]: return True
            else: return False
        def is_right(self,x,y):
            if self.lower_left[1] <= y <= self.upper_left[1] and x == self.upper_left[0]: return True
            else: return False
        def is_left(self,x,y):
            if self.lower_right[1] <= y <= self.upper_right[1] and x == self.upper_right[0]: return True
            else: return False
        def is_inside(self,x,y):
            if self.lower_left[1] <= y <= self.upper_left[1] and self.lower_left[0] <= x <= self.lower_right[0]: 
                return True
            else: return False