import heterocl as hcl

# Based on odp/dynamics/DubinsCapture.py (same system, our params)
class Air3D:
    def __init__(self, x=[0, 0, 0], uMode="max", dMode="min"):
        self.x = x
        self.speed = 0.75
        self.wMax = 3.0
        self.dMax = 3.0
        self.uMode = uMode
        self.dMode = dMode

    def opt_ctrl(self, t, state, spat_deriv):
        opt_w = hcl.scalar(self.wMax, "opt_w")
        in3 = hcl.scalar(0, "in3")
        in4 = hcl.scalar(0, "in4")

        a_term = hcl.scalar(0, "a_term")
        a_term[0] = spat_deriv[0] * state[1] - spat_deriv[1] * state[0] - spat_deriv[2]

        with hcl.if_(a_term[0] >= 0):
            with hcl.if_(self.uMode == "min"):
                opt_w[0] = -opt_w[0]
        with hcl.elif_(a_term[0] < 0):
            with hcl.if_(self.uMode == "max"):
                opt_w[0] = -opt_w[0]
        return (opt_w[0], in3[0], in4[0])

    def opt_dstb(self, t, state, spat_deriv):
        d1 = hcl.scalar(self.dMax, "d1")
        d2 = hcl.scalar(0, "d2")
        d3 = hcl.scalar(0, "d3")

        b_term = hcl.scalar(0, "b_term")
        b_term[0] = spat_deriv[2]

        with hcl.if_(b_term[0] >= 0):
            with hcl.if_(self.dMode == "min"):
                d1[0] = -d1[0]
        with hcl.elif_(b_term[0] < 0):
            with hcl.if_(self.dMode == "max"):
                d1[0] = -d1[0]
        return (d1[0], d2[0], d3[0])

    def dynamics(self, t, state, uOpt, dOpt):
        x_dot = hcl.scalar(0, "x_dot")
        y_dot = hcl.scalar(0, "y_dot")
        theta_dot = hcl.scalar(0, "theta_dot")

        x_dot[0] = -self.speed + self.speed * hcl.cos(state[2]) + uOpt[0] * state[1]
        y_dot[0] = self.speed * hcl.sin(state[2]) - uOpt[0] * state[0]
        theta_dot[0] = dOpt[0] - uOpt[0]

        return (x_dot[0], y_dot[0], theta_dot[0])
