import torch
import torch.nn as nn
import torch.autograd as autograd

class HIsarnaPINN(nn.Module):
    def __init__(self):
        super().__init__()
        # Architecture: 308 -> 256 -> 256 -> 128 -> 64 -> 8
        self.network = nn.Sequential(
            nn.Linear(308, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh()
        )
        # Separate the last layers so we can extract the latent representation (Layer 3 output)
        self.layer3 = nn.Linear(128, 64)
        self.layer3_act = nn.Tanh()
        self.output_layer = nn.Linear(64, 8)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x, return_latent=False):
        hidden_128 = self.network(x)
        latent_64 = self.layer3_act(self.layer3(hidden_128))
        out = self.output_layer(latent_64)
        if return_latent:
            return out, latent_64
        return out

class PINNLoss(nn.Module):
    def __init__(self, lambda_data=1.0, lambda_NS=0.01, lambda_HT=1e-8, lambda_BC=0.1):
        super().__init__()
        self.lambda_data = lambda_data
        self.lambda_NS = lambda_NS
        self.lambda_HT = lambda_HT
        self.lambda_BC = lambda_BC
        self.mse = nn.MSELoss()
        
        # Physics constants
        self.rho = 1.2
        self.mu = 2.5e-5
        self.Cp = 1050.0
        self.k = 0.045
        self.Q_reaction = 500000.0

    def compute_gradients(self, y, x):
        """Helper to compute derivatives using autograd"""
        grad = autograd.grad(
            outputs=y, inputs=x,
            grad_outputs=torch.ones_like(y),
            create_graph=True, retain_graph=True,
            only_inputs=True
        )[0]
        return grad

    def forward(self, model_input, y_pred, y_true):
        # 1. Data Loss
        L_data = self.mse(y_pred, y_true)
        
        # Slicing inputs for physics gradients
        # The coordinates are the last 3 elements of the 308-dim input
        # model_input shape: (batch_size, 308)
        # But wait, to take higher order derivatives, we need x, y, t as independent variables.
        # If we use compute_gradients wrt model_input, it returns (batch_size, 308).
        # We can extract the last three columns.
        
        u_pred = y_pred[:, 0]
        v_pred = y_pred[:, 1]
        T_pred = y_pred[:, 2]
        P_pred = y_pred[:, 3]
        phi_O2_pred = y_pred[:, 4]
        phi_CO_pred = y_pred[:, 5]
        
        # Gradients wrt inputs
        du_dinput = self.compute_gradients(u_pred, model_input)
        dv_dinput = self.compute_gradients(v_pred, model_input)
        dT_dinput = self.compute_gradients(T_pred, model_input)
        dP_dinput = self.compute_gradients(P_pred, model_input)
        
        # Extract derivatives wrt x (idx -3), y (idx -2), t (idx -1)
        du_dx = du_dinput[:, -3]
        du_dy = du_dinput[:, -2]
        du_dt = du_dinput[:, -1]
        
        dv_dx = dv_dinput[:, -3]
        dv_dy = dv_dinput[:, -2]
        dv_dt = dv_dinput[:, -1]
        
        dT_dx = dT_dinput[:, -3]
        dT_dy = dT_dinput[:, -2]
        dT_dt = dT_dinput[:, -1]
        
        dP_dx = dP_dinput[:, -3]
        dP_dy = dP_dinput[:, -2]
        
        # Higher order derivatives
        # To get d2u/dx2, we need derivative of du_dx wrt input
        du_dx_dinput = self.compute_gradients(du_dx, model_input)
        du_dy_dinput = self.compute_gradients(du_dy, model_input)
        d2u_dx2 = du_dx_dinput[:, -3]
        d2u_dy2 = du_dy_dinput[:, -2]
        
        dv_dx_dinput = self.compute_gradients(dv_dx, model_input)
        dv_dy_dinput = self.compute_gradients(dv_dy, model_input)
        d2v_dx2 = dv_dx_dinput[:, -3]
        d2v_dy2 = dv_dy_dinput[:, -2]
        
        dT_dx_dinput = self.compute_gradients(dT_dx, model_input)
        dT_dy_dinput = self.compute_gradients(dT_dy, model_input)
        d2T_dx2 = dT_dx_dinput[:, -3]
        d2T_dy2 = dT_dy_dinput[:, -2]
        
        # 2. Navier-Stokes Residuals
        f_cont = du_dx + dv_dy
        f_mom_x = self.rho * (du_dt + u_pred * du_dx + v_pred * du_dy) + dP_dx - self.mu * (d2u_dx2 + d2u_dy2)
        f_mom_y = self.rho * (dv_dt + u_pred * dv_dx + v_pred * dv_dy) + dP_dy - self.mu * (d2v_dx2 + d2v_dy2)
        
        L_NS = torch.mean(f_cont**2 + f_mom_x**2 + f_mom_y**2)
        
        # 3. Heat Transfer Residual
        f_heat = self.rho * self.Cp * (dT_dt + u_pred * dT_dx + v_pred * dT_dy) - self.k * (d2T_dx2 + d2T_dy2) - self.Q_reaction
        L_HT = torch.mean(f_heat**2)
        
        # 4. Boundary Conditions / Constraints
        # T in [900, 1600]
        BC_T = torch.relu(T_pred - 1600.0) + torch.relu(900.0 - T_pred)
        # Gas fractions sum constraint (~60%)
        BC_gas = (phi_O2_pred + phi_CO_pred - 55.0)**2
        # Velocity magnitude > 0
        BC_vel = torch.relu(-u_pred) + torch.relu(-v_pred)
        
        L_BC = torch.mean(BC_T**2 + BC_gas + BC_vel**2)
        
        # Total
        L_total = self.lambda_data * L_data + self.lambda_NS * L_NS + self.lambda_HT * L_HT + self.lambda_BC * L_BC
        
        return L_total, L_data, L_NS, L_HT, L_BC
