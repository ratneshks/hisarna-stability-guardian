import numpy as np
import scipy.linalg
import torch

class StabilityEnvelope:
    def __init__(self):
        self.mu_stable = None
        self.Sigma_inv = None
        self.d_95th_percentile = 1.0

    def fit(self, latent_vectors):
        """
        Fit the Mahalanobis distance envelope on stable latent vectors.
        latent_vectors: numpy array of shape (N, 64)
        """
        self.mu_stable = np.mean(latent_vectors, axis=0)
        Sigma_stable = np.cov(latent_vectors, rowvar=False)
        
        # Regularization for numerical stability
        reg = 1e-6 * np.eye(Sigma_stable.shape[0])
        self.Sigma_inv = scipy.linalg.inv(Sigma_stable + reg)
        
        # Compute distances for the training set to find 95th percentile
        distances = []
        for v in latent_vectors:
            diff = v - self.mu_stable
            d = np.sqrt(diff.T @ self.Sigma_inv @ diff)
            distances.append(d)
        
        self.d_95th_percentile = np.percentile(distances, 95)
        # Avoid division by zero
        if self.d_95th_percentile < 1e-6:
            self.d_95th_percentile = 1.0

    def compute_distance(self, latent_vector):
        """
        Compute normalized Mahalanobis distance for a single latent vector.
        latent_vector: numpy array of shape (64,)
        """
        if self.mu_stable is None:
            return 0.0 # Not fitted
            
        diff = latent_vector - self.mu_stable
        d = np.sqrt(np.clip(diff.T @ self.Sigma_inv @ diff, 0, None))
        d_norm = d / self.d_95th_percentile
        return d_norm

    def classify(self, d_norm):
        if d_norm < 0.6:
            return "STABLE", "#22c55e"
        elif d_norm < 1.0:
            return "WARNING", "#eab308"
        else:
            return "CRITICAL", "#ef4444"

    def get_recommendation(self, status, sensor_data):
        if status == "STABLE":
            return "Process within nominal envelope. No action required."
            
        elif status == "WARNING":
            # Simple heuristic matching the spec
            if sensor_data.get('phi_O2', 4.0) < 2.0:
                return "Reduce ore injection rate by 10–15%. Monitor O2 fraction recovery. Expected stabilization: 45–60 seconds."
            elif sensor_data.get('T_cyclone', 1450.0) > 1480.0:
                return "Increase blast volume by 5%. Check gas composition balance. Expected stabilization: 30–45 seconds."
            else:
                return "Monitor process parameters closely. Instability detected."
                
        elif status == "CRITICAL":
            return "IMMEDIATE ACTION: Reduce ore feed to minimum. Verify tuyere integrity. Alert shift supervisor. Do NOT increase blast pressure."
        
        return "Unknown state."
