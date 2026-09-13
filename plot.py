import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("sir_alpha_test.txt")

beta = data[:, 0]
sigma = data[:, 1]
prob = data[:, 2]

betas = np.unique(beta)
sigmas = np.unique(sigma)

Z = prob.reshape(len(sigmas), len(betas))

plt.figure(figsize=(7, 5))
plt.pcolormesh(betas, sigmas, Z, shading="auto", cmap="viridis")
plt.yscale("log")
plt.colorbar(label="Major outbreak probability")
plt.xlabel("beta")
plt.ylabel("sigma")
plt.title("SIR alpha phase diagram")
plt.tight_layout()
plt.savefig("sir_alpha_phase_diagram.png", dpi=200)
plt.show()