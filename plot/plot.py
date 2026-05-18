import pickle
import matplotlib.pyplot as plt

with open("plot/trajectory.pkl", "rb") as f:
    pickle.load(f)
plt.show(block=False)

# with open("plot/acc_phase.pkl", "rb") as f:
#     pickle.load(f)
# plt.show(block=False)

with open("plot/data.pkl", "rb") as f:
    pickle.load(f)
plt.show()