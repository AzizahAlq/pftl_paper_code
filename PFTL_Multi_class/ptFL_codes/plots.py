import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

data = """round,local_acc,mixed_acc,switch_on,gamma_global
1,0.342508,0.349163,1,0.515890
2,0.732639,0.754097,1,0.520112
3,0.716659,0.718560,0,0.504647
4,0.721820,0.713581,0,0.485775
5,0.708330,0.706564,0,0.469442
6,0.767180,0.785378,0,0.469891
7,0.751064,0.772522,0,0.464227
8,0.777230,0.784925,0,0.448119
9,0.759303,0.778859,0,0.440600
10,0.746944,0.783250,1,0.442906
11,0.738072,0.763015,0,0.436216
12,0.733499,0.765052,1,0.440789
13,0.756496,0.783839,1,0.450490
14,0.742191,0.783658,1,0.475798
15,0.723178,0.737392,0,0.465176
"""

df = pd.read_csv(StringIO(data))

# Plot accuracies
plt.figure()
plt.plot(df["round"], df["local_acc"], label="Local Accuracy")
plt.plot(df["round"], df["mixed_acc"], label="Mixed Accuracy")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.title("Local vs Mixed Accuracy per Round")
plt.legend()
plt.show()

# Plot gamma
plt.figure()
plt.plot(df["round"], df["gamma_global"], label="Gamma Global")
plt.xlabel("Round")
plt.ylabel("Gamma")
plt.title("Gamma Global per Round")
plt.legend()
plt.show()
