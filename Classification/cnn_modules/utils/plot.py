import matplotlib.pyplot as plt


of_losses = []
torch_losses = []

with open("of_losses.txt", "r") as lines:
    for line in lines:
        line = line.strip()
        of_losses.append(float(line))

with open("torch_losses.txt", "r") as lines:
    for line in lines:
        line = line.strip()
        torch_losses.append(float(line))


indes = [i for i in range(len(of_losses))]


plt.plot(indes, of_losses, label = "oneflow")
plt.plot(indes, torch_losses, label = "pytorch")

plt.xlabel('iter - axis')
# Set the y axis label of the current axis.
plt.ylabel('loss - axis')
# Set a title of the current axes.
plt.title('compare ')
# show a legend on the plot
plt.legend()
# Display a figure.
plt.show()
