import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#data = np.load('save.npy')
#print(data.shape)
#plot_data = data[0]
#print(plot_data)
#sns.heatmap(plot_data)

#plt.show()


pred = np.load('result/predict/bohai_1_pred.npy')

label = np.load('result/predict/bohai_1_label.npy')

#pred = pred.squeeze()
#label = label.squeeze()
plt.plot(pred[:,1],label[:,1])

#print(pred[:,1])
#print(label[:,1])
plt.show()