import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import torch

# load model
model = tf.keras.models.load_model('trajectory_model.keras')

test_tensors = []

scenes = ['eth', 'hotel', 'univ', 'zara1', 'zara2']

for scene in scenes:
    test_tensors.append(torch.load(f'datasets/{scene}/test.pt'))

test_data = torch.cat(test_tensors, dim=0)

# add velocity
test_vel = test_data[:, 1:, :] - test_data[:, :-1, :]
zero_pad_test = torch.zeros(test_data.shape[0], 1, 2)
test_v = torch.cat([zero_pad_test, test_vel], dim=1)
test_data = torch.cat([test_data, test_v], dim=2)

# split
X_test = test_data[:, :8, :].numpy()
Y_test = test_data[:, 8:, :-2].numpy()

# predict
pred = model.predict(X_test)

# evaluate
test_loss = model.evaluate(X_test, Y_test)
print(test_loss)

def plot_examples(trajectory_index):
    """print 20 trajectories (every 5 starting from trajectory_index)
    trajectory_index: starting index of trajectories to print
    """
    fig,ax = plt.subplots(4,5,figsize=(15,10))
    ax=ax.flatten()

    # start of trajectories to show
    n = trajectory_index
    for i in range(20):
        n += 5
        ax[i].plot(X_test[n,:,0], X_test[n,:,1], label='Observed')
        ax[i].plot(Y_test[n,:,0], Y_test[n,:,1], label='True Future')
        ax[i].plot(pred[n,:,0], pred[n,:,1], label='Predicted Future')

        # add arrows
        ax[i].annotate('', 
                xy=(pred[n,6,0], pred[n,6,1]), 
                xytext=(pred[n,5,0], pred[n,5,1]),
                arrowprops=dict(arrowstyle="->", color='green', lw=2))
        ax[i].annotate('', 
                xy=(Y_test[n,-3,0], Y_test[n,-3,1]), 
                xytext=(Y_test[n,-4,0], Y_test[n,-4,1]),
                arrowprops=dict(arrowstyle="->", color='orange', lw=2))
        ax[i].annotate('', 
                xy=(X_test[n,-1,0], X_test[n,-1,1]), 
                xytext=(X_test[n,-2,0], X_test[n,-2,1]),
                arrowprops=dict(arrowstyle="->", color='blue', lw=1))


    plt.legend(bbox_to_anchor=(-5,4))
    plt.suptitle(f'Trajectories {trajectory_index}-{n} (every 5 trajectory)', x=.04, y=.8)
    plt.show()