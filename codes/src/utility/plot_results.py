import os
import numpy as np
import matplotlib.pyplot as plt


def loss_and_accuracy(train_loss, val_loss, train_acc, val_acc, dest_path):
    fig, ax = plt.subplots(1,2,figsize=(10,4))

    ax[0].plot(train_loss, c='blue', label='Train')
    ax[0].plot(val_loss, c='orange', label='Validation')
    ax[0].set_xlabel('Iteration', fontsize=12)
    ax[0].set_ylabel('BCE-Loss', fontsize=12)
    ax[0].set_title('Classification Loss', fontsize=14)
    ax[0].legend(fontsize=12)

    ax[1].plot(train_acc, c='blue', label='Train')
    ax[1].plot(val_acc, c='orange', label='Validation')
    ax[1].set_xlabel('Iteration', fontsize=12)
    ax[1].set_ylabel('Accuracy', fontsize=12)
    ax[1].set_title('Accuracy Variation', fontsize=14)
    ax[1].legend(fontsize=12)

    plt.savefig(os.path.join(dest_path,'loss_and_acc_curves.jpg'))
    plt.show()



def performance(accuracy, sensitivity, specificity, f1_score, epochs, dest_path):
    plt.figure(figsize=(8,4))
    it = np.arange(5, epochs+1, 5)
    
    plt.plot(it, accuracy, 'o--', c='blue', label='accuracy')
    plt.plot(it, sensitivity, 'o--', c='red', label='sensitivity')
    plt.plot(it, specificity, 'o--', c='green', label='specificity')
    plt.plot(it, f1_score, 'o--', c='purple', label='f1_score')
    
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Value', fontsize=16)
    plt.legend(fontsize=12)
    plt.savefig(os.path.join(dest_path,f'performance_metrics.jpg'))
    plt.show()