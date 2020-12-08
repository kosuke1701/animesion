import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
import torchvision

from data.datasets import moeImoutoDataset

def data_loading(batch_size, visualization=True):

    dataset = moeImoutoDataset().getImageFolder()
    classes = dataset.classes
    
    # split into training, validation, and testing
    # https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets/50544887#50544887
    dataset_loader = data.DataLoader(dataset,
                    batch_size=batch_size, shuffle=True,
                    num_workers=4)

    if visualization==True:
        classes_print(dataset)
        img_grid(classes, dataset_loader, batch_size=32)

    return dataset_loader, classes

def classes_print(dataset, no_examples=10):
    print(dataset)
    print(dataset.classes[:no_examples])
    print(dict([key, dataset.class_to_idx[key]] for key in dataset.classes[:no_examples]))
    print(dataset.samples[0])
    print(dataset.targets[0])

def img_grid(classes, dataset_loader, batch_size=16):
    # get some random training images
    dataiter = iter(dataset_loader)
    images, labels = dataiter.next()
    
    # print labels
    print('Numbered left to right, top to bottom:')
    print('\t'.join('{}: {}'.format(
    j+1, classes[labels[j]]) for j in range(batch_size)))

    # show images
    imshow(torchvision.utils.make_grid(images[:batch_size]), 'sample_grid')
    
def imshow(img, name):
    # functions to show an image
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.figure()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #plt.show()
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    plt.savefig(os.path.join(results_dir, '{}.png'.format(name)), dpi=300)


def show_results(device, loader, model, classes, batch_size=8):
    images, labels = iter(loader).next()[:batch_size]
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    
    print('\n'.join('Correct: {}, Predicted: {}'.format(
    classes[labels[j]], classes[predicted[j]]) for j in range(batch_size)))

    imshow(torchvision.utils.make_grid(images.cpu()[:batch_size]), 'class_results')
    

def plot_losses(training_proc_avg, test_proc_avg):
    # to plot learning curves
    x = np.arange(1, len(training_proc_avg)+1)
    x_2 = np.linspace(1, len(training_proc_avg)+1, len(test_proc_avg))

    fig = plt.figure()
    axs = fig.add_subplot(1, 1, 1)
    axs.plot(x, training_proc_avg, label='Training loss')
    axs.plot(x_2, test_proc_avg, label='Testing loss')
    axs.set_xlabel('Epoch no.')
    axs.set_ylabel('Average loss for epoch')
    axs.set_title('Loss as training progresses')
    axs.legend()
    
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    fig.savefig(os.path.join(results_dir, 'training_loss.png'), dpi=300)


def update_lr(optimizer, lr): 
    # For updating learning rate   
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr