import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
import torchvision

from data.datasets import moeImoutoDataset, danbooruFacesCrops, moeImouto

def data_loading(dataset_name, split, batch_size, visualization=True, transform=False):

    if dataset_name == 'moeImouto':
        #dataset = moeImoutoDataset(transform=transform).getImageFolder()
        dataset = moeImouto(input_size=128, train=False)
        
        classes = dataset.classes
        
    if dataset_name == 'danbooruFacesCrops':
        dataset = danbooruFacesCrops(split=split, transform=True)
        classes = dataset.classes
        dataset.stats()
    # split into training, validation, and testing
    # https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets/50544887#50544887
    dataset_loader = data.DataLoader(dataset,
                    batch_size=batch_size, shuffle=True,
                    num_workers=1)

    if visualization==True:
        classes_print(dataset)
        img_grid(classes, dataset_loader, batch_size=16)

    return dataset_loader, classes

def classes_print(dataset, no_examples=10):
    print(dataset)
    print(dataset.classes[:no_examples])
    #print(dict([key, dataset.class_to_idx[key]] for key in dataset.classes[:no_examples]))
    #print(dataset[0])
    #print(dataset.samples[0])
    #print(dataset.targets[0])

def img_grid(classes, dataset_loader, batch_size=8):
    # get some random training images
    dataiter = iter(dataset_loader)
    images, labels = dataiter.next()
    print(images.shape, labels.shape)
    
    # print labels
    print('Numbered left to right, top to bottom:')
    print('\t'.join('{}: {}'.format(
    j+1, classes[labels[j]]) for j in range(batch_size)))

    # show images
    imshow(torchvision.utils.make_grid(images[:batch_size]), 'sample_grid')
    
def imshow(img, name, bottom_text=None):
    # functions to show an image
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.figure()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #plt.show()
    
    if bottom_text is not None:
        plt.text(0.01, 0.01, bottom_text, fontsize=4, bbox=dict(facecolor='gray', alpha=0.2))

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
    bottom_text = '\n'.join('Correct: {}, Predicted: {}'.format(
    classes[labels[j]], classes[predicted[j]]) for j in range(batch_size))
    imshow(torchvision.utils.make_grid(images.cpu()[:batch_size]), 'class_results', bottom_text)
    

def plot_losses(training_proc_avg, test_proc_avg, results_dir):
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
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    fig.savefig(os.path.join(results_dir, 'training_loss.png'), dpi=300)

def calc_stats(samples_per_class, no_classes, class_names):
	samples_per_class = samples_per_class.groupby('class_id', as_index=True).count()['dir'].squeeze()
		
	set_mean = samples_per_class.mean()
	set_median = samples_per_class.median()
	set_std = samples_per_class.std()
	print('Dataset no of classes: {}\n'
	'Mean number of samples per class: {}\n'
	'Median number of samples per class: {}\n'
	'Standard deviation of samples per class: {}'.format(
	no_classes, set_mean, set_median, set_std))

	samples_per_class_ordered = samples_per_class.sort_values(0, ascending=False)
	print(samples_per_class_ordered.head())
	print('Shape of samples per class df:')
	print(samples_per_class_ordered.shape)

	print('Characters with most number of samples: ')
	print('\t'.join('No. {}: {} (Class ID: {}) with {} samples'.format(
    j+1, class_names[samples_per_class_ordered.index[j], 0], 
	class_names[samples_per_class_ordered.index[j], 1], 
	samples_per_class_ordered.iloc[j]) for j in range(10)))

	print('\nCharacters with least number of samples: ')
	print('\t'.join('No. {}: {} (Class ID: {}) with {} samples'.format(
    j+1, class_names[samples_per_class_ordered.index[-1-j], 0], 
	class_names[samples_per_class_ordered.index[-1-j], 1], 
	samples_per_class_ordered.iloc[-1-j]) for j in range(10)))

	fig, axs = plt.subplots(1)
	fig.suptitle('Histogram of Classes for DanbooruFace Dataset')

	# only plot first 100
	bins = 100
	axs.bar(np.arange(bins), samples_per_class_ordered.iloc[0:bins].to_numpy())
		
	axs.set_ylabel('No. of samples per class')
	axs.set_title('Ordered based on no. of samples')
		
	results_dir = 'results'
	if not os.path.exists(results_dir):
		os.makedirs(results_dir)
	fig.savefig(os.path.join(results_dir, 
	'histogram_danbooruFacesCrops.png'), dpi=300)