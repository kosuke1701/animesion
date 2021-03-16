import os
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 

import torch
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary

from train import accuracy
from inference import environment_loader

def evaluate(args, device, model, data_set, data_loader):
    
    file_name = 'evaluate_{}.txt'.format(os.path.splitext(os.path.split(args.checkpoint_path)[1])[0])
    f = open(os.path.join(args.results_dir, '{}'.format(file_name)), 'w')

    # related to dataset
    no_classes = data_set.no_classes
    classid_classname_dic = data_set.classes
    total_step = len(data_loader)
    curr_line = 'Total no. of samples in test set: {}\nTotal no. of classes: {}\n'.format(
        len(data_set), no_classes)
    print(curr_line)
    f.write(curr_line)
    
    # Test the model (test set)
    # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    # dropout probability goes to 0
    model.eval()
    # prints model summary (layers, parameters by giving it a sample input)
    if args.vis_arch:
        summary(model, input_size=iter(data_loader).next()[0].shape[1:])
    
    with torch.no_grad():
        correct_1 = 0
        correct_5 = 0
        total = 0

        class_correct = list(0. for i in range(no_classes))
        class_total = list(0. for i in range(no_classes))
        
        for i, (images, labels) in enumerate(data_loader):

            # prints current evaluation step after each 10 steps
            if (i % 10) == 0:
                print ("Step [{}/{}]".format(i+1, total_step))

            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # calculate top-k (1 and 5) accuracy
            total += labels.size(0)
            curr_corr_list = accuracy(outputs.data, labels, (1, 5, ))
            correct_1 += curr_corr_list[0]
            correct_5 += curr_corr_list[1]    
            
            # calculate per-class accuracy
            _, predicted = torch.max(outputs.data, 1)
            c = (predicted == labels).squeeze()
            if c.ndim == 0:
                print(images.size())
                class_correct[labels[0]] += c.item()
                class_total[labels[0]] += 1
            else:
                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        # compute epoch accuracy in percentages
        curr_top1_acc = 100 * correct_1/total
        curr_line = 'Val/Test Top-1 Accuracy of the model on the test images: {:.4f} %\n'.format(curr_top1_acc)
        print(curr_line)
        f.write(curr_line)

        curr_top5_acc = 100 * correct_5/total
        curr_line = 'Val/Test Top-1 Accuracy of the model on the test images: {:.4f} %\n'.format(curr_top5_acc)
        print(curr_line)
        f.write(curr_line)

        # compute per class accuracy
        for i in range(no_classes):
            class_accuracy = 100 * class_correct[i] / class_total[i]
            class_name = classid_classname_dic.loc[classid_classname_dic['class_id']==i, 'class_name'].item()
            curr_line = 'Total objects in class no. {} ({}): {}. Top-1 Accuracy: {}\n'.format(
            i, class_name, class_total[i], class_accuracy)
            print(curr_line)
            f.write(curr_line)

    f.close()

def main():
  
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", choices=["moeImouto", "danbooruFaces", "cartoonFace", "ZACI20"], 
                        default="moeImouto", help="Which dataset to use.")
    parser.add_argument("--dataset_path", 
                        default="/home2/yan/disk/edwin/personal/data/Danbooru2018AnimeCharacterRecognitionDataset_Revamped/",
                        help="Path for the dataset.")
    parser.add_argument("--image_size", choices=[128, 224], default=128, type=int,
                        help="Image (square) resolution size")
    parser.add_argument("--model_type", choices=["shallow", 'resnet18', 'resnet152', 
                        'B_16', 'B_32', 'L_16', 'L_32'], default='L_32',
                        help="Which model architecture to use")
    parser.add_argument("--checkpoint_path", type=str, 
                        default="checkpoints/danbooruFaces_l32_ptTrue_batch64_imageSize128_50epochs_epochDecay20.ckpt",
                        help="Path for model checkpoint to load.")    
    parser.add_argument("--results_dir", default="results_inference", type=str,
                        help="The directory where results will be stored.")
    parser.add_argument("--pretrained", type=bool, default=True,
                        help="DON'T CHANGE! Always true since always loading when doing inference.")
    parser.add_argument("--batch_size", default=64, type=int,
                        help="Batch size for train/val/test.")                        
    parser.add_argument("--vis_arch", type=bool, default=True,
                        help="Visualize architecture through model summary.")

               
    args = parser.parse_args()
    print(args)

    device, model, data_set, data_loader = environment_loader(args) 

    evaluate(args, device, model, data_set, data_loader)
              

if __name__ == '__main__':
    main()
