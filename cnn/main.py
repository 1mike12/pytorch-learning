import torch
import torchvision
import torchvision.transforms as transforms
from timeit import default_timer as timer
import datetime
import matplotlib.pyplot as plt
import numpy as np

from cnn.CNN import CNN

DATA_DIR = '../../data'
RUN_ON_GPU = False
BATCH_SIZE = 4  # default 4
NEED_TO_DL_DATASET = False

EPOCHS = 2 #default 2
LOG_EVERY_X_BATCHES = int(8000 / BATCH_SIZE)  #default 2000


def main():
    if RUN_ON_GPU:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if not torch.cuda.is_available(): raise Exception("trying to run on gpu but no cuda available")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    ########################################################################
    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1].

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainingSet = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=NEED_TO_DL_DATASET, transform=transform)
    trainingLoader = torch.utils.data.DataLoader(trainingSet, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    testSet = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=NEED_TO_DL_DATASET, transform=transform)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    ########################################################################
    # Let us show some of the training images, for fun.

    # functions to show an image

    def showImage(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    # get some random training images
    # data = iter(trainingLoader)
    # images, labels = data.next()

    # show images
    # showImage(torchvision.utils.make_grid(images))
    # print labels
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    cnn = CNN(device)

    ########################################################################
    # 4. Train the network
    # ^^^^^^^^^^^^^^^^^^^^
    #
    # This is when things start to get interesting.
    # We simply have to loop over our data iterator, and feed the inputs to the
    # network and optimize.
    start = timer()
    print(datetime.datetime.now())

    runs = 0
    for epoch in range(EPOCHS):  # loop over the dataset multiple times

        runningLoss = 0.0
        for i, data in enumerate(trainingLoader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # print statistics
            runningLoss += cnn.runAndGetLoss(inputs, labels)
            runs = runs + 1
            # if runs > 2000:
            #     break
            if i % LOG_EVERY_X_BATCHES == (LOG_EVERY_X_BATCHES - 1):
                print(f'[{epoch + 1}, {i+1}] loss: {epoch + 1, i + 1, runningLoss / LOG_EVERY_X_BATCHES}')
                runningLoss = 0.0
                print(timer() - start)
                start = timer()

    print('Finished Training')
    print(datetime.datetime.now())
    print("\n")

    ########################################################################
    # 5. Test the network on the test data
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #
    # We have trained the network for 2 passes over the training dataset.
    # But we need to check if the network has learnt anything at all.
    #
    # We will check this by predicting the class label that the neural network
    # outputs, and checking it against the ground-truth. If the prediction is
    # correct, we add the sample to the list of correct predictions.
    #
    # Okay, first step. Let us display an image from the test set to get familiar.

    data = iter(testLoader)
    images, labels = data.next()
    images = images.to(device)
    labels = labels.to(device)

    # print images
    # showImage(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    ########################################################################
    # Okay, now let us see what the neural network thinks these examples above are:

    outputs = cnn(images)

    ########################################################################
    # The outputs are energies for the 10 classes.
    # Higher the energy for a class, the more the network
    # thinks that the image is of the particular class.
    # So, let's get the index of the highest energy:
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

    ########################################################################
    # The results seem pretty good.
    #
    # Let us look at how the network performs on the whole dataset.

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testLoader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = cnn(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    ########################################################################
    # That looks waaay better than chance, which is 10% accuracy (randomly picking
    # a class out of 10 classes).
    # Seems like the network learnt something.
    #
    # Hmmm, what are the classes that performed well, and the classes that did
    # not perform well:

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testLoader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = cnn(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(len(classes)):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == '__main__':
    main()
