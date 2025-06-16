# -*- coding: utf-8 -*-
'''

The functions and classes defined in this module will be called by a marker script. 
You should complete the functions and classes according to their specified interfaces.

No partial marks will be awarded for functions that do not meet the specifications
of the interfaces.

Last modified 2024-05-07 by Anthony Vanderkop.
Hopefully without introducing new bugs.
'''


### LIBRARY IMPORTS HERE ###
import os
import numpy as np
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt
from PIL import Image
import scipy as sc
from tensorflow import keras
import tensorflow.keras.applications as ka
import seaborn as sns

class MetricsRecorder(tf.keras.callbacks.Callback):
    """
    Function extracts the training and validation accuracy and losses
    Args:
        tf (tf object): tf library object
    """
    def __init__(self):
        super(MetricsRecorder, self).__init__()
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    def on_epoch_end(self, epoch, logs=None):
        self.train_losses.append(logs.get('loss'))
        self.train_accuracies.append(logs.get('accuracy'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_accuracies.append(logs.get('val_accuracy'))

def plot_training_history(metrics_recorder):
    """
    Creates a graph with epoch on the x axis and loss and/or accuracy on the y axis. Showcases both the validation and training curves.
    Args:
        metrics_recorder (MetricsRecorder class): Class created from MetricsRecorder used to access the callbacks of the model compiling
    """
    epochs = len(metrics_recorder.train_losses)
    plt.figure(figsize=(12, 5))

    # Plot training and validation losses
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), metrics_recorder.train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), metrics_recorder.val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training and validation accuracies
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), metrics_recorder.train_accuracies, label='Training Accuracy')
    plt.plot(range(1, epochs + 1), metrics_recorder.val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
    return [(11078472, " Samuel", "van Zuylen"), (10937951, "Paul", "Rodchue"), (11241691, "Martino", "Nguyen")]
    
def load_model():
    '''
    Load and modify a pre-trained MobileNetV2 model for flower classification.

    This function performs the following steps:
    1. Loads the MobileNetV2 model pre-trained on ImageNet.
    2. Freezes the pre-trained layers so they are not updated during training.
    3. Adds a new Dense layer with 5 units for flower classification.

    Returns:
        new_model: A modified MobileNetV2 model ready for flower classification.
    '''
    # Retrieve MobileNetV2 Data
    base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
        input_shape=(224, 224, 3),
        alpha=1.0,
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        pooling=None,
        classes=1000,
        classifier_activation='softmax'
    )

    base_model.trainable = False

    # Replace the last layer with a Dense layer for flower classification
    new_predictions = keras.layers.Dense(5)(base_model.layers[-2].output)
    new_model = keras.Model(inputs=base_model.inputs, outputs=new_predictions)

    return new_model

def load_data(path):
    '''
    Load in the dataset from its home path. Path should be a string of the path
    to the home directory the dataset is found in. Should return a numpy array
    with paired images and class labels.
    
    Inputs:
            - path: string of the folder directory
    Outputs: 
            - X: np.array of data from the flower images
            - Y: np.array of data containing the class labels of those flowers
    '''
    Y = []
    X = []
    target_size = (224,224)
    # Finding image shape
    
    # Iterate through each folder (flower type)
    for class_label, class_name in enumerate(os.listdir(path)):
        class_dir = os.path.join(path, class_name)
        
        # Iterate through each image in the folder
        for image_file in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_file)
            
            # Load image and convert to numpy array
            image = Image.open(image_path)
            # Resizing image to convert to numpy
            image_resized = image.resize(target_size)
            
            image_np = np.array(image_resized)
            
            # Append image data and class label 
            Y.append(class_label)
            X.append(image_np)
    Y = np.array(Y)
    X = np.array(X)
    return X, Y
    
def split_data(X, Y, train_fraction, randomize=False, eval_set=False):
    """
    Split the data into training and testing sets. If eval_set is True, also create
    an evaluation dataset. There should be two outputs if eval_set there should
    be three outputs (train, test, eval), otherwise two outputs (train, test).
    
    To see what type train, test, and eval should be, refer to the inputs of 
    transfer_learning().
    
    Inputs:
            - X: np.array of data from the flower images
            - Y: np.array of data containing the class labels of those flowers
            - train_fraction: float, the fraction to which splits the data from training and test datasets
            - randomize: boolean. If true, randomizes the dataset
            - eval_set: boolean. If true, create an evaluation dataset
    Outputs:
            - train_set: list. The dataset of the training dataset
            - test_set: list. The dataset of the test dataset
            - val_set: list. The dataset of the evaluation dataset
    """
    # Get the dimensions of X and Y
    Nx, _,_,_ = X.shape
    Ny = Y.shape[0]

    
    # Check if X and Y hae the same number of rows
    if Nx != Ny:
        raise ValueError('X and Y should have the same number of rows')
    
    # Calculate the number of samples for training
    Ne = round(train_fraction * Nx)
    
    # Randomize the data if randomize is true 
    if randomize:
        indices = np.random.permutation(Nx)
        X = X[indices]
        Y = Y[indices]
        
    # Split data into training and testing sets
    Xtr, Xte = X[:Ne], X[Ne:]
    Ytr, Yte = Y[:Ne], Y[Ne:]
    
    # if eval_set is true, create an evaulation dataset
    if eval_set:
        # Splitting training set and evaluation with 50/50 split
        Ne_eval = int(0.5* len(Xte))
        train_set = [Xtr, Ytr]
        Xte, Xev = Xte[:Ne_eval],Xte[Ne_eval:]
        Yte, Yev = Yte[:Ne_eval],Yte[Ne_eval:]
        val_set = [Xev,Yev]
        test_set = [Xte,Yte]
        return train_set, test_set, val_set
    else:
        train_set = [Xtr,Ytr]
        test_set = [Xte,Yte]
        return train_set, test_set

def confusion_matrix(predictions, ground_truth, plot=True, all_classes=None):
    '''
    Given a set of classifier predictions and the ground truth, calculate and
    return the confusion matrix of the classifier's performance.

    Inputs:
        - predictions: np.ndarray of length n where n is the number of data
                       points in the dataset being classified and each value
                       is the class predicted by the classifier
        - ground_truth: np.ndarray of length n where each value is the correct
                        value of the class predicted by the classifier
        - plot: boolean. If true, create a plot of the confusion matrix with
                either matplotlib or with sklearn.
        - classes: a set of all unique classes that are expected in the dataset.
                   If None is provided we assume all relevant classes are in 
                   the ground_truth instead.
    Outputs:
        - cm: type np.ndarray of shape (c,c) where c is the number of unique  
              classes in the ground_truth
              
              Each row corresponds to a unique class in the ground truth and
              each column to a prediction of a unique class by a classifier
    '''
    
    # Initialise classes
    classes = all_classes if all_classes is not None else np.unique(ground_truth)
    
    # Initialise confusion matrix
    cm = np.zeros((len(classes), len(classes)), dtype=np.int32)
    
    # Populate matrix
    for i in range(len(predictions)):
        true_class_index = np.where(classes == ground_truth[i])[0][0]
        pred_class_index = np.where(classes == predictions[i])[0][0]
        cm[true_class_index][pred_class_index] += 1
    
    if plot:
        # Plot the confusion matrix using Matplotlib and Seaborn
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
    
    # Return confusion matrix
    return cm
    

def precision(predictions, ground_truth):
    '''
    Given a set of classifier predictions and the ground truth, calculate and
    return the precision of the classifier's performance for each class.
    
    Inputs:
        - predictions: np.ndarray of length n where n is the number of data
                       points in the dataset being classified and each value
                       is the class predicted by the classifier
        - ground_truth: np.ndarray of length n where each value is the correct
                        value of the class predicted by the classifier
    Outputs:
        - precision: type np.ndarray of length c,
                    values are the precision for each class
    '''
    
    # Initialise confusion matrix
    cm = confusion_matrix(predictions, ground_truth, False)
    
    # Get true positives and false positives from cm
    true_positives = np.diag(cm)
    false_positives = np.sum(cm, axis=0) - true_positives
    
    # Calculate precision using P = TP / (TP + FP)
    precision = true_positives / ( true_positives + false_positives )
    
    # Return precision value
    return precision

def recall(predictions, ground_truth):
    '''
    Given a set of classifier predictions and the ground truth, calculate and
    return the recall of the classifier's performance for each class.
    
    Inputs:
        - predictions: np.ndarray of length n where n is the number of data
                       points in the dataset being classified and each value
                       is the class predicted by the classifier
        - ground_truth: np.ndarray of length n where each value is the correct
                        value of the class predicted by the classifier
    Outputs:
        - recall: type np.ndarray of length c,
                    values are the recall for each class
    '''
    
    # Initialise confusion matrix
    cm = confusion_matrix(predictions, ground_truth, False)
    
    # Get true positives and false negatives from cm
    true_positives = np.diag(cm)
    false_negatives = np.sum(cm, axis=1) - true_positives    
    
    # Calculate recall using R = TP / (TP + FN)
    recall = true_positives / ( true_positives + false_negatives )
    
    # Return recall value
    return recall

def f1(predictions, ground_truth):
    '''
    Given a set of classifier predictions and the ground truth, calculate and
    return the f1 scores of the classifier's performance for each class.
    
    Inputs:
        - predictions: np.ndarray of length n where n is the number of data
                       points in the dataset being classified and each value
                       is the class predicted by the classifier
        - ground_truth: np.ndarray of length n where each value is the correct
                        value of the class predicted by the classifier
    Outputs:
        - f1: type nd.ndarry of length c where c is the number of classes
    '''
    
    # Initialise precision and recall
    prec = precision(predictions, ground_truth)
    rec = recall(predictions, ground_truth)
    
    # Calculate f1 score using f1 = (2PR) / (P + R)
    f1 = ( 2 * prec * rec ) / ( prec + rec )
    
    # Return f1 score
    return f1

def k_fold_cross_validation(X, Y, model, parameters, k=5):
    """
    Perform k-fold cross-validation.

    Inputs:
        - X: input features (images)
        - Y: labels
        - model: an instance of tf.keras.applications.MobileNetV2
        - parameters: list or tuple of parameters to use during training:
            (learning_rate, momentum, nesterov)
        - k: number of folds for cross-validation (default is 5)

    Outputs:
        - avg_metrics: average metrics over all folds
        - sigma_metrics: standard deviation of metrics across folds
    """

    # Calculate number of samples per fold
    fold_size = len(X) // k
    
    # Shuffle the data
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    
    # Initialize lists to store metrics for each fold
    all_metrics = []
    
    # Perform k-fold cross-validation
    for i in range(k):
        print(f"Processing fold {i+1}/{k}")
        # Split data into training and evaluation sets
        val_X = X[i * fold_size: (i + 1) * fold_size]
        val_Y = Y[i * fold_size: (i + 1) * fold_size]
        
        train_X = np.concatenate([X[:i * fold_size], X[(i + 1) * fold_size:]], axis=0)
        train_Y = np.concatenate([Y[:i * fold_size], Y[(i + 1) * fold_size:]], axis=0)
        
        # Perform transfer learning
        trained_model, metrics = transfer_learning((train_X, train_Y), (val_X, val_Y),
                                                   (val_X, val_Y), model, parameters)
        
        all_metrics.append(metrics)
    
    # Calculate average metrics and standard deviation over all folds
    avg_metrics = np.mean(all_metrics, axis=0)
    sigma_metrics = np.std(all_metrics, axis=0)
    
    return avg_metrics, sigma_metrics


##################### MAIN ASSIGNMENT CODE FROM HERE ######################

def transfer_learning(train_set, eval_set, test_set, model, parameters):
    '''
    Implement and perform standard transfer learning here.

    Inputs:
        - train_set: list or tuple of the training images and labels in the
            form (images, labels) for training the classifier
        - eval_set: list or tuple of the images and labels used in evaluating
            the model during training, in the form (images, labels)
        - test_set: list or tuple of the training images and labels in the
            form (images, labels) for testing the classifier after training
        - model: an instance of tf.keras.applications.MobileNetV2
        - parameters: list or tuple of parameters to use during training:
            (learning_rate, momentum, nesterov)


    Outputs:
        - model : an instance of tf.keras.applications.MobileNetV2
        - metrics : list of classwise recall, precision, and f1 scores of the 
            model on the test_set (list of np.ndarray)

    '''
    # Unpack Parameters
    learning_rate, momentum, nesterov = parameters
    
    # Unpack dataset
    train_image, train_labels = train_set
    eval_image,eval_labels = eval_set
    test_image,test_labels = test_set
    
    # Normalize image
    train_images = train_image.astype('float32')/255.0
    eval_images = eval_image.astype('float32')/255.0
    test_images = test_image.astype('float32')/255.0
    
    
    # Freeze convolutional base
    model.trainable = False
    
    # Create new layers for classification
    inputs = model.input
    x = model.layers[-2].output  # Use the second to last layer's output
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(len(np.unique(train_labels)), activation='softmax')(x)
    
    new_model = tf.keras.Model(inputs=inputs,outputs = outputs)
    
    # Compile the model
    new_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=nesterov),
                loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Initialize MetricsRecorder callback
    metrics_recorder = MetricsRecorder()
    
    # Train the model
    new_model.fit(train_images, train_labels, validation_data=(eval_images, eval_labels), epochs = 10, callbacks=[metrics_recorder])
    
    # Evaluate the model on the test set
    test_predictions = new_model.predict(test_images)
    test_predictions = np.argmax(test_predictions, axis=1)
    
    # Calculate class-wise precision, recall, and F1 scores
    precision_val = precision(test_predictions,test_labels)
    recall_val = recall(test_predictions,test_labels)
    f1_val = f1(test_predictions,test_labels)
    
    # Metrics as a list of numpy arrays
    metrics = [precision_val, recall_val, f1_val]
    
    # Plot the training and validation metrics
    plot_training_history(metrics_recorder)
    
    return new_model, metrics

def accelerated_learning(train_set, eval_set, test_set, model, parameters):
    '''
    Implement and perform accelerated transfer learning here.

    Inputs:
        - train_set: list or tuple of the training images and labels in the
            form (images, labels) for training the classifier
        - eval_set: list or tuple of the images and labels used in evaluating
            the model during training, in the form (images, labels)
        - test_set: list or tuple of the training images and labels in the
            form (images, labels) for testing the classifier after training
        - model: an instance of tf.keras.applications.MobileNetV2
        - parameters: list or tuple of parameters to use during training:
            (learning_rate, momentum, nesterov)

    Outputs:
        - new_model : an instance of tf.keras.applications.MobileNetV2
        - metrics : list of classwise recall, precision, and f1 scores of the 
            model on the test_set (list of np.ndarray)
    '''
    # Unpack training parameters
    learning_rate, momentum, nesterov = parameters
    
    # Unpack training, evaluation, and test sets
    train_images, train_labels = train_set
    eval_images, eval_labels = eval_set
    test_images, test_labels = test_set
    
    # Normalize images
    train_images = train_images.astype('float32') / 255.0
    eval_images = eval_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0
    
    # Freeze convolutional base
    model.trainable = False
    
    # Precompute the activations of the base mode
    train_activations = model.predict(train_images)
    eval_activations = model.predict(eval_images)
    test_activations = model.predict(test_images)
    
    # Create new layers for classification
    activation_input = tf.keras.Input(shape=train_activations.shape[1:])
    x = tf.keras.layers.Dense(128, activation='relu')(activation_input)
    output_layer = tf.keras.layers.Dense(len(np.unique(train_labels)), activation='softmax')(x)
    
    # Create a new model with the precomputed activations as input
    new_model = tf.keras.Model(inputs=activation_input, outputs=output_layer)
    
    # Compile the new model
    new_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=nesterov),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Initialize MetricsRecorder callback
    metrics_recorder = MetricsRecorder()
    
    # Train the new model
    new_model.fit(train_activations, train_labels, validation_data=(eval_activations, eval_labels), epochs=10, callbacks=[metrics_recorder])
    
    # Evaluate the model on the test set and obtain predictions
    test_predictions = new_model.predict(test_activations)
    test_predictions = np.argmax(test_predictions, axis=1)
    
    # Calculate class-wise precision, recall, and F1 scores
    precision_val = precision(test_predictions, test_labels)
    recall_val = recall(test_predictions, test_labels)
    f1_val = f1(test_predictions, test_labels)
    
    # Return the model and metrics
    metrics = [precision_val, recall_val, f1_val]
    
    # Plot the training and validation metrics
    plot_training_history(metrics_recorder)
    return new_model, metrics


def task_7():
    model = load_model()
    X, Y = load_data("small_flower_dataset")
    parameters = [0.01, 0.0, False]
    train_fraction = 0.70
    train_set,test_set,val_set = split_data(X,Y,train_fraction,True,True)
    model, metrics = transfer_learning(train_set, val_set, test_set, model, parameters)
    
def task_8():
    # Setting random seed for repoducibility
    np.random.seed(42)
    X, Y = load_data("small_flower_dataset")
    train_fraction = 0.70
    train_set,test_set,val_set = split_data(X,Y,train_fraction,True,True)
    
    # List of learning rates to test
    learning_rates = [0.1, 0.001, 0.0001]
    
    for lr in learning_rates:
        # Load a fresh model for each learning rate
        model = load_model()
        parameters = [lr,0.0, False]
        model, metrics = transfer_learning(train_set, val_set, test_set, model, parameters)
        
def task_9_10():
    model = load_model()
    X, Y = load_data("small_flower_dataset")
    train_fraction = 0.70
    train_set, test_set, val_set = split_data(X, Y, train_fraction, randomize=True, eval_set=True)
    # Set the best parameters found during experimentation
    best_learning_rate = 0.001 
    parameters = [best_learning_rate, 0.0, False]
    # Train the model using transfer learning
    model, metrics = transfer_learning(train_set, val_set, test_set, model, parameters)
    
    # Unpack the test set
    test_images, test_labels = test_set
    
    # Normalize test images
    test_images = test_images.astype('float32') / 255.0
    
    # Predict the test dataset
    test_predictions = model.predict(test_images)
    test_predictions = np.argmax(test_predictions, axis=1)
    
    # Compute and display the confusion matrix
    cm = confusion_matrix(test_predictions, test_labels, plot=True)
    print("Precision")
    print(metrics[0])
    print("Recall")
    print(metrics[1])
    print("F1")
    print(metrics[2])
    

def task_11():
    """
    Perform k-fold cross-validation with k=3, 5, and 10.

    Outputs:
        - results: dictionary containing average and sigma metrics for each k value
    """
    
    model = load_model()
    X, Y = load_data("small_flower_dataset")
    
    # Set the best parameters found during experimentation
    best_learning_rate = 0.001 
    parameters = [best_learning_rate, 0.0, False]

    results = {}

    # Perform k-fold cross-validation for k=3, 5, and 10
    for k_value in [3, 5, 10]:
        print(f"Performing k-fold cross-validation with k={k_value}")
        avg_metrics, sigma_metrics = k_fold_cross_validation(X, Y, model, parameters, k=k_value)
        results[f'k_{k_value}'] = {'avg_metrics': avg_metrics, 'sigma_metrics': sigma_metrics}
        print(f"Results for k={k_value}:")
        print("Average Metrics:")
        print(avg_metrics)
        print("Sigma Metrics:")
        print(sigma_metrics)
        print()
    
    return results
    
        
def task_12():
    model = load_model()
    X, Y = load_data("small_flower_dataset")
    train_fraction = 0.70
    train_set, test_set, val_set = split_data(X, Y, train_fraction, randomize=True, eval_set=True)
    
    # Set the best learning rate found during experimentation
    best_learning_rate = 0.001 
    
    # Set different momentum levels
    momentums = [0.1, 0.5, 0.9]
    
    results = {}

    for momentum in momentums:
        print(f"Training with momentum={momentum}")
        parameters = [best_learning_rate, momentum, False]
        
        # Train the model using transfer learning
        trained_model, metrics = transfer_learning(train_set, val_set, test_set, model, parameters)
        results[f'momentum_{momentum}'] = {'model': trained_model, 'metrics': metrics}
    
        # Unpack the test set
        test_images, test_labels = test_set
    
        # Normalize test images
        test_images = test_images.astype('float32') / 255.0
    
        # Predict the test dataset
        test_predictions = trained_model.predict(test_images)
        test_predictions = np.argmax(test_predictions, axis=1)
    
        # Compute and display the confusion matrix
        cm = confusion_matrix(test_predictions, test_labels, plot=True)
        print(f"For momentum={momentum}")
        print("Precision")
        print(metrics[0])
        print("Recall")
        print(metrics[1])
        print("F1")
        print(metrics[2])
    
    return results

def task_14():
    model = load_model()
    X, Y = load_data("small_flower_dataset")
    parameters = [0.01, 0.0, False]
    train_fraction = 0.70
    train_set,test_set,val_set = split_data(X,Y,train_fraction,True,True)
    acc_model, acc_metrics = accelerated_learning(train_set, val_set, test_set, model, parameters)
    

if __name__ == "__main__":
    task_7()
    #task_8()
    #task_9_10()
    #task_11()
    #task_12()
    #task_14()
#########################  CODE GRAVEYARD  #############################
