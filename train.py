import model
import cv2
import os
import numpy as np
import utils

#hyperparameter
#PS: hm_epoch means how many epoch
hm_epoch = 20
hm_batch = 32
image_size = 224


model_path = './data/models'

def train_image():
    x_train, x_valid, x_test, y_train, y_valid, y_test = utils.get_image_data(image_size)

    #instantiate the class
    models = model.Image_classification_models(num_classes=len(utils.image_label_list), \
            image_size=image_size)

    model_list = models.get_model_list()
    trained_model = []

    #train all the models in model_list
    for one_model in model_list:
        one_model.fit(x_train, y_train, epochs=hm_epoch, batch_size=hm_batch)
        trained_model.append(one_model)

    #evluate the trained models
    best_acc = 0
    for one_model in trained_model:
        train_metric_result = one_model.evaluate(x_train, y_train, batch_size=hm_batch, verbose=0)
        validation_metric_result = one_model.evaluate(x_valid, y_valid, batch_size=hm_batch, verbose=0)

        print('Model name:', one_model.name)
        print('Train Accuracy:', train_metric_result[1])
        print('Train Precision:', train_metric_result[2])
        print('Train Recall:', train_metric_result[3])
        print('Train F1:', train_metric_result[4])
        print('Validation Accuracy:', validation_metric_result[1])
        print('Validation Precision:', validation_metric_result[2])
        print('Validation Recall:', validation_metric_result[3])
        print('Validation F1:', validation_metric_result[4])
        print('\n')

        #pick the best model based on their accuracy
        if validation_metric_result[1] > best_acc:
            best_acc = validation_metric_result[1]
            best_model_precision = validation_metric_result[2]
            best_model_recall = validation_metric_result[3]
            best_model_f1 = validation_metric_result[4]
            best_model = one_model

    print('The best model is: %s with' %(best_model.name))
    print('%.2f %% accuracy' %(best_acc))
    print('%.2f %% precision' %(best_model_precision))
    print('%.2f %% recall' %(best_model_recall))
    print('%.2f %% f1' %(best_model_f1))

    #test the model on test set
    best_model_train = best_model.evaluate(x_train, y_train, batch_size=hm_batch, verbose=0)
    best_model_test = best_model.evaluate(x_test, y_test, batch_size=hm_batch, verbose=0)

    print('Train loss: %.3f acc: %.3f precision: %.3 recall: %.3f f1: %.3f' \
            %(best_model_train[0], best_model_train[1], best_model_train[2], best_model_train[3], best_model_train[4]))
    print('Test loss: %.3f acc: %.3f precision: %.3 recall: %.3f f1: %.3f' \
            %(best_model_test[0], best_model_test[1], best_model_test[2], best_model_test[3], best_model_test[4]))

    #save the best model
    print('Saving model...')
    best_model.save(model_path + '/best_image_model.h5')

if __name__ == "__main__":
    train_image()
