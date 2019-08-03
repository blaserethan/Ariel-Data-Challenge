import pickle
import argparse

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn import preprocessing
import matplotlib.pyplot as plt


def build_FullyConnected(firstDense= [256], secondDense= [128,32], dropout= .5, ): # TODO parameterize model 

    input2d = tf.keras.Input(shape=(55,300), name="input_2d")
    
    #resize = layers.Reshape((55,300,1))(input2d)
    #layer2d = layers.Conv2D(32, (5,6), strides=(1,1))(resize)
    #layer2d = layers.BatchNormalization(momentum=0.75)(layer2d) 
    #layer2d = layers.Conv2D(16, (5,6), strides=(1,1))(layer2d)
    #layer2d = layers.BatchNormalization(momentum=0.75)(layer2d) 
    #layer2d = layers.Conv2D(8, (5,6), strides=(1,1))(layer2d)
    #layer2d = layers.BatchNormalization(momentum=0.75)(layer2d) 
    layer2d = layers.Flatten()(input2d)

    input1d = tf.keras.Input(shape=(55), name="input_1d")
    #resize = layers.Reshape((55,1))(input1d)
    #layer1d = layers.Conv1D(4, 3, strides=1)(resize)
    #layer1d = layers.BatchNormalization(momentum=0.75)(layer1d) 
    #layer1d = layers.Flatten()(layer1d)

    #layerc = layers.Concatenate()([layer1d,layer2d])
    layerc = layers.Concatenate()([input1d,layer2d])

    #first set of fully-connected layers
    for size in firstDense:
        layerc= layers.Dense(size, activation= 'relu')(layerc)
    output = layers.Dense(55, activation='relu')(layerc)
    layerc = layers.Dropout(dropout)(layerc)

    #second set of fully-connected layers
    for size in secondDense:
        layerc= layers.Dense(size, activation= 'relu')(layerc)

    #output layer
    output = layers.Dense(55, activation='relu')(layerc)

    return tf.keras.Model(inputs=[input2d,input1d], outputs=output, name='regressor')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_, default="regressor.h5")
    help_ = "Number of training epochs"
    parser.add_argument("-e", "--epochs", help=help_, default=20 ,type=int)
    args = parser.parse_args()

    truths = pickle.load( open("pickle_files/train_truths.pkl",'rb') )*100
    estimates = pickle.load( open("pickle_files/train_estimates.pkl",'rb') )*100
    residuals = pickle.load( open("pickle_files/train_residuals.pkl",'rb') )*100
    
    model = build_FullyConnected()
    model.summary() 

    #tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)
    
    #try:
    #    model.load_weights(args.weights)
    #except:
    #    print('load weights failed')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), # TODO parameterize optimizer 
        #loss=tf.keras.losses.MeanSquaredError(),
        loss=tf.keras.losses.MeanAbsolutePercentageError(),
        metrics=['accuracy']
    )
    
    history = model.fit(
        [residuals,estimates], truths,
        epochs=args.epochs, 
        batch_size=32,
        validation_split=0.1
    )
    
    model.save_weights(args.weights)

    # Plot training & validation loss values
    f,ax = plt.subplots(1)
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    ax.set_ylabel('Loss')
    ax.set_xlabel('Training Epoch')
    ax.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    #plt.savefig('nn_training.pdf',bbox_inches='tight')
    #plt.close()

    mse = np.sum( (truths-estimates)**2 )

    debiased = model.predict( [residuals,estimates] ) 
    mse_db = np.sum( (truths-debiased)**2 )

    print('starting mse:',mse)
    print('debiased mse:',mse_db)

    f, ax = plt.subplots( 5,10, figsize=(13,7) )
    plt.subplots_adjust(top=0.93,bottom=0.04,left=0.01,right=0.985)
    for i in range(5): 
        for j in range(10):
            ri = np.random.randint( truths.shape[0] )
            ax[i,j].plot( truths[ri], 'g-', label='Truth')
            ax[i,j].plot( estimates[ri], 'r--', label='Prior Estimate')
            ax[i,j].plot( debiased[ri], 'k-', label='NN Estimate')
            ax[i,j].axis('off')

    f.suptitle('Prediction on Training Samples')
    plt.show()



    # TODO mosaic for test data 
    '''
    f, ax = plt.subplots( 5,10 )
    for i in range(5): 
        for j in range(10):
            ri = np.random.randint( Xs.shape[0] )
            ax[i,j].plot( zts[ri], 'r-', label='Prior Estimate')
            ax[i,j].plot( ytp[ri], 'k-', label='NN Estimate')
            ax[i,j].axis('off')

    f.sup_title('Prediction on Test Samples ')
    plt.tight_layout()
    plt.show()
    '''