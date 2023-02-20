from termcolor import colored
import tensorflow as tf
from tensorflow import keras
import sys




# This is a list of all possible models to create
model_list = {
    "resnet_50": keras.applications.resnet50.ResNet50, 
    "resnet_VGG16": keras.applications.VGG16, 
    "InceptionV3": keras.applications.InceptionV3,
    "ResNet50V2": keras.applications.ResNet50V2, 
    "Xception": keras.applications.Xception
}

custom_models = {
    "3dcnn": ""
}

class TrainingModel:

    def __init__(self, hyperparameters, model_type, target_height, target_width, class_names, model_code_path):

        """ Creates and prepares a model for training.
            
        Args:
            hyperparameters (dict): The configuration's hyperparameters.
            model_type (str): Type of model to create.
            target_height (int): Height of input.
            target_width (int): Width of input.
            class_names (list of str): A list of classes.
            
        Returns:
            model (keras Model): The prepared keras model.
            model_type (str): The name of the model type.
        """
        print("ici model_type : ", model_type)
        # Catch if the model is not in the model list
            
        if model_type not in model_list:
            print("essai 2 using custom value")
            
            # TODO: Create a custom model
            #self.model = get_model()
            # Return the prepared model #TODO: adapt that part according to the custom model
            #avg =  keras.layers.GlobalAveragePooling2D()(base_model.output)
            #out =  keras.layers.Dense(len(class_names), activation="softmax")(avg)
            #self.model = keras.models.Model(inputs=base_model.input, outputs=out)
            
            width=185
            height=210
            depth=185
            self.model_type = model_type
            inputs = tf.keras.Input((width, height, depth, 1))

            x = tf.keras.layers.Conv3D(filters=20, kernel_size=10, activation="relu")(inputs)
            x = tf.keras.layers.AveragePooling3D(pool_size=5)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Conv3D(filters=20, kernel_size=10, activation="relu")(x)
            x = tf.keras.layers.BatchNormalization()(x)    
            x = tf.keras.layers.Conv3D(filters=50, kernel_size=10, activation="relu")(x)
            x = tf.keras.layers.BatchNormalization()(x)

            x = tf.keras.layers.GlobalAveragePooling3D()(x)
            x = tf.keras.layers.Dense(units=10, activation="relu")(x)
            #x = tf.keras.layers.Dropout(0.5)(x)

            outputs = tf.keras.layers.Dense(units=3, activation="softmax")(x)

            # Define the model.
            cmodel = tf.keras.Model(inputs, outputs, name="3dcnn")

            self.model = cmodel
            # Create optimizer and add to model
            optimizer = keras.optimizers.Adam(
                lr=hyperparameters['learning_rate']
            )
            self.model.compile(
                loss="sparse_categorical_crossentropy", 
                optimizer=optimizer,
                metrics=["accuracy"]
            )
        else:
            self.model_type = model_type
            # Get the model base
            base_model = model_list[model_type](
                include_top=False,
                weights=None,
                input_tensor=None,
                input_shape=(target_height, target_width, hyperparameters['channels']),
                pooling=None
            )
            # Return the prepared model #TODO: adapt that part according to the custom model
            avg =  keras.layers.GlobalAveragePooling2D()(base_model.output)
            out =  keras.layers.Dense(len(class_names), activation="softmax")(avg)
            self.model = keras.models.Model(inputs=base_model.input, outputs=out)
            
            # Create optimizer and add to model
            optimizer = keras.optimizers.SGD(
                lr=hyperparameters['learning_rate'], 
                momentum=hyperparameters['momentum'], 
                nesterov=True, 
                decay=hyperparameters['decay']
            )
            self.model.compile(
                loss="sparse_categorical_crossentropy", 
                optimizer=optimizer,
                metrics=["accuracy"]
            )