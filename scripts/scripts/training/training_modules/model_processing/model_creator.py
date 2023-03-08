from termcolor import colored
from tensorflow import keras


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
    def __init__(self, hyperparameters, model_type, target_height, target_width, class_names):
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

        # Catch if the model is not in the model list
        if model_type not in model_list:
            print(colored(f"Warning: Model '{model_type}' not found in the list of possible models: {list(model_list.keys())}"))
            
            # TODO: Create a custom model

        self.model_type = model_type
            
        # Get the model base
        base_model = model_list[model_type](
            include_top=False,
            weights=None,
            input_tensor=None,
            input_shape=(target_height, target_width, hyperparameters['channels']),
            pooling=None
        )

        # Return the prepared model
        avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
        out = keras.layers.Dense(len(class_names), activation="softmax")(avg)
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
