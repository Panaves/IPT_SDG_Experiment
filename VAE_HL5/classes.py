import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import pickle
import numpy as np
import pandas as pd
import time


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.base import BaseEstimator, TransformerMixin
from functools import partial

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras import applications
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import backend as K

class xplor:
    def __init__(self, df):
        self.df = df.copy()
        self.numerics = self.df.select_dtypes(exclude = ['object','category']).columns
        self.objects  = self.df.select_dtypes(include = ['object','category']).columns
        self.nulls = None
        self.unique_objs = None
        self.dates = None
        self.variance = None
        
    def check_nulls(self, level = None, select = False):
        
        nulls = (self.df.isnull().sum()/self.df.shape[0]*100).round(2)
        if level != None:
            if type(level) in [np.int, np.float]:
                null_mask = nulls > level
                print("level selected:", level,".", null_mask.sum() ," Variables indentified")
                if select:
                    self.nulls = list(nulls[null_mask].index)
                plot_line = True
        else:
            null_mask = nulls >= 0
            plot_line = False
            print("No level selected, taking 0 as default.", null_mask.sum() ," Variables indentified")
            if select:
                self.nulls = list(nulls[null_mask].index)
        
        if len( self.nulls ) > 0:
            fig, ax = plt.subplots(1,1, figsize = (20,4))
            null_mask = nulls > 0
            nulls[null_mask].sort_values(ascending = False).plot(kind = 'bar', ax = ax, label = '% Nulos')
            if plot_line:
                ax.axhline(y=level, color = 'r', linestyle = '--',label = 'level = '+str(level))

            ax.legend()
            ax.set_ylabel("% de nulos", fontdict = {'size':15})
            ax.set_xlabel("Variáveis", fontdict = {'size':15})
            ax.set_title('Gráfico de Pareto para valores nulos - '+str(((null_mask.sum()/null_mask.shape)*100).round(2)[0])+'% de vars com %nulos>0', fontdict = {'size':15})
            ax.tick_params(axis='x', labelsize=12)
        else:
            self.nulls = []
            
        
        
    def check_unique_objects(self, level_unique = None, select = False, plot_line = True):
        
        obj_desc = self.df[self.objects].describe()
        
        if (level_unique != None) and (type(level_unique) == int):

            obj_delete = list( set( obj_desc.loc[:, obj_desc.loc['unique',:] > level_unique].columns.to_list() ))
            obj_retain = obj_desc.loc[:, obj_desc.loc['unique',:] <= level_unique].columns

            print("Variables with number of unique values greater than ", level_unique)
            print((obj_desc.loc[:, obj_desc.loc['unique',:] > level_unique]).columns)#obj_desc.loc[:, obj_desc.loc['unique',:] > level_unique]

            if len(list((obj_desc.loc[:, obj_desc.loc['unique',:] > level_unique]).columns)) > 0:
                fig, ax = plt.subplots(1,1, figsize = (20,4))
                obj_desc.loc['unique',:].sort_values(ascending = False)[:20].plot(kind = 'bar', ax = ax, label = 'Quantidades')
                if plot_line:
                    ax.axhline(y=level_unique, color = 'r', linestyle = '--',label = 'level = '+str(level_unique))

                ax.legend()
                ax.set_ylabel("# Valores Únicos", fontdict = {'size':15})
                ax.set_xlabel("Variáveis", fontdict = {'size':15})
                ax.set_title('Gráfico de Pareto para valores unicos', fontdict = {'size':15})
                ax.tick_params(axis='x', labelsize=12)


                if select:
                    self.unique_objs = list((obj_desc.loc[:, obj_desc.loc['unique',:] > level_unique]).columns)
            else:
                self.unique_objs = []
                
        else:
            assert False, "Put a valid number of level_unique. It neds to be a int"
        
    def check_dates(self, select = False):
        
        dates = []
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                try:
                    pd.to_datetime(self.df[col])
                    dates.append(col)
                except ValueError:
                    pass
        print(len(dates), " date variables found:", dates)
        print(self.df[dates].head())
        if len(dates) > 0:
            if select:
                self.dates = dates
        else:
            self.dates = []
               
    def check_var(self, low = 0, high = 0.3, plot_number = 10, select = False, plot_line = True):
        mms = MinMaxScaler()

        num_desc = pd.DataFrame(mms.fit_transform(self.df[self.numerics]), columns = self.numerics).describe()

        fig, ax = plt.subplots(2,1, figsize = (20,12))

        num_desc.loc['std',:].sort_values(ascending = False)[:plot_number].plot(kind = 'bar', ax = ax[0])
        ax[0].set_ylabel("Variância Normalizada", fontdict = {'size':15})
        ax[0].set_xlabel("Variáveis", fontdict = {'size':15})
        ax[0].set_title('Variáveis com maior variância normalizada', fontdict = {'size':15})
        ax[0].tick_params(axis='x', labelsize=12, labelrotation = 45)

        num_desc.loc['std',:].sort_values(ascending = True)[:plot_number].plot(kind = 'bar', ax = ax[1])
        ax[1].set_ylabel("Variância Normalizada", fontdict = {'size':15})
        ax[1].set_xlabel("Variáveis", fontdict = {'size':15})
        ax[1].set_title('Variáveis com menor variância normalizada', fontdict = {'size':15})
        ax[1].tick_params(axis='x', labelsize=12, labelrotation = 45)
        
        if plot_line:
            ax[0].axhline(y=high, color = 'r', linestyle = '--',label = 'High Var = '+str(high))
            ax[1].axhline(y=low, color = 'r', linestyle = '--',label = 'Low Var = '+str(low))
        
        fig.tight_layout()
        low_var  = num_desc.loc[:,num_desc.loc['std',:] == 0  ].columns
        high_var = num_desc.loc[:,num_desc.loc['std',:]  > 0.3].columns
        print("low_vars:" , low_var)
        print("high_vars:", high_var)
        
        if len(list(low_var)) + len(list(high_var)) > 0:
            if select:
                self.variance = list(low_var) + list(high_var)
        else:
            self.variance = []
            
    def clean_data(self):
        exclusion = []
        
        print("Data Cleansing")
        
        if len(self.nulls) > 0 :
            print("\n - Columns with lots of nulls values")
            print("   - ",self.nulls)
            exclusion += self.nulls
            
        if len(self.unique_objs) > 0 :
            print("\n - Object columns with lots of unique values:")
            print("   - ",self.unique_objs)
            exclusion += self.unique_objs
            
        if len(self.dates) > 0 :
            print("\n - Date columns")
            print("   - ",self.dates)
            exclusion += self.dates
            
        if len(self.variance) > 0 :
            print("\n - Columns with low and high variance")
            print("   - ",self.variance)
            exclusion += self.variance

        print("\nTotal excluded:",len(set(exclusion)))
        self.df.drop(list(set(exclusion)), axis = 1, inplace = True)
        return self.df

class LowerCase(BaseEstimator, TransformerMixin):
    """
    This class will transform any string variable to lower case
    Should be used only if you want to standardize the string variables. Don't use if you don't need.
    
    We can have this type of parameter description:
    
    :param client: A handle to the :class:`simpleble.SimpleBleClient` client
        object that detected the device
    :type client: class:`simpleble.SimpleBleClient`
    
    or we can use a different approach like this:
    
    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.

    Returns:
        bool: The return value. True for success, False otherwise.
    
    .. code-block:: python
    
        lc = LowerCase()
        lc.fit(X)
        X_lc = lc.transform(X.copy())
     """

    def __init__(self):
        pass

    def fit(self, X, Y=None):
        """
        The fit process wont change anything in your data. It will learn the name of all string variables
        Input:
            - X (pandas DataFrame):
                X needs to be a pandas dataframe.
                
        Output:
            - The same pandas Dataframe of the input
        """
        print("\n - LowerCase:\n   -> Find String Variables")
        self.except_numeric_cols = X.select_dtypes(
            include=["object", "category"]
        ).columns
        return self

    def transform(self, X, Y=None):
        """
        The transform process will take all the string variables and will change their format to lower case.
        Input:
            - X (pandas DataFrame):
                X needs to be a pandas dataframe.
                
        Output:
            - A pandas Dataframe where all string variables are in lower case format.
        """
        print("\n - LowerCase:\n   -> Transforming String Variables")
        X[self.except_numeric_cols] = X[self.except_numeric_cols].apply(
            lambda x: x.str.lower()
        )
        return X

    def inverse_transform(self, X, Y=None):
        """
        The inverse Transform there is no use for this class. 
        Input:
            - X (pandas DataFrame):
                X needs to be a pandas dataframe.
          
        Output:
            - The same pandas Dataframe of the input
        """
        return X
    
    
class MissingValue(BaseEstimator, TransformerMixin):
    """
    This class will replace any numerical missing with a single numeric value.
    Inputs:
        - value (int, float):
            Value tha will be used to replace missing values in all numeric values. Must be an integer or float.
           
    """
    def __init__(self, num_value='mean', value_num = None,  obj_value = 'NULL'):
        self.num_value = num_value
        self.value_num = value_num
        self.obj_value = obj_value
        

    def fit(self, X, Y=None):
        """
        The fit process wont change anything in your data. It will learn the name of all numeric variables
        Input:
            - X (pandas DataFrame):
                X needs to be a pandas dataframe.
                
        Output:
            - The same pandas Dataframe of the input
        """
        print("\n - Missing:\n   -> Find Numerical and Object Variables")
        self.numeric_cols = X.select_dtypes(exclude=["object", "category"]).columns
        self.objects_cols = X.select_dtypes(include=["object", "category"]).columns
        return self

    def transform(self, X, Y=None):
        """
        The transform process will take all the numeric variables and will change the missing values.
        Input:
            - X (pandas DataFrame):
                X needs to be a pandas dataframe.
                
        Output:
            - A pandas Dataframe where all numerical missing are equal to the "value" class input.
        """

        print(
            "\n - Missing:\n   -> Inputting missing values"
            )
        if self.num_value == 'mean':
            X[self.numeric_cols] = X[self.numeric_cols].fillna(X[self.numeric_cols].mean())
        elif self.num_value == 'value':
            X[self.numeric_cols] = X[self.numeric_cols].fillna(self.value_num)
        else:
            assert False,'Missing numeric method is not recognized'
            
        X[self.objects_cols] = X[self.objects_cols].fillna(self.obj_value)
        
        return X

    def inverse_transform(self, X, Y=None):
        """
        The inverse Transform there is no use for this class. 
        Input:
            - X (pandas DataFrame):
                X needs to be a pandas dataframe.
          
        Output:
            - The same pandas Dataframe of the input
        """
        return X



class Outliers(BaseEstimator, TransformerMixin):
    def __init__(self,level = 3, train = True):
        self.level = level
        self.train = train
        
    def fit(self, X, y = None):
        print('- Outlier: Fitting process')
        self.numerics = X.select_dtypes(exclude = ['object','category']).columns
        describe = X[self.numerics].describe()

        mask_out = describe.loc['25%',:] < describe.loc['75%',:]

        self.lower_bound = describe.loc['25%',mask_out] - self.level*(describe.loc['25%',mask_out] - describe.loc['75%',mask_out]).abs() 
        self.upper_bound = describe.loc['75%',mask_out] + self.level*(describe.loc['25%',mask_out] - describe.loc['75%',mask_out]).abs()
        
        return self
        
    def transform(self, X, y = None, test = False):
        if self.train:
            print('- Outlier: Transforming process')
            mask_out = ( (X[self.numerics] > self.upper_bound) |  (X[self.numerics] < self.lower_bound) ).max(axis = 1)
            return X.drop(mask_out.index[mask_out], axis = 0)
        else:
            return X
    
    def inverse_transform(self, X, y = None):
        return X
    
# column transform pipeline
class CharEncoder(BaseEstimator, TransformerMixin):  # (base and mix):
    def __init__(self, methods="onehot"):
        if methods == "onehot":
            self.enc = OneHotEncoder(handle_unknown="ignore", dtype=np.int16)
        else:
            assert False, "Choose a valid method"
        self.methods = methods

    def fit(self, X, Y=None):
        print("\n - Fitting Encoder")
        self.original_columns = X.columns.values
        self.except_numeric_cols2 = X.select_dtypes(
            include=["object", "category"]
        ).columns
        print(
            "  - {} string columns found!".format(len(self.except_numeric_cols2))
        )
        print("  - {}".format(self.except_numeric_cols2.values))

        self.enc.fit(X.loc[:, list(self.except_numeric_cols2.values)])
        print('1')
        return self

    def transform(self, X, Y=None):
        print('2')
        transformed = self.enc.transform(X[list(self.except_numeric_cols2.values)]).toarray()

        print("   -> Applying Encoder to dataframe")
        # Creating new columns
        feature_names = self.enc.get_feature_names()
        temp_df = pd.DataFrame(  transformed
                               , columns = feature_names 
                               , index = X.index)

        X[feature_names] = temp_df[feature_names]
        print("   -> Deleting original variables")
        X = X.drop(self.except_numeric_cols2, axis=1)
        return X

    def inverse_transform(self, X, Y=None):
        X[self.enc.get_feature_names()] = np.round(X[self.enc.get_feature_names()])
        X[self.except_numeric_cols2] = pd.DataFrame(
            self.enc.inverse_transform(X[self.enc.get_feature_names()]),
            columns=self.except_numeric_cols2.values,
        )
        #        X[self.except_numeric_cols2] = pd.DataFrame(self.enc.inverse_transform(X[self.enc.get_feature_names()].apply(lambda x: np.round(x,0))), columns = self.except_numeric_cols2.values)
        X.drop(list(self.enc.get_feature_names()), axis=1, inplace=True)
        return X[self.original_columns]


class ScalingTreatmeant(BaseEstimator, TransformerMixin):  # (base and mix):
    def __init__(self, methods2="minmax"):
        if methods2 == "minmax":
            self.scaler = MinMaxScaler()
        else:
            assert False, "Choose a valid method"
        self.methods2 = methods2

    def fit(self, X, Y=None):
        print("\n - Fitting Scaling Method")
        self.original_columns = X.columns
        print('1')
        self.scaler.fit(X)
        print('2')
        return self

    def transform(self, X, Y=None):
        print("   -> Applying Scaler to Dataframe")
        x_scaled = self.scaler.transform(X)
        return pd.DataFrame(x_scaled, columns=X.columns)

    def inverse_transform(self, X, Y=None):
        print(X.shape)
        print(len(self.original_columns))
        return pd.DataFrame(
            self.scaler.inverse_transform(X), columns=self.original_columns
        )



class Autoencoder(BaseEstimator, TransformerMixin):
    """
        Create a object model using the keras library.
        
        - layers_ (list format):
            list containing numbers between 0 and 1 where each position corresponds a different encoder's hidden layer size. That way, the inverse order of this list represents the decoder's hidden layer size.
            The number will represent a percentage of the total number of columns in the training data. 
            Example: Using the default parameter and considering your training data has 100 columns you are going to have an autoencoder with the folowwing hidden layers size: 100 -> 80 -> 30 -> 80 -> 100
            
        - activations (list format):
            list containing strings representing activations functions recognized by keras where each position corresponds a different encoder's hidden layer size. That way, the inverse order of this list represents the decoder's hidden layer size.
            Example: Using the default parameter and considering your training data has 100 columns you are going to have an autoencoder with the folowing hidden layers setup:
                encoder:
                    - input 100 nodes
                    - First Hidden Layer created using ReLU with 80 Nodes
                    - Center Hidden Layer created using ReLU with 30 Nodes
                    
                decoder:
                    - input 30 nodes
                    - First Hidden Layer created using ReLU with 80 nodes
                    - Output Layer created using ReLU with 100 nodes.
                    
        - final_act (string) optional:
            Can be used to change the activation function of the decoder's output layer. If None, it will use the first activation function from 'activations' parameter. 
            If it's a valid keras activation function it will replace the first activation function from 'activations' parameter in decoder's output layer .
            
        - loss (string):
            The loss function of the model. it can be any valid keras loss function.
            
        - optimizer (string):
            The optimizer of the model. it can be any valid keras optimizer.
            Current version doesn't allow optimizer customisation.
            
        Length of 'layers_' and 'activations' need to be the same and need to be at least length = 1.
        For both 'layers_' and 'activations', the original list will beused to create the encoder setup. To create the decoder model, the function will use the inverse order of those lists.
       
       """

    def __init__(
        self,
        layers_=[0.8, 0.3],
        activations=["relu", "relu"],
        final_act=None,
        loss="mean_squared_error",
        optimizer="adam",
        bias = True,
        learning_rate = 0.01
    ):
        self.layers_ = layers_
        self.activations = activations
        self.final_act = final_act
        self.loss = loss
        self.optimizer = optimizer
        self.bias = bias
        self.lr = learning_rate

    def fit(
        self,
        X,
        Y=None,
        epochs=100,
        batch_size=200,
        shuffle=True,
        verbose=0,
        save=0,
        proj=0,
    ):
        """
        Will train a given model. You can choose the data to train/test and some others parameters of the keras 'fit' method.
        
        - train (pandas DataFrame):
            Data that will be used to train the model
       
        - test (pandas DataFrame):
            Data that will be used to validate the model
            
        - epochs (integer):
            Number of epochs to train the neural network. 1 Epoch means the entire data will go through the model.
            
        - batch_size (integer):
            Size of the bacth that will be used during the model training.
            
        - shuffle (bool):
            If True, will randomly shuffle the training data during the 'fit' process.
            
        - verbose (bool):
            if 0, won't output anything during training. If 1, will output real-time results of each epoch.
        """
#        print("Your data has a shape of {}".format(str(X.shape)))
        if (len(self.layers_) == len(self.activations)) and (len(self.layers_) > 0):
            base_dim = X.shape[1]
        if base_dim == None:
#            logger.exception("Input dim cannot be None")
            assert False, "Insert validate data"
        if any(layer < 1 for layer in self.layers_):
            self.layers_ = [int(layer * base_dim) for layer in self.layers_]
        elif all(layer > 1 for layer in self.layers_):
            pass

#        print("\n - Model Setup")
        aux = 0
        for layer, activation in zip(self.layers_, self.activations):
            if aux == 0:
                input_layer = Input(shape=(base_dim,))
                encoded = Dense(layer, activation=activation, use_bias = self.bias)(input_layer)
            else:
                encoded = Dense(layer, activation=activation, use_bias = self.bias )(encoded)
            aux += 1
        self.encoder = Model(inputs=input_layer, outputs=encoded)

        aux = 0
        for layer, activation in zip(
            reversed(self.layers_), reversed(self.activations)
        ):

            if aux == 0:
                input_decoder = Input(shape=(layer,))
                aux_l = layer
            elif aux == 1:
                decoded = Dense(
                    layer, input_dim=aux_l, activation=activation, use_bias = self.bias
                )(input_decoder)
                aux_l = layer
            elif aux < len(self.layers_):
                decoded = Dense(
                    layer, input_dim=aux_l, activation=activation, use_bias = self.bias
                )(decoded)
                aux_l = layer
            aux += 1
        if self.final_act == None:
            decode_output = Dense(base_dim, activation=activation, use_bias = self.bias)(decoded)
        else:
            decode_output = Dense(base_dim, activation=self.final_act, use_bias = self.bias)(
                decoded
            )
        self.decoder = Model(inputs=input_decoder, outputs=decode_output)

        outputs = self.decoder(self.encoder(input_layer))
        self.autoencoder = Model(inputs=input_layer, outputs=outputs)
        
        opt = tf.keras.optimizers.Adam(learning_rate=self.lr)
        
        self.autoencoder.compile(
            optimizer=opt, loss=self.loss, metrics=["mean_squared_error"] #accuracy
        )

        train, test = train_test_split(X, test_size=0.3, random_state=101)
#        print("\n - Starting Model Training")
        start = time.time()
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=int(epochs*0.05))
        self.autoencoder.fit(
            train,
            train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            validation_data=(test, test),
            verbose=verbose, callbacks=[callback],
        )
#       print(
#            "   -> Model training took {} seconds to train".format(
#                np.round(time.time() - start, 4)
#            ),
#        )

        return self

    def transform(self, X, Y=None):
#        print(" - Model Prediction")
        return (
            pd.DataFrame(
                self.decoder.predict(self.encoder.predict(X)), columns=X.columns
            ),
            pd.DataFrame(self.encoder.predict(X)),
            X,
        )

    def inverse_transform(self, X, Y=None):
        return X
    

def sampling(args):
    '''
    Draws samples from a standard normal and scales the samples with
    standard deviation of the variational distribution and shifts them
    by the mean.

    Args:
        args: sufficient statistics of the variational distribution.

    Returns:
        Samples from the variational distribution.
    '''
    t_mean, t_log_var = args
    t_sigma = K.sqrt(K.exp(t_log_var))
    epsilon = K.random_normal(shape=K.shape(t_mean), mean=0., stddev=1.)
    return t_mean + t_sigma * epsilon
    
class VariationalAutoencoder(BaseEstimator, TransformerMixin):
    """
        Create a object model using the keras library.
        
        - layers_ (list format):
            list containing numbers between 0 and 1 where each position corresponds a different encoder's hidden layer size. That way, the inverse order of this list represents the decoder's hidden layer size.
            The number will represent a percentage of the total number of columns in the training data. 
            Example: Using the default parameter and considering your training data has 100 columns you are going to have an autoencoder with the folowwing hidden layers size: 100 -> 80 -> 30 -> 80 -> 100
            
        - activations (list format):
            list containing strings representing activations functions recognized by keras where each position corresponds a different encoder's hidden layer size. That way, the inverse order of this list represents the decoder's hidden layer size.
            Example: Using the default parameter and considering your training data has 100 columns you are going to have an autoencoder with the folowing hidden layers setup:
                encoder:
                    - input 100 nodes
                    - First Hidden Layer created using ReLU with 80 Nodes
                    - Center Hidden Layer created using ReLU with 30 Nodes
                    
                decoder:
                    - input 30 nodes
                    - First Hidden Layer created using ReLU with 80 nodes
                    - Output Layer created using ReLU with 100 nodes.
                    
        - final_act (string) optional:
            Can be used to change the activation function of the decoder's output layer. If None, it will use the first activation function from 'activations' parameter. 
            If it's a valid keras activation function it will replace the first activation function from 'activations' parameter in decoder's output layer .
            
        - loss (string):
            The loss function of the model. it can be any valid keras loss function.
            
        - optimizer (string):
            The optimizer of the model. it can be any valid keras optimizer.
            Current version doesn't allow optimizer customisation.
            
        Length of 'layers_' and 'activations' need to be the same and need to be at least length = 1.
        For both 'layers_' and 'activations', the original list will beused to create the encoder setup. To create the decoder model, the function will use the inverse order of those lists.
       
       """

    def __init__(
        self,
        layers_=[0.8, 0.3],
        activations=["relu", "relu"],
        final_act=None,
        loss="mean_squared_error",
        optimizer="adam",
        bias = True,
        learning_rate = 0.01
    ):
        self.layers_ = layers_
        self.activations = activations
        self.final_act = final_act
        self.loss = loss
        self.optimizer = optimizer
        self.bias = bias
        self.lr = learning_rate

    def fit(
        self,
        X,
        Y=None,
        epochs=100,
        batch_size=200,
        shuffle=True,
        verbose=0,
        save=0,
        proj=0,
    ):
        """
        Will train a given model. You can choose the data to train/test and some others parameters of the keras 'fit' method.
        
        - train (pandas DataFrame):
            Data that will be used to train the model
       
        - test (pandas DataFrame):
            Data that will be used to validate the model
            
        - epochs (integer):
            Number of epochs to train the neural network. 1 Epoch means the entire data will go through the model.
            
        - batch_size (integer):
            Size of the bacth that will be used during the model training.
            
        - shuffle (bool):
            If True, will randomly shuffle the training data during the 'fit' process.
            
        - verbose (bool):
            if 0, won't output anything during training. If 1, will output real-time results of each epoch.
        """
#        print("Your data has a shape of {}".format(str(X.shape)))
        if (len(self.layers_) == len(self.activations)) and (len(self.layers_) > 0):
            base_dim = X.shape[1]
        if base_dim == None:
#            logger.exception("Input dim cannot be None")
            assert False, "Insert validate data"
        if any(layer < 1 for layer in self.layers_):
            self.layers_ = [int(layer * base_dim) for layer in self.layers_]
        elif all(layer > 1 for layer in self.layers_):
            pass

#        print("\n - Model Setup")
        aux = 0
        for layer, activation in zip(self.layers_, self.activations):
            if aux == 0:
                input_layer = Input(shape=(base_dim,))
                encoded = Dense(layer, activation=activation, use_bias = self.bias)(input_layer)
            elif aux + 1 < len(self.layers_):
                encoded = Dense(layer, activation=activation, use_bias = self.bias )(encoded)
            else:
                t_mean = Dense(layer, activation=activation, use_bias = self.bias   )(encoded)
                t_log_var = Dense(layer, activation=activation, use_bias = self.bias  )(encoded)
                z = Lambda(sampling, output_shape=(layer,), name='z')([t_mean, t_log_var])
            aux += 1
            
        self.encoder = Model(inputs=input_layer, outputs=[t_mean, t_log_var,z])      
        
        aux = 0
        for layer, activation in zip(
            reversed(self.layers_), reversed(self.activations)
        ):

            if aux == 0:
                input_decoder = Input(shape=(layer,))
                aux_l = layer
            elif aux == 1:
                decoded = Dense(
                    layer, input_dim=aux_l, activation=activation, use_bias = self.bias
                )(input_decoder)
                aux_l = layer
            elif aux < len(self.layers_):
                decoded = Dense(
                    layer, input_dim=aux_l, activation=activation, use_bias = self.bias
                )(decoded)
                aux_l = layer
            aux += 1
        if self.final_act == None:
            decode_output = Dense(base_dim, activation=activation, use_bias = self.bias)(decoded)
        else:
            decode_output = Dense(base_dim, activation=self.final_act, use_bias = self.bias)(
                decoded
            )
        self.decoder = Model(inputs=input_decoder, outputs=decode_output)
       
        outputs = self.decoder(self.encoder(input_layer)[2])
        
        self.autoencoder = Model(inputs=input_layer, outputs=outputs)
        
        opt = tf.keras.optimizers.Adam(learning_rate=self.lr)
        
        #@tf.function
        #def vae_loss(input_layer, outputs):
        #    reconstruction_loss = K.sum(K.binary_crossentropy(input_layer,outputs),axis=-1)
        #    return loss
        
        self.autoencoder.compile(
            optimizer=opt, loss=self.loss, metrics=["mean_squared_error"] #accuracy
        )

        train, test = train_test_split(X, test_size=0.3, random_state=101)
#        print("\n - Starting Model Training")
        start = time.time()
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=int(epochs*0.05))
        self.autoencoder.fit(
            train,
            train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            validation_data=(test, test),
            verbose=verbose, callbacks=[callback],
        )
#       print(
#            "   -> Model training took {} seconds to train".format(
#                np.round(time.time() - start, 4)
#            ),
#        )

        return self

    def transform(self, X, Y=None):
#        print(" - Model Prediction")
        return (
            pd.DataFrame(
                self.decoder.predict(self.encoder.predict(X)[2]), columns=X.columns
            ),
            pd.DataFrame(self.encoder.predict(X)[2]),
            X,
        )

    def inverse_transform(self, X, Y=None):
        return X
 

def compare_metric(df_real,dfs_synth, metric = 'wasserstein'):
    from scipy.stats import ks_2samp,wasserstein_distance
    W_df = pd.DataFrame([])
    for i,df in enumerate(dfs_synth):
        print(i)
        df_synth = df.copy()
        df_ = df_real.copy()
        W_List = {}
        for column in df_.columns:
            print(column)
            W_List[column] = []

            if (df_[column].dtype == 'object') | pd.CategoricalDtype.is_dtype(df_[column]):

                if metric == 'wasserstein':
                    le = LabelEncoder()
                    le.fit(df_[column])

                    mms = MinMaxScaler()
                    mms.fit(le.transform(df_[column]).reshape(-1,1))
                    try:
                        W_List[column] = wasserstein_distance( np.squeeze(mms.transform(le.transform(df_synth[column].dropna()).reshape(-1,1)))
                                                           ,np.squeeze(mms.transform(le.transform(df_      [column]         ).reshape(-1,1)))
                                                          )
                                
                    except:
                        print('except:', column)
                        W_List[column] = [None]

            else:           
                if metric == 'wasserstein':
                    try:
                        mms = MinMaxScaler()
                        mms.fit(df_[[column]])

                        W_List[column] = wasserstein_distance( np.squeeze(mms.transform(df_synth[[column]]))
                                                           ,np.squeeze(mms.transform(df_[[column]]))
                                                          )

                    except:
                        print('except:', column)
                        W_List[column] = [None]
        W_df[['Data_'+str(i)]] = pd.DataFrame(W_List, index = ['Data_'+str(i)]).T
    return W_df

