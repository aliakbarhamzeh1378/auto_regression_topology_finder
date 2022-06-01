import json
import os
import uuid
from typing import Dict

import numpy as np
import pandas
import pandas as pd
import pymongo
import tensorflow as tf
from bson import ObjectId
from keras import Sequential
from keras.layers import Normalization, Dense
from keras.models import load_model


class Logs:
    def __init__(self):
        pass

    def error(self, string):
        print("Error , ", string)
        exit(0)


class Dataset:
    def __init__(self, train="", outputs: str = "", inputs: str = "", seperator=",", divisions=None, test=None,
                 validation=None) -> None:
        if divisions is None:
            divisions = [60, 20, 20]

        self.__train_path = train
        self.__test_path = test
        self.__validation_path = validation
        self.__divisions = divisions
        self.__inputs = inputs
        self.__outputs = outputs
        self.__input_keys = []
        self.__output_keys = []
        self.__seperator = seperator
        self.__train_data = None
        self.__train_label = None
        self.__test_data = None
        self.__validation_data = None
        self.__validation_label = None
        self.__column_names = []

    def load(self):
        self.__initializer()

    def __file_loader(self, path: str):
        suffix = path.split(".")[-1]

        if suffix in ['csv', 'txt']:
            raw_data = pd.read_csv(path,
                                   na_values='?', comment='\t',
                                   sep=self.__seperator, skipinitialspace=True)
        elif suffix in ['xls', 'xlsx', 'xlsm', 'xlsb', 'odf', 'ods']:
            raw_data = pd.read_excel(path, sheet_name=1,
                                     na_values='?', comment='\t'
                                     )
        elif suffix == 'json':
            raw_data = pd.read_json(path,
                                    na_values='?', comment='\t',
                                    sep=self.__seperator, skipinitialspace=True)

        else:
            Logs().error("train file format not acceptable")
        return raw_data

    def __preprocess_data(self, data):
        data.isna().sum()
        print(data)
        for key in data.keys():
            if key not in self.__column_names:
                data.drop(key, inplace=True, axis=1)
        print(data)
        return data

    def __extract_label_data(self, data):
        features = data.copy()
        labels = pandas.DataFrame()
        for output in self.__output_keys:
            labels[output] = features.pop(output)
        return features, labels

    def __check_assertion(self):
        if self.__train_path == "" or None:
            Logs().error("train file not inserted")
        if (self.__test_path == "" or self.__test_path is None) or (
                self.__validation_path == "" or self.__validation_path is None):
            Logs().error("test and validation input incorrect")
        if self.__inputs == "" or self.__outputs == "":
            Logs().error("input output not inserted")

    def __parse_keys(self):

        self.__input_keys = self.__inputs.split(self.__seperator)
        self.__output_keys = self.__outputs.split(self.__seperator)
        self.__column_names = self.__input_keys + self.__output_keys

    def __initializer(self):
        # check all parameter is correct
        self.__check_assertion()
        self.__parse_keys()
        self.__train_data = self.__file_loader(self.__train_path)
        # prepare train data
        train_data = self.__file_loader(self.__train_path)
        train_data = self.__preprocess_data(train_data)
        self.__train_data, self.__train_label = self.__extract_label_data(train_data)

        test_data = self.__file_loader(self.__test_path)
        test_data = self.__preprocess_data(test_data)
        self.__test_data, _ = self.__extract_label_data(test_data)
        validation_data = self.__file_loader(self.__validation_path)
        validation_data = self.__preprocess_data(validation_data)
        self.__validation_data, self.__validation_label = self.__extract_label_data(validation_data)

    def get_data(self):
        return {
            'train_data': self.__train_data,
            'train_label': self.__train_label,
            'validation_data': self.__validation_data,
            'validation_label': self.__validation_label,
            'test_data': self.__test_data
        }

    def load_data_for_predict(self, file_path):
        self.__parse_keys()

        data = self.__file_loader(file_path)
        data = self.__preprocess_data(data)
        data, _ = self.__extract_label_data(data)
        labels = self.__output_keys
        return data, labels


class ConfigParser:
    def __init__(self, config_file_path: str) -> None:
        self.config_file = config_file_path
        # ___________ dataset ____________
        self.train_dataset_path = ""
        self.test_dataset_path = ""
        self.validation_dataset_path = ""
        self.dataset_split = []
        self.__outputs = ""
        self.__inputs = ""
        self.__seperator = ""
        # ___________ model ____________
        self.__max_layer = 3
        self.__max_neurons = 30
        self.__batch_size = 100
        self.__epoch = 100
        self.__lr = 0.001
        # ___________ database ____________

        self.__monogo_connection = ""
        self.__temp = ""
        self.__main = ""
        self.__project_name = ""

        self.__initial()

    def __initial(self):
        config_string = open(self.config_file)
        config = json.load(config_string)
        # ___________ dataset ____________

        self.train_dataset_path = str(config["dataset"]["train"])
        self.test_dataset_path = str(config["dataset"]["test"])
        self.validation_dataset_path = str(config["dataset"]["validation"])
        try:
            self.dataset_split = str(config["dataset"]["divisions"]).split(",")
        except Exception:
            self.dataset_split = [60, 20, 20]
        self.__outputs = str(config["dataset"]["outputs"])
        self.__inputs = str(config["dataset"]["inputs"])
        self.__seperator = str(config["dataset"]["seperator"])
        # ___________ model ____________
        self.__lr = float(config["model"]["lr"])
        self.__epoch = int(config["model"]["epoch"])
        self.__batch_size = int(config["model"]["batch_size"])
        self.__max_layer = int(config["model"]["max_layer"])
        self.__max_neurons = int(config["model"]["max_neurons"])
        # ___________ database _________
        self.__monogo_connection = config["save_path"]["monogo_connection"]
        self.__temp = config["save_path"]["temp"]
        self.__main = config["save_path"]["main"]
        self.__project_name = config["save_path"]["project_name"]

    def get_dataset_data(self) -> Dict[str, str]:
        return {
            'train': self.train_dataset_path, 'test': self.test_dataset_path,
            'validation': self.validation_dataset_path, 'inputs': self.__inputs, 'outputs': self.__outputs,
            'seperator': self.__seperator
        }

    def get_model_data(self) -> Dict:
        return {
            'lr': self.__lr, 'batch_size': self.__batch_size,
            'epoch': self.__epoch, 'max_layer': self.__max_layer, 'max_neurons': self.__max_neurons,
        }

    def get_database_info(self):
        return {
            "connection": self.__monogo_connection,
            "temp": self.__temp,
            "main": self.__main,
            "project": self.__project_name
        }


class DNN:
    def __init__(self, dataset, model, database):
        # ___________ dataset ____________

        self.__dataset = dataset
        self.__train_data = None
        self.__train_label = None
        self.__validation_data = None
        self.__validation_label = None
        self.__test_data = None
        self.__prepare_dataset()
        self.normalizer = None
        # ___________ model ____________
        self.__model = model

        self.__max_layer = None
        self.__max_neurons = None
        self.__batch_size = None
        self.__epoch = None
        self.__lr = None
        self.__prepare_model()
        self.__topologies = []

        # ___________ database  ____________
        self.connection = database['connection']
        self.client = None
        self.db = "reports"
        self.col = None
        self.project_name = database['project']

        # __________ storage _____________
        self.temp_path = database['temp']
        self.main_path = database['main']
        self.__connect_database()
        try:
            os.mkdir(self.temp_path)
        except:
            pass
        try:
            os.mkdir(self.main_path)
        except:
            pass
        pass

    def __connect_database(self):
        self.client = pymongo.MongoClient(self.connection)
        self.db = self.client[self.db]
        self.col = self.db[self.project_name]
        db2 = self.client['statistics']
        self.col2 = db2[self.project_name]

    def __normalize_data(self):
        self.normalizer = Normalization(axis=-1)
        self.normalizer.adapt(np.array(self.__train_data))

    def __model_builder(self, topology):
        model = Sequential()
        model.add(self.normalizer)
        for neuron in topology:
            model.add(Dense(neuron, activation='relu'))

        model.add(Dense(self.__train_label.shape[1], name="output"))
        return model

    def __prepare_dataset(self):
        self.__train_data = self.__dataset['train_data']
        self.__train_label = self.__dataset['train_label']
        self.__validation_data = self.__dataset['validation_data']
        self.__validation_label = self.__dataset['validation_label']
        self.__test_data = self.__dataset['test_data']

    def __train(self, topology, loss="mean_absolute_error"):
        self.__normalize_data()
        model = self.__model_builder(topology)
        model.compile(loss=loss,
                      optimizer=tf.keras.optimizers.Adam(0.01),
                      metrics=["mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error",
                               "cosine_proximity"])
        history = model.fit(
            self.__train_data,
            self.__train_label,
            batch_size=self.__batch_size,
            verbose=1, epochs=self.__epoch)
        self.__save_report(model, topology)

    def __get_accuracy_loss(self, model, topology):
        history2 = model.evaluate(self.__validation_data, self.__validation_label)
        print(topology, history2)

    def __save_report(self, model, topology):
        data = model.evaluate(self.__validation_data, self.__validation_label)
        uu = str(uuid.uuid4())
        name = self.temp_path + '/' + self.project_name + "_" + uu
        topology.insert(0, self.__train_data.shape[1])
        topology.insert(len(topology), self.__train_label.shape[1])
        metrics = {
            'loss': data[0],
            'mean_squared_error': data[1],
            'mean_absolute_error': data[2],
            'mean_absolute_percentage_error': data[3],
            'cosine_proximity': data[4],
            "path": name,
            'model_name': self.project_name + "_" + uu,
            'topology': topology

        }
        model.save(name)
        self.col.insert_one(metrics)
        pass

    def __get_minimum_item(self):
        item = self.col.find().sort('loss', 1).limit(1)[0]
        print(item['path'])
        model = load_model(item['path'])
        model.summary()
        os.system('mv ' + item['path'] + ' ' + self.main_path + '/' + item['model_name'])
        os.system('rm -r ' + self.temp_path)
        tmp_item = dict(item)
        tmp_item['path'] = self.main_path + '/' + item['model_name']
        self.col2.insert_one(tmp_item)
        self.col.drop()
        return str(item['_id'])

    def run(self):
        self.__loop_maker()
        print(len(self.__topologies), '  state to check')
        for topology in self.__topologies:
            self.__train(topology)
        return self.__get_minimum_item()

    def __loop_maker(self):
        max_number = self.__max_layer
        space = 1
        file_name = "compute_data.py"
        with open(file_name, 'w') as f:
            f.write("def compute_data(max_number):\n")
            f.write("    " * space + "numbers = [] \n")
            for i in range(max_number):
                f.write("    " * space + "for i" + str(i) + " in range(max_number):\n")
                space += 1
            conition = ""
            for i in range(max_number):
                conition += 'i' + str(i)
                if i != max_number - 1:
                    conition += " + "

            conition2 = ""
            for i in range(max_number):
                conition2 += 'i' + str(i) + " != 0 "
                if i != max_number - 1:
                    conition2 += " and "
            f.write(
                "    " * (max_number + 1) + "if " + conition + " == max_number and " + conition2 + " : \n")

            conition3 = ""
            for i in range(max_number):
                conition3 += 'i' + str(i)
                if i != max_number - 1:
                    conition3 += " , "
            f.write(
                "     " * (max_number + 2) + "numbers.append([" + conition3 + "]) \n")
            f.write("    return numbers")

        os.system('python3 ' + file_name)
        from compute_data import compute_data

        self.__topologies = compute_data(self.__max_neurons)

    def __prepare_model(self):
        self.__lr = self.__model['lr']
        self.__batch_size = self.__model['batch_size']
        self.__max_neurons = self.__model['max_neurons']
        self.__epoch = self.__model['epoch']
        self.__max_layer = self.__model['max_layer']
        pass


class Prediction:
    def __init__(self, database, model_id):
        # ___________ database  ____________
        self.connection = database['connection']
        self.client = None
        self.project_name = database['project']
        self.model_id = model_id
        self.model = None
        self.col2 = None
        self.__connect_database()
        pass

    def __connect_database(self):
        self.client = pymongo.MongoClient(self.connection)
        db2 = self.client['statistics']
        self.col2 = db2[self.project_name]

    def load_model(self):
        try:
            item = self.col2.find_one({'_id': ObjectId(self.model_id)})
        except:
            print("item not found")
            exit(0)
        self.model = load_model(item['path'])
        self.model.summary()

    def predict(self, items, labels, output):
        print(items)
        predictions = self.model.predict(items).flatten()
        num = len(predictions)
        predictions = np.asarray(predictions)
        predictions = np.split(predictions, num / len(labels))
        prediction_dataframe = pd.DataFrame(columns=labels)
        for andis, data in enumerate(predictions):
            prediction_dataframe.loc[andis] = data
        result = pd.concat([items, prediction_dataframe], axis=1)
        print(result)

        result.to_csv(output, index=True)


def train(config: str) -> str:
    config = ConfigParser(config)
    b = Dataset(**config.get_dataset_data())
    b.load()
    dnn = DNN(dataset=b.get_data(), model=config.get_model_data(), database=config.get_database_info())
    best_model_id = dnn.run()
    return best_model_id


def predict(config: str, model_id: str, output: str):
    config = ConfigParser(config)
    items, labels = Dataset(**config.get_dataset_data()).load_data_for_predict('dataset/auto-mpg2.txt')

    predict = Prediction(database=config.get_database_info(), model_id=model_id)
    predict.load_model()
    predict.predict(items, labels, output)


# 'validation_data': self.__validation_data,
# 'validation_label': self.__validation_label,
if __name__ == '__main__':
    model_id = train("config.json")
    predict("config.json", model_id=model_id, output="output.csv")
