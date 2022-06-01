# requirements
+ tensorflow > 2.0.0
+ keras > 2.0.0
+ pandas
+ pymongo
+ mongodb

# Steps
1- install MongoDb and fill the connection ip:server in config.json at save_path -> monogo_connection.

2- set path of dataset in config.json and seperator word 

3- set input and output of model in config.json

4- set maximum of layer and max_neurons and some other model config in config.json

5- run the code

# results
while training all the results save on the ./tmp folder and you can use all weights and show thats chart using tensorboard

after complete the training step , the best model with minimum loss save on the ./main folder with all chart and can view the chart using tensorboard
