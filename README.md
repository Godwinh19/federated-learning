# Federated Learning Library

This is a Golang library for implementing federated learning, a machine learning approach that allows multiple 
parties to collaboratively train a model without sharing their data.

## `@WIP`

## Usage

To use this library, you will need to import the packages for the main components of federated learning:

- `aggregator`: Package for implementing the algorithm for aggregating the model parameters received from the clients.
- `client`: Package for implementing the client-side logic of federated learning, including training the local model and sending the updates to the server.
- `dataset`: Package for loading and partitioning the dataset into subsets for each client.
- `model`: Package for defining the machine learning model architecture and operations, such as forward and backward propagation.
- `server`: Package for implementing the server-side logic of federated learning, including receiving the model updates from the clients and aggregating them.

You can then use these packages to implement your own federated learning system for your specific use case.

## License

This library is licensed under the MIT License. See the LICENSE file for more information.
