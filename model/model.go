package model

import "federated-learning/dataset"

//Typically a model is a collection of layers
// Each layer have its own parameters and gradients
type LayerType struct {
	Params []float64
}
type Model struct {
	name string
	Layer []LayerType
}

// initialize a new model
func (m *Model) New() *Model {
	return &Model{
		name: "new-version",
	}
}

// trains the model on a given dataset
func (m *Model) Train(dataSet *dataset.DataSet) {
	// Implement training algorithm for the model
}

// evaluates the model on a given dataset and returns the accuracy
func (m *Model) Evaluate(dataSet *dataset.DataSet) float64 {
	// Implement evaluation algorithm for the model
	return 0.0
}
