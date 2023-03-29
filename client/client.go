package client

import (
	"github.com/Godwinh19/federated-learning/dataset"
	"github.com/Godwinh19/federated-learning/model"
)

// Client represents a client participating in the federated learning system
type Client struct {
	ID       int
	DataSet  *dataset.DataSet
	Model    *model.Model
	Accuracy float64
}

// Train trains the client's model on the client's dataset
func (c *Client) Train() {
	c.Model.Train(c.DataSet)
}

// Evaluate evaluates the client's model on the client's dataset and
// calculates the accuracy of the model
func (c *Client) Evaluate() {
	c.Accuracy = c.Model.Evaluate(c.DataSet)
}
