package client

import (
	"github.com/Godwinh19/federated-learning/model"
)

// Client represents a client participating in the federated learning system
type Client struct {
	ID    int
	Model *model.Model
	Loss  float64
}

// Train trains the client's model on the client's dataset
func (c *Client) Train() float64 {
	c.Model.Net()
	c.Evaluate()
	return c.Loss
}

// Evaluate evaluates the client's model on the client's dataset and
// calculates the loss of the model
func (c *Client) Evaluate() {
	c.Loss = c.Model.Loss
}
