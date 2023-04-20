package client

import (
	"github.com/Godwinh19/federated-learning/model"
	"net/url"
)

// Client represents a client participating in the federated learning system
type Client struct {
	ID    string       `json:"id"`
	Model *model.Model `json:"model"`
	Loss  float64      `json:"loss"`
	Url   url.URL      `json:"url"`
}

// Train trains the client's model on the client's dataset
func (c *Client) Train() *Client {
	c.Model.Net()
	c.Evaluate()
	return c
}

// Evaluate evaluates the client's model on the client's dataset and
// calculates the loss of the model
func (c *Client) Evaluate() {
	c.Loss = c.Model.Loss
}
