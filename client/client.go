package client

import (
	"github.com/Godwinh19/federated-learning/model"
	"github.com/gin-gonic/gin"
	"net/http"
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

func welcome(c *gin.Context) {
	//get a client and load the train process. At the end,
	//send its params

	c.IndentedJSON(http.StatusOK, gin.H{
		"client": "Client server",
	})
}

func trainHandler(c *gin.Context) {
	//name := c.Query("name")
	clientId := c.Query("clientId")
	modelId := c.Query("modelId")

	_client := Client{
		ID: clientId,
		Model: &model.Model{
			Id: modelId,
		},
	}
	_client.Train()
	_client.Url = url.URL{
		Scheme: "http",
		Host:   "localhost:8080",
		Path:   "/train",
	}

	c.Set(_client.ID, _client)

	c.IndentedJSON(http.StatusOK, gin.H{
		"_client": _client,
	})
}

func Run() {
	r := gin.Default()
	r.GET("/", welcome)
	r.GET("/train", trainHandler)

	r.Run("localhost:8080")
}
