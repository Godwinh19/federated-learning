package server

import (
	"github.com/Godwinh19/federated-learning/aggregator"
	"github.com/Godwinh19/federated-learning/client"
	"github.com/Godwinh19/federated-learning/model"
	"github.com/gin-gonic/gin"
	"net/http"
)

// Server represents the central server in the federated learning system
type Server struct {
	ID          string
	Clients     []*client.Client
	GlobalModel *model.Model
}

var server = Server{
	ID: "sid1",
	Clients: []*client.Client{
		{ID: "client1", Model: &model.Model{Id: "mc1"}},
		{ID: "client2", Model: &model.Model{Id: "mc2"}},
		{ID: "client3", Model: &model.Model{Id: "mc3"}},
	},
	GlobalModel: &model.Model{
		Id: "globalModel"},
}

// Load train process
// if local client ready to send weights, receive it through an endpoint
// and then aggregate instantly

func getServer(c *gin.Context) {
	//get a client and load the train process. At the end,
	//send its params

	c.IndentedJSON(http.StatusOK, server)
}

func startClientTraining(c *gin.Context) {
	//get a client and load the train process. At the end,
	//send its params
	var currentClient client.Client
	if err := c.BindQuery(&currentClient); err != nil {
		return
	}

	c.IndentedJSON(http.StatusOK, currentClient.Train())
}

// AggregateModels aggregates the models received from the clients
// and returns a new model with the aggregated weights
func (s *Server) AggregateModels() *client.Client {
	var models []*model.Model
	for _, c := range s.Clients {
		models = append(models, c.Model)
	}
	aggregatedModel := aggregator.AggregateModels(models)
	return &client.Client{Model: aggregatedModel}
}

func Run() {
	r := gin.Default()
	r.GET("/", getServer)
	r.GET("/train", startClientTraining)
	r.GET("/info", getServer)

	r.Run("localhost:8080")
}
