package server

import (
	"federated-learning/aggregator"
	"federated-learning/client"
	"federated-learning/model"
)

// Server represents the central server in the federated learning system
type Server struct {
	ID         int
	Clients    []*client.Client
	GlobalData *client.Client
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
