package server

import (
	"encoding/json"
	"fmt"
	"github.com/Godwinh19/federated-learning/aggregator"
	"github.com/Godwinh19/federated-learning/client"
	"github.com/Godwinh19/federated-learning/model"
	"github.com/gin-gonic/gin"
	"io"
	"log"
	"net/http"
	"net/url"
	"sync"
)

// Server represents the central server in the federated learning system
type Server struct {
	ID          string
	Clients     []client.Client
	GlobalModel model.Model
}

var server = Server{
	ID: "sid1",
	Clients: []client.Client{
		{ID: "client1", Model: &model.Model{Id: "mc1"}, Url: url.URL{Path: "http://localhost:8080"}},
		{ID: "client2", Model: &model.Model{Id: "mc2"}, Url: url.URL{Path: "http://localhost:8080"}},
		{ID: "client3", Model: &model.Model{Id: "mc3"}, Url: url.URL{Path: "http://localhost:8080"}},
	},
	GlobalModel: model.Model{
		Id: "globalModel"},
}

// Load train process
// if local client ready to send weights, receive it through an endpoint
// and then aggregate instantly

func getServer(c *gin.Context) {
	//get a client and load the train process. At the end,
	//send its params

	c.IndentedJSON(http.StatusOK, gin.H{
		"server info": server,
	})
}

// TrainClients We can train clients model concurrently
func (s *Server) TrainClients() <-chan client.Client {
	out := make(chan client.Client)
	var wg sync.WaitGroup
	for _, c := range s.Clients {
		wg.Add(1)
		go func(currentClient client.Client) {
			defer wg.Done()
			fmt.Println("Start training for client: ", currentClient.ID)
			endpoint, err := url.Parse(currentClient.Url.Path + "/train?clientId=" + currentClient.ID + "&modelId=" + currentClient.Model.Id)
			if err != nil {
				panic(err)
			}
			resp, err := http.Get(endpoint.String())
			body, err := io.ReadAll(resp.Body)
			if err != nil {
				panic(err)
			}
			var data map[string]interface{}
			err = json.Unmarshal(body, &data)
			// Read the response body into a byte array
			log.Printf("resp %v", data)
			//_client.(*client.Client)
			_ = currentClient.Train()
			fmt.Println("End training for client: ", currentClient.ID)
			//out <- _client.(client.Client)
			out <- currentClient
		}(c)
	}
	go func() {
		wg.Wait()
		close(out)
	}()
	return out
}

func startClientTraining(c *gin.Context) {
	//get a client and load the train process. At the end,
	//send its params

	clientsChan := server.TrainClients()
	idx := 0
	for c := range clientsChan {
		log.Printf("chan %v", c.ID)
		server.Clients[idx] = c
		idx++
	}

	server.GlobalModel = *server.AggregateModels()
	fmt.Println(server.GlobalModel)

	c.IndentedJSON(http.StatusOK, gin.H{
		"message": "Training finished",
	})
}

// AggregateModels aggregates the models received from the clients
// and returns a new model with the aggregated weights
func (s *Server) AggregateModels() *model.Model {
	var models []*model.Model
	for _, c := range s.Clients {
		models = append(models, c.Model)
	}
	aggregatedModel := aggregator.AggregateModels(models)
	return aggregatedModel
}

func Run() {
	r := gin.Default()
	r.GET("/", getServer)
	r.GET("/train", startClientTraining)
	r.GET("/info", getServer)

	r.Run("localhost:8081")
}
