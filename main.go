package main

import (
	"fmt"
	"github.com/Godwinh19/federated-learning/client"
	"github.com/Godwinh19/federated-learning/model"
	"github.com/Godwinh19/federated-learning/server"
)

func oneModel() {
	m := model.Model{Id: "c1"}
	m.Net()
	for k, v := range m.Params {
		fmt.Printf("Key: %s\n", k)
		fmt.Printf("b: %v\n", v["b"])
		fmt.Printf("w: %v\n", v["w"])
	}

}

func multipleTraining() {
	// Train the model with the client's data
	clients := []client.Client{
		{ID: "client1", Model: &model.Model{Id: "mc1"}},
		{ID: "client2", Model: &model.Model{Id: "mc2"}},
		{ID: "client3", Model: &model.Model{Id: "mc3"}},
	}
	lossChan := server.TrainClients(clients)

	for l := range lossChan {
		fmt.Println(l)
	}
}

func main() {
	server.Run()
}
