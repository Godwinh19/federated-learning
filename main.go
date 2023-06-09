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

//func multipleTraining() {
//	// Train the model with the client's data
//	clients := []client.Client{
//		{ID: "client1", Model: &model.Model{Id: "mc1"}, Url: url.URL{Path: "http://localhost:8080"}},
//		{ID: "client2", Model: &model.Model{Id: "mc2"}, Url: url.URL{Path: "http://localhost:8080"}},
//		{ID: "client3", Model: &model.Model{Id: "mc3"}, Url: url.URL{Path: "http://localhost:8080"}},
//	}
//	s := server.Server{Clients: clients}
//	lossChan := s.TrainClients()
//
//	for l := range lossChan {
//		fmt.Println(l.Loss)
//	}
//}

func main() {
	go client.Run()
	go server.Run()
	var input string
	fmt.Scanln(&input)

}
