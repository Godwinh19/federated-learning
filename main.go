package main

import (
	"fmt"
	"github.com/Godwinh19/federated-learning/client"
	"github.com/Godwinh19/federated-learning/model"
	"sync"
)

func oneModel() {
	m := model.Model{Id: 1}
	m.Net()
	for k, v := range m.Params {
		fmt.Printf("Key: %s\n", k)
		fmt.Printf("b: %v\n", v["b"])
		fmt.Printf("w: %v\n", v["w"])
	}

}

// We can train clients model concurrently
func trainClients(clientDataChan []client.Client) <-chan float64 {
	out := make(chan float64)
	var wg sync.WaitGroup
	for _, c := range clientDataChan {
		wg.Add(1)
		go func(currentClient client.Client) {
			defer wg.Done()
			fmt.Println("Start training for client: ", currentClient.ID)
			loss := currentClient.Train()
			fmt.Println("End training for client: ", currentClient.ID)
			out <- loss
		}(c)
	}
	go func() {
		wg.Wait()
		close(out)
	}()
	return out
}

func main() {
	// Train the model with the client's data
	clients := []client.Client{
		{ID: 1, Model: &model.Model{Id: 1}},
		{ID: 2, Model: &model.Model{Id: 2}},
		{ID: 3, Model: &model.Model{Id: 3}},
	}
	lossChan := trainClients(clients)
	for l := range lossChan {
		fmt.Println(l)
	}
}
