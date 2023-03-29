package main

import (
	"encoding/json"
	"fmt"
	"github.com/Godwinh19/federated-learning/model"
	"log"
	"net/http"
)

func trainClients(clientDataChan chan []float64) {
	for clientData := range clientDataChan {
		go func(data []float64) {
			// train the client data here
			// ...
			fmt.Println("Trained client data:", data)
		}(clientData)
	}
}

func handleTrain(w http.ResponseWriter, r *http.Request) {
	// Parse request body to get the client's data
	var data []float64
	err := json.NewDecoder(r.Body).Decode(&data)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Train the model with the client's data
	model.Net()

	// Send a response back to the client
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("Model trained successfully"))
}

func serve() {
	// Create a new HTTP server
	server := http.NewServeMux()

	// Register the `handleTrain` function as an endpoint
	server.HandleFunc("/train", handleTrain)

	// Start the server
	log.Fatal(http.ListenAndServe(":8080", server))
}
