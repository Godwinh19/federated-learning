package main

import (
	"encoding/json"
	"github.com/Godwinh19/federated-learning/client"
	"github.com/Godwinh19/federated-learning/model"
	"log"
	"net/http"
)

func handleTrain(w http.ResponseWriter, r *http.Request) {
	// Parse request body to get the client's data
	var data []float64
	err := json.NewDecoder(r.Body).Decode(&data)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Train the model with the client's data
	clients := []client.Client{
		{ID: 1, Model: &model.Model{Id: 1}},
	}
	trainClients(clients)

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
