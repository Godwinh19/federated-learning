package aggregator

import (
	"github.com/Godwinh19/federated-learning/model"
	"github.com/Godwinh19/gotorch/torch/tensor"
)

// convert slice to channel
func sliceToChannel(numb []float64) <-chan float64 {
	out := make(chan float64)
	go func() {
		for _, n := range numb {
			out <- n
		}
		close(out)
	}()
	return out

}

// AggregateModels calculates the average of the model weights
// received from the clients and returns a new model with these weights
func AggregateModels(models []*model.Model) *model.Model {
	// Initialize channel to receive parameters from each model
	W := make(chan tensor.Tensor, len(models))
	b := make(chan tensor.Tensor, len(models))

	// Collect parameters from each model in separate goroutine
	go func() {
		for _, m := range models {
			go func(mod *model.Model) {
				for _, v := range mod.Params {
					W <- v["w"]
					b <- v["b"]
					close(W)
					close(b)
				}
			}(m)
		}
	}()

	// Create goroutine to aggregate parameters
	done := make(chan bool)
	// Wait for aggregation to complete
	<-done

	return &model.Model{}
}
