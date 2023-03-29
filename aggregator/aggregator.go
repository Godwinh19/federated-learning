package aggregator

import "github.com/Godwinh19/federated-learning/model"

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
	paramsChan := make(chan []float64, len(models))

	// Collect parameters from each model in separate goroutine
	for _, m := range models {
		go func(mod *model.Model) {
			for _, layer := range mod.Layer {
				paramsChan <- layer.Params
			}
		}(m)
	}

	// Create goroutine to aggregate parameters
	done := make(chan bool)
	go func() {
		var sum []float64
		count := 0
		for params := range paramsChan {
			if sum == nil {
				sum = make([]float64, len(params))
			}
			for i, p := range params {
				sum[i] += p
			}
			count++
		}
		for i := range sum {
			sum[i] /= float64(count)
		}
		result := &model.Model{
			name:  "newVersion",
			Layer: []LayerType{Params: sum},
		}
		done <- true
	}()

	// Wait for aggregation to complete
	<-done

	return result
}
