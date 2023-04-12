package aggregator

import (
	"github.com/Godwinh19/federated-learning/model"
	t "github.com/Godwinh19/gotorch/torch/tensor"
	"log"
	"sync"
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
	var wg sync.WaitGroup
	var mutex sync.Mutex
	sumParams := models[0].Params
	count := float64(len(models))
	//log.Printf("values %v", sumParams["1"]["b"])

	// Create a Goroutine for each model in the list
	for _, clientModel := range models {
		wg.Add(1)
		go func(m model.Model) {
			defer wg.Done()

			// Compute the sum of values for each key in the Params map
			for idx, param := range m.Params {
				for key, value := range param {
					log.Printf("key, value %v", param)
					mutex.Lock()
					//log.Printf("values %v %v", sumParams[idx][key], value)
					sumParams[idx][key] = t.TensorOpsTensor(value, value, "+")
					//log.Printf("The value of idx, key is %s %s %v", idx, key, sumParams[idx][key])
					mutex.Unlock()
				}
			}
		}(*clientModel)
	}

	// Wait for all Goroutines to finish before dividing the sum by count
	wg.Wait()

	// Divide the sum of each key by the count to get the mean value
	for idx, param := range sumParams {
		for key, _ := range param {
			sumParams[idx][key] = t.TensorOpsScalar(sumParams[idx][key], count, "/")
		}
	}
	log.Printf("The value of sumParams %v", sumParams)
	return &model.Model{Id: "AggregatedM", Params: sumParams}
}
