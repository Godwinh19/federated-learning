package aggregator

import (
	"github.com/Godwinh19/federated-learning/model"
	t "github.com/Godwinh19/gotorch/torch/tensor"
	"sync"
)

// AggregateModels calculates the average of the model weights
// received from the clients and returns a new model with these weights
func AggregateModels(models []*model.Model) *model.Model {
	var wg sync.WaitGroup
	var mutex sync.Mutex
	sumParams := &models[0].Params // initialize the params with the first model values
	count := float64(len(models))

	// Create a Goroutine for each model in the list
	for _, clientModel := range models {
		wg.Add(1)
		go func(clientModel *model.Model) {
			defer wg.Done()

			// Compute the sum of values for each key in the Params map
			for idx, param := range clientModel.Params {
				for key, value := range param {
					//log.Printf("idx, key, value %v %v %v", idx, key, value)
					mutex.Lock()
					//if _, ok := (*sumParams)[idx][key]; !ok {
					//	(*sumParams)[idx] = make(map[string]t.Tensor)
					//	(*sumParams)[idx][key] = value
					//	log.Printf("Actual sumparams %v %v", idx, key)
					//}
					(*sumParams)[idx][key] = add((*sumParams)[idx][key], value)
					mutex.Unlock()
				}
			}
		}(clientModel)
	}

	// Wait for all Goroutines to finish before dividing the sum by count
	wg.Wait()

	// Divide the sum of each key by the count to get the mean value
	for idx, param := range *sumParams {
		for key := range param {
			(*sumParams)[idx][key] = *t.TensorOpsScalar((*sumParams)[idx][key], count, "/")
		}
	}
	return &model.Model{Id: "AggregatedM", Params: *sumParams}
}

func add(a, b t.Tensor) t.Tensor {
	result := t.Zeros(a.Dim)

	for i, _ := range a.Data {
		result.Data[i] = a.Data[i] + b.Data[i]
	}

	return *result
}
