package model

import (
	"encoding/json"
	"fmt"
	"github.com/Godwinh19/federated-learning/dataset"
	"github.com/Godwinh19/gotorch/torch/nn"
	t "github.com/Godwinh19/gotorch/torch/tensor"
	"os"
)

func Net(args ...t.Tensor) {
	var x, y t.Tensor
	switch len(args) {
	case 0:
		x, y = dataset.LoadIrisData("dataset/iris.csv")
	case 2:
		x, y = args[0], args[1]
	default:
		panic("Provide the inputs and labels")
	}

	a1 := nn.Activation{Name: "tanh"}
	linear_1 := nn.Linear{InputSize: x.Shape()[1], OutputSize: 5, Activation: a1}
	a2 := nn.Activation{Name: "relu"}
	linear_2 := nn.Linear{InputSize: 5, OutputSize: 1, Activation: a2}

	var output, grad t.Tensor
	var params interface{}
	var currentLoss float64
	net := nn.NeuralNet{NLinear: []*nn.Linear{&linear_1, &linear_2}}
	lr := 0.0001
	optim := nn.SGD{Lr: lr}
	scheduler := nn.StepLRScheduler(lr, 10, 0.5)
	loss := nn.MSELoss{Actual: y}

	for i := 0; i < 100; i++ {

		// Zero gradients for every batch!
		optim.ZeroGradients(net)

		// Make predictions for this batch
		output, params = net.Forward(x)

		// Compute the loss and its gradients
		loss.Predicted = output
		grad = nn.Gradient(loss)

		currentLoss = float64(nn.Loss(loss).Data[0][0])
		net.Backward(grad)

		// Adjust learning weights
		optim.Step(net)
		optim.Lr = scheduler.Next()

		if i%5 == 0 {
			fmt.Println(currentLoss)
		}
	}
	// saving the weights
	fmt.Printf("\nParams for layers %v\n", params)
	saveToJson(params)
}

func saveToJson(weights interface{}) {
	data, err := json.Marshal(weights)
	if err != nil {
		fmt.Println(err)
	}
	err = os.WriteFile("data.json", data, 0644)
}
