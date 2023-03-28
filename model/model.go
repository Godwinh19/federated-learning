package model

import (
	"encoding/json"
	"federated-learning/dataset"
	"fmt"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"log"
	"os"
)

type model struct {
	W1, B1, W2, B2 *tensor.Dense
}

func Net() {
	// Load Iris dataset
	data, labels, err := dataset.LoadIrisDataset("dataset/iris.data")
	if err != nil {
		log.Fatal(err)
	}

	// Define neural network architecture
	g := gorgonia.NewGraph()
	x := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(data.Shape()[0], 4), gorgonia.WithName("x"))
	y := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(data.Shape()[0], 1), gorgonia.WithName("y"))
	w1 := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(4, 8), gorgonia.WithName("w1"))
	//b1 := gorgonia.NewVector(g, tensor.Float64, gorgonia.WithShape(150, 8), gorgonia.WithName("b1"))
	w2 := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(8, 1), gorgonia.WithName("w2"))
	//b2 := gorgonia.NewVector(g, tensor.Float64, gorgonia.WithShape(1), gorgonia.WithName("b2"))

	// Define neural network operations
	hidden := gorgonia.Must(gorgonia.Mul(x, w1))
	//hidden = gorgonia.Must(gorgonia.Add(hidden, b1))
	hidden = gorgonia.Must(gorgonia.Rectify(hidden))
	output := gorgonia.Must(gorgonia.Mul(hidden, w2))
	//output = gorgonia.Must(gorgonia.Add(output, b2))

	// Define loss function
	cost := gorgonia.Must(gorgonia.Mean(gorgonia.Must(gorgonia.Square(gorgonia.Must(gorgonia.Sub(output, y))))))

	// Define gradients
	grads, err := gorgonia.Grad(cost, w1, w2)
	if err != nil {
		log.Fatal(err)
	}

	// Define VM
	machine := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(w1, w2))

	// Train neural network
	solver := gorgonia.NewAdamSolver(gorgonia.WithLearnRate(0.1))
	epochs := 20
	for i := 0; i < epochs; i++ {
		numExamples := data.Shape()[0]
		for i := 0; i < numExamples; i++ {
			xT, err := data.Slice(gorgonia.S(i, i+1), gorgonia.S(0, 4))
			yT, err := labels.Slice(gorgonia.S(i, i+1), gorgonia.S(0, 1))
			_ = gorgonia.Let(x, xT)
			_ = gorgonia.Let(y, yT)
			_, err = gorgonia.Grad(cost, x, y)
			if err != nil {
				fmt.Println(err)
			}
			if err := machine.RunAll(); err != nil {
				fmt.Println(err)
			}
			if err := solver.Step(gorgonia.NodesToValueGrads(grads)); err != nil {
				fmt.Println(err)
			}
			machine.Reset()
		}
		gorgonia.Let(x, data)
		gorgonia.Let(y, labels)
		if err := machine.RunAll(); err != nil {
			fmt.Println(err)
		}
		costVal := cost.Value()
		fmt.Printf("Epoch: %d, Cost: %f\n", i+1, costVal)
	}
}

func saveModel(mod *model, filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	enc := json.NewEncoder(file)
	if err := enc.Encode(mod); err != nil {
		return err
	}
	return nil
}
