package dataset

import (
	"encoding/csv"
	"os"
	"strconv"

	"gorgonia.org/tensor"
)

func LoadIrisDataset(filename string) (data, labels *tensor.Dense, err error) {
	// Load data from file
	f, err := os.Open(filename)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	rows, err := r.ReadAll()
	if err != nil {
		return nil, nil, err
	}

	// Parse data and labels
	numSamples := len(rows)
	data = tensor.New(tensor.WithShape(numSamples, 4), tensor.WithBacking(make([]float64, numSamples*4)))
	labels = tensor.New(tensor.WithShape(numSamples, 1), tensor.WithBacking(make([]int, numSamples)))

	for i, row := range rows {
		// Parse data
		for j := 0; j < 4; j++ {
			val, err := strconv.ParseFloat(row[j], 64)
			if err != nil {
				return nil, nil, err
			}
			data.SetAt(val, i, j)
		}

		// Parse label
		switch row[4] {
		case "Iris-setosa":
			labels.SetAt(0, i, 0)
		case "Iris-versicolor":
			labels.SetAt(1, i, 0)
		case "Iris-virginica":
			labels.SetAt(2, i, 0)
		}
	}

	return data, labels, nil
}
