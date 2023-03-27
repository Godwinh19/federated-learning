package dataset

// DataSet represents a dataset of training examples
type DataSet struct {
	data []Example
}

// Example represents a single training example
type Example struct {
	Features []float64
	Label    int
}
