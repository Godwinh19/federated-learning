package dataset

import (
	"fmt"
	"github.com/Godwinh19/gotorch/torch/tensor"
	"github.com/Godwinh19/gotorch/torch/utils"
	"path/filepath"
)

func LoadIrisData(filename string) (x, y tensor.Tensor) {
	var data, _ = filepath.Abs(filename)
	records := utils.ReadCsvFile(data)
	x, y, _ = utils.SplitXandY(records)
	fmt.Println("Shape of dataset: ", x.Shape(), y.Shape())
	return
}
