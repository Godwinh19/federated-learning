package dataset

import "testing"

func TestLoadIrisData(t *testing.T) {
	_, _ = LoadIrisData("./iris.csv")
}
