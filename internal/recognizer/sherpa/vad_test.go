package sherpa

import (
	"reflect"
	"testing"
)

func TestBatchBounds(t *testing.T) {
	cases := []struct {
		total, max int
		want       [][2]int
	}{
		{0, 16, nil},
		{1, 16, [][2]int{{0, 1}}},
		{16, 16, [][2]int{{0, 16}}},
		{17, 16, [][2]int{{0, 16}, {16, 17}}},
		{40, 16, [][2]int{{0, 16}, {16, 32}, {32, 40}}},
		{5, 0, [][2]int{{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}}}, // max<1 clamps to 1
	}
	for _, c := range cases {
		got := batchBounds(c.total, c.max)
		if !reflect.DeepEqual(got, c.want) {
			t.Errorf("batchBounds(%d, %d) = %v, want %v", c.total, c.max, got, c.want)
		}
	}
}

// batchBounds must cover every item exactly once, in order.
func TestBatchBounds_CoversAll(t *testing.T) {
	got := batchBounds(100, 16)
	prev := 0
	for _, b := range got {
		if b[0] != prev {
			t.Fatalf("gap or overlap: expected start %d, got %d", prev, b[0])
		}
		if b[1] <= b[0] || b[1]-b[0] > 16 {
			t.Fatalf("invalid batch %v", b)
		}
		prev = b[1]
	}
	if prev != 100 {
		t.Fatalf("did not cover all items: ended at %d", prev)
	}
}
