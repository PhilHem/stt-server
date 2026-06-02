package sherpa

import sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"

// Diarization tuning. The clustering threshold lets the speaker count be
// discovered automatically (vs. fixing NumClusters); 0.5 is sherpa's default
// for the campplus-style embedders. MinDuration* smooth over very short
// speech/silence blips.
const (
	diarClusterThreshold = 0.5
	diarMinDurationOn    = 0.3
	diarMinDurationOff   = 0.5
)

// newDiarizer builds an offline speaker-diarization pipeline (pyannote
// segmentation + speaker embeddings + clustering). Returns nil if sherpa-onnx
// cannot construct it. provider/threads mirror the recognizer.
func newDiarizer(segModel, embModel, provider string, numThreads int) *sherpa.OfflineSpeakerDiarization {
	cfg := &sherpa.OfflineSpeakerDiarizationConfig{
		Segmentation: sherpa.OfflineSpeakerSegmentationModelConfig{
			Pyannote:   sherpa.OfflineSpeakerSegmentationPyannoteModelConfig{Model: segModel},
			NumThreads: numThreads,
			Provider:   provider,
		},
		Embedding: sherpa.SpeakerEmbeddingExtractorConfig{
			Model:      embModel,
			NumThreads: numThreads,
			Provider:   provider,
		},
		Clustering:     sherpa.FastClusteringConfig{Threshold: diarClusterThreshold},
		MinDurationOn:  diarMinDurationOn,
		MinDurationOff: diarMinDurationOff,
	}
	return sherpa.NewOfflineSpeakerDiarization(cfg)
}

// diarize runs the pipeline and turns each speaker turn into one or more
// recognition segments: a turn longer than the recognizer's safe window is
// split (at quiet points) into sub-segments that all carry the turn's speaker.
// Segments come back in chronological order.
func diarize(d *sherpa.OfflineSpeakerDiarization, samples []float32, sampleRate int) []segment {
	turns := d.Process(samples) // sorted by start time; empty if no speech
	var segs []segment
	for _, t := range turns {
		start := int(t.Start * float32(sampleRate))
		end := int(t.End * float32(sampleRate))
		if start < 0 {
			start = 0
		}
		if end > len(samples) {
			end = len(samples)
		}
		if end <= start {
			continue
		}
		turn := samples[start:end]
		for _, w := range splitWindows(turn, sampleRate, chunkTargetSeconds, chunkSearchSeconds) {
			segs = append(segs, segment{
				start:   start + w.start,
				samples: turn[w.start:w.end],
				speaker: t.Speaker,
			})
		}
	}
	return segs
}
