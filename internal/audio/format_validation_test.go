package audio

import (
	"testing"
)

func TestIsKnownAudioFormat_WAV(t *testing.T) {
	// RIFF header: "RIFF" + size + "WAVE"
	data := []byte("RIFF\x00\x00\x00\x00WAVEfmt ")
	if !IsKnownFormat(data) {
		t.Error("expected WAV (RIFF header) to be recognized")
	}
}

func TestIsKnownAudioFormat_MP3_ID3(t *testing.T) {
	// ID3v2 header
	data := []byte("ID3\x04\x00\x00\x00\x00\x00\x00")
	if !IsKnownFormat(data) {
		t.Error("expected MP3 (ID3 header) to be recognized")
	}
}

func TestIsKnownAudioFormat_FLAC(t *testing.T) {
	// fLaC magic bytes
	data := []byte("fLaC\x00\x00\x00\x22")
	if !IsKnownFormat(data) {
		t.Error("expected FLAC (fLaC header) to be recognized")
	}
}

func TestIsKnownAudioFormat_OGG(t *testing.T) {
	// OggS magic bytes
	data := []byte("OggS\x00\x02\x00\x00")
	if !IsKnownFormat(data) {
		t.Error("expected OGG (OggS header) to be recognized")
	}
}

func TestIsKnownAudioFormat_Unknown(t *testing.T) {
	// Random bytes that don't match any known format
	data := []byte{0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04}
	if IsKnownFormat(data) {
		t.Error("expected random bytes to be rejected")
	}
}

func TestIsKnownAudioFormat_TooShort(t *testing.T) {
	// 1-byte input is too short for any format signature
	data := []byte{0x42}
	if IsKnownFormat(data) {
		t.Error("expected 1-byte input to be rejected")
	}
}

func TestIsKnownAudioFormat_Empty(t *testing.T) {
	if IsKnownFormat([]byte{}) {
		t.Error("expected empty input to be rejected")
	}
}
