package model

import (
	"os"
	"path/filepath"
	"testing"
)

func TestResolveModel_LocalPath(t *testing.T) {
	// Create a temp directory to act as a local model
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "tokens.txt"), []byte("test"), 0o644)

	result, err := Resolve(dir, "")
	if err != nil {
		t.Fatalf("Resolve(%s): %v", dir, err)
	}
	if result != dir {
		t.Errorf("expected %s, got %s", dir, result)
	}
}

func TestResolveModel_CachedModel(t *testing.T) {
	cacheDir := t.TempDir()
	modelName := "test-model"

	// Pre-create a cached model directory with tokens.txt and a valid ONNX file
	// that passes verifyModelFiles (non-empty tokens.txt, ONNX >= 1 MB with 0x08 header).
	modelDir := filepath.Join(cacheDir, modelName)
	os.MkdirAll(modelDir, 0o755)
	os.WriteFile(filepath.Join(modelDir, "tokens.txt"), []byte("test"), 0o644)

	// Create a fake ONNX file: starts with 0x08 (protobuf field 1 varint),
	// padded to minOnnxFileSize so it passes the size check.
	fakeOnnx := make([]byte, minOnnxFileSize)
	fakeOnnx[0] = 0x08
	fakeOnnx[1] = 0x07
	os.WriteFile(filepath.Join(modelDir, "model.onnx"), fakeOnnx, 0o644)

	result, err := Resolve(modelName, cacheDir)
	if err != nil {
		t.Fatalf("Resolve(%s): %v", modelName, err)
	}
	if result != modelDir {
		t.Errorf("expected %s, got %s", modelDir, result)
	}
}

func TestResolveModel_NonExistentModel(t *testing.T) {
	cacheDir := t.TempDir()

	// This should try to download and fail (no network in tests)
	_, err := Resolve("nonexistent-model-that-does-not-exist", cacheDir)
	if err == nil {
		t.Error("expected error for non-existent model")
	}
}
