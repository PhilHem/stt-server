package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestResolveModel_LocalPath(t *testing.T) {
	// Create a temp directory to act as a local model
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "tokens.txt"), []byte("test"), 0o644)

	result, err := resolveModel(dir, "")
	if err != nil {
		t.Fatalf("resolveModel(%s): %v", dir, err)
	}
	if result != dir {
		t.Errorf("expected %s, got %s", dir, result)
	}
}

func TestResolveModel_CachedModel(t *testing.T) {
	cacheDir := t.TempDir()
	modelName := "test-model"

	// Pre-create a cached model directory
	modelDir := filepath.Join(cacheDir, modelName)
	os.MkdirAll(modelDir, 0o755)
	os.WriteFile(filepath.Join(modelDir, "tokens.txt"), []byte("test"), 0o644)

	result, err := resolveModel(modelName, cacheDir)
	if err != nil {
		t.Fatalf("resolveModel(%s): %v", modelName, err)
	}
	if result != modelDir {
		t.Errorf("expected %s, got %s", modelDir, result)
	}
}

func TestResolveModel_NonExistentModel(t *testing.T) {
	cacheDir := t.TempDir()

	// This should try to download and fail (no network in tests)
	_, err := resolveModel("nonexistent-model-that-does-not-exist", cacheDir)
	if err == nil {
		t.Error("expected error for non-existent model")
	}
}
