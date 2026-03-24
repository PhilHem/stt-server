package main

import (
	"archive/tar"
	"compress/bzip2"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

const (
	// All sherpa-onnx ASR models follow this URL pattern.
	sherpaReleaseURL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/%s.tar.bz2"

	defaultCacheDir = "/opt/stt/cache"
	localCacheDir   = ".cache/stt-server"
)

// resolveModel takes a --model value and returns the path to a local model directory.
// If the value is an existing directory, it's returned as-is.
// Otherwise, it's treated as a model name and downloaded to the cache directory.
func resolveModel(model, cacheDir string) (string, error) {
	// Already a local directory?
	if info, err := os.Stat(model); err == nil && info.IsDir() {
		return model, nil
	}

	// Treat as model name — resolve cache directory
	if cacheDir == "" {
		cacheDir = defaultCacheDir
		// Fall back to home directory if /opt/stt/cache is not writable
		if err := os.MkdirAll(cacheDir, 0o755); err != nil {
			home, _ := os.UserHomeDir()
			cacheDir = filepath.Join(home, localCacheDir)
		}
	}

	if err := os.MkdirAll(cacheDir, 0o755); err != nil {
		return "", fmt.Errorf("create cache dir: %w", err)
	}

	modelDir := filepath.Join(cacheDir, model)

	// Already cached?
	if _, err := os.Stat(filepath.Join(modelDir, "tokens.txt")); err == nil {
		slog.Info("using cached model", "path", modelDir)
		return modelDir, nil
	}

	// Download from sherpa-onnx GitHub releases
	url := fmt.Sprintf(sherpaReleaseURL, model)
	slog.Info("downloading model", "model", model, "url", url)

	if err := downloadAndExtract(url, cacheDir); err != nil {
		return "", fmt.Errorf("download model %s: %w", model, err)
	}

	// Verify extraction produced the expected directory
	if _, err := os.Stat(filepath.Join(modelDir, "tokens.txt")); err != nil {
		return "", fmt.Errorf("model downloaded but tokens.txt not found in %s", modelDir)
	}

	slog.Info("model cached", "path", modelDir)
	return modelDir, nil
}

// downloadAndExtract fetches a .tar.bz2 URL and extracts it to destDir.
func downloadAndExtract(url, destDir string) error {
	resp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("HTTP GET: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP %d from %s", resp.StatusCode, url)
	}

	if resp.ContentLength > 0 {
		slog.Info("download started", "size_mb", resp.ContentLength/(1024*1024))
	}

	return extractTarBz2(resp.Body, destDir)
}

// extractTarBz2 extracts a .tar.bz2 stream to destDir.
func extractTarBz2(r io.Reader, destDir string) error {
	bzr := bzip2.NewReader(r)
	tr := tar.NewReader(bzr)

	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("read tar: %w", err)
		}

		// Sanitize path to prevent directory traversal
		target := filepath.Join(destDir, filepath.Clean(hdr.Name))
		if !strings.HasPrefix(target, filepath.Clean(destDir)+string(os.PathSeparator)) {
			continue // skip entries that would escape destDir
		}

		switch hdr.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(target, 0o755); err != nil {
				return err
			}
		case tar.TypeReg:
			if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
				return err
			}
			f, err := os.OpenFile(target, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, os.FileMode(hdr.Mode))
			if err != nil {
				return err
			}
			if _, err := io.Copy(f, tr); err != nil {
				f.Close()
				return err
			}
			f.Close()
		}
	}

	return nil
}
