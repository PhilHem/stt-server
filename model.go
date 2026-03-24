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
	"regexp"
	"strings"
	"time"
)

const (
	// All sherpa-onnx ASR models follow this URL pattern.
	sherpaReleaseURL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/%s.tar.bz2"

	defaultCacheDir = "/opt/stt/cache"
	localCacheDir   = ".cache/stt-server"

	// maxExtractBytes is the total extraction size limit (2 GB).
	maxExtractBytes = 2 << 30
	// maxFileBytes is the per-file extraction size limit (1 GB).
	maxFileBytes = 1 << 30
	// minOnnxFileSize is the minimum acceptable size for ONNX model files.
	minOnnxFileSize = 1 << 20 // 1 MB
)

// validModelName matches only safe model names: must start with alphanumeric,
// then alphanumeric/dots/hyphens/underscores. Rejects ".", "..", and similar.
var validModelName = regexp.MustCompile(`^[a-zA-Z0-9][a-zA-Z0-9._-]*$`)

// resolveModel takes a --model value and returns the path to a local model directory.
// If the value is an existing directory, it's returned as-is.
// Otherwise, it's treated as a model name and downloaded to the cache directory.
func resolveModel(model, cacheDir string) (string, error) {
	// Already a local directory?
	if info, err := os.Stat(model); err == nil && info.IsDir() {
		return model, nil
	}

	// H5: Validate model name before using it in URL construction.
	if !validModelName.MatchString(model) {
		return "", fmt.Errorf("invalid model name %q: must contain only alphanumeric characters, dots, hyphens, and underscores", model)
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
		os.RemoveAll(modelDir) // clean up partial extraction
		return "", fmt.Errorf("download model %s: %w", model, err)
	}

	// M4: Verify extraction produced expected files with reasonable sizes.
	if err := verifyModelFiles(modelDir); err != nil {
		// Clean up partial/invalid download
		os.RemoveAll(modelDir)
		return "", fmt.Errorf("model verification failed for %s: %w", model, err)
	}

	slog.Info("model cached", "path", modelDir)
	return modelDir, nil
}

// downloadAndExtract fetches a .tar.bz2 URL and extracts it to destDir.
func downloadAndExtract(url, destDir string) error {
	// M6: Custom client with redirect limits, HTTPS enforcement, and timeout.
	client := &http.Client{
		Timeout: 10 * time.Minute,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= 5 {
				return fmt.Errorf("too many redirects")
			}
			if req.URL.Scheme != "https" {
				return fmt.Errorf("refusing redirect to non-HTTPS URL: %s", req.URL)
			}
			return nil
		},
	}

	resp, err := client.Get(url)
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

	// M3: Track total bytes extracted to prevent decompression bombs.
	var totalBytes int64

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
			// L4: Mask file mode to remove setuid/setgid/sticky bits.
			mode := os.FileMode(hdr.Mode) & 0o777
			f, err := os.OpenFile(target, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, mode)
			if err != nil {
				return err
			}
			// M3: Limit per-file size and track total extraction size.
			limited := &io.LimitedReader{R: tr, N: maxFileBytes}
			written, err := io.Copy(f, limited)
			f.Close()
			if err != nil {
				return err
			}
			if limited.N <= 0 {
				return fmt.Errorf("extraction size limit exceeded: file %s exceeds per-file limit", hdr.Name)
			}
			totalBytes += written
			if totalBytes > maxExtractBytes {
				return fmt.Errorf("extraction size limit exceeded")
			}
		default:
			// L5: Log skipped entry types (symlinks, hardlinks, etc.) for visibility.
			slog.Debug("skipping tar entry with unsupported type", "name", hdr.Name, "typeflag", hdr.Typeflag)
		}
	}

	return nil
}

// verifyModelFiles checks that a downloaded model directory contains expected files
// with reasonable sizes. This provides basic integrity verification (M4).
func verifyModelFiles(modelDir string) error {
	// tokens.txt must exist and be non-empty.
	tokensInfo, err := os.Stat(filepath.Join(modelDir, "tokens.txt"))
	if err != nil {
		return fmt.Errorf("tokens.txt not found in %s", modelDir)
	}
	if tokensInfo.Size() == 0 {
		return fmt.Errorf("tokens.txt is empty in %s", modelDir)
	}
	slog.Info("verified tokens.txt", "path", modelDir, "size_bytes", tokensInfo.Size())

	// Check for at least one ONNX model file with a reasonable size.
	// Models use either encoder/decoder/joiner pattern or a single model.onnx.
	onnxPatterns := []string{
		"encoder*.onnx",
		"decoder*.onnx",
		"joiner*.onnx",
		"model.onnx",
		"*.onnx",
	}

	var foundOnnx bool
	for _, pattern := range onnxPatterns {
		matches, err := filepath.Glob(filepath.Join(modelDir, pattern))
		if err != nil {
			continue
		}
		for _, match := range matches {
			info, err := os.Stat(match)
			if err != nil {
				continue
			}
			if info.Size() < minOnnxFileSize {
				return fmt.Errorf("ONNX file %s is suspiciously small (%d bytes, expected >= %d)", filepath.Base(match), info.Size(), minOnnxFileSize)
			}
			// Verify ONNX protobuf magic byte (field 1, varint = 0x08)
			if err := checkOnnxMagic(match); err != nil {
				return err
			}
			slog.Info("verified ONNX file", "path", match, "size_mb", info.Size()/(1024*1024))
			foundOnnx = true
		}
	}

	if !foundOnnx {
		return fmt.Errorf("no ONNX model files found in %s", modelDir)
	}

	return nil
}

// checkOnnxMagic verifies that a file starts with the ONNX protobuf header.
// ONNX uses protobuf: field 1 (ir_version) as varint → first byte is 0x08.
func checkOnnxMagic(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("cannot open %s: %w", filepath.Base(path), err)
	}
	defer f.Close()

	magic := make([]byte, 2)
	if _, err := io.ReadFull(f, magic); err != nil {
		return fmt.Errorf("cannot read %s header: %w", filepath.Base(path), err)
	}
	// ONNX protobuf: 0x08 = field 1 varint. Some models start with
	// 0x08 0x0N (ir_version N). Accept any value after 0x08.
	if magic[0] != 0x08 {
		return fmt.Errorf("ONNX file %s has invalid header (expected protobuf, got 0x%02x)", filepath.Base(path), magic[0])
	}
	return nil
}
