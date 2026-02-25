# test_diffusion_fixes.py - Diagnostic script for llama-diffusion-cli

import subprocess
import os
import sys

LLAMA_DIFFUSION_CLI = r"D:\llama.cpp\build\bin\Release\llama-diffusion-cli.exe"
MODEL_PATH = r"D:\models\bartowski\Dream-org_Dream-v0-Instruct-7B-GGUF\Dream-org_Dream-v0-Instruct-7B-Q4_K_M.gguf"

def run_test(name, cmd, timeout=60):
    """Run a test case and report results"""
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd[:8])}...")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding='utf-8',
            errors='ignore'
        )
        
        print(f"Return code: {result.returncode}")
        
        if result.stdout:
            print(f"STDOUT ({len(result.stdout)} chars):")
            print(result.stdout[:500])
        else:
            print("STDOUT: (empty)")
            
        if result.stderr:
            print(f"STDERR ({len(result.stderr)} chars):")
            # Print last 1000 chars of stderr (most relevant)
            print(result.stderr[-1000:])
        else:
            print("STDERR: (empty)")
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT after {timeout}s")
        return False
    except Exception as e:
        print(f"EXCEPTION: {e}")
        return False

def main():
    # Create prompt file
    prompt = "User: Hi\nAssistant:"
    with open("test_prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)
    
    print("Starting diffusion CLI diagnostics...")
    print(f"Model: {MODEL_PATH}")
    print(f"CLI: {LLAMA_DIFFUSION_CLI}")
    
    # Test 1: Basic with file input (avoid -p escaping issues)
    run_test("File input, minimal args", [
        LLAMA_DIFFUSION_CLI,
        "-m", MODEL_PATH,
        "-f", "test_prompt.txt",
        "-n", "16",
        "--diffusion-steps", "16",
        "-ngl", "99",
    ])
    
    # Test 2: With --diffusion-visual (might be needed for output)
    run_test("With --diffusion-visual flag", [
        LLAMA_DIFFUSION_CLI,
        "-m", MODEL_PATH,
        "-f", "test_prompt.txt",
        "-n", "16",
        "--diffusion-steps", "16",
        "-ngl", "99",
        "--diffusion-visual",
    ])
    
    # Test 3: Without -n (let it use default)
    run_test("Without -n parameter", [
        LLAMA_DIFFUSION_CLI,
        "-m", MODEL_PATH,
        "-f", "test_prompt.txt",
        "--diffusion-steps", "16",
        "-ngl", "99",
    ])
    
    # Test 4: With -sys system prompt
    run_test("With system prompt", [
        LLAMA_DIFFUSION_CLI,
        "-m", MODEL_PATH,
        "-sys", "You are a helpful assistant.",
        "-f", "test_prompt.txt",
        "-n", "16",
        "--diffusion-steps", "16",
        "-ngl", "99",
    ])
    
    # Test 5: CPU only (isolate GPU issue)
    run_test("CPU only (-ngl 0)", [
        LLAMA_DIFFUSION_CLI,
        "-m", MODEL_PATH,
        "-f", "test_prompt.txt",
        "-n", "16",
        "--diffusion-steps", "16",
        "-ngl", "0",
    ])
    
    # Test 6: Check if we need specific algorithm
    run_test("With origin algorithm (0)", [
        LLAMA_DIFFUSION_CLI,
        "-m", MODEL_PATH,
        "-f", "test_prompt.txt",
        "-n", "16",
        "--diffusion-steps", "16",
        "--diffusion-algorithm", "0",
        "-ngl", "99",
    ])
    
    # Test 7: Very minimal - just prompt and model
    run_test("Absolute minimal", [
        LLAMA_DIFFUSION_CLI,
        "-m", MODEL_PATH,
        "-p", "Hello",
    ])
    
    # Cleanup
    if os.path.exists("test_prompt.txt"):
        os.remove("test_prompt.txt")
    
    print(f"\n{'='*70}")
    print("Diagnostics complete")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()