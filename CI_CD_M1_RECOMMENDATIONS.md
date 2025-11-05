# CI/CD Recommendations for M1 Testing

**Version**: 4.1.2
**Date**: 2025-11-05
**Status**: Recommended for Production CI/CD

---

## Overview

This document provides recommendations for setting up continuous integration and continuous deployment (CI/CD) testing on Apple Silicon (M1/M2/M3) runners to ensure compatibility and performance optimization.

---

## GitHub Actions M1 Support

### Current Status

GitHub Actions now provides M1 (ARM64) macOS runners:
- **Runner label**: `macos-14` (M1 Mac)
- **Architecture**: ARM64 (Apple Silicon)
- **macOS Version**: 14.x (Sonoma) or later
- **Availability**: Generally available for GitHub Team and Enterprise plans

### Recommended Workflow

```yaml
name: M1 Compatibility Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test-m1:
    name: Test on Apple Silicon M1
    runs-on: macos-14  # M1 runner

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Verify ARM64 architecture
        run: |
          python3 -c "import platform; print(f'Machine: {platform.machine()}, System: {platform.system()}')"
          if [ "$(python3 -c 'import platform; print(platform.machine())')" != "arm64" ]; then
            echo "ERROR: Not running on ARM64"
            exit 1
          fi

      - name: Install Homebrew dependencies
        run: |
          # Homebrew should be pre-installed on GitHub M1 runners
          brew --version

      - name: Install Miniconda (for FAISS)
        uses: conda-incubator/setup-miniconda@v3
        with:
          miniconda-version: "latest"
          python-version: '3.11'
          activate-environment: rag_env

      - name: Install dependencies via conda
        shell: bash -l {0}
        run: |
          conda activate rag_env
          conda install -c conda-forge faiss-cpu=1.8.0 numpy requests
          conda install -c pytorch sentence-transformers pytorch
          pip install urllib3==2.2.3 rank-bm25==0.2.2

      - name: Verify dependencies
        shell: bash -l {0}
        run: |
          conda activate rag_env
          python3 -c "import numpy, requests, sentence_transformers, torch, rank_bm25, faiss; print('✅ All dependencies OK')"

      - name: Verify PyTorch MPS
        shell: bash -l {0}
        run: |
          conda activate rag_env
          python3 -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

      - name: Run M1 compatibility tests
        shell: bash -l {0}
        run: |
          conda activate rag_env
          bash scripts/m1_compatibility_test.sh

      - name: Run acceptance tests
        shell: bash -l {0}
        run: |
          conda activate rag_env
          bash scripts/acceptance_test.sh

      - name: Run smoke tests
        shell: bash -l {0}
        run: |
          conda activate rag_env
          bash scripts/smoke.sh

      - name: Upload test logs
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: m1-test-logs
          path: |
            *.log
            benchmark_results.csv

  test-intel:
    name: Test on Intel Mac
    runs-on: macos-13  # Intel runner

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Verify x86_64 architecture
        run: |
          python3 -c "import platform; print(f'Machine: {platform.machine()}, System: {platform.system()}')"
          if [ "$(python3 -c 'import platform; print(platform.machine())')" != "x86_64" ]; then
            echo "WARNING: Not running on x86_64"
          fi

      - name: Install dependencies via pip
        run: |
          python3 -m venv rag_env
          source rag_env/bin/activate
          pip install -r requirements.txt

      - name: Run acceptance tests
        run: |
          source rag_env/bin/activate
          bash scripts/acceptance_test.sh

      - name: Run smoke tests
        run: |
          source rag_env/bin/activate
          bash scripts/smoke.sh

  compare-performance:
    name: Compare M1 vs Intel Performance
    needs: [test-m1, test-intel]
    runs-on: ubuntu-latest

    steps:
      - name: Download M1 artifacts
        uses: actions/download-artifact@v4
        with:
          name: m1-test-logs
          path: m1-results

      - name: Generate performance comparison
        run: |
          echo "# M1 vs Intel Performance Comparison" > comparison.md
          echo "" >> comparison.md

          if [ -f "m1-results/benchmark_results.csv" ]; then
            echo "## Benchmark Results" >> comparison.md
            cat m1-results/benchmark_results.csv >> comparison.md
          fi

          echo "" >> comparison.md
          echo "See uploaded artifacts for full logs." >> comparison.md

      - name: Upload comparison report
        uses: actions/upload-artifact@v4
        with:
          name: performance-comparison
          path: comparison.md
```

---

## Alternative: Self-Hosted M1 Runners

If GitHub Actions M1 runners are not available or too expensive, use self-hosted runners.

### Setup Self-Hosted M1 Runner

1. **Prepare M1 Mac**:
   ```bash
   # Install Homebrew
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

   # Install Python
   brew install python@3.11

   # Install Miniforge (conda for ARM)
   brew install miniforge
   conda init
   ```

2. **Register Runner**:
   - Go to GitHub repo Settings → Actions → Runners
   - Click "New self-hosted runner"
   - Select macOS and ARM64
   - Follow installation instructions

3. **Configure Runner**:
   ```bash
   # Download and extract runner
   mkdir actions-runner && cd actions-runner
   curl -o actions-runner-osx-arm64-2.311.0.tar.gz -L \
     https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-osx-arm64-2.311.0.tar.gz
   tar xzf ./actions-runner-osx-arm64-2.311.0.tar.gz

   # Configure
   ./config.sh --url https://github.com/YOUR_ORG/YOUR_REPO --token YOUR_TOKEN

   # Install as service
   ./svc.sh install
   ./svc.sh start
   ```

4. **Label Runner**: Add label `m1` or `apple-silicon` to the runner

5. **Use in Workflow**:
   ```yaml
   jobs:
     test-m1:
       runs-on: [self-hosted, macos, m1]
       steps:
         # ... same as above
   ```

---

## GitLab CI/CD

### Using GitLab Runners on M1

```yaml
# .gitlab-ci.yml
stages:
  - test
  - benchmark

test-m1:
  stage: test
  tags:
    - macos
    - arm64
  script:
    - python3 -c "import platform; print(f'Machine: {platform.machine()}')"
    - conda create -n rag_env python=3.11 -y
    - conda activate rag_env
    - conda install -c conda-forge faiss-cpu=1.8.0 numpy requests -y
    - conda install -c pytorch sentence-transformers pytorch -y
    - pip install urllib3==2.2.3 rank-bm25==0.2.2
    - bash scripts/m1_compatibility_test.sh
    - bash scripts/acceptance_test.sh
  artifacts:
    paths:
      - "*.log"
      - "benchmark_results.csv"
    when: always

test-intel:
  stage: test
  tags:
    - macos
    - x86_64
  script:
    - python3 -c "import platform; print(f'Machine: {platform.machine()}')"
    - python3 -m venv rag_env
    - source rag_env/bin/activate
    - pip install -r requirements.txt
    - bash scripts/acceptance_test.sh
  artifacts:
    paths:
      - "*.log"
    when: always

benchmark-m1:
  stage: benchmark
  tags:
    - macos
    - arm64
  script:
    - conda activate rag_env
    - bash scripts/benchmark.sh
  artifacts:
    paths:
      - "benchmark_*.log"
      - "benchmark_results.csv"
  only:
    - main
    - develop
```

---

## CircleCI

### M1 Support

CircleCI offers M1 runners with the `macos.m1.medium` resource class.

```yaml
# .circleci/config.yml
version: 2.1

jobs:
  test-m1:
    macos:
      xcode: "15.0.0"
    resource_class: macos.m1.medium.gen1

    steps:
      - checkout

      - run:
          name: Verify ARM64
          command: |
            python3 -c "import platform; print(f'Machine: {platform.machine()}')"

      - run:
          name: Install dependencies
          command: |
            brew install miniforge
            conda create -n rag_env python=3.11 -y
            conda activate rag_env
            conda install -c conda-forge faiss-cpu=1.8.0 numpy requests -y
            conda install -c pytorch sentence-transformers pytorch -y
            pip install urllib3==2.2.3 rank-bm25==0.2.2

      - run:
          name: Run M1 tests
          command: |
            conda activate rag_env
            bash scripts/m1_compatibility_test.sh

      - store_artifacts:
          path: m1_compatibility.log

workflows:
  test-all:
    jobs:
      - test-m1
```

---

## Testing Strategy

### 1. Platform Matrix Testing

Test on multiple platforms in parallel:
- **M1 Mac** (ARM64): Primary target for optimization
- **Intel Mac** (x86_64): Backward compatibility
- **Linux** (x86_64): Production server environment

### 2. Test Coverage

**Minimum Tests** (run on every PR):
- `scripts/acceptance_test.sh` - Validates core functionality
- Platform detection verification
- Dependency import tests

**Extended Tests** (run on main branch):
- `scripts/smoke.sh` - Full build and query tests
- `scripts/m1_compatibility_test.sh` - M1-specific tests
- `scripts/benchmark.sh` - Performance benchmarks

### 3. Performance Regression Detection

Track performance metrics over time:
```yaml
- name: Check performance regression
  run: |
    # Extract build time from benchmark
    BUILD_TIME=$(grep "Build time:" benchmark_*.log | awk '{print $3}' | sed 's/s//')

    # Compare with baseline (stored in repo)
    BASELINE=30  # seconds for M1
    if [ "$BUILD_TIME" -gt "$((BASELINE + 5))" ]; then
      echo "WARNING: Build time regression detected: ${BUILD_TIME}s > ${BASELINE}s"
      exit 1
    fi
```

### 4. Artifact Comparison

Compare artifacts between platforms to ensure consistency:
```yaml
- name: Compare chunk counts
  run: |
    M1_CHUNKS=$(wc -l < m1-results/chunks.jsonl)
    INTEL_CHUNKS=$(wc -l < intel-results/chunks.jsonl)

    if [ "$M1_CHUNKS" != "$INTEL_CHUNKS" ]; then
      echo "ERROR: Chunk count mismatch: M1=$M1_CHUNKS, Intel=$INTEL_CHUNKS"
      exit 1
    fi
```

---

## Best Practices

### 1. Cache Dependencies

Cache conda/pip dependencies to speed up builds:
```yaml
- name: Cache conda packages
  uses: actions/cache@v4
  with:
    path: ~/miniconda3/pkgs
    key: ${{ runner.os }}-${{ runner.arch }}-conda-${{ hashFiles('requirements-m1.txt') }}
    restore-keys: |
      ${{ runner.os }}-${{ runner.arch }}-conda-
```

### 2. Fail Fast

Use fail-fast strategy for quick feedback:
```yaml
strategy:
  fail-fast: true
  matrix:
    os: [macos-14, macos-13, ubuntu-latest]
```

### 3. Parallel Execution

Run independent tests in parallel:
- M1 tests
- Intel tests
- Linux tests

### 4. Conditional Runs

Skip expensive tests on draft PRs:
```yaml
if: github.event.pull_request.draft == false
```

---

## Cost Optimization

### GitHub Actions Pricing

- **M1 runners** (macos-14): $0.16/minute
- **Intel runners** (macos-13): $0.08/minute
- **Ubuntu runners**: $0.008/minute

### Optimization Strategies

1. **Run M1 tests only on main/release branches**:
   ```yaml
   if: github.ref == 'refs/heads/main' || startsWith(github.ref, 'refs/tags/')
   ```

2. **Use caching aggressively** to reduce build times

3. **Run full benchmarks only weekly**:
   ```yaml
   on:
     schedule:
       - cron: '0 0 * * 0'  # Weekly on Sunday
   ```

4. **Use self-hosted runners** for frequent testing (no per-minute cost)

---

## Monitoring and Alerts

### Set up Slack/Email notifications

```yaml
- name: Notify on failure
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    text: 'M1 compatibility tests failed on ${{ github.ref }}'
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

### Track metrics over time

- Store benchmark results as artifacts
- Use GitHub API to track trends
- Set up dashboards (e.g., Grafana) for performance monitoring

---

## Troubleshooting CI/CD Issues

### Issue: FAISS fails to install on M1 runner

**Solution**:
```yaml
- name: Install FAISS with retry
  shell: bash -l {0}
  run: |
    conda activate rag_env
    conda install -c conda-forge faiss-cpu=1.8.0 -y || \
    conda install -c conda-forge faiss-cpu=1.8.0 --force-reinstall -y
```

### Issue: PyTorch MPS not available

**Solution**:
```yaml
- name: Verify macOS version
  run: |
    sw_vers
    # Ensure macOS 12.3+ for MPS support
```

### Issue: Platform detection fails

**Solution**:
```yaml
- name: Debug platform info
  run: |
    python3 -c "
    import platform
    print(f'System: {platform.system()}')
    print(f'Machine: {platform.machine()}')
    print(f'Processor: {platform.processor()}')
    print(f'Platform: {platform.platform()}')
    "
```

---

## Example Complete Workflow

See `.github/workflows/m1-ci.yml` (to be created) for a complete, production-ready workflow.

---

## Additional Resources

- **GitHub Actions M1 Runners**: https://github.blog/2024-01-30-github-actions-introducing-the-new-m1-macos-runner/
- **CircleCI M1 Support**: https://circleci.com/docs/using-macos/#apple-silicon-support
- **Self-Hosted Runners**: https://docs.github.com/en/actions/hosting-your-own-runners

---

**Maintained by**: Clockify RAG Team
**Last Updated**: 2025-11-05
**Version**: 4.1.2
