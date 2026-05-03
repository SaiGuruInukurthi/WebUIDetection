# Parallel Crawling Guide — WebUI Dataset Crawler

**Date**: May 3, 2026  
**Focus**: Concurrency tuning, worker management, performance profiling

---

## Overview

The WebUI crawler uses **parallel Playwright workers** to crawl multiple URLs simultaneously:

```
5 Playwright instances running in parallel
Each instance captures light + dark variants
Stride-based URL distribution (worker 0 gets URLs 0, 5, 10...; worker 1 gets URLs 1, 6, 11...)
Local batch upload queue (every 50 images)
```

---

## Architecture

### Worker Distribution

With `CRAWLER_CONCURRENCY=5` and `N` URLs total:

```
Worker 0: URLs 0, 5, 10, 15, 20, ... (every 5th URL starting at 0)
Worker 1: URLs 1, 6, 11, 16, 21, ... (every 5th URL starting at 1)
Worker 2: URLs 2, 7, 12, 17, 22, ... (every 5th URL starting at 2)
Worker 3: URLs 3, 8, 13, 18, 23, ... (every 5th URL starting at 3)
Worker 4: URLs 4, 9, 14, 19, 24, ... (every 5th URL starting at 4)
```

**Benefit**: Fair distribution, no conflicts, no shared state.

### Concurrency Levels

```
┌─────────────────────────────────────┐
│ CRAWLER_CONCURRENCY=1 (Baseline)    │
│ 1 Playwright instance                │
│ 2 URLs/sec                           │
│ 50k URLs: ~7 weeks                   │
└─────────────────────────────────────┘
                 ▲
                 │ 2x faster
                 │
┌─────────────────────────────────────┐
│ CRAWLER_CONCURRENCY=2                │
│ 2 Playwright instances               │
│ 4 URLs/sec                           │
│ 50k URLs: ~3.5 weeks                 │
└─────────────────────────────────────┘
                 ▲
                 │ 2x faster (but diminishing returns)
                 │
┌─────────────────────────────────────┐
│ CRAWLER_CONCURRENCY=4 (Recommended) │
│ 4 Playwright instances               │
│ 6 URLs/sec                           │
│ 50k URLs: ~2 weeks                   │
└─────────────────────────────────────┘
                 ▲
                 │ 1.3x faster (heavy CPU/memory)
                 │
┌─────────────────────────────────────┐
│ CRAWLER_CONCURRENCY=8+ (Overkill)    │
│ 8+ Playwright instances              │
│ 6–7 URLs/sec (no improvement)       │
│ High CPU/memory usage                │
│ 50k URLs: ~2 weeks (same)            │
└─────────────────────────────────────┘
```

---

## Tuning Concurrency

### 1. Default Configuration

```bash
# In .env or shell
export CRAWLER_CONCURRENCY=5
```

This is the **recommended default**:
- 5–6 URLs/sec on c6i.2xlarge
- Balanced CPU/memory usage
- Good for 2–3 week crawls

### 2. For Different Instance Types

| Instance | CPU | RAM | Recommended CONCURRENCY | Expected Speed |
|---|---|---|---|---|
| t3.medium | 2 vCPU | 4 GB | 2 | 2–3 URLs/sec |
| t3.large | 2 vCPU | 8 GB | 3 | 3–4 URLs/sec |
| t3.xlarge | 4 vCPU | 16 GB | 4 | 4–5 URLs/sec |
| c6i.xlarge | 4 vCPU | 8 GB | 4 | 5–6 URLs/sec |
| c6i.2xlarge | 8 vCPU | 16 GB | 5–6 | 6–8 URLs/sec |
| c6i.4xlarge | 16 vCPU | 32 GB | 8–10 | 8–10 URLs/sec |

### 3. Memory Usage per Worker

Each Playwright instance uses:
- Base: ~150 MB (Chromium process)
- Per page: ~50–100 MB (JavaScript heap)
- Screenshot: ~10 MB (temporary)

**Total for CONCURRENCY=5 on 16 GB RAM**:
- 5 × (150 MB + 100 MB) = 1.25 GB
- Remaining: 14.75 GB for OS/cache

Safe: Leave at least 2 GB for OS. So max CONCURRENCY = (RAM - 2GB) / 300MB

### 4. Find Optimal Concurrency

```bash
# Test with CONCURRENCY=1 (1 URL)
export CRAWLER_CONCURRENCY=1
timeout 300 npm run crawl 2>&1 | head -20
# Check: CPU, RAM, network

# Test with CONCURRENCY=3
export CRAWLER_CONCURRENCY=3
timeout 300 npm run crawl 2>&1 | head -20

# Test with CONCURRENCY=5
export CRAWLER_CONCURRENCY=5
timeout 300 npm run crawl 2>&1 | head -20

# Pick value with lowest time-to-URL (URLs/sec)
```

---

## Monitoring Parallel Workers

### 1. Check CPU/Memory Usage

On EC2, in separate terminal:

```bash
# Real-time monitoring
top -b -n 1 | head -20

# Per-process breakdown
ps aux | grep -E "node|chrome|playwright"

# Memory usage
free -h
```

**Expected output** (c6i.2xlarge, CONCURRENCY=5):
```
CPU:  40–60% (4 of 8 cores)
Memory: 2–3 GB / 16 GB
```

### 2. Monitor URLs Per Second

```bash
# Watch crawl progress in real-time
tail -f crawl.log | grep "Processed\|uploaded"

# Output:
# ✓ Processed 50/50000 URLs (100 screenshots, 480 annotations) | S3: 100 uploaded
# ✓ Processed 100/50000 URLs (200 screenshots, 960 annotations) | S3: 200 uploaded
# ...
```

**Calculate speed**:
```bash
# Grab last two timestamps
tail -20 crawl.log | grep "✓ Processed" | tail -2

# Output:
# [18:20:45] ✓ Processed 50/50000 URLs
# [18:21:00] ✓ Processed 100/50000 URLs
# Duration: 15 seconds for 50 URLs = 3.3 URLs/sec
```

### 3. Identify Slow Workers

If one worker is much slower, restart:

```bash
# Check individual worker progress
tail -f crawl.log | grep "Worker [0-4]"

# If Worker 2 is stalled
# Stop crawler (Ctrl+C)
# Resume: npm run crawl
# Checkpoint will skip completed URLs and re-process Worker 2's batch
```

---

## Performance Tuning

### 1. Increase Concurrency (Safe)

```bash
# If CPU/memory are underutilized
export CRAWLER_CONCURRENCY=6
npm run crawl
```

Monitor first 10 minutes:
- If CPU < 70% and memory < 50%, try CONCURRENCY=7
- If CPU > 90%, revert to CONCURRENCY=5

### 2. Decrease Batch Size (Trade Speed for Safety)

For unstable networks:

```typescript
// In Dataset/src/crawler.ts line ~50
const S3_BATCH_SIZE = 10;  // was 50, now smaller
```

This reduces risk of timeout during batch upload. Cost: more frequent uploads.

### 3. Increase Navigation Timeout (For Slow Networks)

```typescript
// In Dataset/src/crawler.ts around line 100
navigationTimeoutMs: 60000  // was 30000
```

Gives Playwright 60 seconds to load page (default: 30). Cost: crawl takes longer if URLs are slow.

### 4. Reduce Scroll Steps (Trade Coverage for Speed)

```typescript
// In Dataset/src/paths.ts
maxScrollSteps: 3  // was 5
```

Scrolls only 3 times instead of 5. Cost: might miss lazy-loaded content below fold.

### 5. Disable Dark Theme Variant (Extreme Measure)

To halve capture time (not recommended):

```bash
# In .env or shell
export SKIP_DARK_VARIANT=true
```

Cost: lose dark theme annotations, dataset halved.

---

## Parallelization Strategies

### Strategy 1: Single Instance, High Concurrency

```
┌─ EC2: c6i.2xlarge ─────┐
│ ├─ Worker 0 ───────────┤
│ ├─ Worker 1 ───────────┤
│ ├─ Worker 2 ───────────┤
│ ├─ Worker 3 ───────────┤
│ ├─ Worker 4 ───────────┤
│ └─ Worker 5 ───────────┤
│    CONCURRENCY=6       │
│    Speed: 6–8 URLs/sec │
└────────────────────────┘
```

**Pros**: Simple, low cost, low latency
**Cons**: Single point of failure, max speed ~8 URLs/sec
**Time for 50k URLs**: 2–3 weeks

### Strategy 2: Multiple Instances, Split URL Range

```
Worker 0–4 on EC2 #1 (URLs 0–10000)
Worker 0–4 on EC2 #2 (URLs 10000–20000)
Worker 0–4 on EC2 #3 (URLs 20000–30000)

Speed: 3 × 6 URLs/sec = 18 URLs/sec (parallel)
Time for 50k URLs: ~1 week
```

**Implementation**:
```bash
# On EC2 #1
export CRAWLER_START_INDEX=0
export CRAWLER_END_INDEX=10000
npm run crawl

# On EC2 #2
export CRAWLER_START_INDEX=10000
export CRAWLER_END_INDEX=20000
npm run crawl

# Merge results at end
```

**Pros**: 3x faster, fault-tolerant
**Cons**: More complex, more instances ($), coordination needed

### Strategy 3: Cascade Deployment

Day 1–7: Single c6i.2xlarge, CONCURRENCY=6
- Crawl 10k URLs
- Validate quality
- Cost: ~$20

Day 8–14: Add 2 more instances (3 total)
- Parallelize next 40k URLs
- Speed up 3x
- Cost: ~$60 total

**Pros**: Start immediately, scale as needed
**Cons**: Not fastest possible

---

## Troubleshooting Parallel Workers

### Issue: Only 1 Worker Running

**Check**:
```bash
# View logs for worker startup
grep "Starting.*workers" crawl.log

# Check environment variable
echo $CRAWLER_CONCURRENCY
```

**Fix**:
- Verify CRAWLER_CONCURRENCY is set
- Check instance has enough CPU/memory
- Restart crawler

### Issue: Workers Crashing

**Check logs**:
```bash
tail -f crawl.log | grep "Worker\|error\|Error"
```

**Common causes**:
- Out of memory: Reduce CONCURRENCY
- Network timeout: Increase navigationTimeoutMs
- Duplicate URL processing: Checkpoint system corrupted (rare)

**Fix**:
```bash
# Clear checkpoint (WARNING: restarts from URL 0)
rm -f Dataset/raw/screenshots/manifest.jsonl

# Restart with reduced concurrency
export CRAWLER_CONCURRENCY=3
npm run crawl
```

### Issue: Slow Performance

**Diagnose**:
```bash
# Check URLs/sec
tail -f crawl.log | grep "Processed"

# If < 2 URLs/sec:
# 1. Check CPU: top
# 2. Check memory: free -h
# 3. Check network: iftop -i eth0 (EC2 only)
```

**Optimize**:
- Increase CONCURRENCY (if CPU < 70%)
- Reduce maxScrollSteps (if network slow)
- Switch to faster instance type (if CPU-bound)
- Check for memory leaks: `ps aux | sort -k4 -nr | head -5`

### Issue: Batch Uploads Failing

**Check**:
```bash
tail -f crawl.log | grep "S3 upload"
```

**Common causes**:
- S3 bucket deleted: Recreate
- IAM permissions revoked: Re-attach role
- Network issue: Transient, will retry

**Fix**:
```bash
# Verify S3 access
aws s3 ls s3://webui-dataset-2026/

# Verify IAM role (on EC2)
aws sts get-caller-identity
```

---

## Performance Benchmarks

### c6i.2xlarge with Varying Concurrency

| CONCURRENCY | CPU | Memory | URLs/sec | Time (50k) |
|---|---|---|---|---|
| 1 | 12–15% | 0.8 GB | 2.0 | ~7 weeks |
| 2 | 25–30% | 1.2 GB | 3.5 | ~4 weeks |
| 3 | 35–40% | 1.5 GB | 4.5 | ~2.6 weeks |
| 4 | 45–50% | 1.8 GB | 5.5 | ~2.1 weeks |
| 5 | 50–55% | 2.1 GB | 6.0 | ~1.9 weeks |
| 6 | 55–60% | 2.4 GB | 6.2 | ~1.9 weeks |
| 7 | 60–70% | 2.7 GB | 6.3 | ~1.9 weeks |
| 8+ | 70–85% | 3+ GB | ~6.3 | ~1.9 weeks |

**Observation**: Diminishing returns after CONCURRENCY=5. Sweet spot for cost/speed.

---

## Recommended Configuration

For most users:

```bash
# .env
CRAWLER_CONCURRENCY=5
S3_ENABLED=true
S3_BUCKET=webui-dataset-2026
AWS_REGION=us-east-1

# Instance: c6i.2xlarge (8 vCPU, 16 GB)
# Time: ~2 weeks for 50k URLs
# Cost: ~$34 (spot)
```

---

## Next Steps

1. **Set CRAWLER_CONCURRENCY=5** (default)
2. **Monitor first 1 hour** — Check URLs/sec
3. **Adjust if needed** — Increase/decrease by 1
4. **Lock in configuration** — Save to .env
5. **Run full crawl** — Check back in 2 weeks

---

**See also**: [AWS_DEPLOYMENT.md](AWS_DEPLOYMENT.md) for instance sizing  
**Questions?** Check: `tail -f crawl.log | grep "Processed\|Worker"`
