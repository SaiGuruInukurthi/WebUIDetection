# AWS Documentation Index

**WebUI Dataset Crawler — AWS Deployment Guides**

---

## 📚 Documentation Overview

This folder contains comprehensive guides for deploying the WebUI dataset crawler to AWS with parallel workers and S3 storage.

### Quick Navigation

**🚀 Want to get started immediately?**
→ Start with [AWS_QUICKSTART.md](AWS_QUICKSTART.md) (30 minutes)

**📋 Need detailed setup?**
→ Follow [AWS_DEPLOYMENT.md](AWS_DEPLOYMENT.md) (complete guide)

**⚙️ Want to understand the implementation?**
→ Read [AWS_IMPLEMENTATION_SUMMARY.md](AWS_IMPLEMENTATION_SUMMARY.md)

---

## 📄 Document Index

### 1. **AWS_QUICKSTART.md** — 30-Minute Quick Start
**For**: Users who want to get crawling quickly  
**Time**: 30 minutes  
**Content**:
- TL;DR setup (copy-paste commands)
- Configuration reference
- Instance sizing recommendations
- Running and monitoring the crawler
- Troubleshooting common errors
- Scaling strategies

**When to use**: First time deploying, want fast results

---

### 2. **AWS_DEPLOYMENT.md** — Complete AWS Setup Guide
**For**: Users setting up AWS infrastructure from scratch  
**Time**: 1–2 hours  
**Content**:
- Architecture diagram
- AWS account setup (S3 bucket, IAM roles)
- EC2 instance launch (with user data script)
- Environment configuration
- Running the crawler
- Monitoring and logging
- Downloading results
- Cost analysis
- Troubleshooting guide

**When to use**: First AWS deployment, need step-by-step instructions

---

### 3. **S3_CONFIGURATION.md** — S3 Bucket & IAM Setup
**For**: Users configuring S3 storage and permissions  
**Time**: 30 minutes  
**Content**:
- 5-minute S3 bucket creation
- Versioning and lifecycle policies
- IAM role setup (full and minimal policies)
- IAM user creation for local development
- AWS CLI commands reference
- Cost analysis
- Batch upload tuning
- Performance optimization

**When to use**: Setting up S3 for the first time, need AWS credentials

---

### 4. **PARALLEL_CRAWLING.md** — Concurrency & Performance Tuning
**For**: Users optimizing crawler performance  
**Time**: 30 minutes  
**Content**:
- Parallel worker architecture
- Concurrency levels and recommendations
- Per-instance tuning guide
- Memory usage calculations
- Worker monitoring
- Performance benchmarks (URLs/sec by instance type)
- Troubleshooting worker issues
- Scaling strategies (single instance, multi-instance)

**When to use**: Tuning for faster crawling, optimizing CPU/memory

---

### 5. **AWS_IMPLEMENTATION_SUMMARY.md** — Technical Summary
**For**: Developers and technical reviewers  
**Time**: 15 minutes  
**Content**:
- Code changes (parallel workers, S3 integration)
- Environment variables reference
- Verification results (tests passing, compilation clean)
- Architecture diagrams
- Performance expectations
- Known limitations and future enhancements
- Deployment checklist
- Phase-based approach (validation → full crawl → analysis)

**When to use**: Understanding the implementation, code review, architecture planning

---

## 🎯 Recommended Reading Path

### Path 1: I want to start crawling immediately
1. [AWS_QUICKSTART.md](AWS_QUICKSTART.md) — Follow TL;DR section (30 min)
2. Monitor logs and check S3 progress daily
3. Return here for troubleshooting if needed

### Path 2: I want a complete understanding
1. [AWS_IMPLEMENTATION_SUMMARY.md](AWS_IMPLEMENTATION_SUMMARY.md) — Understand what was built (15 min)
2. [AWS_DEPLOYMENT.md](AWS_DEPLOYMENT.md) — Full AWS setup (1 hour)
3. [S3_CONFIGURATION.md](S3_CONFIGURATION.md) — S3 details (30 min)
4. [PARALLEL_CRAWLING.md](PARALLEL_CRAWLING.md) — Performance tuning (30 min)
5. Deploy and monitor

### Path 3: I'm an AWS expert, just need to implement
1. [AWS_IMPLEMENTATION_SUMMARY.md](AWS_IMPLEMENTATION_SUMMARY.md) — Understand architecture (10 min)
2. [AWS_QUICKSTART.md](AWS_QUICKSTART.md) — Copy setup commands (5 min)
3. Customize and deploy

### Path 4: I'm troubleshooting an issue
1. Find your issue in the "Troubleshooting" section of the relevant guide:
   - **S3 upload errors** → [S3_CONFIGURATION.md](S3_CONFIGURATION.md#troubleshooting)
   - **Slow performance** → [PARALLEL_CRAWLING.md](PARALLEL_CRAWLING.md#troubleshooting-parallel-workers)
   - **Worker crashes** → [PARALLEL_CRAWLING.md](PARALLEL_CRAWLING.md#troubleshooting-parallel-workers)
   - **General AWS issues** → [AWS_DEPLOYMENT.md](AWS_DEPLOYMENT.md#troubleshooting)
   - **Quick issues** → [AWS_QUICKSTART.md](AWS_QUICKSTART.md#troubleshooting)

---

## 🔑 Key Concepts

### Parallel Workers
- **What**: Multiple Playwright instances running simultaneously
- **Why**: 5–8x faster crawling (5–8 URLs/sec vs 1–2)
- **Default**: 5 concurrent workers
- **Where to learn**: [PARALLEL_CRAWLING.md](PARALLEL_CRAWLING.md)

### S3 Batch Upload
- **What**: Queue screenshots locally, upload in batches of 50
- **Why**: Reduces network overhead, faster uploads
- **Where to learn**: [S3_CONFIGURATION.md](S3_CONFIGURATION.md#batch-upload-configuration)

### Checkpoint-Resume
- **What**: Automatically resume from last checkpoint on interruption
- **Why**: Tolerate EC2 spot instance interruptions (70% cheaper)
- **Where to learn**: [AWS_DEPLOYMENT.md](AWS_DEPLOYMENT.md#phase-4-run-crawler)

### Cost Optimization
- **Spot instances**: 70% cheaper than on-demand ($0.10/hour vs $0.34)
- **Batch uploads**: Reduce per-request S3 costs
- **Total cost**: ~$35 for 50,000 URLs
- **Where to learn**: [AWS_DEPLOYMENT.md](AWS_DEPLOYMENT.md#cost-optimization-tips)

---

## 📊 Performance Expectations

### Speed by Instance Type

| Instance | vCPU | Memory | Speed | Time (50k) | Cost (14d) |
|---|---|---|---|---|---|
| t3.xlarge | 4 | 16GB | 4–5 URLs/s | 3–4 weeks | ~$16 |
| c6i.xlarge | 4 | 8GB | 5–6 URLs/s | 2–3 weeks | ~$21 |
| **c6i.2xlarge** | 8 | 16GB | **6–8 URLs/s** | **2 weeks** | **$34** |

### Recommended Configuration
```
Instance: c6i.2xlarge (8 vCPU, 16 GB)
Concurrency: 5 parallel workers
Time: 2 weeks for 50,000 URLs
Cost: ~$35 (compute) + ~$1 (storage)
```

**For details**: See [AWS_DEPLOYMENT.md — Performance Expectations](AWS_DEPLOYMENT.md#performance-expectations) or [PARALLEL_CRAWLING.md — Performance Benchmarks](PARALLEL_CRAWLING.md#performance-benchmarks)

---

## 🛠️ Environment Variables

Set before running `npm run crawl`:

```bash
# Required
S3_ENABLED=true               # Enable S3 uploads
S3_BUCKET=webui-dataset-2026  # Your S3 bucket
AWS_REGION=us-east-1         # AWS region

# Recommended
CRAWLER_CONCURRENCY=5         # Parallel workers (1-10)

# Optional
CRAWLER_START_INDEX=0         # URL range start
CRAWLER_END_INDEX=50000       # URL range end
SKIP_DARK_VARIANT=false       # Skip dark theme
```

**Where to learn**: [S3_CONFIGURATION.md](S3_CONFIGURATION.md#environment-variables)

---

## 🚀 Quick Commands

### Deploy to AWS (30 minutes)

```bash
# 1. Set up AWS account and IAM (5 min)
# Follow: AWS_QUICKSTART.md → TL;DR: AWS Setup

# 2. Launch EC2 instance (10 min)
# Follow: AWS_QUICKSTART.md → TL;DR: Launch EC2

# 3. SSH and configure (5 min)
ssh -i your-key.pem ubuntu@<public-ip>
cd /opt/WebUIDetection/Dataset
cat > .env << EOF
S3_ENABLED=true
S3_BUCKET=webui-dataset-2026
AWS_REGION=us-east-1
CRAWLER_CONCURRENCY=5
EOF

# 4. Start crawling (1 min)
source .env
npm run crawl

# 5. Monitor progress (ongoing)
tail -f crawl.log
```

### Monitor S3 Progress (From Your Laptop)

```bash
watch -n 10 'aws s3 ls s3://webui-dataset-2026/screenshots/ --recursive --summarize'
```

### Download Results (After Crawl Completes)

```bash
# Metrics only (fast, ~1 MB)
aws s3 cp s3://webui-dataset-2026/screenshots/dataset-metrics.json ./

# All data (large, ~100 GB)
aws s3 sync s3://webui-dataset-2026/screenshots/ ./screenshots/
```

---

## 📞 Support & FAQ

### Q: How long does crawling 50,000 URLs take?
**A**: With c6i.2xlarge (recommended): ~2 weeks  
See [PARALLEL_CRAWLING.md — Performance Benchmarks](PARALLEL_CRAWLING.md#performance-benchmarks)

### Q: Can I interrupt the crawl and resume?
**A**: Yes! Checkpoint system auto-resumes. Just run `npm run crawl` again.  
See [AWS_DEPLOYMENT.md — Resume If Interrupted](AWS_DEPLOYMENT.md#phase-4-run-crawler)

### Q: How much does it cost?
**A**: ~$35 for full 50k crawl (compute) + ~$1/month (storage)  
See [AWS_DEPLOYMENT.md — Cost Breakdown](AWS_DEPLOYMENT.md#cost-optimization-tips)

### Q: Can I run multiple instances in parallel?
**A**: Yes! Split URL ranges across instances. See [PARALLEL_CRAWLING.md — Strategy 2](PARALLEL_CRAWLING.md#strategy-2-multiple-instances-split-url-range)

### Q: What if my EC2 spot instance gets interrupted?
**A**: Checkpoint system resumes from where it left off. Just re-launch instance.  
See [AWS_DEPLOYMENT.md — Recovery](AWS_DEPLOYMENT.md#troubleshooting)

### Q: How do I optimize for speed?
**A**: Increase CONCURRENCY or use a faster instance type.  
See [PARALLEL_CRAWLING.md — Tuning Concurrency](PARALLEL_CRAWLING.md#tuning-concurrency)

---

## 🎯 Phase-Based Approach

### Phase 1: Validation (This Week)
- [ ] Set up AWS account (5 min)
- [ ] Launch test instance (10 min)
- [ ] Run crawler on 100 URLs (15 min)
- [ ] Verify S3 uploads and metrics
- [ ] Cost: < $1

### Phase 2: Full Crawl (Next 2 Weeks)
- [ ] Launch production instance
- [ ] Configure with CONCURRENCY=5
- [ ] Start crawl of 50,000 URLs
- [ ] Monitor daily progress
- [ ] Cost: ~$35

### Phase 3: Analysis (Following Week)
- [ ] Review dataset metrics
- [ ] Assess quality statistics
- [ ] Validate annotations (spot-check)
- [ ] Plan next iteration
- [ ] Cost: Storage only (~$0.50)

---

## 📖 Related Documentation

**From Phase 1 (Quality Improvements)**:
- [QUALITY_MONITORING_SYSTEM.md](QUALITY_MONITORING_SYSTEM.md) — Quality flags and metrics
- [QUALITY_IMPROVEMENTS.md](QUALITY_IMPROVEMENTS.md) — 7 dataset quality improvements
- [QUALITY_QUICK_REFERENCE.md](QUALITY_QUICK_REFERENCE.md) — Quick diagnostics

**From Code**:
- [Dataset/src/crawler.ts](../Dataset/src/crawler.ts) — Parallel workers + S3 integration
- [Dataset/src/paths.ts](../Dataset/src/paths.ts) — Configuration constants
- [Dataset/package.json](../Dataset/package.json) — Dependencies

---

## ✅ Implementation Status

| Component | Status | Link |
|---|---|---|
| Parallel workers (5 concurrent) | ✅ Complete | [crawler.ts](../Dataset/src/crawler.ts#L519-L590) |
| S3 batch upload | ✅ Complete | [crawler.ts](../Dataset/src/crawler.ts#L432-L475) |
| Checkpoint-resume system | ✅ Complete | [crawler.ts](../Dataset/src/crawler.ts#L213-L305) |
| TypeScript compilation | ✅ Clean | `npm run typecheck` |
| Unit tests | ✅ 45/45 passing | `npm test` |
| AWS SDK integration | ✅ Installed | package.json |
| Documentation (4 guides) | ✅ Complete | This folder |

---

## 🔗 Quick Links

- **AWS Setup**: [AWS_DEPLOYMENT.md](AWS_DEPLOYMENT.md)
- **S3 Configuration**: [S3_CONFIGURATION.md](S3_CONFIGURATION.md)
- **Performance Tuning**: [PARALLEL_CRAWLING.md](PARALLEL_CRAWLING.md)
- **Quick Start**: [AWS_QUICKSTART.md](AWS_QUICKSTART.md)
- **Implementation Details**: [AWS_IMPLEMENTATION_SUMMARY.md](AWS_IMPLEMENTATION_SUMMARY.md)

---

**Ready to get started?** → [AWS_QUICKSTART.md](AWS_QUICKSTART.md) 🚀

Last updated: May 3, 2026
