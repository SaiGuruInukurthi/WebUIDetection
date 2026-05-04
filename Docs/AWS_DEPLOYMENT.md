# AWS Deployment Guide — WebUI Dataset Crawler

**Date**: May 3, 2026  
**Status**: ✅ Ready for AWS deployment

---

## Overview

This guide covers deploying the WebUI dataset crawler to AWS using:
- **EC2** for computation (crawler runs here)
- **S3** for persistent storage (screenshots, annotations, metrics)
- **Spot Instances** for 50–70% cost savings
- **Parallel workers** for 5–8x faster crawling

---

## Architecture

```
┌─ LOCAL (Development) ──┐
│ scraped URLs           │
│ config.yaml            │
└─────────────┬──────────┘
              │ (git push)
              ▼
┌─ AWS EC2 ──────────────────┐
│ ┌─ Worker 1 ────────────┐  │
│ │ Playwright instance   │  │
│ │ (processes URLs 0, N, 2N...)│
│ └─────────┬──────────────┘  │
│           │ saves .webp/.json
│ ┌─ Worker 2 ────────────┐  │
│ │ Playwright instance   │  │
│ │ (processes URLs 1, N+1, 2N+1...)│
│ └─────────┬──────────────┘  │
│           │ queues files    │
│ ┌─ Batch Upload (every 50) │
│ │ .webp → S3              │
│ │ .json → S3              │
│ └─────────┬──────────────┘  │
└──────────┼─────────────────┘
           │
           ▼
     ┌──────────────┐
     │   AWS S3     │
     │   Bucket     │
     │              │
     │ screenshots/ │
     │ metrics/     │
     │ manifest/    │
     └──────────────┘
```

---

## Phase 1: AWS Account Setup

### 1.1 Create S3 Bucket

```bash
# Set bucket name (must be globally unique)
BUCKET_NAME="webui-dataset-2026-ap"

# Create bucket in ap-south-2 region
aws s3 mb s3://${BUCKET_NAME} --region ap-south-2
```

**Verify**:
```bash
aws s3 ls s3://${BUCKET_NAME}
```

### 1.2 Create IAM Role for EC2

**Permission policy** (`s3-crawler-policy.json`):
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::webui-dataset-2026-ap",
        "arn:aws:s3:::webui-dataset-2026-ap/*"
      ]
    }
  ]
}
```

**Create role**:
```bash
# Create role
aws iam create-role --role-name webui-crawler-role \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "ec2.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

# Attach policy
aws iam put-role-policy --role-name webui-crawler-role \
  --policy-name s3-access \
  --policy-document file://s3-crawler-policy.json

# Create instance profile
aws iam create-instance-profile --instance-profile-name webui-crawler-profile
aws iam add-role-to-instance-profile \
  --instance-profile-name webui-crawler-profile \
  --role-name webui-crawler-role
```

---

## Phase 2: Launch EC2 Instance

### 2.1 Recommended Instance Configuration

**For initial testing** (1,000–5,000 URLs):
```
Instance Type: t3.xlarge (4 vCPU, 16 GB RAM)
Cost: ~$0.1664/hour on-demand, ~$0.05/hour spot
AMI: Ubuntu 22.04 LTS (ami-0c55b159cbfafe1f0 for ap-south-2)
Storage: 100 GB gp3 (root volume)
Spot: YES (70% discount)
```

**For production** (50,000 URLs):
```
Instance Type: c6i.2xlarge (8 vCPU, 16 GB RAM)
Cost: ~$0.34/hour on-demand, ~$0.10/hour spot
Storage: 200 GB gp3
Spot: YES
```

### 2.2 Launch Command

```bash
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type t3.xlarge \
  --key-name your-keypair \
  --iam-instance-profile Name=webui-crawler-profile \
  --security-group-ids sg-xxxxxxxx \
  --subnet-id subnet-xxxxxxxx \
  --instance-market-options 'MarketType=spot,SpotOptions={MaxPrice=0.06,SpotInstanceType=persistent}' \
  --block-device-mappings DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp3,DeleteOnTermination=true} \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=webui-crawler-1}]' \
  --user-data file://setup.sh
```

### 2.3 User Data Script (`setup.sh`)

```bash
#!/bin/bash
set -e

echo "Installing dependencies..."
sudo apt update
sudo apt install -y \
  curl \
  wget \
  git \
  build-essential \
  python3-pip

echo "Installing Node.js..."
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

echo "Installing Playwright dependencies..."
sudo apt install -y \
  libnss3 \
  libatk1.0-0 \
  libatk-bridge2.0-0 \
  libcups2 \
  libdrm2 \
  libxkbcommon0 \
  libxdamage1

echo "Cloning repository..."
cd /opt
sudo git clone https://github.com/yourusername/WebUIDetection.git
cd WebUIDetection/Dataset
sudo chown -R ubuntu:ubuntu /opt/WebUIDetection

echo "Installing npm dependencies..."
npm install
npx playwright install chromium

echo "Setup complete!"
```

---

## Phase 3: Configure Crawler for AWS

### 3.1 Set Environment Variables

On the EC2 instance:

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@<instance-public-ip>

# Set environment variables
export S3_ENABLED=true
export S3_BUCKET=webui-dataset-2026-ap
export AWS_REGION=ap-south-2
export CRAWLER_CONCURRENCY=5

# Verify
echo "S3_ENABLED=$S3_ENABLED"
echo "S3_BUCKET=$S3_BUCKET"
echo "AWS_REGION=$AWS_REGION"
echo "CRAWLER_CONCURRENCY=$CRAWLER_CONCURRENCY"
```

### 3.2 Persistent Configuration

Create `/opt/WebUIDetection/Dataset/.env`:

```bash
S3_ENABLED=true
S3_BUCKET=webui-dataset-2026-ap
AWS_REGION=ap-south-2
CRAWLER_CONCURRENCY=5
```

Load before crawling:

```bash
source .env
npm run crawl
```

---

## Phase 4: Run Crawler

### 4.1 Start Crawling

```bash
cd /opt/WebUIDetection/Dataset

# Source environment
source .env

# Option 1: Foreground (attach to terminal)
npm run crawl

# Option 2: Background (detach with nohup)
nohup npm run crawl > crawl.log 2>&1 &

# Option 3: Background with screen (recommended)
screen -S crawl -d -m npm run crawl
screen -S crawl -X logfile crawl.log
```

### 4.2 Monitor Progress

**Check logs**:
```bash
tail -f crawl.log
```

**Expected output**:
```
Starting 5 parallel workers...
📤 S3 upload enabled: bucket="webui-dataset-2026", region="us-east-1"

[Worker 0] Crawling 1/50000: https://example.com
[Worker 1] Crawling 2/50000: https://example.org
[Worker 2] Crawling 3/50000: https://example.net
...
✓ Processed 50/50000 URLs (100 screenshots, 480 annotations, 8 low-quality) | S3: 100 uploaded
```

### 4.3 Resume If Interrupted

Crawler automatically resumes from last checkpoint:

```bash
npm run crawl

# Output:
# Checkpoint: 5000/50000 URLs already completed (10000 screenshots).
# Resuming from URL 5001/50000 (44999 URLs remaining).
```

---

## Phase 5: Download Results

### 5.1 After Crawl Completes

**Check local results**:
```bash
# Manifest and metrics
ls -lh Dataset/raw/screenshots/manifest.jsonl
ls -lh Dataset/url-sources/dataset-metrics.json

# Count images
ls Dataset/raw/screenshots/*.webp | wc -l
```

**Download from S3**:
```bash
# Option 1: Download all (slow for 100k files)
aws s3 sync s3://webui-dataset-2026-ap/screenshots/ ./Dataset/raw/screenshots/

# Option 2: Download only metrics (fast)
aws s3 cp s3://webui-dataset-2026-ap/screenshots/manifest.jsonl \
  ./Dataset/raw/screenshots/manifest.jsonl

aws s3 cp s3://webui-dataset-2026-ap/screenshots/dataset-metrics.json \
  ./Dataset/url-sources/dataset-metrics.json
```

### 5.2 Keep Instance or Terminate

**Keep for further work**:
```bash
# Stop (not terminate) to save state
aws ec2 stop-instances --instance-ids i-xxxxxxxx
```

**Terminate to save costs**:
```bash
# Terminate (WARNING: data on disk is lost, but S3 data persists)
aws ec2 terminate-instances --instance-ids i-xxxxxxxx
```

---

## Performance Expectations

### Crawling Speed

| Instance | Concurrency | Speed | Time (50k URLs) |
|---|---|---|---|
| t3.large | 2 | 2–3 URLs/sec | ~6–8 weeks |
| t3.xlarge | 4 | 4–5 URLs/sec | ~3–4 weeks |
| c6i.xlarge | 4 | 5–6 URLs/sec | ~2–3 weeks |
| c6i.2xlarge | 6 | 6–8 URLs/sec | ~2 weeks |

### Storage Cost (AWS S3)

Assuming 100,000 images, ~10 GB total:

```
Storage: $0.023 per GB/month = $0.23/month
Transfer out: ~$0.01/GB = $0.10 for initial download
Total: ~$0.33/month (very cheap)
```

### Compute Cost (EC2 Spot)

Crawling 50,000 URLs in 2 weeks on c6i.2xlarge:

```
336 hours × $0.10/hour (spot) = $33.60
vs on-demand: 336 × $0.34 = $114.24
Savings: $80.64 (71% discount)
```

---

## Cost Optimization Tips

### 1. Use Spot Instances
- 50–70% cheaper than on-demand
- OK for batch jobs (can be interrupted, but checkpoint system handles it)
- Can interrupt for 30–90 seconds, then resume

### 2. Choose Right Instance Size
- Start with t3.xlarge ($0.05/hour spot)
- If too slow, upgrade to c6i.xlarge ($0.07/hour spot)
- Only use c6i.2xlarge if you need speed badly

### 3. Turn Off Instance When Done
```bash
aws ec2 stop-instances --instance-ids i-xxxxxxxx
# No cost while stopped (only storage charges for root volume)
```

### 4. Phase Approach
- Phase 1: Test with 1,000 URLs on t3.xlarge (< 1 hour, < $0.10)
- Phase 2: Run 10,000 URLs (< 1 day, < $1)
- Phase 3: Full 50,000 URLs (2–3 weeks, $30–50)

---

## Troubleshooting

### Instance Terminated by Spot

**What happens**: AWS terminates your instance due to demand

**Recovery**:
```bash
# Spot was terminated, re-launch
aws ec2 run-instances ... # Same command as before

# Re-connect and resume
npm run crawl
# Will pickup from checkpoint
```

### S3 Upload Failures

**Check logs**:
```bash
tail crawl.log | grep "S3 upload failed"
```

**Common causes**:
- IAM permissions incorrect (fix and restart)
- Network issue (temporary, will retry)
- Bucket name wrong (check S3_BUCKET env var)

### Out of Disk Space

**Check usage**:
```bash
df -h /
```

**Fix**:
- Clean local screenshots: `rm -rf Dataset/raw/screenshots/*.webp`
- S3 copies are preserved
- Re-run crawler (will re-capture and upload)

### Slow Crawling

**Check**:
```bash
# View worker load
top

# Check network
iftop -i eth0
```

**Optimize**:
- Increase CONCURRENCY (if CPU/RAM available)
- Use faster instance type (c6i.xlarge instead of t3.xlarge)
- Check network connectivity to S3

---

## Next Steps

1. **Set up S3 bucket** — 5 minutes
2. **Create IAM role** — 5 minutes
3. **Launch EC2** — 15 minutes (instance starts)
4. **Configure crawler** — 5 minutes
5. **Start crawling** — Check back in 2–3 weeks
6. **Download results** — 30 minutes
7. **Begin training** — Next phase

---

## Deployment Checklist

- [ ] S3 bucket created
- [ ] IAM role created and attached
- [ ] EC2 instance launched
- [ ] SSH access verified
- [ ] Environment variables set
- [ ] Crawler started
- [ ] Logs being written
- [ ] S3 uploads happening
- [ ] Checkpoint system working

---

**Reference**: See [S3_CONFIGURATION.md](S3_CONFIGURATION.md) for detailed S3 setup.
**Questions?** Check crawler logs: `tail -f crawl.log`
