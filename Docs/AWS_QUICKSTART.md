# AWS Quick Start — WebUI Dataset Crawler

**Date**: May 3, 2026  
**Audience**: Ready to deploy to AWS  
**Time**: 30 minutes to first crawl

---

## TL;DR: Get Running in 30 Minutes

### Step 1: AWS Setup (5 min)

```bash
# Create S3 bucket
BUCKET=webui-dataset-2026-ap
aws s3api create-bucket --bucket $BUCKET --region ap-south-2 --create-bucket-configuration LocationConstraint=ap-south-2

# Enable versioning
aws s3api put-bucket-versioning --bucket $BUCKET \
  --versioning-configuration Status=Enabled
```

### Step 2: Launch EC2 (10 min)

```bash
# Create IAM role (one-time)
aws iam create-role --role-name webui-crawler-role \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "ec2.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

# Attach S3 policy
aws iam put-role-policy --role-name webui-crawler-role \
  --policy-name s3-access \
  --policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Action": ["s3:*"],
      "Resource": [
        "arn:aws:s3:::'"$BUCKET"'",
        "arn:aws:s3:::'"$BUCKET"'/*"
      ]
    }]
  }'

# Create instance profile
aws iam create-instance-profile --instance-profile-name webui-crawler
aws iam add-role-to-instance-profile \
  --instance-profile-name webui-crawler \
  --role-name webui-crawler-role
```

### Step 3: Launch Instance (5 min)

```bash
# Get latest Ubuntu 22.04 AMI ID
AMI_ID=$(aws ec2 describe-images \
  --owners 099720109477 \
  --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
  --query "sort_by(Images, &CreationDate)[-1].[ImageId]" \
  --output text)

# Launch instance (t3.xlarge for testing, ~$0.05/hour spot)
INSTANCE=$(aws ec2 run-instances \
  --image-id $AMI_ID \
  --instance-type t3.xlarge \
  --iam-instance-profile Name=webui-crawler \
  --security-groups default \
  --instance-market-options 'MarketType=spot' \
  --user-data '#!/bin/bash
set -e
sudo apt update
sudo apt install -y curl git build-essential
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
sudo apt install -y libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libxkbcommon0 libxdamage1
cd /opt
sudo git clone https://github.com/yourusername/WebUIDetection.git
cd WebUIDetection/Dataset
sudo chown -R ubuntu:ubuntu /opt/WebUIDetection
npm install
npx playwright install chromium' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=webui-crawler-1}]' \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "Instance launched: $INSTANCE"
echo "Wait 5 minutes for startup..."
sleep 300
```

### Step 4: SSH and Configure (5 min)

```bash
# Get public IP
IP=$(aws ec2 describe-instances --instance-ids $INSTANCE \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

echo "SSH: ssh -i your-key.pem ubuntu@$IP"

# SSH in
ssh -i your-key.pem ubuntu@$IP

# Inside instance:
cd /opt/WebUIDetection/Dataset

# Create .env
cat > .env << EOF
S3_ENABLED=true
S3_BUCKET=webui-dataset-2026-ap
AWS_REGION=ap-south-2
CRAWLER_CONCURRENCY=5
EOF

# Start crawling
source .env
npm run crawl &
disown
```

### Step 5: Verify (5 min)

```bash
# Back on your laptop
# Wait 2 minutes, then check S3
aws s3 ls s3://webui-dataset-2026/screenshots/ --recursive --summarize

# Output should show:
# Total Objects: 10-20
# Total Size: 100-200 MB
```

---

## Configuration Reference

### Environment Variables

Set in `.env` or shell:

```bash
# Required
S3_ENABLED=true
S3_BUCKET=webui-dataset-2026-ap
AWS_REGION=ap-south-2

# Optional (defaults shown)
CRAWLER_CONCURRENCY=5           # parallelism (1-10)
CRAWLER_START_INDEX=0           # URL range start
CRAWLER_END_INDEX=50000         # URL range end
SKIP_DARK_VARIANT=false         # skip dark theme
```

### Instance Sizing Guide

| Goal | Instance | Cost/Hour | Speed | Time (50k) |
|---|---|---|---|---|
| Test | t3.xlarge | $0.05 (spot) | 4 URLs/s | 3.5 weeks |
| Production | c6i.2xlarge | $0.10 (spot) | 6 URLs/s | 2 weeks |
| Fast | c6i.4xlarge | $0.20 (spot) | 8 URLs/s | 1.5 weeks |

---

## Running the Crawler

### Start Crawling

```bash
# Option 1: Foreground
npm run crawl

# Option 2: Background (nohup)
nohup npm run crawl > crawl.log 2>&1 &

# Option 3: Detached screen session
screen -S crawl -d -m npm run crawl
screen -S crawl -X logfile crawl.log

# Option 4: tmux
tmux new-session -d -s crawl "npm run crawl"
```

### Monitor Progress

```bash
# Watch logs in real-time
tail -f crawl.log

# Expected output:
# [18:20:30] Starting WebUI crawler...
# [18:20:31] Starting 5 parallel workers...
# [18:20:32] 📤 S3 upload enabled: bucket="webui-dataset-2026-ap"
# [18:20:35] [Worker 0] Crawling 1/50000: https://example.com
# [18:20:40] ✓ Processed 5/50000 URLs | S3: 10 uploaded
```

### Resume If Interrupted

```bash
# Crawler auto-detects checkpoint
npm run crawl

# Output:
# Checkpoint: 5000/50000 URLs completed
# Resuming from 5001/50000...
```

### Stop Crawler

```bash
# Foreground (Ctrl+C)
^C

# Background (nohup)
pkill -f "npm run crawl"

# Screen session
screen -S crawl -X quit
```

---

## Monitoring

### Check S3 Progress

```bash
# On your laptop
watch -n 10 'aws s3 ls s3://webui-dataset-2026-ap/screenshots/ --recursive --summarize'

# Or one-time:
aws s3 ls s3://webui-dataset-2026/screenshots/ --recursive --summarize

# Output:
# Total Objects: 1000
# Total Size: 10.5 GiB
```

### Check Instance Health

```bash
# CPU/memory on EC2
top

# Network bandwidth
iftop -i eth0

# Disk usage
df -h
```

---

## Downloading Results

### After Crawl Completes

```bash
# Option 1: Just metrics (fast, ~1 MB)
aws s3 cp s3://webui-dataset-2026-ap/screenshots/manifest.jsonl ./
aws s3 cp s3://webui-dataset-2026-ap/screenshots/dataset-metrics.json ./

# Option 2: All JSON annotations (large, ~5 GB)
aws s3 sync s3://webui-dataset-2026-ap/screenshots/ ./screenshots/ \
  --exclude "*.webp" --include "*.json"

# Option 3: All data (very large, ~100 GB)
aws s3 sync s3://webui-dataset-2026-ap/screenshots/ ./screenshots/

# Option 4: Specific files
aws s3 cp s3://webui-dataset-2026-ap/screenshots/00001_light_apnews.com_article-*.webp ./
```

---

## Cost Breakdown

### Compute (EC2 Spot c6i.2xlarge)

```
14 days × 24 hours × $0.10/hour = $33.60
Savings vs on-demand: ~$80 (71% discount)
```

### Storage (S3)

```
100,000 images = 10 GB
Storage: 10 GB × $0.023/month = $0.23
Requests: ~$0.50
Total: ~$0.73/month
```

### Total

```
One-time (50k URLs): ~$35
Per month after: ~$1
```

---

## Troubleshooting

### "Cannot find module '@aws-sdk/client-s3'"

```bash
# Re-run npm install
npm install
npm run typecheck
```

### "S3 upload failed: Access Denied"

```bash
# Verify IAM role
aws sts get-caller-identity

# Verify S3 bucket exists
aws s3 ls s3://webui-dataset-2026/

# Restart crawler
npm run crawl
```

### "NoSuchBucket" error

```bash
# Verify bucket name in .env
cat .env | grep S3_BUCKET

# Check actual bucket
aws s3 ls

# Create if missing
aws s3api create-bucket --bucket webui-dataset-2026 --region us-east-1
```

### Slow crawling (< 2 URLs/sec)

```bash
# Check CPU
top

# Increase concurrency (if CPU < 70%)
export CRAWLER_CONCURRENCY=6
npm run crawl

# Or upgrade instance type
```

---

## Scaling Up

### Phase 1: Test (1,000 URLs)
- Instance: t3.xlarge
- Time: 5 minutes
- Cost: < $0.01

### Phase 2: Validate (10,000 URLs)
- Instance: t3.xlarge
- Time: 1 hour
- Cost: < $0.10

### Phase 3: Full (50,000 URLs)
- Instance: c6i.2xlarge
- Time: 2 weeks
- Cost: ~$35

### Phase 4: Multi-Instance (100,000 URLs)
- Instances: 2× c6i.2xlarge
- Time: 2 weeks
- Cost: ~$70

---

## Quick Reference

### AWS CLI Commands

```bash
# List all instances
aws ec2 describe-instances --query 'Reservations[*].Instances[*].[InstanceId,InstanceType,State.Name]'

# Stop instance (keep data)
aws ec2 stop-instances --instance-ids i-xxxxxxxx

# Start instance (resume)
aws ec2 start-instances --instance-ids i-xxxxxxxx

# Terminate instance (delete)
aws ec2 terminate-instances --instance-ids i-xxxxxxxx

# Sync S3 (download all)
aws s3 sync s3://webui-dataset-2026/ ./data/

# Check costs (CloudWatch)
aws ce get-cost-and-usage \
  --time-period Start=2026-05-01,End=2026-05-31 \
  --granularity MONTHLY \
  --metrics "UnblendedCost" \
  --group-by Type=DIMENSION,Key=SERVICE
```

### SSH Commands

```bash
# Connect to EC2
ssh -i your-key.pem ubuntu@<public-ip>

# Copy file from EC2 to laptop
scp -i your-key.pem ubuntu@<public-ip>:/opt/WebUIDetection/crawl.log ./

# Copy file from laptop to EC2
scp -i your-key.pem ./config.json ubuntu@<public-ip>:/opt/WebUIDetection/Dataset/

# Port forward (if needed)
ssh -i your-key.pem -L 3000:localhost:3000 ubuntu@<public-ip>
```

---

## Next Steps

1. **Follow TL;DR above** (30 min)
2. **Monitor first hour** — Check CPU, URLs/sec
3. **Let run for 2 weeks** — Check progress daily
4. **Download metrics** — Review quality
5. **Begin training** — Use dataset

---

## Need Help?

- **AWS Deployment**: See [AWS_DEPLOYMENT.md](AWS_DEPLOYMENT.md)
- **S3 Configuration**: See [S3_CONFIGURATION.md](S3_CONFIGURATION.md)
- **Parallel Crawling**: See [PARALLEL_CRAWLING.md](PARALLEL_CRAWLING.md)
- **Quality Monitoring**: See [QUALITY_MONITORING_SYSTEM.md](QUALITY_MONITORING_SYSTEM.md) (from phase 1)

---

**Ready?** Start with the TL;DR above! 🚀
