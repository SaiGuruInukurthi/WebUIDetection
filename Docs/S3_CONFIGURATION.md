# S3 Configuration Guide — WebUI Dataset Crawler

**Date**: May 3, 2026  
**Focus**: S3 bucket setup, IAM permissions, batch upload tuning

---

## Quick Setup (5 minutes)

### 1. Create S3 Bucket

```bash
# Set variables
BUCKET_NAME="webui-dataset-2026"
AWS_REGION="us-east-1"

# Create bucket
aws s3api create-bucket \
  --bucket ${BUCKET_NAME} \
  --region ${AWS_REGION} \
  --create-bucket-configuration LocationConstraint=${AWS_REGION}

# Verify
aws s3 ls s3://${BUCKET_NAME}
# Output: (empty bucket, no objects)
```

### 2. Enable Versioning (Recommended)

Protect against accidental deletes:

```bash
aws s3api put-bucket-versioning \
  --bucket ${BUCKET_NAME} \
  --versioning-configuration Status=Enabled
```

### 3. Set Bucket Policy (Public Read for Web Preview)

If you want to view screenshots in browser:

```bash
aws s3api put-bucket-policy \
  --bucket ${BUCKET_NAME} \
  --policy '{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Sid": "PublicRead",
        "Effect": "Allow",
        "Principal": "*",
        "Action": "s3:GetObject",
        "Resource": "arn:aws:s3:::'"${BUCKET_NAME}"'/*",
        "Condition": {
          "StringEquals": {
            "aws:PrincipalAccount": "'"$(aws sts get-caller-identity --query Account --output text)"'"
          }
        }
      }
    ]
  }'
```

---

## S3 Directory Structure

The crawler creates this structure in S3:

```
s3://webui-dataset-2026/
├── screenshots/
│   ├── 00001_light_apnews.com_article-*.webp
│   ├── 00001_light_apnews.com_article-*.json
│   ├── 00002_light_www.wsj.com_news-*.webp
│   ├── 00002_light_www.wsj.com_news-*.json
│   ├── ... (50,000 image pairs)
│   ├── manifest.jsonl              (checkpoint, quality flags)
│   └── dataset-metrics.json        (global statistics)
└── metadata/
    └── crawl-logs/
        └── 2026-05-03-crawl.log    (optional)
```

### Naming Convention

```
{INDEX}_{VARIANT}_{DOMAIN}_{PATH}_{HASH}.{EXT}

Example:
00001_light_apnews.com_article-carter-habitat-housing-affordable-atlanta-community_9fc7874b74.webp

Breakdown:
- 00001    = URL index (1-50000)
- light    = "light" or "dark" color scheme variant
- apnews.com = domain
- article-carter-...  = URL path (truncated)
- 9fc7874b74 = hash (unique per URL)
- .webp    = WebP image (or .json for annotation)
```

---

## IAM Configuration

### Full Permission Policy

For EC2 instance to write to S3:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "S3CrawlerAccess",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::webui-dataset-2026",
        "arn:aws:s3:::webui-dataset-2026/*"
      ]
    }
  ]
}
```

### Minimal Permission Policy (Read-Only)

For downloading results from your laptop:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "S3ReadOnly",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::webui-dataset-2026",
        "arn:aws:s3:::webui-dataset-2026/*"
      ]
    }
  ]
}
```

### Create IAM User (For Local Development)

```bash
# Create user
aws iam create-user --user-name webui-dataset-user

# Create access key
aws iam create-access-key --user-name webui-dataset-user

# Output saved as: credentials (SAVE SOMEWHERE SAFE)
# {
#   "AccessKeyId": "AKIA...",
#   "SecretAccessKey": "..."
# }

# Attach S3 policy
aws iam put-user-policy \
  --user-name webui-dataset-user \
  --policy-name s3-access \
  --policy-document file://policy.json
```

### Configure AWS CLI Locally

```bash
aws configure

# Prompts:
# AWS Access Key ID: AKIA... (from above)
# AWS Secret Access Key: ... (from above)
# Default region: us-east-1
# Default output format: json

# Verify
aws s3 ls s3://webui-dataset-2026
```

---

## Environment Variables

Set on EC2 before running crawler:

### In `.env` File

```bash
# Required: Enable S3 uploads
S3_ENABLED=true

# Required: S3 bucket name
S3_BUCKET=webui-dataset-2026

# Optional: AWS region (default: us-east-1)
AWS_REGION=us-east-1

# Optional: Parallel concurrency (default: 5)
CRAWLER_CONCURRENCY=5
```

### In Shell

```bash
export S3_ENABLED=true
export S3_BUCKET=webui-dataset-2026
export AWS_REGION=us-east-1
export CRAWLER_CONCURRENCY=5

# Verify
echo "S3_ENABLED=$S3_ENABLED"
```

### In Docker (If Using Containers)

```dockerfile
ENV S3_ENABLED=true
ENV S3_BUCKET=webui-dataset-2026
ENV AWS_REGION=us-east-1
```

---

## Batch Upload Configuration

### How Batch Upload Works

1. Crawler captures screenshot + annotations
2. Queue locally: `s3UploadQueue.push({ localPath, s3Key })`
3. When queue reaches 50 items: `uploadBatchToS3(queue)`
4. Upload all 50 to S3 in parallel
5. Clear queue, repeat
6. At end, upload remaining items

### Tuning Batch Size

Edit [Dataset/src/crawler.ts](../Dataset/src/crawler.ts) to change batch size:

```typescript
// Line ~50
const S3_BATCH_SIZE = 50;  // Increase for faster but larger batches
                             // Decrease for smaller, more frequent uploads
```

**Guidance**:
- `S3_BATCH_SIZE = 10` — Small batches, more uploads, more network overhead (slower)
- `S3_BATCH_SIZE = 50` — **Recommended** (default)
- `S3_BATCH_SIZE = 100` — Large batches, fewer uploads, more memory (faster but riskier)

### Expected Throughput

With c6i.2xlarge + CONCURRENCY=5 + S3_BATCH_SIZE=50:

```
Capture speed:  6–8 URLs/sec
Upload speed:   50 images/batch = ~20 seconds/batch
Net speed:      5–6 URLs/sec (after S3 overhead)

50,000 URLs @ 5 URLs/sec = 10,000 seconds = ~2.8 hours per batch
Total time: 50,000 URLs @ 5–6 URLs/sec = ~2.5–2.8 weeks
```

---

## Monitoring Uploads

### View Logs in Terminal

```bash
tail -f crawl.log | grep "S3 upload\|uploaded"

# Output:
# ✓ Processed 50/50000 URLs | S3: 100 uploaded (2.3s)
# ✓ Processed 100/50000 URLs | S3: 200 uploaded (4.6s)
# ...
```

### Check S3 Bucket in AWS Console

```bash
# From CLI
aws s3 ls s3://webui-dataset-2026/screenshots/ --recursive --human-readable --summarize

# Output:
# Total Objects: 100000
# Total Size: 10.5 GiB
```

### Monitor Network Traffic (On EC2)

```bash
# Install iftop
sudo apt install -y iftop

# Monitor S3 connections
sudo iftop -i eth0 -f "dst port 443"  # HTTPS to S3
```

---

## Cost Analysis

### Storage Costs

```
Assuming 50,000 URLs = 100,000 images (light + dark)
Compressed WebP: 100 KB avg = 10 GB total
JSON annotations: 5 KB avg = 500 MB total

S3 Storage: 11 GB @ $0.023/GB = $0.253/month
S3 Requests (PutObject): 100,000 @ $0.0000055 = $0.55
S3 Requests (GetObject): ~5,000 = $0.0275

Total: ~$0.83/month
```

### Data Transfer Costs

```
Outbound traffic: 11 GB @ $0.01/GB (first 1 TB) = $0.11
Inbound (uploads): FREE

Total: $0.11
```

### Total Cost (50,000 URLs)

```
Compute (EC2 c6i.2xlarge, 2 weeks):  ~$34
Storage (S3 first month):             ~$0.83
Total:                                ~$35
```

---

## Troubleshooting

### "Cannot find module '@aws-sdk/client-s3'"

**Cause**: AWS SDK not installed

**Fix**:
```bash
cd Dataset
npm install @aws-sdk/client-s3
npm run typecheck
```

### "S3 upload failed: Access Denied"

**Cause**: IAM permissions or credentials wrong

**Check**:
```bash
# Verify IAM role (on EC2)
aws sts get-caller-identity

# Verify S3 access
aws s3 ls s3://webui-dataset-2026/
```

**Fix**:
1. Check EC2 has IAM role attached
2. Check role has S3 permissions
3. Check bucket name matches S3_BUCKET env var

### "S3 upload failed: NoSuchBucket"

**Cause**: Bucket name wrong or doesn't exist

**Fix**:
```bash
# List your buckets
aws s3 ls

# Check S3_BUCKET env var
echo $S3_BUCKET

# Create bucket if missing
aws s3api create-bucket --bucket webui-dataset-2026
```

### Large S3 Bill?

**Check**:
```bash
# Analyze bucket size
aws s3 ls s3://webui-dataset-2026 --recursive --summarize

# Check for unintended uploads (multiple batches?)
aws s3api list-object-versions --bucket webui-dataset-2026 --query 'Versions[*].[Key,LastModified,VersionId]' --output table
```

**Fix**:
- Delete old batches: `aws s3 rm s3://webui-dataset-2026/screenshots/ --recursive`
- Disable versioning if not needed: `aws s3api put-bucket-versioning --bucket webui-dataset-2026 --versioning-configuration Status=Suspended`

---

## AWS CLI Commands Reference

### List Objects

```bash
# All objects
aws s3 ls s3://webui-dataset-2026/screenshots/

# Recursive with human-readable sizes
aws s3 ls s3://webui-dataset-2026/screenshots/ --recursive --human-readable

# Summary
aws s3 ls s3://webui-dataset-2026/screenshots/ --recursive --summarize
```

### Download Objects

```bash
# Single file
aws s3 cp s3://webui-dataset-2026/screenshots/manifest.jsonl ./

# Entire directory
aws s3 sync s3://webui-dataset-2026/screenshots/ ./screenshots/

# Exclude certain files
aws s3 sync s3://webui-dataset-2026/screenshots/ ./screenshots/ --exclude "*.webp" --include "*.json"
```

### Delete Objects

```bash
# Single file
aws s3 rm s3://webui-dataset-2026/screenshots/00001_light_apnews.com_article-*.webp

# All objects in bucket
aws s3 rm s3://webui-dataset-2026/screenshots/ --recursive

# Keep versioning enabled (keeps history)
aws s3api delete-object-version --bucket webui-dataset-2026 --key screenshots/00001_light_apnews.com_article-*.webp --version-id XXXX
```

### Upload Objects

```bash
# Single file
aws s3 cp ./local-file.webp s3://webui-dataset-2026/screenshots/

# Entire directory
aws s3 sync ./screenshots/ s3://webui-dataset-2026/screenshots/

# With metadata
aws s3 cp ./file.webp s3://webui-dataset-2026/screenshots/ --metadata quality=high,source=crawler
```

---

## Performance Optimization

### 1. Use S3 Transfer Acceleration

For faster uploads from distant regions:

```bash
aws s3api put-bucket-accelerate-configuration \
  --bucket webui-dataset-2026 \
  --accelerate-configuration Status=Enabled
```

Update crawler to use:
```bash
export AWS_S3_ACCELERATE=true
```

### 2. Use CloudFront Distribution

Cache screenshots for web access:

```bash
aws cloudfront create-distribution \
  --origin-domain-name webui-dataset-2026.s3.amazonaws.com \
  --default-root-object manifest.jsonl \
  # ... more config
```

### 3. S3 Multipart Upload

Crawler already uses multipart uploads for files > 100 MB (handled by AWS SDK automatically).

### 4. Lifecycle Policies

Auto-delete old versions after 30 days:

```bash
aws s3api put-bucket-lifecycle-configuration \
  --bucket webui-dataset-2026 \
  --lifecycle-configuration '{
    "Rules": [{
      "Id": "DeleteOldVersions",
      "Filter": {},
      "NoncurrentVersionExpiration": {
        "NoncurrentDays": 30
      }
    }]
  }'
```

---

## Next Steps

1. **Create bucket** — 2 minutes
2. **Set permissions** — 3 minutes
3. **Configure crawler** — 5 minutes
4. **Run test batch** — 30 minutes
5. **Monitor uploads** — Ongoing

---

**See also**: [AWS_DEPLOYMENT.md](AWS_DEPLOYMENT.md) for full EC2 setup  
**Questions?** Check: `tail -f crawl.log | grep "S3\|uploaded"`
