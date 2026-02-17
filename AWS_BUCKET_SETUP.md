# AWS Bucket Setup (S3)

Follow these steps to configure S3 for storing tiles and processed artifacts.

## 1) Install AWS CLI

Check if it is installed:

```bash
aws --version
```

If missing (Ubuntu):

```bash
sudo apt-get update
sudo apt-get install -y awscli
```

## 2) Configure Credentials (One-Time)

Run:

```bash
aws configure
```

Enter:
- **AWS Access Key ID**: your access key (starts with `AKIA...`)
- **AWS Secret Access Key**: your secret key (long string)
- **Default region name**: `us-east-1` (or your preferred region)
- **Default output format**: press Enter (defaults to `json`) or type `json`

## 3) Create the Bucket (One-Time)

Pick a globally unique bucket name, then run:

```bash
aws s3api create-bucket --bucket scenicdriver-data --region us-east-1
```

## 4) Set the Bucket Env Var

```bash
export SCENIC_S3_BUCKET=scenicdriver-data
```

## 5) Sync Local Data to S3

```bash
bash scripts/s3_sync.sh
```

Notes:
- Script may not be executable by default; `bash ...` avoids permission issues.
- Keep tiles under `raw/images/{satellite|terrain}/z{zoom}/{region}/{x}_{y}.png`.

## 6) Apply Lifecycle Rules (Optional)

Moves older data to cheaper storage.

```bash
aws s3api put-bucket-lifecycle-configuration \
  --bucket "$SCENIC_S3_BUCKET" \
  --lifecycle-configuration file://scripts/s3_lifecycle.json
```
