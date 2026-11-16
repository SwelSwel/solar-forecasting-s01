# Deployment Guide — S01 Solar Forecast

## Quick Start (Local)

```bash
# 1. Export the model from the notebook (run the export cell)
# 2. Start the app
pip install -r requirements.txt
streamlit run app.py
# Opens at http://localhost:8501
```

## AWS EC2 Deployment (Student Account)

### Step 1: Create EC2 Instance

1. Go to **AWS Console → EC2 → Launch Instance**
2. Settings:
   - **Name:** solar-forecast
   - **AMI:** Ubuntu 22.04 LTS
   - **Instance type:** t3.small (2 vCPU, 2 GB RAM)
   - **Key pair:** Create new or use existing (.pem file)
   - **Security group:** Allow these inbound rules:
     - SSH (port 22) from your IP
     - Custom TCP (port 8501) from 0.0.0.0/0 (for Streamlit)
3. Click **Launch Instance**
4. Note the **Public IPv4 address** from the instance details

### Step 2: Connect and Install

```bash
# SSH into the instance
ssh -i your-key.pem ubuntu@<PUBLIC_IP>

# Install Docker
sudo apt-get update -y
sudo apt-get install -y docker.io
sudo systemctl start docker
sudo usermod -aG docker ubuntu

# Log out and back in for group change
exit
ssh -i your-key.pem ubuntu@<PUBLIC_IP>
```

### Step 3: Upload Your App

From your local machine:

```bash
# Create project folder on EC2
ssh -i your-key.pem ubuntu@<PUBLIC_IP> "mkdir -p ~/solar-app"

# Copy files (from your project root)
scp -i your-key.pem -r \
    app.py requirements.txt Dockerfile .streamlit/ src/ models/ \
    ubuntu@<PUBLIC_IP>:~/solar-app/
```

### Step 4: Build and Run

```bash
ssh -i your-key.pem ubuntu@<PUBLIC_IP>

cd ~/solar-app
docker build -t solar-forecast .
docker run -d --name solar-forecast --restart unless-stopped -p 8501:8501 solar-forecast
```

### Step 5: Access

Open `http://<PUBLIC_IP>:8501` in your browser.

### Useful Commands

```bash
# Check if running
docker ps

# View logs
docker logs solar-forecast

# Restart after code change
docker stop solar-forecast && docker rm solar-forecast
docker build -t solar-forecast . && docker run -d --name solar-forecast --restart unless-stopped -p 8501:8501 solar-forecast

# Stop instance (save money when not demoing)
# Go to AWS Console → EC2 → Select instance → Instance State → Stop
```

## CI/CD (Optional)

The `.github/workflows/deploy.yml` auto-deploys on push to main.

To enable:
1. Add these GitHub secrets (Settings → Secrets):
   - `EC2_HOST`: your EC2 public IP
   - `EC2_SSH_KEY`: contents of your .pem file
2. Push to main — it tests, then deploys

## Project Structure

```
solar-forecasting-s01/
├── app.py                    # Streamlit web app
├── requirements.txt          # Python dependencies
├── Dockerfile                # Container build
├── DEPLOY.md                 # This file
├── .streamlit/config.toml    # Theme and settings
├── .github/workflows/        # CI/CD pipeline
├── src/
│   ├── __init__.py
│   ├── data_quality.py       # Feature engineering
│   ├── modeling.py           # Model training
│   └── fetch_weather.py      # Weather API client
├── models/
│   └── xgboost_v2.pkl        # Trained model (export from notebook)
├── data/                     # gitignored
└── notebooks/
    └── S01_Solar_DayAhead_Forecasting_Progress.ipynb
```

## Cost

- **t3.small:** ~$0.02/hour ($15/month if always on)
- **Student account:** check your credit balance
- **Tip:** Stop the instance when not demoing to save credits
