# Step-by-Step Execution Guide
## MLOps Capstone — From Model Training to Live Deployment

> Follow every step in order. Each phase depends on the one before it.  
> ⚠️ = common mistake / wrong input correction included.

---

## Table of Contents

| Phase | What Happens |
|-------|-------------|
| [Phase 1](#phase-1--prerequisites) | Install tools on your machine |
| [Phase 2](#phase-2--train-the-model) | Train & export the ML model |
| [Phase 3](#phase-3--test-the-api-locally-no-docker) | Run FastAPI locally & verify endpoints |
| [Phase 4](#phase-4--build-docker-images) | Build Docker images for backend & frontend |
| [Phase 5](#phase-5--run-with-docker-compose) | Run full stack locally with Docker Compose |
| [Phase 6](#phase-6--push-code-to-github) | Initialise Git repo & push to GitHub |
| [Phase 7](#phase-7--push-docker-images-to-docker-hub) | Tag & push images to Docker Hub registry |
| [Phase 8](#phase-8--set-up-github-actions-cicd) | Add secrets & trigger the CI/CD pipeline |
| [Phase 9](#phase-9--deploy-to-kubernetes) | Deploy to Kubernetes cluster |
| [Phase 10](#phase-10--verify-the-live-deployment) | Verify everything is running end-to-end |

---

## Phase 1 — Prerequisites

Install the following tools before anything else.

### 1.1 Python 3.10+
```bash
# Verify
python --version
# Expected: Python 3.10.x or higher

# ⚠️ Wrong: python3 --version shows 3.7 or lower
# Fix: Download from https://www.python.org/downloads/
```

### 1.2 Docker Desktop
```bash
# Download: https://www.docker.com/products/docker-desktop/
# Verify after install
docker --version
# Expected: Docker version 24.x.x or higher

docker-compose --version
# Expected: Docker Compose version v2.x.x

# ⚠️ Wrong: 'docker' is not recognized as a command
# Fix: Docker Desktop is not installed or not started — open Docker Desktop app first
```

### 1.3 Git
```bash
git --version
# Expected: git version 2.x.x

# ⚠️ Wrong: 'git' is not recognized
# Fix: Download from https://git-scm.com/downloads
```

### 1.4 kubectl (Kubernetes CLI)
```bash
# Windows (PowerShell as Admin):
choco install kubernetes-cli

# Mac:
brew install kubectl

# Verify
kubectl version --client
# Expected: Client Version: v1.xx.x

# ⚠️ Wrong: kubectl connects to wrong cluster
# Fix: Run `kubectl config current-context` to check which cluster is active
```

### 1.5 Accounts needed
- **GitHub** account → https://github.com
- **Docker Hub** account → https://hub.docker.com

---

## Phase 2 — Train the Model

This step runs the full ML pipeline, trains all 3 models, logs them to MLflow, and exports the best model (`model.pkl` + `scaler.pkl`) into the `backend/model/` folder.

### 2.1 Open a terminal and navigate to the project folder
```bash
cd "E:\2026\mlops\Capstone - Copy"

# ⚠️ Wrong: cd E:\2026\mlops\Capstone - Copy  (without quotes)
# Fix: Always quote paths that contain spaces
```

### 2.2 Create and activate a virtual environment

> ⚠️ **Python version requirement**: `whylogs` only has pre-built binary wheels for
> Python 3.10 and 3.11 on Windows. Using Python 3.12 or 3.13 will trigger a C++ source
> build that fails unless Visual Studio Build Tools are installed.
> **Always create the venv with Python 3.10 or 3.11.**

```bash
# Check available Python versions (Windows):
py -0
# Look for -V:3.10 or -V:3.11 in the list

# Create venv using Python 3.11 explicitly (Windows):
py -3.11 -m venv venv

# Create venv (Mac / Linux — ensure python3.11 is the interpreter):
python3.11 -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — Mac / Linux
source venv/bin/activate

# ⚠️ Wrong: (venv) prompt shows but python --version says 3.13
# Fix: You used `python -m venv venv` which picked the default Python (3.13).
#      Delete the venv folder and recreate with `py -3.11 -m venv venv`

# ⚠️ Wrong: You see (base) instead of (venv) in the prompt
# Fix: You are in a conda environment — run `conda deactivate` first, then activate venv

# ⚠️ Wrong: whylogs fails with "No CMAKE_CXX_COMPILER could be found"
# Fix 1 (easy):  Recreate venv with Python 3.11 (see above)
# Fix 2 (advanced): Install Visual Studio C++ Build Tools from
#                   https://visualstudio.microsoft.com/visual-cpp-build-tools/
#                   Select "Desktop development with C++" → restart terminal → retry

# Verify correct Python version inside venv
python --version   # Must show 3.10.x or 3.11.x
where python       # Windows — should point to venv\Scripts\python.exe
which python       # Mac/Linux — should point to venv/bin/python
```

### 2.3 Install project dependencies
```bash
pip install -r requirements.txt

# ⚠️ Wrong: ERROR: Could not find a version that satisfies the requirement mlflow==3.12.0
# Fix: Upgrade pip first: python -m pip install --upgrade pip

# Verify installs
pip show mlflow scikit-learn whylogs
```

### 2.4 Run the training pipeline
```bash
python capstone_mlops.py
```

**Expected output (last few lines):**
```
Best model  : Linear Regression  (R² = 0.9024)
Deployment  : Registered as 'salary_linear_regression' in MLflow
Capstone pipeline completed successfully!
```

**Files created after this step:**
```
backend/model/model.pkl       ← trained LinearRegression model
backend/model/scaler.pkl      ← fitted StandardScaler
backend/model/metadata.json   ← model metadata (R2, RMSE, etc.)
outputs/model_comparison.csv
outputs/model_comparison_plot.png
outputs/r2_comparison.png
mlruns/                       ← MLflow experiment store
```

### 2.5 (Optional) View MLflow UI
```bash
mlflow ui --backend-store-uri file:///E:/2026/mlops/Capstone - Copy/mlruns
# Open browser → http://127.0.0.1:5000
# Press Ctrl+C to stop the UI server when done
```

---

## Phase 3 — Test the API Locally (No Docker)

Always test the API directly with Python before building Docker images. This catches code errors cheaply.

### 3.1 Install backend dependencies
```bash
pip install fastapi uvicorn[standard] pydantic

# Or use backend's own requirements file:
pip install -r backend/requirements.txt
```

### 3.2 Start the FastAPI server
```bash
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000

# ⚠️ Wrong: ModuleNotFoundError: No module named 'backend'
# Fix: Run from the project root folder, not from inside backend/
# Correct working directory: E:\2026\mlops\Capstone - Copy

# ⚠️ Wrong: FileNotFoundError: model/model.pkl not found
# Fix: Make sure Phase 2 completed and backend/model/ contains the .pkl files
```

### 3.3 Test the endpoints

Open a **second terminal** (keep the server running in the first):

```bash
# Health check
curl http://localhost:8000/health
# Expected: {"status":"ok","model_loaded":true,...}

# Predict salary
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"years_experience\": 5.5}"
# Expected: {"years_experience":5.5,"predicted_salary":76211.xx,"currency":"USD",...}

# Model metadata
curl http://localhost:8000/metadata
# Expected: {"model_type":"LinearRegression","r2_score":0.9024,...}

# ⚠️ Wrong: curl: (7) Failed to connect to localhost port 8000
# Fix: The server is not running — check the first terminal for errors

# ⚠️ Wrong: {"detail":[{"type":"float_parsing",...}]}
# Fix: years_experience must be a number, not a string: use 5.5 not "5.5"

# ⚠️ Wrong: {"detail":"Input should be greater than 0"}
# Fix: years_experience must be > 0. Use 0.5 as minimum.
```

### 3.4 Open Swagger UI (browser)
```
http://localhost:8000/docs
```
You can test all endpoints interactively here. Stop the server with `Ctrl+C` when done.

---

## Phase 4 — Build Docker Images

### 4.1 Make sure Docker Desktop is running
```bash
docker info
# ⚠️ Wrong: Cannot connect to the Docker daemon
# Fix: Open Docker Desktop application and wait for it to say "Engine running"
```

### 4.2 Build the backend image
```bash
# From project root:
docker build -t salary-api:latest ./backend

# Watch for "Successfully built xxxxxxxx" at the end
# ⚠️ Wrong: COPY failed: file not found in build context: model/model.pkl
# Fix: Phase 2 was skipped — run capstone_mlops.py first to generate the .pkl files

# ⚠️ Wrong: ERROR [internal] load metadata for docker.io/library/python:3.10-slim
# Fix: Docker cannot reach the internet — check your internet connection or VPN
```

### 4.3 Build the frontend image
```bash
docker build -t salary-ui:latest ./frontend

# ⚠️ Wrong: COPY failed: nginx.conf not found
# Fix: Make sure frontend/nginx.conf exists (created in Phase 0 setup)
```

### 4.4 Verify images were built
```bash
docker images | grep salary
# Expected output:
# salary-api    latest    abc123def456   1 minute ago   250MB
# salary-ui     latest    789xyz012345   1 minute ago   45MB

# ⚠️ Wrong: Images not showing up
# Fix: Confirm the build command ran without errors (exit code 0)
```

### 4.5 Quick smoke test — run each container alone
```bash
# Test backend container
docker run --rm -p 8000:8000 salary-api:latest &
sleep 5
curl http://localhost:8000/health
docker stop $(docker ps -q --filter ancestor=salary-api:latest)

# ⚠️ Wrong: port is already allocated
# Fix: Something is already using port 8000 — stop it first:
#      Windows: netstat -ano | findstr :8000 → taskkill /PID <pid> /F
#      Mac/Linux: lsof -i :8000 → kill -9 <pid>
```

---

## Phase 5 — Run with Docker Compose

Docker Compose starts both containers together and wires the network between them.

### 5.1 Start the full stack
```bash
# From project root (where docker-compose.yml lives):
docker-compose up --build

# First run takes 2–3 minutes (downloads base images, installs packages)
# Subsequent runs are fast (layers are cached)

# ⚠️ Wrong: docker-compose: command not found
# Fix: Use the newer syntax: docker compose up --build  (space, not hyphen)

# ⚠️ Wrong: Service 'frontend' failed to build
# Fix: Check that frontend/Dockerfile and frontend/nginx.conf both exist
```

### 5.2 Verify both containers are healthy
Open a second terminal:
```bash
docker-compose ps
# Expected:
# NAME          IMAGE             STATUS
# salary-api    salary-api        Up (healthy)
# salary-ui     salary-ui         Up (healthy)

# ⚠️ Wrong: salary-api is "Up (health: starting)" for more than 60 seconds
# Fix: View logs: docker-compose logs backend
#      Common cause: model.pkl missing — run capstone_mlops.py again
```

### 5.3 Test the full stack in browser
```
Frontend UI  →  http://localhost:80
API docs     →  http://localhost:8000/docs
```

- Enter a number in the "Years of Experience" field (e.g. `7`)
- Click **Predict Salary**
- You should see a predicted salary appear

```bash
# ⚠️ Wrong: The UI loads but "Predict" returns a network error
# Fix: The frontend is calling the wrong API URL.
#      Open browser DevTools (F12) → Network tab → look at the failed request URL.
#      If it says http://localhost:8000 it should work — check backend is running.

# ⚠️ Wrong: Frontend shows but model stats (R2, RMSE) are blank
# Fix: Normal if the /metadata endpoint is slow on first load — refresh the page.
```

### 5.4 Stop the stack
```bash
# Stop (keeps containers):
docker-compose stop

# Stop and remove containers:
docker-compose down

# Stop and remove containers + images:
docker-compose down --rmi all
```

---

## Phase 6 — Push Code to GitHub

### 6.1 Create a new repository on GitHub
1. Go to https://github.com → click **New repository**
2. Name it: `mlops-salary-predictor`
3. Set to **Public** (needed for free CI/CD minutes) or Private
4. **Do NOT** initialise with README (we already have one)
5. Click **Create repository**
6. Copy the repository URL: `https://github.com/<your-username>/mlops-salary-predictor.git`

### 6.2 Create a .gitignore file
```bash
# From project root:
cat > .gitignore << 'EOF'
# Python
venv/
__pycache__/
*.pyc
*.pyo
.env

# MLflow
mlruns/

# Model binaries (too large for Git — use DVC or artifact storage in production)
backend/model/*.pkl

# OS
.DS_Store
Thumbs.db

# Outputs
outputs/
EOF
```

> ⚠️ **Important**: We exclude `.pkl` files from Git because binary model files should not live in source control. In a real project use DVC, MLflow Model Registry, or S3/GCS. For this capstone the model is rebuilt during CI.

### 6.3 Initialise Git and make the first commit
```bash
cd "E:\2026\mlops\Capstone - Copy"

git init
git config user.name  "Your Name"
git config user.email "your@email.com"

git add .
git status
# Review what files are staged — make sure no .pkl or mlruns/ appear

git commit -m "Initial commit: MLOps Capstone salary predictor"

# ⚠️ Wrong: nothing to commit (working tree clean) — but git init just ran
# Fix: git add . stages files. If it's still empty, check you're in the right folder.

# ⚠️ Wrong: LF will be replaced by CRLF (Windows warning)
# Fix: This is just a warning, not an error — safe to ignore.
```

### 6.4 Connect to GitHub and push
```bash
git remote add origin https://github.com/<your-username>/mlops-salary-predictor.git
git branch -M main
git push -u origin main

# ⚠️ Wrong: remote: Repository not found
# Fix: Check the URL — it must match exactly what GitHub showed you

# ⚠️ Wrong: Authentication failed
# Fix: GitHub no longer accepts passwords.
#      Use a Personal Access Token (PAT):
#      GitHub → Settings → Developer settings → Personal access tokens → Generate new token
#      Use the token as the password when prompted.

# ⚠️ Wrong: error: src refspec main does not match any
# Fix: You have no commits yet. Run git commit first (Step 6.3).
```

### 6.5 Verify on GitHub
Go to `https://github.com/<your-username>/mlops-salary-predictor` — you should see all your files there.

---

## Phase 7 — Push Docker Images to Docker Hub

### 7.1 Log in to Docker Hub
```bash
docker login
# Enter your Docker Hub username and password when prompted

# ⚠️ Wrong: unauthorized: incorrect username or password
# Fix: Use your Docker Hub username (not email) and password.
#      If 2FA is enabled, create an Access Token:
#      Docker Hub → Account Settings → Security → New Access Token
```

### 7.2 Tag the images with your Docker Hub username
```bash
# Replace <your-dockerhub-username> with your actual username
docker tag salary-api:latest <your-dockerhub-username>/salary-api:latest
docker tag salary-ui:latest  <your-dockerhub-username>/salary-ui:latest

# Example:
docker tag salary-api:latest rahul123/salary-api:latest
docker tag salary-ui:latest  rahul123/salary-ui:latest

# ⚠️ Wrong: Error response from daemon: No such image: salary-api:latest
# Fix: Phase 4 was skipped or failed — run docker build again
```

### 7.3 Push to Docker Hub
```bash
docker push <your-dockerhub-username>/salary-api:latest
docker push <your-dockerhub-username>/salary-ui:latest

# Watch for "latest: digest: sha256:..." — that confirms success

# ⚠️ Wrong: denied: requested access to the resource is denied
# Fix: You forgot to docker login, or the image tag doesn't match your username
```

### 7.4 Update image names in Kubernetes manifests
```bash
# In k8s/backend-deployment.yaml, replace:
#   image: your-dockerhub-username/salary-api:latest
# With your actual username:
#   image: rahul123/salary-api:latest

# Do the same in k8s/frontend-deployment.yaml for salary-ui

# Then commit the change:
git add k8s/
git commit -m "chore: set Docker Hub image names in k8s manifests"
git push
```

---

## Phase 8 — Set Up GitHub Actions CI/CD

Every push to `main` will now automatically: run tests → build images → push to Docker Hub → deploy to Kubernetes.

### 8.1 Add GitHub Secrets
Go to your GitHub repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

Add these three secrets:

| Secret Name | Value |
|-------------|-------|
| `DOCKER_USERNAME` | Your Docker Hub username |
| `DOCKER_PASSWORD` | Your Docker Hub password or Access Token |
| `KUBE_CONFIG` | Base64-encoded kubeconfig (see Step 8.2) |

```bash
# ⚠️ Wrong: Storing raw password in DOCKER_PASSWORD
# Better: Create a Docker Hub Access Token with Read/Write scope and use that instead
```

### 8.2 Generate KUBE_CONFIG secret
```bash
# This encodes your kubeconfig file to base64 for the secret:

# Mac / Linux:
cat ~/.kube/config | base64 | tr -d '\n'

# Windows (PowerShell):
[Convert]::ToBase64String([System.IO.File]::ReadAllBytes("$HOME\.kube\config"))

# Copy the entire output string and paste it as the KUBE_CONFIG secret value

# ⚠️ Wrong: Kubeconfig contains localhost / 127.0.0.1 as server address
# Fix: Kubernetes API must be publicly reachable from GitHub Actions runners.
#      Use your cloud cluster's external IP/hostname in the kubeconfig.
#      Local clusters (Minikube) will NOT work with GitHub Actions — use a cloud cluster.
```

### 8.3 Add a basic test file so the pipeline doesn't fail
```bash
mkdir -p backend/tests
cat > backend/tests/test_api.py << 'EOF'
from fastapi.testclient import TestClient
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from app import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_valid():
    r = client.post("/predict", json={"years_experience": 5.0})
    assert r.status_code == 200
    assert r.json()["predicted_salary"] > 0

def test_predict_invalid_zero():
    r = client.post("/predict", json={"years_experience": 0})
    assert r.status_code == 422   # validation error

def test_metadata():
    r = client.get("/metadata")
    assert r.status_code == 200
    assert "model_type" in r.json()
EOF

git add backend/tests/
git commit -m "test: add API unit tests for CI pipeline"
git push
```

### 8.4 Trigger the pipeline
The pipeline runs automatically on every push to `main`. To check it:

1. Go to your GitHub repo
2. Click the **Actions** tab
3. You should see a run called **"CI/CD — Salary Predictor"**
4. Click it to watch each job (Test → Build & Push → Deploy)

```
# ⚠️ Wrong: Job "test" fails with "ModuleNotFoundError: No module named 'app'"
# Fix: Check that backend/tests/test_api.py has the sys.path.insert line shown above

# ⚠️ Wrong: Job "build-and-push" fails with "denied: access forbidden"
# Fix: DOCKER_USERNAME or DOCKER_PASSWORD secret is wrong — re-enter them

# ⚠️ Wrong: Job "deploy" fails with "error: couldn't get current server API group list"
# Fix: KUBE_CONFIG is wrong or cluster is unreachable — re-generate it (Step 8.2)
```

---

## Phase 9 — Deploy to Kubernetes

### 9.1 Choose your Kubernetes cluster

**Option A — Local (for testing only)**
```bash
# Install Minikube
# https://minikube.sigs.k8s.io/docs/start/

minikube start
minikube status
# Expected: host: Running, kubelet: Running, apiserver: Running
```

**Option B — Cloud (for real deployment)**
```bash
# Google GKE:
gcloud container clusters get-credentials <cluster-name> --region <region>

# AWS EKS:
aws eks update-kubeconfig --name <cluster-name> --region <region>

# Azure AKS:
az aks get-credentials --resource-group <rg> --name <cluster-name>
```

### 9.2 Install the Nginx Ingress Controller
```bash
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/cloud/deploy.yaml

# Wait for it to be ready:
kubectl wait --namespace ingress-nginx \
  --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller \
  --timeout=120s

# ⚠️ Wrong: timed out waiting for the condition
# Fix: On Minikube, use: minikube addons enable ingress  instead
```

### 9.3 Apply all Kubernetes manifests
```bash
kubectl apply -f k8s/backend-deployment.yaml
kubectl apply -f k8s/frontend-deployment.yaml
kubectl apply -f k8s/ingress.yaml

# Expected output:
# deployment.apps/salary-api created
# service/salary-api-service created
# horizontalpodautoscaler.autoscaling/salary-api-hpa created
# deployment.apps/salary-ui created
# service/salary-ui-service created
# ingress.networking.k8s.io/salary-ingress created
```

### 9.4 Watch pods come up
```bash
kubectl get pods --watch
# Wait until all pods show STATUS = Running and READY = 1/1

# ⚠️ Wrong: Pod status is "ImagePullBackOff" or "ErrImagePull"
# Fix: The image name in the YAML is wrong, or Docker Hub image is private.
#      Check: kubectl describe pod <pod-name> | grep -A5 Events
#      Make sure image name matches exactly what you pushed in Phase 7

# ⚠️ Wrong: Pod status is "CrashLoopBackOff"
# Fix: Check logs: kubectl logs <pod-name>
#      Usually means model.pkl is missing — the Docker image didn't include it.
#      Rebuild the backend image after running capstone_mlops.py (Phase 2)
```

---

## Phase 10 — Verify the Live Deployment

### 10.1 Get the external IP
```bash
kubectl get ingress salary-ingress
# Expected:
# NAME              CLASS   HOSTS                ADDRESS        PORTS
# salary-ingress    nginx   salary.example.com   34.x.x.x       80

# ⚠️ Wrong: ADDRESS column is empty / <pending>
# Fix: Cloud load balancer is still provisioning — wait 2–3 minutes and check again.
#      On Minikube, run: minikube tunnel  (in a separate terminal) to get an IP.
```

### 10.2 Test via curl
```bash
# Replace 34.x.x.x with your actual external IP:
curl http://34.x.x.x/health
# Expected: {"status":"ok","model_loaded":true}

curl -X POST http://34.x.x.x/api/predict \
  -H "Content-Type: application/json" \
  -d "{\"years_experience\": 7.0}"
# Expected: {"predicted_salary": 91xxx.xx, ...}
```

### 10.3 Open the frontend in your browser
```
http://34.x.x.x
```
Enter years of experience → click Predict → salary appears.

### 10.4 Check auto-scaling is working
```bash
kubectl get hpa
# NAME             REFERENCE           TARGETS   MINPODS   MAXPODS   REPLICAS
# salary-api-hpa   Deployment/salary-api   5%/70%    2         6         2
```

---

## Complete Flow Summary

```
[Your Machine]
      │
      ├─ python capstone_mlops.py          → trains model, exports pkl files
      ├─ uvicorn backend.app:app           → test API without Docker
      ├─ docker build / docker-compose up  → test full stack locally
      │
[Git / GitHub]
      │
      ├─ git push origin main
      │        │
      │        └─ GitHub Actions triggers automatically:
      │               ├─ pytest (runs tests)
      │               ├─ docker build + docker push → Docker Hub
      │               └─ kubectl apply              → Kubernetes cluster
      │
[Kubernetes Cluster]
      │
      ├─ salary-ui  pods  (Nginx, 2 replicas)   → serves frontend
      ├─ salary-api pods  (FastAPI, 2 replicas)  → serves predictions
      └─ Ingress                                 → routes / and /api/*
```

---

## Quick Reference — Most Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `model.pkl not found` | Phase 2 skipped | Run `python capstone_mlops.py` first |
| `port already in use` | Another process on 8000/80 | Kill the process using that port |
| `ImagePullBackOff` | Wrong image name in YAML | Match tag to Docker Hub push |
| `CrashLoopBackOff` | App crashes on startup | `kubectl logs <pod>` to see the error |
| `Authentication failed` (git push) | Password auth disabled | Use a GitHub Personal Access Token |
| `denied` (docker push) | Not logged in or wrong tag | `docker login` then retag with username |
| `ErrImagePull` | Image is private | Set image to public on Docker Hub or add imagePullSecret |
| `Pipeline skips deploy job` | Push is not to `main` branch | Merge to main or adjust branch filter in ci-cd.yml |
| Blank model stats in UI | Backend unreachable | Check CORS and that backend container is healthy |
