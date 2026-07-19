# 🚀 How to Push This Project to GitHub

## Step 1: Create a New Repo on GitHub
1. Go to https://github.com
2. Click "New Repository"
3. Name it: `loan-approval-prediction`
4. Keep it Public
5. Do NOT add README (we already have one)
6. Click "Create Repository"

---

## Step 2: Open Terminal in Your Project Folder

```bash
cd loan-approval-prediction
```

---

## Step 3: Initialize Git & Push

```bash
# Initialize git
git init

# Add all files
git add .

# First commit
git commit -m "Initial commit: Loan Approval Prediction ML project"

# Connect to your GitHub repo (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/loan-approval-prediction.git

# Push to GitHub
git branch -M main
git push -u origin main
```

---

## Step 4: Future Updates

```bash
# After making changes:
git add .
git commit -m "describe what you changed"
git push
```

---

## ✅ After Pushing
- Go to your GitHub profile
- You'll see the project with the README displayed
- Copy the repo link and add it to your resume!

## 📝 Resume Line
> "Built a Loan Approval Prediction ML model using Random Forest (82% accuracy) with EDA, feature engineering, and classification report. [GitHub Link]"
