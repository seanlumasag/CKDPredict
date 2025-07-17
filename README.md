# CKD Risk Predictor

A full-stack web app serving as a risk predictor for Chronic Kidney Disease (CKD) based on user health inputs. Built with a modern tech stack including Spring Boot, React, and a PyTorch machine learning microservice.

---

## 🧠 Features

- Responsive UI for entering patient data
- Spring Boot backend exposing RESTful endpoints
- Python microservice running a trained PyTorch machine learning model
- PostgreSQL database storing all predictions
- Deployment-ready (frontend + backend hosted separately)

---

## 🚀 Live Demo

🌐 https://ckdpredict.vercel.app

---

## ⚙️ Tech Stack

- **Frontend:** React, JavaScript, HTML, CSS  
- **Backend:** Spring Boot (Java)  
- **ML Service:** PyTorch, FastAPI (Python)
- **Database:** PostgreSQL  
- **Deployment:** Render (Java API & ML), Vercel (React), Neon (DB)

---

## 🛠️ How to Run Locally

### Prerequisites

- Java 17+
- Node.js and npm
- Python 3.9+
- PostgreSQL

### 1. Clone the Repository
```bash
git clone https://github.com/seanlumasag/CKDPredict
cd CKDPredict
```

### 2. Setup Backend
```bash
cd backend
./mvnw spring-boot:run
```

### 3. Setup Frontend
```bash
cd frontend
npm install
npm run dev
```

### 4. Setup ML-Service
```bash
cd ml-service
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
