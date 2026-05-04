# AI-Based Online Exam Cheating Detection System (Hybrid ML)

## Overview
This project presents a **hybrid AI system** to detect cheating in online exams using **behavioral**, **relational (group/collusion)**, **textual**, and **biometric (typing-based)** analysis.

Online exams lack physical supervision, so cheating becomes easier. Traditional proctoring or single-model detectors often fail to catch:
- **Group cheating / collusion**
- **AI-generated or copied answers**
- **Identity fraud / impersonation**

This system combines multiple algorithms—each specialized for a cheating dimension—so the final detection is more **robust** and **multi-layered**.

---

## Why These Algorithms?
### 1) Isolation Forest (Behavior Anomaly Detection)
Cheating behavior is often **rare** and **different** from normal behavior.
Isolation Forest is effective for **unsupervised anomaly detection** when labels are not available.

**Signals used (example):**
- time_taken
- tab_switches
- typing_speed
- idle_time

### 2) Graph-Based Model (Collusion / Group Cheating – GNN Concept)
Students involved in collusion often show **similar behavior patterns**.  
We build a **similarity graph**:
- Node = student
- Edge = strong similarity between two students
- Clusters/connected components = potential **group cheating**

> In this prototype, `networkx` is used to create and visualize the graph. In future work, this can be upgraded to a full **GNN** pipeline.

### 3) BERT Stylometry / Sentence Embeddings (Text Similarity)
Used to detect:
- unusually similar answers
- writing-style mismatch across attempts (stylometry concept)

We encode answers using a transformer embedding model (e.g., SentenceTransformer) and compute cosine similarity to flag suspicious pairs.

### 4) Variational Autoencoder (VAE) (Unknown Pattern Detection)
A VAE-style reconstruction approach can flag **unseen cheating patterns** not captured by standard rules.
If reconstruction error is high, the sample may be abnormal.

> Note: This repo demonstrates the concept. A complete VAE typically includes mean/variance sampling + KL divergence loss.

### 5) Keystroke Dynamics (Identity Verification)
Typing behavior can act like a biometric signature.  
Large deviation in typing dynamics can indicate **impersonation**.

> In this prototype, typing variance is simulated. Real deployment requires genuine keystroke timing features.

---

## System Workflow (High Level)
1. **Data collection** (behavior + answers + typing signals)
2. **Preprocessing & scaling**
3. Run detectors:
   - Isolation Forest → behavior anomalies
   - Similarity graph → group cheating
   - Text embeddings similarity → copied/AI-like similarity signals
   - VAE reconstruction → unknown anomalies
   - Keystroke dynamics → identity fraud
4. **Final decision fusion** (rule-based combination in this prototype)

---

## Installation
```bash
pip install pandas numpy scikit-learn sentence-transformers networkx torch matplotlib
```

---

## Example (Prototype)
This repo includes a prototype pipeline that:
- generates a **simulated dataset** for ~50 students
- detects anomalies using Isolation Forest
- creates a similarity graph for collusion detection
- computes text similarity using transformer embeddings
- trains an autoencoder-like model (VAE concept)
- flags identity anomalies using typing variance
- outputs a final cheating category

---

## Output Categories
- **Normal**
- **Behavior Cheating**
- **Group Cheating**
- **Identity Cheating**

(Extendable to: Text Cheating / AI-generated suspicion, etc.)

---

## Visualizations
- **Bar chart** showing distribution of final flags
- **Collusion graph** showing student clusters (possible group cheating)

---

## Limitations
- Dataset is **simulated**, not real-world
- **False positives** may occur
- Keystroke signals are approximated
- Real-time deployment needs more infrastructure (streaming + monitoring)

---

## Future Scope
- Integrate **webcam/video-based monitoring**
- Real-time pipeline + dashboard
- Stronger stylometry and AI-generated answer detection
- Replace graph heuristic with a full **Graph Neural Network (GNN)**
- Deploy as a web application (Flask/FastAPI + frontend)

---

## Conclusion
This project demonstrates a **hybrid AI-based cheating detection system** for online exams.

By combining:
- Behavioral anomaly detection
- Graph-based collusion detection
- Text similarity detection
- Identity verification

…the system becomes more accurate and robust than single-model approaches.
