import { useState } from "react";

function App() {
  // State for form inputs
  const [age, setAge] = useState("");
  const [bp, setBp] = useState("");
  const [sg, setSg] = useState("");
  const [al, setAl] = useState("");

  // State for prediction result and loading/error
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Handle form submit
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch("http://localhost:8000/api/predict/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          age: Number(age),
          bp: Number(bp),
          sg: Number(sg),
          al: Number(al),
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to get prediction");
      }

      const data = await response.json();
      setResult(data.ckd_risk ? "High CKD Risk" : "Low CKD Risk");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h1>CKDPredict</h1>

      <form onSubmit={handleSubmit}>
        <label>
          Age:
          <input
            type="number"
            value={age}
            onChange={(e) => setAge(e.target.value)}
            required
          />
        </label>
        <label>
          Blood Pressure:
          <input
            type="number"
            value={bp}
            onChange={(e) => setBp(e.target.value)}
            required
          />
        </label>
        <label>
          Specific Gravity:
          <input
            type="number"
            step="0.01"
            value={sg}
            onChange={(e) => setSg(e.target.value)}
            required
          />
        </label>
        <label>
          Albumin:
          <input
            type="number"
            value={al}
            onChange={(e) => setAl(e.target.value)}
            required
          />
        </label>
        <button type="submit" disabled={loading}>
          {loading ? "Predicting..." : "Predict CKD Risk"}
        </button>
      </form>

      {result && (
        <p>
          <strong>Result:</strong> {result}
        </p>
      )}
      {error && <p style={{ color: "red" }}>Error: {error}</p>}
    </div>
  );
}

export default App;
