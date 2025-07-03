import { useState, useEffect } from "react";

function App() {
  const [age, setAge] = useState("");
  const [bp, setBp] = useState("");
  const [sg, setSg] = useState("");
  const [al, setAl] = useState("");

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [history, setHistory] = useState([]);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [historyError, setHistoryError] = useState(null);

  // Edit mode state
  const [isEditing, setIsEditing] = useState(false);
  const [editId, setEditId] = useState(null);

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    setLoadingHistory(true);
    setHistoryError(null);
    try {
      const res = await fetch("http://localhost:8000/api/recent/");
      if (!res.ok) throw new Error("Failed to fetch history");
      const data = await res.json();
      setHistory(data);
    } catch (err) {
      setHistoryError(err.message);
    } finally {
      setLoadingHistory(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    const payload = {
      age: Number(age),
      bp: Number(bp),
      sg: Number(sg),
      al: Number(al),
    };

    const url = isEditing
      ? `http://localhost:8000/api/patient/${editId}/`
      : "http://localhost:8000/api/predict/";

    const method = isEditing ? "PUT" : "POST";

    try {
      const response = await fetch(url, {
        method,
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error("Failed to save data");
      }

      const data = await response.json();
      setResult(data.ckd_risk ? "High CKD Risk" : "Low CKD Risk");
      setIsEditing(false);
      setEditId(null);
      setAge("");
      setBp("");
      setSg("");
      setAl("");
      fetchHistory();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleEdit = (patient) => {
    setAge(patient.age);
    setBp(patient.bp);
    setSg(patient.sg);
    setAl(patient.al);
    setIsEditing(true);
    setEditId(patient.id);
    setResult(null);
    setError(null);
  };

  const handleDelete = async (id) => {
    if (!window.confirm("Are you sure you want to delete this record?")) return;

    try {
      const res = await fetch(
        `http://localhost:8000/api/patient/${id}/delete/`,
        {
          method: "DELETE",
        }
      );

      if (!res.ok) throw new Error("Failed to delete");

      // Refresh history after delete
      await fetchHistory();
    } catch (err) {
      alert(err.message);
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
          />
        </label>
        <label>
          Blood Pressure:
          <input
            type="number"
            value={bp}
            onChange={(e) => setBp(e.target.value)}
          />
        </label>
        <label>
          Specific Gravity:
          <input
            type="number"
            step="0.01"
            value={sg}
            onChange={(e) => setSg(e.target.value)}
          />
        </label>
        <label>
          Albumin:
          <input
            type="number"
            value={al}
            onChange={(e) => setAl(e.target.value)}
          />
        </label>
        <button type="submit" disabled={loading}>
          {loading
            ? isEditing
              ? "Saving..."
              : "Predicting..."
            : isEditing
            ? "Update Entry"
            : "Predict CKD Risk"}
        </button>
      </form>

      {result && (
        <p>
          <strong>Result:</strong> {result}
        </p>
      )}
      {error && <p style={{ color: "red" }}>Error: {error}</p>}

      <h2>Recent Predictions</h2>
      {loadingHistory && <p>Loading history...</p>}
      {historyError && <p style={{ color: "red" }}>Error: {historyError}</p>}
      {history.length === 0 && !loadingHistory && <p>No recent data.</p>}

      <table>
        <thead>
          <tr>
            <th>Age</th>
            <th>BP</th>
            <th>SG</th>
            <th>AL</th>
            <th>Prediction</th>
            <th>Timestamp</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {history.map((patient) => (
            <tr key={patient.id}>
              <td>{patient.age}</td>
              <td>{patient.bp}</td>
              <td>{patient.sg}</td>
              <td>{patient.al}</td>
              <td>{patient.prediction ? "High Risk" : "Low Risk"}</td>
              <td>{new Date(patient.created_at).toLocaleString()}</td>
              <td>
                <button onClick={() => handleEdit(patient)}>Edit</button>
                <button onClick={() => handleDelete(patient.id)}>Delete</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default App;
