import { useState, useEffect } from "react";
import PredictionForm from "./components/PredictionForm";
import PredictionsTable from "./components/PredictionsTable";

// Read backend base URL from environment variable
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

function App() {
  // Form input states
  const [age, setAge] = useState("");
  const [bp, setBp] = useState("");
  const [sg, setSg] = useState("");
  const [al, setAl] = useState("");

  // Prediction result and error handling
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // History of recent predictions
  const [history, setHistory] = useState([]);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [historyError, setHistoryError] = useState(null);

  // Edit mode tracking
  const [isEditing, setIsEditing] = useState(false);
  const [editId, setEditId] = useState(null);

  // Fetch history once when component mounts
  useEffect(() => {
    fetchHistory();
  }, []);

  // Fetches most recent prediction records from backend
  const fetchHistory = async () => {
    setLoadingHistory(true);
    setHistoryError(null);
    try {
      const res = await fetch(`${API_BASE_URL}/api/recent`);
      if (!res.ok) throw new Error("Failed to fetch history");
      const data = await res.json();
      setHistory(data);
    } catch (err) {
      setHistoryError(err.message);
    } finally {
      setLoadingHistory(false);
    }
  };

  // Handles submitting the form (predict or edit)
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    // Build request body from form values
    const payload = {
      age: Number(age),
      bp: Number(bp),
      sg: Number(sg),
      al: Number(al),
    };

    // Choose endpoint and HTTP method depending on mode
    const url = isEditing
      ? `${API_BASE_URL}/api/patient/${editId}`
      : `${API_BASE_URL}/api/predict`;
    // Put to edit, post to create
    const method = isEditing ? "PUT" : "POST";

    try {
      const response = await fetch(url, {
        method,
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) throw new Error("Failed to save data");

      const data = await response.json();

      // Show result as human-readable message
      setResult(data.prediction ? "High Risk" : "Low Risk");

      // Reset form and states
      setIsEditing(false);
      setEditId(null);
      setAge("");
      setBp("");
      setSg("");
      setAl("");

      // Refresh recent predictions
      fetchHistory();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Called when user clicks edit on a record
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

  // Called when user clicks delete on a record
  const handleDelete = async (id) => {
    if (!window.confirm("Are you sure you want to delete this record?")) return;
    try {
      const res = await fetch(`${API_BASE_URL}/api/patient/${id}`, {
        method: "DELETE",
      });
      if (!res.ok) throw new Error("Failed to delete");

      // Called when user clicks delete on a record
      await fetchHistory();
    } catch (err) {
      alert(err.message);
    }
  };

  return (
    <div className="app">
      <div className="predictions-container">
        <PredictionForm
          age={age}
          bp={bp}
          sg={sg}
          al={al}
          setAge={setAge}
          setBp={setBp}
          setSg={setSg}
          setAl={setAl}
          loading={loading}
          isEditing={isEditing}
          handleSubmit={handleSubmit}
          result={result}
          error={error}
        />
        <PredictionsTable
          history={history}
          loadingHistory={loadingHistory}
          historyError={historyError}
          handleEdit={handleEdit}
          handleDelete={handleDelete}
        />
      </div>
      <div className="responsiveness">
        <h1>Please use wider resolution</h1>
      </div>

    </div>
  );
}

export default App;
