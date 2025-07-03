import { useState, useEffect } from "react";
import PredictionForm from "./components/PredictionForm";
import PredictionsTable from "./components/PredictionsTable";

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

      if (!response.ok) throw new Error("Failed to save data");

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
      await fetchHistory();
    } catch (err) {
      alert(err.message);
    }
  };

  return (
    <div className="app">
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
  );
}

export default App;
