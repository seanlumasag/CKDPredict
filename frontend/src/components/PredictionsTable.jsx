function PredictionsTable({
  history,
  loadingHistory,
  historyError,
  handleEdit,
  handleDelete,
}) {
  if (loadingHistory) return <p>Loading history...</p>;
  if (historyError)
    return <p style={{ color: "red" }}>Error: {historyError}</p>;
  if (history.length === 0) return <p>No recent data.</p>;

  return (
    <div className="predictions-table">
      <h2>Past Predictions</h2>
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
              <td>{new Date(patient.created_at).toLocaleDateString()}</td>
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

export default PredictionsTable;
