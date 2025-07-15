function PredictionsTable({
  history,
  loadingHistory,
  historyError,
  handleEdit,
  handleDelete,
}) {
  if (historyError)
    return <p style={{ color: "red" }}>Error: {historyError}</p>;
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
            <th>Date</th>
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
              <td>
                <div className="result-container">
                  <p className="result">
                    {patient.prediction ? "High Risk" : "Low Risk"}
                  </p>
                </div>
              </td>
              <td>{new Date(patient.createdAt).toLocaleDateString()}</td>
              <td>
                <div className="button-container">
                  <button onClick={() => handleEdit(patient)}>Edit</button>
                  <button onClick={() => handleDelete(patient.id)}>
                    Delete
                  </button>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default PredictionsTable;
